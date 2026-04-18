"""Agnes — Supply Chain Agent with long-context Master Sourcing View.

Pulls all relational tables from SQLite, joins them into a single denormalized
view (one row per BOM component, enriched with product category, owning company
and supplier list), and pins that view into the Gemini system instruction so
the 1M-token window handles retrieval directly — no keyword search required.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DB_PATH = Path("./db/db.sqlite")
MODEL_NAME = "gemini-2.5-pro"

SYSTEM_INSTRUCTION_TEMPLATE = """You are Agnes, a Senior Supply Chain Manager.

You have direct, authoritative access to the MASTER SOURCING VIEW below. Every
row is one (produced product, consumed component) relationship enriched with
category (finished-good / raw-material), owning company, the full list of
qualified suppliers for the consumed component, and a **BOM_Frequency** score.

## Key metric — BOM_Frequency

`BOM_Frequency` = number of distinct ProducedSKUs that consume a given
ConsumedSKU. It is the substitute for missing quantity data and is your primary
criticality signal.

- **High BOM_Frequency** → the ConsumedSKU is a **Common Part** supporting many
  production lines. These are your highest priority for sourcing stability and
  consolidation. Treat them as strategic.
- **High BOM_Frequency + multiple entries in `ConsumedSuppliers`** → **RED FLAG
  for supplier fragmentation**. A critical common part sourced from many
  suppliers is a consolidation opportunity and a governance risk.
- **High BOM_Frequency + a single entry in `ConsumedSuppliers`** → single-source
  risk on a critical part; recommend qualifying a second source.
- **Low BOM_Frequency** → niche / one-off component; lower consolidation value.

## Reasoning Protocol — always respond in this exact order

1. **LIST — SKUs in scope.** Enumerate the specific SKUs from the MASTER SOURCING
   VIEW that match the user's question. Quote them verbatim with their
   BOM_Frequency. If none match, say so. When the user asks for "critical",
   "high-volume", "important", or "strategic" materials, rank by BOM_Frequency
   descending.
2. **ANALYZE — why they are candidates.** For each SKU, explain the signal
   using the data: BOM_Frequency, shared category, supplier fragmentation,
   overlapping BOMs, single-source risk, shared owning company. Flag the
   red-flag pattern above explicitly when it occurs.
3. **RECOMMEND — concrete action.** Close with a specific sourcing consolidation
   or functional-substitute recommendation. Call out trade-offs.

## Data-source priority

- For any SKU-, supplier-, or company-specific answer: use ONLY the MASTER
  SOURCING VIEW. Never invent SKUs, suppliers, or companies.
- For questions outside the view (market trends, regulations, vendors not
  listed, general knowledge): explicitly say the local data does not cover it,
  then invoke the Google Search tool if it is available.

## MASTER SOURCING VIEW (authoritative internal data)

{view_markdown}
"""


def _api_key() -> str | None:
    """Resolve the Gemini API key: prefer st.secrets (prod), fall back to .env."""
    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GOOGLE_API_KEY")


@st.cache_data(show_spinner="Building Master Sourcing View…")
def load_master_view(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        product = pd.read_sql_query("SELECT * FROM Product", conn)
        company = pd.read_sql_query("SELECT * FROM Company", conn)
        supplier = pd.read_sql_query("SELECT * FROM Supplier", conn)
        bom = pd.read_sql_query("SELECT * FROM BOM", conn)
        bom_comp = pd.read_sql_query("SELECT * FROM BOM_Component", conn)
        supp_prod = pd.read_sql_query("SELECT * FROM Supplier_Product", conn)

    suppliers_by_product = (
        supp_prod
        .merge(supplier.rename(columns={"Id": "SupplierId", "Name": "SupplierName"}),
               on="SupplierId")
        .groupby("ProductId")["SupplierName"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .rename("Suppliers")
        .reset_index()
    )

    product_enriched = (
        product
        .merge(company.rename(columns={"Id": "CompanyId", "Name": "CompanyName"}),
               on="CompanyId", how="left")
        .merge(suppliers_by_product, left_on="Id", right_on="ProductId", how="left")
        .drop(columns=["ProductId"])
    )
    product_enriched["Suppliers"] = product_enriched["Suppliers"].fillna("—")

    produced = product_enriched.rename(columns={
        "Id": "ProducedProductId",
        "SKU": "ProducedSKU",
        "Type": "ProducedCategory",
        "CompanyName": "ProducedCompany",
        "Suppliers": "ProducedSuppliers",
    })[["ProducedProductId", "ProducedSKU", "ProducedCategory",
        "ProducedCompany", "ProducedSuppliers"]]

    consumed = product_enriched.rename(columns={
        "Id": "ConsumedProductId",
        "SKU": "ConsumedSKU",
        "Type": "ConsumedCategory",
        "CompanyName": "ConsumedCompany",
        "Suppliers": "ConsumedSuppliers",
    })[["ConsumedProductId", "ConsumedSKU", "ConsumedCategory",
        "ConsumedCompany", "ConsumedSuppliers"]]

    bom_joined = bom.merge(bom_comp, left_on="Id", right_on="BOMId")

    view = (
        bom_joined
        .merge(produced, on="ProducedProductId", how="left")
        .merge(consumed, on="ConsumedProductId", how="left")
    )

    view["BOM_Frequency"] = view.groupby("ConsumedSKU")["ProducedSKU"].transform("nunique")

    return view[[
        "ProducedSKU", "ProducedCategory", "ProducedCompany",
        "ConsumedSKU", "ConsumedCategory", "ConsumedCompany",
        "ConsumedSuppliers", "BOM_Frequency",
    ]].sort_values(
        ["BOM_Frequency", "ProducedCompany", "ProducedSKU", "ConsumedSKU"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def view_to_markdown(view: pd.DataFrame) -> str:
    return view.to_markdown(index=False)


@st.cache_resource(show_spinner=False)
def get_client() -> genai.Client | None:
    key = _api_key()
    if not key:
        return None
    return genai.Client(api_key=key)


def _history_to_contents(history: list[dict]) -> list[types.Content]:
    return [
        types.Content(
            role="user" if turn["role"] == "user" else "model",
            parts=[types.Part.from_text(text=turn["content"])],
        )
        for turn in history
    ]


def _generate(
    client: genai.Client,
    history: list[dict],
    system_instruction: str,
    use_web: bool,
) -> str:
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_web else None
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
    )
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=_history_to_contents(history),
            config=config,
        )
        return resp.text or "(empty response)"
    except Exception as e:
        return f"Gemini call failed: {e}"


def main() -> None:
    st.set_page_config(page_title="Agnes — Supply Chain Agent",
                       page_icon="📦", layout="wide")
    st.title("📦 Agnes — Supply Chain Agent")
    st.caption(f"{MODEL_NAME} · long-context Master Sourcing View")

    with st.sidebar:
        st.subheader("Configuration")
        if _api_key():
            st.success("GOOGLE_API_KEY loaded (st.secrets / .env)")
        else:
            st.error("GOOGLE_API_KEY missing. Add to `.env` or `.streamlit/secrets.toml`.")
        use_web = st.toggle(
            "Enable Google Search tool",
            value=False,
            help="Grounds answers with live Google Search. Local data still wins for SKU specifics.",
        )
        if st.button("Reset chat"):
            st.session_state.pop("history", None)
            st.rerun()

    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        return

    view = load_master_view(str(DB_PATH))
    view_md = view_to_markdown(view)
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(view_markdown=view_md)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BOM-component rows", f"{len(view):,}")
    col2.metric("Distinct produced SKUs", f"{view['ProducedSKU'].nunique():,}")
    col3.metric("Distinct consumed SKUs", f"{view['ConsumedSKU'].nunique():,}")
    avg_components = view.groupby("ProducedSKU").size().mean()
    col4.metric("Avg. Components per BOM", f"{avg_components:.1f}")

    with st.expander("Preview Master Sourcing View", expanded=False):
        st.dataframe(view.head(200), use_container_width=True)
        st.caption(f"Context payload ≈ {len(view_md):,} characters "
                   f"(~{len(view_md)//4:,} tokens) pinned to system instruction.")

    client = get_client()
    if client is None:
        return

    history = st.session_state.setdefault("history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input("Ask Agnes about consolidation, substitutes, supplier fragmentation…")
    if not prompt:
        return

    history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        spinner_msg = "Agnes is reasoning over the Master View + web…" if use_web \
            else "Agnes is reasoning over the Master View…"
        with st.spinner(spinner_msg):
            answer = _generate(client, history, system_instruction, use_web)
        st.markdown(answer)
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
