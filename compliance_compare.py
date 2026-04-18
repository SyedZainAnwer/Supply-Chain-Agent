"""
Agnes Compliance Comparison Tool
--------------------------------
Pipeline:
1. Benchmark: Extract certs from current product.
2. Discovery: Search for target product AND site-wide regulatory/sustainability pages.
3. Extraction: Scrape and extract attributes from all combined sources.
4. Comparison: Use LLM to reason across product-specific and facility-wide compliance.
"""

from __future__ import annotations

import os
import json
from typing import List, Optional

import instructor
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

# GEMINI_MODEL = "gemini-1.5-flash" # Use Flash for speed in Hackathon
GEMINI_MODEL = "gemini-2.5-flash"  # Use Pro for better reasoning on compliance
MAX_EXTRACT_CHARS = 25_000

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ProductAttributes(BaseModel):
    """Certifications, quality claims, and price for a single product."""

    product_name: Optional[str] = Field(
        default=None, description="Canonical product / SKU name as seen on the page."
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Normalized certificate + quality-claim names (e.g. Kosher, Halal, ISO 9001)",
    )
    price: Optional[str] = Field(
        default=None,
        description="Price as stated on the page, verbatim with currency/unit. None if not listed.",
    )


class ComparisonResult(BaseModel):
    """Final structured output for Agnes reasoning."""

    benchmark_attributes: ProductAttributes
    target_attributes: ProductAttributes
    match_score: float = Field(ge=0.0, le=1.0)
    missing_claims: list[str]
    extra_claims: list[str]
    price_comparison: str
    reasoning_summary: str
    source_urls: list[str] = Field(default_factory=list)


class _Candidate(BaseModel):
    url: str
    title: Optional[str] = None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def _llm_client() -> instructor.Instructor:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return instructor.from_genai(genai.Client(api_key=api_key))


def _tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    return TavilyClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Phase 1 — Benchmark
# ---------------------------------------------------------------------------


def extract_benchmark_attributes(
    llm: instructor.Instructor,
    *,
    current_product_url: Optional[str] = None,
    current_product_text: Optional[str] = None,
    tavily: Optional[TavilyClient] = None,
) -> ProductAttributes:
    source_text = current_product_text or ""
    if current_product_url and not current_product_text:
        resp = (tavily or _tavily_client()).extract(
            urls=[current_product_url], extract_depth="advanced", format="markdown"
        )
        results = resp.get("results") or []
        source_text = (results[0].get("raw_content") or "")[:MAX_EXTRACT_CHARS]

    return llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=ProductAttributes,
        messages=[
            {
                "role": "system",
                "content": "Extract ALL certifications and quality claims. Normalize names (e.g., 'Non-GMO').",
            },
            {"role": "user", "content": source_text},
        ],
    )


# ---------------------------------------------------------------------------
# Phase 2 — Discovery
# ---------------------------------------------------------------------------


def _normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    for prefix in ("https://", "http://", "www."):
        if d.startswith(prefix):
            d = d[len(prefix) :]
    return d.rstrip("/")


def discover_regulatory_pages(
    tavily: TavilyClient, domain: str, supplier_name: str
) -> list[_Candidate]:
    """Hunts for 'Sustainability' or 'Certificates' pages."""
    query = f"site:{domain} ({supplier_name} certifications OR quality certificates OR kosher OR sustainability)"
    res = tavily.search(query=query, max_results=3, search_depth="advanced")
    return [
        _Candidate(url=h["url"], title=h.get("title")) for h in res.get("results", [])
    ]


def discover_target_candidates(
    tavily: TavilyClient,
    target_supplier_name: str,
    target_supplier_domain: str,
    target_product_name: str,
) -> list[_Candidate]:
    domain = _normalize_domain(target_supplier_domain)
    query = f'site:{domain} "{target_product_name}" (specification OR datasheet OR compliance)'
    res = tavily.search(query=query, max_results=4, search_depth="advanced")
    return [
        _Candidate(url=h["url"], title=h.get("title"))
        for h in res.get("results", [])
        if h.get("url")
    ]


# ---------------------------------------------------------------------------
# Phase 3 — Extraction
# ---------------------------------------------------------------------------


def extract_target_attributes(
    llm: instructor.Instructor,
    tavily: TavilyClient,
    candidates: list[_Candidate],
) -> tuple[List[ProductAttributes], list[str]]:
    if not candidates:
        return [], []

    urls = [c.url for c in candidates[:6]]
    resp = tavily.extract(urls=urls, extract_depth="advanced", format="markdown")
    results = resp.get("results") or []

    merged_content = "\n\n---\n\n".join(
        f"# Source: {r.get('url')}\n\n{r.get('raw_content') or ''}"[:MAX_EXTRACT_CHARS]
        for r in results
    )

    # We use List[ProductAttributes] because one page might list multiple variants
    attrs_list = llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=List[ProductAttributes],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance analyst. You are reading product data AND corporate regulatory pages. "
                    "Rule: If a general sustainability/quality page lists certs for the entire facility or product category "
                    "(e.g., 'All our cellulose products are Halal'), apply those certs to the product. "
                    "Extract the product name, its price if listed, and all relevant certifications."
                ),
            },
            {"role": "user", "content": merged_content},
        ],
    )
    return attrs_list, [r.get("url") for r in results if r.get("url")]


# ---------------------------------------------------------------------------
# Phase 4 — Comparison
# ---------------------------------------------------------------------------


def _compare_attributes(
    llm: instructor.Instructor,
    benchmark: ProductAttributes,
    target: ProductAttributes,
    source_urls: list[str],
) -> ComparisonResult:
    return llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=ComparisonResult,
        messages=[
            {
                "role": "system",
                "content": (
                    "Compare BENCHMARK vs TARGET. Normalize synonyms ('BSE-Free' == 'BSE Free'). "
                    "match_score is fraction of benchmark certs found in target. "
                    "Provide a reasoning_summary explaining the verdict clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"BENCHMARK:\n{benchmark.model_dump_json(indent=2)}\n\n"
                    f"TARGET:\n{target.model_dump_json(indent=2)}\n\n"
                    f"SOURCES:\n{source_urls}"
                ),
            },
        ],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compare_compliance(
    *,
    target_supplier_name: str,
    target_supplier_domain: str,
    target_product_name: str,
    current_product_url: Optional[str] = None,
    current_product_text: Optional[str] = None,
) -> ComparisonResult:
    llm = _llm_client()
    tavily = _tavily_client()

    # 1. Benchmark
    benchmark = extract_benchmark_attributes(
        llm,
        current_product_url=current_product_url,
        current_product_text=current_product_text,
        tavily=tavily,
    )

    # 2. Discovery (Product + Sustainability Hubs)
    domain = _normalize_domain(target_supplier_domain)
    prod_cands = discover_target_candidates(
        tavily, target_supplier_name, domain, target_product_name
    )
    reg_cands = discover_regulatory_pages(tavily, domain, target_supplier_name)

    all_candidates = prod_cands + reg_cands

    # 3. Extraction
    target_list, extracted_urls = extract_target_attributes(llm, tavily, all_candidates)
    if not target_list:
        raise RuntimeError(
            f"No product data found for {target_product_name} at {target_supplier_name}"
        )

    # Pick the best match from the list based on name similarity
    target = target_list[0]
    for item in target_list:
        if target_product_name.lower() in (item.product_name or "").lower():
            target = item
            break

    # 4. Final Comparison
    all_source_urls = list(
        set(extracted_urls + ([current_product_url] if current_product_url else []))
    )
    return _compare_attributes(llm, benchmark, target, all_source_urls)


# ---------------------------------------------------------------------------
# Demo Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        print(f"--- Agnes is investigating Ashland compliance for Cellulose ---")
        result = compare_compliance(
            target_supplier_name="ashland",
            target_supplier_domain="ashland.com",
            target_product_name="cellulose",
            current_product_text="Standard: Cellulose. Required Certs: Kosher, Halal, ISO 9001, Non-GMO.",
        )
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Error: {e}")
