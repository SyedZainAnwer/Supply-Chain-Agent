
"""
Agnes Compliance Tool: Final Unified Demo
-----------------------------------------
Workflow:
1. Profiles any supplier (Current or New) into a 'Passport' object.
2. Compares any two Passports using LLM semantic reasoning.
3. Ranks multiple candidates against a benchmark using a Decision Matrix.
"""

from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import instructor
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

# --- Configurations ---
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # Use Pro for better reasoning on compliance
MAX_EXTRACT_CHARS = 25_000
DATA_PATH = Path(__file__).parent / "compliance_json_data.json"

# --- Pydantic Models ---

class ProductProfile(BaseModel):
    """The 'Passport' for a product at a specific supplier."""
    supplier_name: str
    product_name: Optional[str] = None
    certifications: List[str] = Field(default_factory=list)
    price: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)

class ComparisonResult(BaseModel):
    """Semantic comparison output."""
    match_score: float = Field(description="Percentage (0.0 to 1.0) of benchmark certs met.")
    missing_claims: List[str]
    extra_claims: List[str]
    reasoning: str

# --- Core Logic Functions ---

def get_supplier_profile(
    supplier_name: str, 
    supplier_domain: str, 
    product_name: str
) -> ProductProfile:
    """Live web-extraction of a supplier's product specs and regulatory posture."""
    llm = instructor.from_genai(genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    print(f"🔍 Agnes is profiling {product_name} at {supplier_name}...")

    domain = supplier_domain.replace("https://", "").replace("http://", "").split('/')[0]
    
    # Discovery: Target product specs + Supplier quality hub
    queries = [
        f'site:{domain} "{product_name}" (specification OR datasheet OR "technical data")',
        f'site:{domain} {supplier_name} (certifications OR "regulatory compliance" OR "quality certificates")'
    ]
    
    all_urls = []
    for q in queries:
        res = tavily.search(query=q, max_results=3)
        all_urls.extend([h['url'] for h in res['results']])
    
    # Extraction
    extract_res = tavily.extract(urls=list(set(all_urls))[:5], extract_depth="advanced")
    content = "\n\n".join([r['raw_content'] for r in extract_res['results'] if r['raw_content']])

    # Structuring
    profiles = llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=List[ProductProfile],
        messages=[
            {"role": "system", "content": (
                f"Identify '{product_name}' attributes. "
                "CRITICAL: If a page lists global site certs (e.g. ISO 9001, Kosher), "
                "assume they apply to this product unless stated otherwise."
            )},
            {"role": "user", "content": content[:MAX_EXTRACT_CHARS]}
        ]
    )
    
    # Validation
    best_match = next((p for p in profiles if product_name.lower() in (p.product_name or "").lower()), None)
    if not best_match:
        return ProductProfile(supplier_name=supplier_name, product_name=product_name, certifications=[])
    
    best_match.supplier_name = supplier_name
    best_match.source_urls = list(set(all_urls))
    return best_match

def compare_profiles(current: ProductProfile, target: ProductProfile) -> ComparisonResult:
    """LLM-backed comparison between any two Profile objects."""
    llm = instructor.from_genai(genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))
    
    return llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=ComparisonResult,
        messages=[
            {"role": "system", "content": (
                "You are a compliance auditor. Compare the TARGET against the BENCHMARK. "
                "Normalize synonyms (e.g., 'BSE/TSE Free' == 'BSE Free'). "
                "The match_score is the fraction of BENCHMARK certifications found in TARGET."
            )},
            {"role": "user", "content": f"BENCHMARK: {current.model_dump_json()}\n\nTARGET: {target.model_dump_json()}"}
        ]
    )

# --- Ranking & Utility Functions ---

def parse_price(price_str: Optional[str]) -> float:
    """Helper to convert '$45.00/kg' strings to floats for sorting."""
    if not price_str: return float('inf')
    match = re.search(r"(\d+[\d,.]*)", price_str)
    return float(match.group(1).replace(",", "")) if match else float('inf')

def build_decision_matrix(
    benchmark: ProductProfile,
    candidates: List[ProductProfile],
) -> List[dict]:
    """Runs benchmark-vs-candidate comparisons and returns a ranked list of rows.

    Pure data output (no printing) so Streamlit / other UIs can consume it directly.
    """
    rows = []
    for cand in candidates:
        report = compare_profiles(benchmark, cand)
        price_val = parse_price(cand.price)
        # Selection Index: (Match Score^2 / Price) * 100 to reward high compliance
        idx = ((report.match_score ** 2) / max(price_val, 1)) * 100
        rows.append({
            "supplier": cand.supplier_name,
            "product": cand.product_name,
            "match_score": report.match_score,
            "missing_claims": report.missing_claims,
            "extra_claims": report.extra_claims,
            "certifications": cand.certifications,
            "price": cand.price,
            "price_value": price_val,
            "selection_index": idx,
            "reasoning": report.reasoning,
        })

    ranked = sorted(rows, key=lambda r: (-r["match_score"], -r["selection_index"]))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
        if i == 1 and row["match_score"] >= 0.8:
            row["verdict"] = "OPTIMAL"
        elif row["match_score"] < 0.5:
            row["verdict"] = "RISK"
        else:
            row["verdict"] = "CONSIDER"
    return ranked


def rank_suppliers(benchmark: ProductProfile, candidates: List[ProductProfile]):
    """CLI wrapper: builds the decision matrix and prints it as a table."""
    ranked = build_decision_matrix(benchmark, candidates)

    print(f"\n{'='*85}")
    print(f"🎯 AGNES DECISION MATRIX: {benchmark.product_name.upper()}")
    print(f"{'='*85}\n")
    print(f"{'Rank':<5} | {'Supplier':<22} | {'Score':<6} | {'Price':<12} | {'Verdict'}")
    print("-" * 85)

    icons = {"OPTIMAL": "✅ OPTIMAL", "RISK": "⚠️ RISK", "CONSIDER": "CONSIDER"}
    for row in ranked:
        print(f"{row['rank']:<5} | {row['supplier'][:22]:<22} | "
              f"{row['match_score']:<6.2f} | {row['price'] or 'N/A':<12} | "
              f"{icons[row['verdict']]}")
        print(f"      Reasoning: {row['reasoning']}")
        print("-" * 85)

    return ranked


# --- Dummy-Data Loader ---

def load_compliance_data(path: Path = DATA_PATH) -> Dict[str, Dict[str, object]]:
    """Loads the dummy compliance JSON into `{product_name: {benchmark, candidates}}`.

    Used when the SQLite Master Sourcing View lacks real supplier compliance data.
    """
    raw = json.loads(Path(path).read_text())
    return {
        product: {
            "benchmark": ProductProfile(**entry["benchmark"]),
            "candidates": [ProductProfile(**c) for c in entry["candidates"]],
        }
        for product, entry in raw.items()
    }


# --- Main Execution ---

if __name__ == "__main__":
    data = load_compliance_data()
    for product_name, entry in data.items():
        rank_suppliers(entry["benchmark"], entry["candidates"])