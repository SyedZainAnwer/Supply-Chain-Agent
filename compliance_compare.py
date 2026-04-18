"""Dynamic Compliance Comparison tool for the Agnes agent.

Given a Current (benchmark) product and a Target supplier, determine whether
the Target's equivalent product meets the Current product's quality /
certification standards.

Pipeline
--------
1. Benchmark — LLM extracts quality claims + certificates from the Current
   product's URL or pasted text.
2. Discovery — Tavily `site:` search on the Target supplier's domain, with a
   broader "[Supplier Name] [Product Name] technical specifications" fallback
   when the scoped search returns zero hits.
3. Extraction — Tavily `extract` pulls Markdown for the best candidate URL;
   the LLM extracts certifications + price from it.
4. Comparison — LLM aligns the two attribute sets and emits match_score,
   missing/extra claims, and a price delta summary.

Structured outputs use `instructor` against the existing Gemini client so the
tool stays consistent with the rest of the project.
"""

from __future__ import annotations

import os
from typing import Optional

import instructor
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

GEMINI_MODEL = "gemini-2.5-pro"
MAX_EXTRACT_CHARS = 20_000


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
        description="Normalized certificate + quality-claim names (e.g. 'Kosher', "
        "'Halal', 'BSE Free', 'Non-GMO', 'ISO 22000', 'FSSC 22000', "
        "'Organic (EU)', 'Allergen Free').",
    )
    price: Optional[str] = Field(
        default=None,
        description="Price as stated on the page, verbatim with currency/unit "
        "(e.g. '€12.50 / kg'). None if not listed.",
    )


class ComparisonResult(BaseModel):
    """Final structured output returned by `compare_compliance`."""

    benchmark_attributes: ProductAttributes
    target_attributes: ProductAttributes
    match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of benchmark certificates present in the target.",
    )
    missing_claims: list[str] = Field(
        description="Benchmark certificates absent from the target."
    )
    extra_claims: list[str] = Field(
        description="Target certificates not present in the benchmark (value add)."
    )
    price_comparison: str = Field(
        description="Plain-language summary of the price delta, or a note that "
        "one / both prices were unavailable."
    )
    reasoning_summary: str = Field(
        description="2-4 sentence explanation of the match verdict."
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="All URLs consulted during discovery + extraction.",
    )


class _Candidate(BaseModel):
    url: str
    title: Optional[str] = None
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def _llm_client() -> instructor.Instructor:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment / .env")
    return instructor.from_genai(genai.Client(api_key=api_key))


def _tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set in environment / .env")
    return TavilyClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Phase 1 — Benchmark
# ---------------------------------------------------------------------------


def _fetch_benchmark_markdown(tavily: TavilyClient, url: str) -> str:
    resp = tavily.extract(urls=[url], extract_depth="advanced", format="markdown")
    results = resp.get("results") or []
    if not results:
        raise RuntimeError(f"Tavily extract returned no content for {url}")
    return (results[0].get("raw_content") or "")[:MAX_EXTRACT_CHARS]


def extract_benchmark_attributes(
    llm: instructor.Instructor,
    *,
    current_product_url: Optional[str] = None,
    current_product_text: Optional[str] = None,
    tavily: Optional[TavilyClient] = None,
) -> ProductAttributes:
    if not current_product_url and not current_product_text:
        raise ValueError("Provide either current_product_url or current_product_text.")

    source_text = current_product_text or ""
    if current_product_url and not current_product_text:
        source_text = _fetch_benchmark_markdown(tavily or _tavily_client(), current_product_url)

    return llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=ProductAttributes,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance analyst. From the supplied product "
                    "datasheet, extract ALL certifications and quality claims "
                    "(e.g. 'BSE Free', 'Kosher', 'Halal', 'ISO 22000', "
                    "'Non-GMO', 'Allergen Free', 'Organic (EU)'). Normalize "
                    "abbreviations and drop marketing fluff. Capture price only "
                    "if explicitly stated."
                ),
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


def discover_target_candidates(
    tavily: TavilyClient,
    *,
    target_supplier_name: str,
    target_supplier_domain: str,
    target_product_name: str,
    max_results: int = 5,
) -> list[_Candidate]:
    domain = _normalize_domain(target_supplier_domain)
    scoped_query = f'site:{domain} "{target_product_name}" specification OR certificate OR datasheet'
    scoped = tavily.search(
        query=scoped_query,
        max_results=max_results,
        search_depth="advanced",
    )
    hits = scoped.get("results") or []

    if not hits:
        broad_query = f"{target_supplier_name} {target_product_name} technical specifications"
        broad = tavily.search(
            query=broad_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=[domain],
        )
        hits = broad.get("results") or []

        if not hits:
            broad = tavily.search(
                query=broad_query,
                max_results=max_results,
                search_depth="advanced",
            )
            hits = broad.get("results") or []

    return [
        _Candidate(url=h["url"], title=h.get("title"), score=h.get("score"))
        for h in hits
        if h.get("url")
    ]


# ---------------------------------------------------------------------------
# Phase 3 — Extraction
# ---------------------------------------------------------------------------


def extract_target_attributes(
    llm: instructor.Instructor,
    tavily: TavilyClient,
    candidates: list[_Candidate],
) -> tuple[ProductAttributes, list[str]]:
    if not candidates:
        return ProductAttributes(), []

    urls = [c.url for c in candidates[:3]]
    resp = tavily.extract(urls=urls, extract_depth="advanced", format="markdown")
    results = resp.get("results") or []
    if not results:
        return ProductAttributes(), urls

    merged = "\n\n---\n\n".join(
        f"# Source: {r.get('url')}\n\n{(r.get('raw_content') or '')[:MAX_EXTRACT_CHARS]}"
        for r in results
    )[: MAX_EXTRACT_CHARS * 2]

    attrs = llm.chat.completions.create(
        model=GEMINI_MODEL,
        response_model=ProductAttributes,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance analyst. From the supplied supplier "
                    "pages, extract the target product's certifications, "
                    "quality claims, and price. Use the same normalized "
                    "vocabulary you would use for a benchmark datasheet so the "
                    "two lists are directly comparable."
                ),
            },
            {"role": "user", "content": merged},
        ],
    )
    return attrs, [r.get("url") for r in results if r.get("url")]


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
                    "Compare two product attribute sets. Treat synonyms and "
                    "near-equivalents as matches (e.g. 'BSE-Free' == 'BSE "
                    "Free', 'ISO22000' == 'ISO 22000'). Compute match_score as "
                    "the share of benchmark certificates that appear in the "
                    "target (0.0-1.0). List benchmark items missing from the "
                    "target as missing_claims; list target-only items as "
                    "extra_claims. In price_comparison, summarize the delta in "
                    "plain language; if either price is missing, say so. "
                    "reasoning_summary: 2-4 sentences on the overall verdict."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"BENCHMARK:\n{benchmark.model_dump_json(indent=2)}\n\n"
                    f"TARGET:\n{target.model_dump_json(indent=2)}\n\n"
                    f"SOURCE URLS:\n{source_urls}"
                ),
            },
        ],
    )


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compare_compliance(
    *,
    target_supplier_name: str,
    target_supplier_domain: str,
    target_product_name: str,
    current_product_url: Optional[str] = None,
    current_product_text: Optional[str] = None,
) -> ComparisonResult:
    """Run the full Benchmark → Discovery → Extraction → Comparison pipeline."""
    llm = _llm_client()
    tavily = _tavily_client()

    benchmark = extract_benchmark_attributes(
        llm,
        current_product_url=current_product_url,
        current_product_text=current_product_text,
        tavily=tavily,
    )

    candidates = discover_target_candidates(
        tavily,
        target_supplier_name=target_supplier_name,
        target_supplier_domain=target_supplier_domain,
        target_product_name=target_product_name,
    )

    target, extracted_urls = extract_target_attributes(llm, tavily, candidates)

    source_urls: list[str] = []
    if current_product_url:
        source_urls.append(current_product_url)
    source_urls.extend(c.url for c in candidates)
    source_urls.extend(u for u in extracted_urls if u not in source_urls)

    result = _compare_attributes(llm, benchmark, target, source_urls)
    if not result.source_urls:
        result.source_urls = source_urls
    return result


if __name__ == "__main__":
    import json

    demo = compare_compliance(
        target_supplier_name="Roquette",
        target_supplier_domain="roquette.com",
        target_product_name="Pea Protein Isolate NUTRALYS S85F",
        current_product_text=(
            "Benchmark product: Pea Protein Isolate\n"
            "Certifications: Kosher, Halal, BSE Free, Non-GMO, ISO 22000\n"
            "Price: €9.80 / kg"
        ),
    )
    print(json.dumps(demo.model_dump(), indent=2, ensure_ascii=False))
