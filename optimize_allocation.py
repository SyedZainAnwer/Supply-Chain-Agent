"""
Agnes — Supplier Allocation Optimizer
-------------------------------------
Linear Programming model (PuLP / CBC) that converts a ranked compliance report
into an optimal procurement split across suppliers.

Decision variables
    x_i ∈ [0, max_allocation_cap]  for each supplier i   (volume fraction)

Objective
    minimize   Σ x_i · price_i

Constraints
    Σ x_i                       = 1.0                        (full demand)
    Σ x_i · compliance_score_i ≥ min_compliance_threshold    (risk floor)
    x_i                        ≤ max_allocation_cap          (diversification)

Returns a structured dict so a Streamlit UI can render it directly:

    {
        "status": "optimal" | "infeasible",
        "allocations": {"Supplier A": 0.70, "Supplier B": 0.30},
        "total_cost": 39.95,
        "weighted_compliance": 0.885,
        "constraints": {...},
        "message": "...",
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pulp import (
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
    value,
)


def _validate(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("ranked_results is empty — nothing to optimize.")
    for i, row in enumerate(rows):
        for key in ("supplier", "score", "val"):
            if key not in row:
                raise ValueError(f"ranked_results[{i}] missing required key '{key}'.")
        if row["val"] is None or float(row["val"]) <= 0:
            raise ValueError(
                f"ranked_results[{i}] ({row['supplier']!r}) has invalid price: {row['val']!r}"
            )
        if not (0.0 <= float(row["score"]) <= 1.0):
            raise ValueError(
                f"ranked_results[{i}] ({row['supplier']!r}) score must be in [0, 1]."
            )


def _infeasibility_hint(
    rows: List[Dict[str, Any]],
    min_compliance_threshold: float,
    max_allocation_cap: float,
) -> str:
    """Explain WHY the LP is infeasible — far more useful than a bare CBC status."""
    n = len(rows)
    if n * max_allocation_cap < 1.0 - 1e-9:
        return (
            f"Need at least ⌈1 / {max_allocation_cap:.2f}⌉ = "
            f"{int((1.0 / max_allocation_cap) + 0.999)} suppliers to fill demand "
            f"under the cap; only {n} provided."
        )

    # Upper bound on achievable weighted score: fill the highest-scoring
    # suppliers up to max_cap until demand = 1.
    ordered = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    remaining, best_score = 1.0, 0.0
    for row in ordered:
        take = min(max_allocation_cap, remaining)
        best_score += take * float(row["score"])
        remaining -= take
        if remaining <= 1e-9:
            break
    if best_score < min_compliance_threshold - 1e-9:
        return (
            f"Even filling the top-scoring suppliers to the {max_allocation_cap:.0%} "
            f"cap yields a weighted score of {best_score:.3f}, below the required "
            f"threshold of {min_compliance_threshold:.3f}."
        )

    return "Model reported infeasible; check input data for inconsistencies."


def optimize_supplier_allocation(
    ranked_results: List[Dict[str, Any]],
    min_compliance_threshold: float = 0.85,
    max_allocation_cap: float = 0.70,
    *,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Solve the supplier-allocation LP.

    Args:
        ranked_results: list of dicts with keys ``supplier`` (str),
            ``score`` (float, 0-1 compliance), ``val`` (float, unit price).
        min_compliance_threshold: minimum weighted compliance score required.
        max_allocation_cap: maximum volume share for any single supplier.
        tolerance: rounding threshold below which allocations are dropped.

    Returns:
        Dict with ``status``, ``allocations``, ``total_cost``,
        ``weighted_compliance``, ``constraints``, and ``message``.
        On infeasibility, ``allocations`` is empty and ``message`` explains why.
    """
    _validate(ranked_results)

    if not (0.0 < max_allocation_cap <= 1.0):
        raise ValueError("max_allocation_cap must be in (0, 1].")
    if not (0.0 <= min_compliance_threshold <= 1.0):
        raise ValueError("min_compliance_threshold must be in [0, 1].")

    # Deduplicate supplier names so variable IDs are unique.
    names = [row["supplier"] for row in ranked_results]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate supplier names in ranked_results.")

    prob = LpProblem("agnes_supplier_allocation", LpMinimize)
    x = {
        row["supplier"]: LpVariable(
            f"x_{i}", lowBound=0.0, upBound=max_allocation_cap
        )
        for i, row in enumerate(ranked_results)
    }

    prob += lpSum(x[r["supplier"]] * float(r["val"]) for r in ranked_results), "total_cost"
    prob += lpSum(x[r["supplier"]] for r in ranked_results) == 1.0, "demand"
    prob += (
        lpSum(x[r["supplier"]] * float(r["score"]) for r in ranked_results)
        >= min_compliance_threshold,
        "compliance_floor",
    )

    prob.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[prob.status]

    constraints_meta = {
        "min_compliance_threshold": min_compliance_threshold,
        "max_allocation_cap": max_allocation_cap,
        "supplier_count": len(ranked_results),
    }

    if status != "Optimal":
        return {
            "status": "infeasible",
            "allocations": {},
            "total_cost": None,
            "weighted_compliance": None,
            "constraints": constraints_meta,
            "message": _infeasibility_hint(
                ranked_results, min_compliance_threshold, max_allocation_cap
            ),
        }

    allocations: Dict[str, float] = {}
    for row in ranked_results:
        share = value(x[row["supplier"]]) or 0.0
        if share > tolerance:
            allocations[row["supplier"]] = round(share, 4)

    total_cost = sum(share * float(_lookup(ranked_results, s, "val"))
                     for s, share in allocations.items())
    weighted_compliance = sum(share * float(_lookup(ranked_results, s, "score"))
                              for s, share in allocations.items())

    return {
        "status": "optimal",
        "allocations": allocations,
        "total_cost": round(total_cost, 4),
        "weighted_compliance": round(weighted_compliance, 4),
        "constraints": constraints_meta,
        "message": (
            f"Optimal split across {len(allocations)} supplier(s) — "
            f"weighted compliance {weighted_compliance:.3f} ≥ "
            f"{min_compliance_threshold:.3f}, total unit cost {total_cost:.2f}."
        ),
    }


def _lookup(rows: List[Dict[str, Any]], supplier: str, key: str) -> Any:
    for r in rows:
        if r["supplier"] == supplier:
            return r[key]
    raise KeyError(supplier)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json

    demo_rows = [
        {"supplier": "Global-Cure",   "score": 0.95, "val": 41.0},
        {"supplier": "Eco-Nutrient",  "score": 0.70, "val": 38.5},
        {"supplier": "LowCost-Chem",  "score": 0.50, "val": 30.0},
        {"supplier": "Premium-Bio",   "score": 0.90, "val": 44.0},
    ]

    print("--- Feasible case: threshold 0.85, cap 0.70 ---")
    print(json.dumps(optimize_supplier_allocation(demo_rows), indent=2))

    print("\n--- Infeasible case: threshold 0.99, cap 0.40 ---")
    print(json.dumps(
        optimize_supplier_allocation(
            demo_rows, min_compliance_threshold=0.99, max_allocation_cap=0.40
        ),
        indent=2,
    ))
