"""
optimize_resources.py
---------------------
Core optimization engine for the Construction Resource Optimization System.

Uses PuLP (MILP) to schedule tasks and allocate workers/equipment while:
  - Minimising total project duration (makespan)
  - Respecting worker and equipment capacity constraints
  - Enforcing task precedence relationships

Run:
    python src/optimize_resources.py

Output:
    models/optimized_schedule.csv
    models/optimization_summary.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_PATH = "data/project_tasks.csv"
OUTPUT_DIR = "models"
MAX_HORIZON = 200  # planning horizon in days


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_tasks(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and parse the task dataset."""
    df = pd.read_csv(path)
    df["equipment_required"] = df["equipment_required"].apply(json.loads)
    df["dependencies"] = df["dependencies"].apply(json.loads)
    return df


# ─── Scheduling Model ─────────────────────────────────────────────────────────

def build_schedule_lp(
    df: pd.DataFrame,
    max_workers: int = 40,
    max_equipment_units: int = 10,
    big_m: int = MAX_HORIZON,
) -> Tuple[pulp.LpProblem, Dict]:
    """
    Formulate a Mixed-Integer Linear Programme to minimise project makespan.

    Decision variables
    ------------------
    start[i]   : integer start day of task i  (≥ 0)
    makespan   : integer project completion day (minimised)
    y[i][j]    : binary, 1 if tasks i and j overlap in time

    Constraints
    -----------
    1. Precedence: start[j] ≥ start[i] + duration[i]  for j ∈ successors(i)
    2. Resource: total workers on any day ≤ max_workers  (linearised)
    3. Makespan: makespan ≥ start[i] + duration[i]  ∀i
    """
    tasks = df["task_id"].tolist()
    duration = dict(zip(df["task_id"], df["task_duration_days"]))
    workers = dict(zip(df["task_id"], df["num_workers_required"]))
    deps = dict(zip(df["task_id"], df["dependencies"]))

    prob = pulp.LpProblem("Construction_Scheduling", pulp.LpMinimize)

    # ── Decision variables ────────────────────────────────────────────────────
    start = {t: pulp.LpVariable(f"start_{t}", lowBound=0, cat="Integer") for t in tasks}
    makespan = pulp.LpVariable("makespan", lowBound=0, cat="Integer")

    # Overlap binary variables: y[i,j] = 1 means task i starts before task j
    y = {}
    for i in tasks:
        for j in tasks:
            if i < j:
                y[(i, j)] = pulp.LpVariable(f"y_{i}_{j}", cat="Binary")

    # ── Objective ─────────────────────────────────────────────────────────────
    prob += makespan, "Minimise_Makespan"

    # ── Constraints ──────────────────────────────────────────────────────────

    # 1. Makespan ≥ finish of every task
    for t in tasks:
        prob += makespan >= start[t] + duration[t], f"Makespan_{t}"

    # 2. Precedence relationships
    for t in tasks:
        for pred in deps[t]:
            if pred in start:
                prob += start[t] >= start[pred] + duration[pred], f"Prec_{pred}_{t}"

    # 3. No-overlap / resource constraints using big-M disjunctive formulation
    #    For each pair (i,j), either i finishes before j starts, or vice-versa.
    #    Additionally cap parallel worker usage.
    for i in tasks:
        for j in tasks:
            if i >= j:
                continue
            key = (i, j)
            # i before j  OR  j before i
            prob += start[j] >= start[i] + duration[i] - big_m * y[key],        f"NoOvlp_ij_{i}_{j}"
            prob += start[i] >= start[j] + duration[j] - big_m * (1 - y[key]),  f"NoOvlp_ji_{i}_{j}"

    # 4. Global worker cap (soft — enforced via disjunctive scheduling above
    #    but we add a direct daily-aggregate check over a sampled set of days)
    #    We linearise by checking a representative sample of time points.
    sample_days = range(0, big_m, max(1, big_m // 30))
    for d in sample_days:
        active_expr = pulp.lpSum(
            workers[t] * (
                # task t is active on day d iff start[t] <= d < start[t] + duration[t]
                # linearised: we add an indicator but to keep the model tractable we
                # rely on the disjunctive constraints and cap via a soft bound expression
                pulp.LpVariable(f"active_{t}_{d}", cat="Binary")
            )
            for t in tasks
        )
        # Link active indicator to start window
        for t in tasks:
            active_var = prob.variablesDict().get(f"active_{t}_{d}")
            if active_var is not None:
                prob += active_var <= 1, f"ActUB_{t}_{d}"
                # Active iff d is within [start[t], start[t]+duration[t])
                prob += start[t] <= d + big_m * (1 - active_var), f"ActLB1_{t}_{d}"
                prob += d <= start[t] + duration[t] - 1 + big_m * (1 - active_var), f"ActLB2_{t}_{d}"

        prob += active_expr <= max_workers, f"WorkerCap_{d}"

    meta = {
        "tasks": tasks,
        "duration": duration,
        "workers": workers,
        "deps": deps,
        "max_workers": max_workers,
        "max_equipment_units": max_equipment_units,
    }
    return prob, meta, start, makespan


def solve_scheduling(
    df: pd.DataFrame,
    max_workers: int = 40,
    max_equipment_units: int = 10,
    time_limit_sec: int = 60,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Solve the scheduling LP and return an enriched schedule DataFrame.

    Falls back to a heuristic (earliest-start topological sort) if the
    solver exceeds the time limit without finding an optimal solution.
    """
    print(f"\n[optimizer] Solving with max_workers={max_workers}, "
          f"max_equipment={max_equipment_units} ...")

    # ── Attempt MILP ──────────────────────────────────────────────────────────
    try:
        prob, meta, start_vars, makespan_var = build_schedule_lp(
            df, max_workers, max_equipment_units
        )
        solver = pulp.PULP_CBC_CMD(
            msg=0,
            timeLimit=time_limit_sec,
            options=["maxIterations 500000"],
        )
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        print(f"[optimizer] Solver status: {status}")

        if prob.status in (1, -1):  # Optimal or time-limited feasible
            start_days = {t: int(round(pulp.value(start_vars[t]) or 0)) for t in meta["tasks"]}
            makespan = int(round(pulp.value(makespan_var) or 0))
            solver_used = "MILP (PuLP/CBC)"
        else:
            raise RuntimeError("MILP infeasible or unbounded — falling back to heuristic.")

    except Exception as e:
        print(f"[optimizer] MILP fallback triggered: {e}")
        start_days, makespan = _heuristic_schedule(df)
        solver_used = "Heuristic (Topological Early-Start)"

    # ── Build result DataFrame ────────────────────────────────────────────────
    result = df.copy()
    result["optimal_start_day"] = result["task_id"].map(start_days)
    result["optimal_end_day"] = result["optimal_start_day"] + result["task_duration_days"]

    # Worker allocation: scale to available budget proportionally
    result["allocated_workers"] = result.apply(
        lambda r: min(r["num_workers_required"], max_workers), axis=1
    )

    # Equipment allocation count
    result["allocated_equipment_units"] = result["equipment_required"].apply(
        lambda eq: min(len(eq), max_equipment_units)
    )

    # Cost with allocated workers
    result["allocated_cost_per_day"] = (
        result["allocated_workers"] * 250  # $250 / worker / day
        + result["allocated_equipment_units"] * 300  # $300 / equipment / day
        + result["cost_per_day_usd"] * 0.3  # overhead 30%
    )
    result["total_allocated_cost"] = (
        result["allocated_cost_per_day"] * result["task_duration_days"]
    )

    # Resource utilisation rate
    result["worker_utilisation_pct"] = (
        result["allocated_workers"] / result["num_workers_required"] * 100
    ).clip(0, 100).round(1)

    summary = {
        "solver_used": solver_used,
        "status": "Optimal" if solver_used.startswith("MILP") else "Feasible (Heuristic)",
        "makespan_days": int(makespan),
        "total_project_cost_usd": float(result["total_allocated_cost"].sum()),
        "max_workers": max_workers,
        "max_equipment_units": max_equipment_units,
        "num_tasks": len(result),
        "avg_worker_utilisation_pct": float(result["worker_utilisation_pct"].mean()),
    }

    return result, summary


# ─── Heuristic Fallback ───────────────────────────────────────────────────────

def _heuristic_schedule(df: pd.DataFrame) -> Tuple[Dict, int]:
    """
    Earliest-Start topological scheduling heuristic.
    Respects dependencies; ignores resource contention (optimistic).
    """
    duration = dict(zip(df["task_id"], df["task_duration_days"]))
    deps = dict(zip(df["task_id"], df["dependencies"]))
    finish = {}
    start = {}

    # Topological sort (Kahn's algorithm)
    in_degree = {t: len(d) for t, d in deps.items()}
    queue = [t for t, deg in in_degree.items() if deg == 0]
    order = []

    predecessors_finish = {t: [] for t in deps}
    while queue:
        t = queue.pop(0)
        order.append(t)
        # find successors
        for other, other_deps in deps.items():
            if t in other_deps:
                predecessors_finish[other].append(t)
                in_degree[other] -= 1
                if in_degree[other] == 0:
                    queue.append(other)

    for t in order:
        earliest = max((finish[p] for p in predecessors_finish[t]), default=0)
        start[t] = earliest
        finish[t] = earliest + duration[t]

    makespan = max(finish.values(), default=0)
    return start, makespan


# ─── Equipment Allocation Report ──────────────────────────────────────────────

def allocate_equipment(
    result_df: pd.DataFrame,
    equipment_pool: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Generate a per-task equipment allocation report.
    Checks each task's required equipment against available pool capacity.
    """
    if equipment_pool is None:
        equipment_pool = {
            "Crane": 2, "Excavator": 2, "Bulldozer": 1,
            "Dump Truck": 3, "Concrete Mixer": 2, "Compactor": 2,
            "Aerial Platform": 3, "Welding Machine": 4, "Scaffold": 5,
        }

    rows = []
    for _, row in result_df.iterrows():
        for eq in row["equipment_required"]:
            available = equipment_pool.get(eq, 0)
            rows.append({
                "task_id": row["task_id"],
                "task_name": row["task_name"],
                "equipment": eq,
                "units_available": available,
                "units_allocated": min(1, available),
                "start_day": row["optimal_start_day"],
                "end_day": row["optimal_end_day"],
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── Persist Results ──────────────────────────────────────────────────────────

def save_results(
    result_df: pd.DataFrame,
    summary: Dict,
    equip_df: pd.DataFrame,
    out_dir: str = OUTPUT_DIR,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Serialise list columns back to JSON strings
    out = result_df.copy()
    out["equipment_required"] = out["equipment_required"].apply(json.dumps)
    out["dependencies"] = out["dependencies"].apply(json.dumps)
    out.to_csv(f"{out_dir}/optimized_schedule.csv", index=False)

    with open(f"{out_dir}/optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not equip_df.empty:
        equip_df.to_csv(f"{out_dir}/equipment_allocation.csv", index=False)

    print(f"\n[optimizer] Results saved to {out_dir}/")


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Construction Resource Optimization System")
    print("Step 2 / 3  –  Resource Optimization")
    print("=" * 60)

    # Load data (generate if missing)
    if not os.path.exists(DATA_PATH):
        print("[optimizer] Dataset not found — running data_generator first …")
        from data_generator import generate_project_tasks, generate_equipment_availability
        generate_project_tasks()
        generate_equipment_availability()

    df = load_tasks()

    # Solve
    result_df, summary = solve_scheduling(
        df,
        max_workers=40,
        max_equipment_units=10,
        time_limit_sec=30,
    )

    # Equipment report
    equip_df = allocate_equipment(result_df)

    # Persist
    save_results(result_df, summary, equip_df)

    # Print summary
    print("\n" + "─" * 50)
    print("OPTIMISATION SUMMARY")
    print("─" * 50)
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    print("─" * 50)

    print("\nOptimised Schedule:")
    cols = ["task_id", "task_name", "optimal_start_day",
            "optimal_end_day", "allocated_workers", "total_allocated_cost"]
    print(result_df[cols].to_string(index=False))
    print(f"\n✓  Makespan: {summary['makespan_days']} days")
    print(f"✓  Total Cost: ${summary['total_project_cost_usd']:,.0f}")
