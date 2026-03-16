"""
predict_allocation.py
---------------------
Predictive analytics layer for the Construction Resource Optimization System.

Provides:
  - Duration prediction using regression on task features
  - Cost estimation models
  - What-if scenario analysis
  - Resource efficiency metrics

Run standalone:
    python src/predict_allocation.py
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/project_tasks.csv"
SCHEDULE_PATH = "models/optimized_schedule.csv"


# ─── Feature Engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct model features from raw task data.
    """
    out = df.copy()

    # Parse JSON columns if stored as strings
    if isinstance(out["equipment_required"].iloc[0], str):
        out["equipment_required"] = out["equipment_required"].apply(json.loads)
    if isinstance(out["dependencies"].iloc[0], str):
        out["dependencies"] = out["dependencies"].apply(json.loads)

    # Numerical features
    out["num_equipment_types"] = out["equipment_required"].apply(len)
    out["num_dependencies"] = out["dependencies"].apply(len)
    out["worker_cost_ratio"] = (
        out["num_workers_required"] / out["cost_per_day_usd"].clip(lower=1)
    )
    out["complexity_index"] = (
        out["num_workers_required"] * 0.4
        + out["num_equipment_types"] * 0.3
        + out["num_dependencies"] * 0.3
    )

    # Phase encoding
    phase_order = {
        "Pre-Construction": 1,
        "Foundation": 2,
        "Structure": 3,
        "Interior": 4,
        "Finishing": 5,
        "Closeout": 6,
    }
    out["phase_ordinal"] = out["phase"].map(phase_order).fillna(3)

    # Skill-level encoding
    skill_order = {"General": 1, "Skilled": 2, "Highly Skilled": 3}
    out["skill_ordinal"] = out["skill_level"].map(skill_order).fillna(1)

    return out


# ─── Duration Prediction Model ────────────────────────────────────────────────

def train_duration_model(
    df: pd.DataFrame,
) -> Tuple[Pipeline, Dict]:
    """
    Train a Gradient Boosting regressor to predict task duration.

    Because our dataset is small (10 tasks), we bootstrap synthetic
    variants to give the model enough training signal.
    """
    fe = engineer_features(df)
    augmented = _bootstrap_augment(fe, n_copies=50)

    feature_cols = [
        "num_workers_required",
        "num_equipment_types",
        "num_dependencies",
        "cost_per_day_usd",
        "complexity_index",
        "phase_ordinal",
        "skill_ordinal",
    ]
    X = augmented[feature_cols]
    y = augmented["task_duration_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "mae_days": round(mean_absolute_error(y_test, y_pred), 2),
        "r2": round(r2_score(y_test, y_pred), 3),
    }
    return pipe, metrics, feature_cols


def _bootstrap_augment(df: pd.DataFrame, n_copies: int = 50) -> pd.DataFrame:
    """
    Create synthetic variants of the task data by adding Gaussian noise.
    Used to stabilise model training on small datasets.
    """
    frames = [df]
    rng = np.random.default_rng(seed=0)
    for _ in range(n_copies):
        noisy = df.copy()
        noisy["task_duration_days"] = (
            df["task_duration_days"]
            + rng.normal(0, df["task_duration_days"] * 0.1)
        ).clip(lower=1).round()
        noisy["num_workers_required"] = (
            df["num_workers_required"]
            + rng.integers(-2, 3, size=len(df))
        ).clip(lower=1)
        noisy["cost_per_day_usd"] = (
            df["cost_per_day_usd"]
            * (1 + rng.normal(0, 0.08, size=len(df)))
        ).clip(lower=100)
        frames.append(noisy)
    return pd.concat(frames, ignore_index=True)


# ─── Cost Prediction ─────────────────────────────────────────────────────────

def estimate_cost(
    df: pd.DataFrame,
    max_workers: int = 40,
    overhead_rate: float = 0.15,
    equipment_daily_rate: float = 300.0,
    worker_daily_rate: float = 250.0,
) -> pd.DataFrame:
    """
    Estimate per-task and project-level costs given resource allocations.
    """
    out = df.copy()
    if isinstance(out.get("equipment_required", pd.Series([])).iloc[0], str):
        out["equipment_required"] = out["equipment_required"].apply(json.loads)

    out["estimated_workers"] = out["num_workers_required"].clip(upper=max_workers)
    out["estimated_equipment"] = out["equipment_required"].apply(len)

    out["labour_cost"] = out["estimated_workers"] * worker_daily_rate * out["task_duration_days"]
    out["equipment_cost"] = out["estimated_equipment"] * equipment_daily_rate * out["task_duration_days"]
    out["base_cost"] = out["labour_cost"] + out["equipment_cost"]
    out["overhead_cost"] = out["base_cost"] * overhead_rate
    out["total_estimated_cost"] = out["base_cost"] + out["overhead_cost"]

    return out


# ─── Scenario Analysis ────────────────────────────────────────────────────────

def run_scenarios(
    df: pd.DataFrame,
    worker_levels: List[int] = None,
) -> pd.DataFrame:
    """
    Run what-if analysis across a range of worker availability levels.
    Returns summary metrics for each scenario.
    """
    if worker_levels is None:
        worker_levels = [20, 30, 40, 50, 60]

    rows = []
    for w in worker_levels:
        costed = estimate_cost(df, max_workers=w)
        # Approximate makespan: critical path length (simple heuristic)
        makespan = _critical_path_length(df)
        rows.append({
            "max_workers": w,
            "estimated_makespan_days": makespan,
            "total_cost_usd": costed["total_estimated_cost"].sum(),
            "labour_cost_usd": costed["labour_cost"].sum(),
            "equipment_cost_usd": costed["equipment_cost"].sum(),
            "overhead_cost_usd": costed["overhead_cost"].sum(),
            "avg_worker_utilisation": min(1.0, df["num_workers_required"].sum() / (w * len(df))),
        })

    return pd.DataFrame(rows)


def _critical_path_length(df: pd.DataFrame) -> int:
    """Compute critical path duration via topological forward pass."""
    if isinstance(df["dependencies"].iloc[0], str):
        deps = dict(zip(df["task_id"], df["dependencies"].apply(json.loads)))
    else:
        deps = dict(zip(df["task_id"], df["dependencies"]))
    duration = dict(zip(df["task_id"], df["task_duration_days"]))
    finish = {}
    tasks = df["task_id"].tolist()

    def ef(t):
        if t in finish:
            return finish[t]
        pred_finish = max((ef(p) for p in deps.get(t, [])), default=0)
        finish[t] = pred_finish + duration[t]
        return finish[t]

    for t in tasks:
        ef(t)
    return max(finish.values(), default=0)


# ─── Resource Efficiency Report ───────────────────────────────────────────────

def compute_efficiency_metrics(result_df: pd.DataFrame) -> Dict:
    """
    Compute project-level and task-level efficiency indicators.
    """
    if isinstance(result_df["equipment_required"].iloc[0], str):
        result_df = result_df.copy()
        result_df["equipment_required"] = result_df["equipment_required"].apply(json.loads)

    total_worker_days_required = (
        result_df["num_workers_required"] * result_df["task_duration_days"]
    ).sum()

    total_worker_days_allocated = (
        result_df.get("allocated_workers", result_df["num_workers_required"])
        * result_df["task_duration_days"]
    ).sum()

    critical_path = _critical_path_length(result_df)
    sequential_duration = result_df["task_duration_days"].sum()
    parallelism_gain = sequential_duration - critical_path

    return {
        "critical_path_days": critical_path,
        "sequential_duration_days": sequential_duration,
        "parallelism_gain_days": parallelism_gain,
        "schedule_compression_pct": round(parallelism_gain / sequential_duration * 100, 1),
        "total_worker_days_required": int(total_worker_days_required),
        "total_worker_days_allocated": int(total_worker_days_allocated),
        "labour_efficiency_pct": round(
            min(total_worker_days_required, total_worker_days_allocated)
            / max(total_worker_days_required, 1) * 100, 1
        ),
        "total_equipment_deployments": result_df["equipment_required"].apply(len).sum(),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Construction Resource Optimization System")
    print("Step 3a / 3  –  Predictive Allocation Analysis")
    print("=" * 60)

    # Prefer optimised schedule; fall back to raw tasks
    path = SCHEDULE_PATH if os.path.exists(SCHEDULE_PATH) else DATA_PATH
    df = pd.read_csv(path)

    print(f"\n[predict] Loaded data from: {path}")

    # Feature engineering
    fe_df = engineer_features(df)

    # Train duration model
    model, metrics, feat_cols = train_duration_model(fe_df)
    print(f"\n[predict] Duration Model Metrics → MAE: {metrics['mae_days']} days | R²: {metrics['r2']}")

    # Cost estimates
    costed = estimate_cost(df)
    print(f"\n[predict] Estimated total project cost: ${costed['total_estimated_cost'].sum():,.0f}")

    # Scenario analysis
    scenarios = run_scenarios(df)
    print("\n[predict] Scenario Analysis (workers vs cost):")
    print(scenarios.to_string(index=False))

    # Efficiency metrics
    metrics_out = compute_efficiency_metrics(df)
    print("\n[predict] Efficiency Metrics:")
    for k, v in metrics_out.items():
        print(f"  {k:<40} {v}")
