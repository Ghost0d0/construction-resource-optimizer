"""
data_generator.py
-----------------
Generates synthetic construction project task datasets for the
Construction Resource Optimization System.

Run:
    python src/data_generator.py

Output:
    data/project_tasks.csv
"""

import pandas as pd
import numpy as np
import json
import os

# ─── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ─── Task Templates ───────────────────────────────────────────────────────────
TASK_TEMPLATES = [
    {
        "task_id": "T01",
        "task_name": "Site Preparation & Clearing",
        "phase": "Pre-Construction",
        "task_duration_days": 5,
        "num_workers_required": 8,
        "equipment_required": ["Bulldozer", "Dump Truck"],
        "cost_per_day_usd": 2400.0,
        "dependencies": [],
        "skill_level": "General",
        "priority": 1,
    },
    {
        "task_id": "T02",
        "task_name": "Excavation & Earthworks",
        "phase": "Foundation",
        "task_duration_days": 10,
        "num_workers_required": 12,
        "equipment_required": ["Excavator", "Dump Truck", "Compactor"],
        "cost_per_day_usd": 3800.0,
        "dependencies": ["T01"],
        "skill_level": "General",
        "priority": 1,
    },
    {
        "task_id": "T03",
        "task_name": "Foundation & Concrete Works",
        "phase": "Foundation",
        "task_duration_days": 14,
        "num_workers_required": 15,
        "equipment_required": ["Concrete Mixer", "Crane"],
        "cost_per_day_usd": 5200.0,
        "dependencies": ["T02"],
        "skill_level": "Skilled",
        "priority": 1,
    },
    {
        "task_id": "T04",
        "task_name": "Structural Steel Erection",
        "phase": "Structure",
        "task_duration_days": 18,
        "num_workers_required": 20,
        "equipment_required": ["Crane", "Welding Machine", "Aerial Platform"],
        "cost_per_day_usd": 7500.0,
        "dependencies": ["T03"],
        "skill_level": "Highly Skilled",
        "priority": 2,
    },
    {
        "task_id": "T05",
        "task_name": "Roofing & Waterproofing",
        "phase": "Structure",
        "task_duration_days": 8,
        "num_workers_required": 10,
        "equipment_required": ["Aerial Platform", "Compactor"],
        "cost_per_day_usd": 3100.0,
        "dependencies": ["T04"],
        "skill_level": "Skilled",
        "priority": 2,
    },
    {
        "task_id": "T06",
        "task_name": "MEP Installations",
        "phase": "Interior",
        "task_duration_days": 20,
        "num_workers_required": 18,
        "equipment_required": ["Aerial Platform", "Welding Machine"],
        "cost_per_day_usd": 6000.0,
        "dependencies": ["T04"],
        "skill_level": "Highly Skilled",
        "priority": 2,
    },
    {
        "task_id": "T07",
        "task_name": "Interior Masonry & Plastering",
        "phase": "Interior",
        "task_duration_days": 12,
        "num_workers_required": 14,
        "equipment_required": ["Concrete Mixer", "Scaffold"],
        "cost_per_day_usd": 3900.0,
        "dependencies": ["T05", "T06"],
        "skill_level": "Skilled",
        "priority": 3,
    },
    {
        "task_id": "T08",
        "task_name": "Flooring & Tiling",
        "phase": "Finishing",
        "task_duration_days": 10,
        "num_workers_required": 10,
        "equipment_required": ["Scaffold"],
        "cost_per_day_usd": 2800.0,
        "dependencies": ["T07"],
        "skill_level": "Skilled",
        "priority": 3,
    },
    {
        "task_id": "T09",
        "task_name": "External Facades & Cladding",
        "phase": "Finishing",
        "task_duration_days": 12,
        "num_workers_required": 12,
        "equipment_required": ["Scaffold", "Aerial Platform"],
        "cost_per_day_usd": 4100.0,
        "dependencies": ["T05"],
        "skill_level": "Skilled",
        "priority": 3,
    },
    {
        "task_id": "T10",
        "task_name": "Commissioning & Handover",
        "phase": "Closeout",
        "task_duration_days": 5,
        "num_workers_required": 6,
        "equipment_required": [],
        "cost_per_day_usd": 1500.0,
        "dependencies": ["T08", "T09"],
        "skill_level": "General",
        "priority": 4,
    },
]

# ─── Available Equipment Pool ──────────────────────────────────────────────────
EQUIPMENT_POOL = {
    "Crane": 2,
    "Excavator": 2,
    "Bulldozer": 1,
    "Dump Truck": 3,
    "Concrete Mixer": 2,
    "Compactor": 2,
    "Aerial Platform": 3,
    "Welding Machine": 4,
    "Scaffold": 5,
}


def generate_project_tasks(
    noise_level: float = 0.05,
    save_path: str = "data/project_tasks.csv",
) -> pd.DataFrame:
    """
    Generate the canonical task dataset with optional stochastic noise.

    Parameters
    ----------
    noise_level : float
        Fraction of random ± variation added to durations and costs.
    save_path : str
        Where to write the CSV output.

    Returns
    -------
    pd.DataFrame
    """
    records = []
    for tpl in TASK_TEMPLATES:
        row = tpl.copy()

        # Apply small noise to simulate real-world uncertainty
        duration_noise = int(row["task_duration_days"] * noise_level * np.random.randn())
        cost_noise = row["cost_per_day_usd"] * noise_level * np.random.randn()

        row["task_duration_days"] = max(1, row["task_duration_days"] + duration_noise)
        row["cost_per_day_usd"] = round(max(100.0, row["cost_per_day_usd"] + cost_noise), 2)

        # Serialise lists as JSON strings for CSV compatibility
        row["equipment_required"] = json.dumps(row["equipment_required"])
        row["dependencies"] = json.dumps(row["dependencies"])

        records.append(row)

    df = pd.DataFrame(records)

    # Derived columns
    df["total_task_cost_usd"] = df["task_duration_days"] * df["cost_per_day_usd"]
    df["labour_cost_per_day"] = (df["num_workers_required"] * 250).astype(float)  # $250/worker/day

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_generator] Dataset saved → {save_path}  ({len(df)} tasks)")
    return df


def generate_equipment_availability(
    save_path: str = "data/equipment_pool.csv",
) -> pd.DataFrame:
    """Persist the equipment pool as a reference CSV."""
    rows = [{"equipment": k, "units_available": v} for k, v in EQUIPMENT_POOL.items()]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[data_generator] Equipment pool saved → {save_path}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Construction Resource Optimization System")
    print("Step 1 / 3  –  Data Generation")
    print("=" * 60)
    tasks_df = generate_project_tasks()
    equip_df = generate_equipment_availability()

    print("\nTask Summary:")
    print(tasks_df[["task_id", "task_name", "task_duration_days",
                     "num_workers_required", "cost_per_day_usd"]].to_string(index=False))
    print(f"\nTotal baseline project cost : ${tasks_df['total_task_cost_usd'].sum():,.0f}")
    print(f"Total baseline duration (sequential): {tasks_df['task_duration_days'].sum()} days")
