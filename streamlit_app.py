"""
streamlit_app.py
----------------
Interactive dashboard for the Construction Resource Optimization System.

Launch:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ── Path resolution (allow running from any CWD) ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_project_tasks, generate_equipment_availability
from optimize_resources import load_tasks, solve_scheduling, allocate_equipment
from predict_allocation import (
    engineer_features, estimate_cost, run_scenarios, compute_efficiency_metrics
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Resource Optimizer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; }
code { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    color: #38bdf8;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-sub {
    color: #64748b;
    font-size: 0.72rem;
    margin-top: 0.2rem;
}
.phase-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─── Colour Palette ───────────────────────────────────────────────────────────
PHASE_COLORS = {
    "Pre-Construction": "#f97316",
    "Foundation":       "#eab308",
    "Structure":        "#22c55e",
    "Interior":         "#3b82f6",
    "Finishing":        "#a855f7",
    "Closeout":         "#ec4899",
}
CHART_BG = "#0f172a"
AXIS_COLOR = "#334155"
TEXT_COLOR = "#cbd5e1"


# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_tasks_df() -> pd.DataFrame:
    path = ROOT / "data" / "project_tasks.csv"
    if not path.exists():
        generate_project_tasks(save_path=str(path))
    return load_tasks(str(path))


def run_optimization(max_workers: int, max_equip: int) -> tuple:
    df = get_tasks_df()
    result_df, summary = solve_scheduling(df, max_workers=max_workers,
                                          max_equipment_units=max_equip,
                                          time_limit_sec=20)
    equip_df = allocate_equipment(result_df)
    scenarios = run_scenarios(df)
    efficiency = compute_efficiency_metrics(result_df)
    return result_df, summary, equip_df, scenarios, efficiency


def parse_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if df[col].dtype == object and df[col].iloc[0].startswith("["):
        df[col] = df[col].apply(json.loads)
    return df


# ─── Plot Functions ───────────────────────────────────────────────────────────

def plot_gantt(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    tasks = result_df.sort_values("optimal_start_day")
    y_positions = range(len(tasks))

    for i, (_, row) in enumerate(tasks.iterrows()):
        color = PHASE_COLORS.get(row.get("phase", "Structure"), "#3b82f6")
        ax.barh(
            i,
            row["task_duration_days"],
            left=row["optimal_start_day"],
            height=0.55,
            color=color,
            alpha=0.85,
            edgecolor="#1e293b",
            linewidth=0.8,
        )
        # Label inside bar
        mid = row["optimal_start_day"] + row["task_duration_days"] / 2
        ax.text(mid, i, f"D{row['optimal_start_day']}–{row['optimal_end_day']}",
                ha="center", va="center", fontsize=7.5,
                color="white", fontweight="600",
                fontfamily="monospace")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(
        [f"{r['task_id']} · {r['task_name']}" for _, r in tasks.iterrows()],
        color=TEXT_COLOR, fontsize=8.5
    )
    ax.set_xlabel("Project Day", color=TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[:].set_color(AXIS_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)

    # Phase legend
    patches = [mpatches.Patch(color=c, label=p) for p, c in PHASE_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", framealpha=0.2,
              labelcolor=TEXT_COLOR, fontsize=7.5,
              facecolor="#1e293b", edgecolor=AXIS_COLOR)

    ax.set_title("Optimised Project Schedule (Gantt)", color=TEXT_COLOR,
                 fontsize=11, fontweight="700", pad=12)
    ax.invert_yaxis()
    fig.tight_layout(pad=1.5)
    return fig


def plot_worker_bar(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    x = np.arange(len(result_df))
    w = 0.38
    bars1 = ax.bar(x - w / 2, result_df["num_workers_required"], w,
                   label="Required", color="#3b82f6", alpha=0.7)
    bars2 = ax.bar(x + w / 2, result_df["allocated_workers"], w,
                   label="Allocated", color="#22c55e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(result_df["task_id"], color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Workers", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Worker Allocation: Required vs Allocated", color=TEXT_COLOR,
                 fontsize=11, fontweight="700")
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(AXIS_COLOR)
    ax.legend(facecolor="#1e293b", edgecolor=AXIS_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8.5)

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=7.5,
                color=TEXT_COLOR, fontfamily="monospace")

    fig.tight_layout(pad=1.5)
    return fig


def plot_equipment_pie(equip_df: pd.DataFrame) -> plt.Figure:
    if equip_df.empty:
        fig, ax = plt.subplots()
        fig.patch.set_facecolor(CHART_BG)
        ax.text(0.5, 0.5, "No equipment data", ha="center", va="center",
                color=TEXT_COLOR, transform=ax.transAxes)
        return fig

    counts = equip_df.groupby("equipment")["units_allocated"].sum()
    colors = plt.cm.get_cmap("tab10")(np.linspace(0, 0.9, len(counts)))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.0f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 1.5},
    )
    for t in texts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(8.5)
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)
        at.set_fontweight("700")

    ax.set_title("Equipment Utilisation Distribution", color=TEXT_COLOR,
                 fontsize=11, fontweight="700")
    fig.tight_layout()
    return fig


def plot_cost_breakdown(result_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)

    x = np.arange(len(result_df))
    costed = estimate_cost(result_df)

    ax.bar(x, costed["labour_cost"] / 1000, label="Labour", color="#3b82f6", alpha=0.85)
    ax.bar(x, costed["equipment_cost"] / 1000, bottom=costed["labour_cost"] / 1000,
           label="Equipment", color="#f97316", alpha=0.85)
    ax.bar(x, costed["overhead_cost"] / 1000,
           bottom=(costed["labour_cost"] + costed["equipment_cost"]) / 1000,
           label="Overhead", color="#a855f7", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(result_df["task_id"], color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Cost (USD thousands)", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Cost Breakdown per Task", color=TEXT_COLOR,
                 fontsize=11, fontweight="700")
    ax.tick_params(colors=TEXT_COLOR)
    ax.spines[:].set_color(AXIS_COLOR)
    ax.legend(facecolor="#1e293b", edgecolor=AXIS_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8.5)
    fig.tight_layout(pad=1.5)
    return fig


def plot_scenario_lines(scenarios_df: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(CHART_BG)
    ax1.set_facecolor(CHART_BG)

    ax2 = ax1.twinx()
    ax2.set_facecolor(CHART_BG)

    ax1.plot(scenarios_df["max_workers"], scenarios_df["total_cost_usd"] / 1000,
             color="#38bdf8", lw=2.2, marker="o", markersize=5, label="Total Cost (k$)")
    ax2.plot(scenarios_df["max_workers"], scenarios_df["estimated_makespan_days"],
             color="#f97316", lw=2.2, marker="s", markersize=5, linestyle="--",
             label="Makespan (days)")

    ax1.set_xlabel("Max Workers Available", color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel("Total Cost (USD thousands)", color="#38bdf8", fontsize=9)
    ax2.set_ylabel("Makespan (days)", color="#f97316", fontsize=9)

    for ax in [ax1, ax2]:
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.spines[:].set_color(AXIS_COLOR)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#1e293b", edgecolor=AXIS_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8.5)
    ax1.set_title("Scenario Analysis: Workers vs Cost & Duration",
                  color=TEXT_COLOR, fontsize=11, fontweight="700")
    fig.tight_layout()
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size:1.8rem">🏗️</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                        color:#64748b; text-transform:uppercase; letter-spacing:0.1em">
                Resource Optimizer
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Constraints")
        max_workers = st.slider(
            "Available Workers", min_value=10, max_value=80, value=40, step=5,
            help="Total workforce available across all tasks simultaneously."
        )
        max_equipment = st.slider(
            "Equipment Units Available", min_value=3, max_value=20, value=10, step=1,
            help="Total number of equipment units in the project pool."
        )

        st.markdown("---")
        st.markdown("### 📋 Project Info")
        st.markdown("""
        <div style="font-size:0.78rem; color:#64748b; line-height:1.6;">
        <b>Solver:</b> PuLP / CBC MILP<br>
        <b>Method:</b> Makespan minimisation<br>
        <b>Tasks:</b> 10 construction phases<br>
        <b>Constraints:</b> Precedence + Resource caps
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        run_btn = st.button("🚀 Run Optimisation", type="primary", use_container_width=True)

        st.markdown("""
        <div style="font-size:0.65rem; color:#475569; margin-top:2rem; line-height:1.5;">
        This project simulates realistic construction resource allocation scenarios
        using operations research techniques.
        </div>
        """, unsafe_allow_html=True)

    return max_workers, max_equipment, run_btn


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="border-bottom: 1px solid #1e3a5f; padding-bottom: 1rem; margin-bottom: 1.5rem;">
        <h1 style="margin:0; font-size:1.9rem; color:#f1f5f9;">
            🏗️ Construction Resource Optimization System
        </h1>
        <p style="margin:0.3rem 0 0 0; color:#64748b; font-size:0.85rem;
                  font-family:'IBM Plex Mono',monospace;">
            Operations Research · MILP Scheduling · Resource Allocation · MSc Portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)

    max_workers, max_equipment, run_btn = render_sidebar()

    # Run optimization on button press or first load
    if run_btn or "result_df" not in st.session_state:
        with st.spinner("Solving mixed-integer programme …"):
            try:
                result_df, summary, equip_df, scenarios, efficiency = run_optimization(
                    max_workers, max_equipment
                )
                st.session_state.result_df = result_df
                st.session_state.summary = summary
                st.session_state.equip_df = equip_df
                st.session_state.scenarios = scenarios
                st.session_state.efficiency = efficiency
                st.success(f"Optimisation complete — {summary['status']}")
            except Exception as e:
                st.error(f"Optimisation error: {e}")
                st.stop()

    result_df = st.session_state.result_df
    summary = st.session_state.summary
    equip_df = st.session_state.equip_df
    scenarios = st.session_state.scenarios
    efficiency = st.session_state.efficiency

    # ── KPI Row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "MAKESPAN", f"{summary['makespan_days']}", "days"),
        (c2, "TOTAL COST", f"${summary['total_project_cost_usd']/1000:.0f}k", "USD"),
        (c3, "TASKS", f"{summary['num_tasks']}", "construction phases"),
        (c4, "WORKER UTIL.", f"{summary['avg_worker_utilisation_pct']:.0f}%", "average"),
        (c5, "SCHEDULE COMPRESSION", f"{efficiency['schedule_compression_pct']}%",
         f"{efficiency['parallelism_gain_days']} days saved"),
    ]
    for col, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Schedule", "👷 Workers", "🔧 Equipment", "💰 Costs", "📊 Scenarios"
    ])

    with tab1:
        st.subheader("Optimised Gantt Schedule")
        st.pyplot(plot_gantt(result_df), use_container_width=True)

        st.markdown("#### Task Details")
        display_cols = [
            "task_id", "task_name", "phase", "skill_level",
            "optimal_start_day", "optimal_end_day", "task_duration_days",
            "allocated_workers", "worker_utilisation_pct"
        ]
        st.dataframe(
            result_df[display_cols].rename(columns={
                "task_id": "ID", "task_name": "Task",
                "optimal_start_day": "Start Day", "optimal_end_day": "End Day",
                "task_duration_days": "Duration", "allocated_workers": "Workers",
                "worker_utilisation_pct": "Util %",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.subheader("Worker Allocation Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(plot_worker_bar(result_df), use_container_width=True)
        with col2:
            st.markdown("#### Labour Metrics")
            st.metric("Critical Path (days)", efficiency["critical_path_days"])
            st.metric("Total Worker-Days Required",
                      f"{efficiency['total_worker_days_required']:,}")
            st.metric("Labour Efficiency",
                      f"{efficiency['labour_efficiency_pct']}%")
            st.metric("Parallelism Gain",
                      f"{efficiency['parallelism_gain_days']} days")

    with tab3:
        st.subheader("Equipment Utilisation")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(plot_equipment_pie(equip_df), use_container_width=True)
        with col2:
            if not equip_df.empty:
                st.markdown("#### Equipment Schedule")
                st.dataframe(
                    equip_df[["equipment", "task_name", "start_day", "end_day",
                               "units_allocated"]].rename(columns={
                        "task_name": "Task", "start_day": "Start",
                        "end_day": "End", "units_allocated": "Units",
                    }),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No equipment data available.")

    with tab4:
        st.subheader("Project Cost Breakdown")
        st.pyplot(plot_cost_breakdown(result_df), use_container_width=True)

        costed = estimate_cost(result_df)
        st.markdown("#### Cost Summary Table (USD)")
        cost_cols = ["task_name", "labour_cost", "equipment_cost",
                     "overhead_cost", "total_estimated_cost"]
        st.dataframe(
            costed[cost_cols].rename(columns={
                "task_name": "Task",
                "labour_cost": "Labour ($)",
                "equipment_cost": "Equipment ($)",
                "overhead_cost": "Overhead ($)",
                "total_estimated_cost": "Total ($)",
            }).style.format({
                "Labour ($)": "${:,.0f}",
                "Equipment ($)": "${:,.0f}",
                "Overhead ($)": "${:,.0f}",
                "Total ($)": "${:,.0f}",
            }),
            use_container_width=True, hide_index=True
        )

    with tab5:
        st.subheader("What-If Scenario Analysis")
        st.pyplot(plot_scenario_lines(scenarios), use_container_width=True)

        st.markdown("#### Scenario Table")
        st.dataframe(
            scenarios.style.format({
                "total_cost_usd": "${:,.0f}",
                "labour_cost_usd": "${:,.0f}",
                "equipment_cost_usd": "${:,.0f}",
                "avg_worker_utilisation": "{:.1%}",
            }),
            use_container_width=True, hide_index=True
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3rem; border-top:1px solid #1e3a5f; padding-top:1rem;
                font-size:0.72rem; color:#475569; font-family:'IBM Plex Mono',monospace;">
        Construction Resource Optimization System · PuLP MILP Solver ·
        MSc Research Portfolio · Anthropic Claude
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
