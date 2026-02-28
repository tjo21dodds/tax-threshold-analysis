"""
UK Income Tax Threshold Analysis – Interactive Dashboard
=========================================================
Run with:  streamlit run dashboard.py
"""

import matplotlib.pyplot as plt
import streamlit as st

from tax_analysis.tax_analysis import (
    ANNUAL_INFLATION,
    ANNUAL_RPI,
    ANNUAL_WAGE_GROWTH,
    NUM_TAXPAYERS,
    PROJECTION_YEARS,
    build_scenarios,
)

st.set_page_config(
    page_title="UK Income Tax Threshold Analysis",
    layout="wide",
)

st.title("UK Income Tax Threshold Analysis")
st.markdown(
    "Explore how different threshold-uprating policies affect income tax revenue "
    "over the next five years.  Use the controls in the sidebar to adjust the "
    "economic assumptions."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Economic Assumptions")

wage_rate = (
    st.sidebar.slider(
        "Annual Wage Growth – AWE (%)",
        min_value=0.0,
        max_value=10.0,
        value=round(ANNUAL_WAGE_GROWTH * 100, 1),
        step=0.1,
        help="Average Weekly Earnings growth that shifts the income distribution.",
    )
    / 100
)

cpi_rate = (
    st.sidebar.slider(
        "Annual CPI Inflation (%)",
        min_value=0.0,
        max_value=10.0,
        value=round(ANNUAL_INFLATION * 100, 1),
        step=0.1,
        help="Consumer Prices Index used to uprate thresholds in the CPI scenario.",
    )
    / 100
)

rpi_rate = (
    st.sidebar.slider(
        "Annual RPI Inflation (%)",
        min_value=0.0,
        max_value=10.0,
        value=round(ANNUAL_RPI * 100, 1),
        step=0.1,
        help=(
            "Retail Prices Index used as a proxy for expected government "
            "expenditure increases. Drawn as a reference line on both charts."
        ),
    )
    / 100
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Base year: 2024/25 · {NUM_TAXPAYERS:,} taxpayers · "
    f"{PROJECTION_YEARS}-year projection"
)

# ── Build scenarios ───────────────────────────────────────────────────────────

df = build_scenarios(rpi_rate=rpi_rate, cpi_rate=cpi_rate, wage_rate=wage_rate)

# ── Plot 1 – revenue by scenario ─────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Threshold Scenario")
    st.caption(
        "All scenarios use the same wage growth rate — only the threshold "
        "uprating policy differs.  The dashed grey line shows how government "
        "spending is expected to grow (RPI proxy)."
    )

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(
        df["Tax Year"], df["Frozen Thresholds (£bn)"],
        marker="o", linewidth=2, label="Frozen Thresholds",
    )
    ax1.plot(
        df["Tax Year"], df["CPI-Uprated (£bn)"],
        marker="s", linewidth=2, label=f"CPI-Uprated ({cpi_rate:.1%})",
    )
    ax1.plot(
        df["Tax Year"], df["Wage-Growth-Uprated (£bn)"],
        marker="^", linewidth=2, label=f"Wage-Growth-Uprated ({wage_rate:.1%})",
    )
    ax1.plot(
        df["Tax Year"], df["RPI-Uprated (£bn)"],
        marker="*", linewidth=2, label=f"RPI-Uprated ({rpi_rate:.1%})",
    )
    ax1.plot(
        df["Tax Year"], df["RPI Spending Baseline (£bn)"],
        linewidth=2, linestyle="--", color="grey",
        label=f"RPI Spending Baseline ({rpi_rate:.1%})",
    )
    ax1.set_xlabel("Tax Year")
    ax1.set_ylabel("Income Tax Revenue (£ billion)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# ── Plot 2 – revenue gap vs RPI ───────────────────────────────────────────────

with col2:
    st.subheader("Revenue vs RPI Spending Baseline")
    st.caption(
        "Shows how much more (or less) revenue each scenario raises compared "
        "to the RPI spending baseline.  Positive = revenue grows faster than "
        "expected government expenditure."
    )

    baseline = df["RPI Spending Baseline (£bn)"]

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(
        df["Tax Year"],
        df["Frozen Thresholds (£bn)"] - baseline,
        marker="o", linewidth=2, label="Frozen Thresholds vs RPI Spending",
    )
    ax2.plot(
        df["Tax Year"],
        df["CPI-Uprated (£bn)"] - baseline,
        marker="s", linewidth=2,
        label=f"CPI ({cpi_rate:.1%}) vs RPI ({rpi_rate:.1%}) Spending",
    )
    ax2.plot(
        df["Tax Year"],
        df["Wage-Growth-Uprated (£bn)"] - baseline,
        marker="^", linewidth=2,
        label=f"Wage-Growth ({wage_rate:.1%}) vs RPI ({rpi_rate:.1%}) Spending",
    )
    ax2.plot(
        df["Tax Year"],
        df["RPI-Uprated (£bn)"] - baseline,
        marker="s", linewidth=2,
        label=f"RPI ({rpi_rate:.1%}) vs RPI ({rpi_rate:.1%}) Spending",
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Tax Year")
    ax2.set_ylabel("Revenue minus RPI Spending Baseline (£ billion)")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Data table ────────────────────────────────────────────────────────────────

with st.expander("Show full data table"):
    st.dataframe(df, use_container_width=True)

    drag = df[["Tax Year"]].copy()
    drag["Frozen vs CPI (£bn)"] = (
        df["Frozen Thresholds (£bn)"] - df["CPI-Uprated (£bn)"]
    ).round(1)
    drag["Frozen vs Wages (£bn)"] = (
        df["Frozen Thresholds (£bn)"] - df["Wage-Growth-Uprated (£bn)"]
    ).round(1)
    drag["Frozen vs RPI (£bn)"] = (
        df["Frozen Thresholds (£bn)"] - df["RPI-Uprated (£bn)"]
    ).round(1)

    st.subheader("Fiscal Drag (extra revenue from frozen thresholds)")
    st.dataframe(drag, use_container_width=True)
