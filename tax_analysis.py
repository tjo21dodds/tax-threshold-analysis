"""
UK Income Tax Threshold Analysis
=================================
Analyses income tax revenue under three threshold scenarios:

  1. Frozen Thresholds  – thresholds held at 2024/25 levels (fiscal drag)
  2. Inflation-Uprated  – thresholds rise each year with CPI
  3. Wage-Growth-Uprated – thresholds rise each year with Average Weekly Earnings

In all scenarios wage growth (AWE) drives the underlying income distribution
upward — only the threshold policy differs between scenarios.

An RPI spending baseline is included as a reference line representing how
government expenditure is expected to grow (indexed from the base year).

Base year: 2024/25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ── Base year parameters (2024/25) ────────────────────────────────────────────
BASE_YEAR          = 2024
PERSONAL_ALLOWANCE = 12_570   # £ – frozen since 2021/22
BASIC_RATE_LIMIT   = 50_270   # £ – upper bound of the basic-rate band
HIGHER_RATE_LIMIT  = 125_140  # £ – upper bound of the higher-rate band
TAPER_THRESHOLD    = 100_000  # £ – income above which personal allowance tapers

TAX_RATES = {
    "basic":      0.20,
    "higher":     0.40,
    "additional": 0.45,
}

# HMRC estimate of UK income taxpayers (2024)
NUM_TAXPAYERS = 34_700_000

# ── Economic assumptions ──────────────────────────────────────────────────────
ANNUAL_INFLATION   = 0.025  # CPI: Bank of England target
ANNUAL_RPI         = 0.035  # RPI: typically ~1 pp above CPI
ANNUAL_WAGE_GROWTH = 0.040  # AWE: OBR central forecast

PROJECTION_YEARS = 5  # 2025/26 – 2029/30

# ── Income distribution ───────────────────────────────────────────────────────
# Calibrated to ONS ASHE 2024: median ≈ £35k, mean ≈ £42k
# A lognormal distribution is a standard model for earnings.
# For X ~ LN(μ, σ): median = exp(μ), mean = exp(μ + σ²/2)
# ⟹ σ² = 2(ln(mean) − ln(median))
_MEDIAN_INCOME = 35_000
_MEAN_INCOME   = 42_000
_LN_MU         = np.log(_MEDIAN_INCOME)
_LN_SIGMA      = np.sqrt(2.0 * (np.log(_MEAN_INCOME) - _LN_MU))
INCOME_DIST    = stats.lognorm(s=_LN_SIGMA, scale=np.exp(_LN_MU))

# Integration grid (base-year incomes; scaled later to account for wage growth)
_MAX_INCOME           = 600_000  # upper bound; lognormal density is negligible above this
_INTEGRATION_POINTS   = 200_000  # number of quadrature points
_INCOMES  = np.linspace(1, _MAX_INCOME, _INTEGRATION_POINTS)
_BASE_PDF = INCOME_DIST.pdf(_INCOMES)


# ── Tax calculation helpers ───────────────────────────────────────────────────

def effective_personal_allowance(income: float, pa: float) -> float:
    """
    Return the personal allowance after applying the income taper.

    Above £100,000 the allowance reduces by £1 for every £2 of additional
    income, reaching zero at £100,000 + 2 × pa.
    """
    if income <= TAPER_THRESHOLD:
        return pa
    reduction = (income - TAPER_THRESHOLD) / 2.0
    return max(0.0, pa - reduction)


def compute_tax(income: float, pa: float, basic_limit: float, higher_limit: float) -> float:
    """
    Calculate income tax for a single taxpayer.

    Parameters
    ----------
    income       : gross annual income (£)
    pa           : personal allowance threshold (£)
    basic_limit  : upper limit of the basic-rate band (£)
    higher_limit : upper limit of the higher-rate band (£)
    """
    eff_pa  = effective_personal_allowance(income, pa)
    taxable = max(0.0, income - eff_pa)
    if taxable == 0.0:
        return 0.0

    basic_band  = basic_limit  - pa
    higher_band = higher_limit - basic_limit

    tax = 0.0

    basic_portion = min(taxable, basic_band)
    tax     += basic_portion * TAX_RATES["basic"]
    taxable -= basic_portion

    if taxable > 0:
        higher_portion = min(taxable, higher_band)
        tax     += higher_portion * TAX_RATES["higher"]
        taxable -= higher_portion

    if taxable > 0:
        tax += taxable * TAX_RATES["additional"]

    return tax


def total_revenue(pa: float, basic_limit: float, higher_limit: float,
                  income_scale: float = 1.0) -> float:
    """
    Estimate total income tax revenue (£ billion) via numerical integration.

    ``income_scale`` captures cumulative wage growth relative to the base year:
    each taxpayer's income is assumed to be ``income_scale`` times higher than
    in the base year distribution.  We integrate

        E[tax(s·X)] = ∫ tax(s·x, pa, bl, hl) · f(x) dx

    over the base-year distribution f, then multiply by the number of taxpayers.
    """
    scaled_incomes = _INCOMES * income_scale
    taxes = np.vectorize(compute_tax)(scaled_incomes, pa, basic_limit, higher_limit)
    expected_tax = np.trapezoid(taxes * _BASE_PDF, _INCOMES)
    return expected_tax * NUM_TAXPAYERS / 1e9


# ── Scenario projection ───────────────────────────────────────────────────────

def build_scenarios(
    cpi_rate: float = ANNUAL_INFLATION,
    rpi_rate: float = ANNUAL_RPI,
    wage_rate: float = ANNUAL_WAGE_GROWTH,
) -> pd.DataFrame:
    """
    Project income tax revenue for each scenario over PROJECTION_YEARS years.

    Parameters
    ----------
    cpi_rate  : annual CPI/inflation rate used to uprate thresholds in scenario 2
    rpi_rate  : annual RPI rate used as a government-spending growth proxy
    wage_rate : annual wage-growth rate (AWE); identical across all scenarios

    The ``wage_rate`` shifts every taxpayer's income each year equally in all
    three threshold scenarios — only the threshold policy differs.

    An "RPI Spending Baseline" column is computed as the base-year revenue grown
    at ``rpi_rate`` each year, representing expected government expenditure growth.

    Returns a DataFrame with one row per tax year (including the base year).
    """
    # Base-year revenue: used to anchor the RPI spending growth baseline
    base_rev = total_revenue(PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)

    rows = []
    for t in range(PROJECTION_YEARS + 1):  # t=0 is the base year 2024/25
        year_label = f"{BASE_YEAR + t}/{str(BASE_YEAR + t + 1)[-2:]}"
        wage_scale = (1 + wage_rate) ** t
        inf_scale  = (1 + cpi_rate)  ** t
        rpi_scale  = (1 + rpi_rate)  ** t

        # Scenario 1: thresholds frozen at 2024/25 levels
        rev_frozen = total_revenue(
            PERSONAL_ALLOWANCE,
            BASIC_RATE_LIMIT,
            HIGHER_RATE_LIMIT,
            income_scale=wage_scale,
        )

        # Scenario 2: thresholds uprated annually with CPI inflation
        rev_inflation = total_revenue(
            PERSONAL_ALLOWANCE * inf_scale,
            BASIC_RATE_LIMIT   * inf_scale,
            HIGHER_RATE_LIMIT  * inf_scale,
            income_scale=wage_scale,
        )

        # Scenario 3: thresholds uprated annually with wage growth
        rev_wages = total_revenue(
            PERSONAL_ALLOWANCE * wage_scale,
            BASIC_RATE_LIMIT   * wage_scale,
            HIGHER_RATE_LIMIT  * wage_scale,
            income_scale=wage_scale,
        )

        # RPI spending baseline: base-year revenue grown at RPI — proxy for
        # expected government expenditure increases over time
        rpi_spending_baseline = base_rev * rpi_scale

        rows.append({
            "Tax Year":                      year_label,
            "Frozen Thresholds (£bn)":       round(rev_frozen, 1),
            "Inflation-Uprated (£bn)":       round(rev_inflation, 1),
            "Wage-Growth-Uprated (£bn)":     round(rev_wages, 1),
            "RPI Spending Baseline (£bn)":   round(rpi_spending_baseline, 1),
        })

    return pd.DataFrame(rows)


# ── Output helpers ────────────────────────────────────────────────────────────

def print_results(df: pd.DataFrame) -> None:
    """Print the revenue table and fiscal-drag summary to stdout."""
    print("\nUK Income Tax Revenue Projections")
    print("=" * 80)
    print("Assumptions:")
    print(f"  Annual wage growth (AWE):  {ANNUAL_WAGE_GROWTH:.1%}  [identical across all scenarios]")
    print(f"  Annual CPI inflation:      {ANNUAL_INFLATION:.1%}")
    print(f"  Annual RPI inflation:      {ANNUAL_RPI:.1%}  [spending growth proxy]")
    print(f"  Number of taxpayers:       {NUM_TAXPAYERS:,}")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Fiscal-drag columns (extra revenue because thresholds were not uprated)
    drag = df[["Tax Year"]].copy()
    drag["vs Inflation (£bn)"] = (
        df["Frozen Thresholds (£bn)"] - df["Inflation-Uprated (£bn)"]
    ).round(1)
    drag["vs Wage Growth (£bn)"] = (
        df["Frozen Thresholds (£bn)"] - df["Wage-Growth-Uprated (£bn)"]
    ).round(1)

    print("Fiscal Drag (extra revenue from frozen thresholds vs each scenario):")
    print(drag.to_string(index=False))
    print()


def plot_results(df: pd.DataFrame, output_path: str = "tax_revenue_scenarios.png") -> None:
    """Save a line chart comparing the three scenarios with an RPI spending baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["Tax Year"], df["Frozen Thresholds (£bn)"],
            marker="o", linewidth=2, label="Frozen Thresholds")
    ax.plot(df["Tax Year"], df["Inflation-Uprated (£bn)"],
            marker="s", linewidth=2, label="Inflation-Uprated (CPI)")
    ax.plot(df["Tax Year"], df["Wage-Growth-Uprated (£bn)"],
            marker="^", linewidth=2, label="Wage-Growth-Uprated (AWE)")
    ax.plot(df["Tax Year"], df["RPI Spending Baseline (£bn)"],
            linewidth=2, linestyle="--", color="grey",
            label="RPI Spending Baseline (govt. expenditure proxy)")

    ax.set_title("UK Income Tax Revenue by Threshold Scenario", fontsize=14)
    ax.set_xlabel("Tax Year")
    ax.set_ylabel("Income Tax Revenue (£ billion)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Chart saved to {output_path}")


def plot_rpi_comparison(
    df: pd.DataFrame,
    output_path: str = "tax_revenue_vs_rpi.png",
) -> None:
    """
    Save a chart showing how each scenario's revenue compares to the RPI
    spending baseline (a proxy for expected government expenditure growth).

    A positive value means the scenario raises *more* revenue than spending is
    expected to increase; a negative value means revenue lags behind spending.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline = df["RPI Spending Baseline (£bn)"]

    ax.plot(
        df["Tax Year"],
        df["Frozen Thresholds (£bn)"] - baseline,
        marker="o", linewidth=2, label="Frozen Thresholds vs RPI Spending",
    )
    ax.plot(
        df["Tax Year"],
        df["Inflation-Uprated (£bn)"] - baseline,
        marker="s", linewidth=2, label="CPI-Uprated vs RPI Spending",
    )
    ax.plot(
        df["Tax Year"],
        df["Wage-Growth-Uprated (£bn)"] - baseline,
        marker="^", linewidth=2, label="Wage-Growth-Uprated vs RPI Spending",
    )
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(
        "Revenue vs RPI Spending Baseline\n"
        "(positive = revenue grows faster than expected govt. expenditure)",
        fontsize=14,
    )
    ax.set_xlabel("Tax Year")
    ax.set_ylabel("Revenue minus RPI Spending Baseline (£ billion)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"RPI comparison chart saved to {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_scenarios()
    print_results(df)
    plot_results(df)
    plot_rpi_comparison(df)
