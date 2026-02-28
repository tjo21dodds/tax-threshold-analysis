"""
Unit tests for tax_analysis.py
"""

import math
import pytest
import pandas as pd

from tax_analysis.tax_analysis import (
    PERSONAL_ALLOWANCE,
    BASIC_RATE_LIMIT,
    HIGHER_RATE_LIMIT,
    TAPER_THRESHOLD,
    TAX_RATES,
    ANNUAL_RPI,
    effective_personal_allowance,
    compute_tax,
    total_revenue,
    build_scenarios,
    PROJECTION_YEARS,
)


# ── effective_personal_allowance ─────────────────────────────────────────────

class TestEffectivePersonalAllowance:
    def test_below_taper_threshold(self):
        """PA is unchanged for income at or below £100k."""
        assert effective_personal_allowance(50_000, PERSONAL_ALLOWANCE) == PERSONAL_ALLOWANCE

    def test_at_taper_threshold(self):
        """PA is unchanged exactly at the £100k boundary."""
        assert effective_personal_allowance(TAPER_THRESHOLD, PERSONAL_ALLOWANCE) == PERSONAL_ALLOWANCE

    def test_taper_reduces_allowance(self):
        """£10k above the taper start should reduce PA by £5k."""
        result = effective_personal_allowance(110_000, PERSONAL_ALLOWANCE)
        expected = PERSONAL_ALLOWANCE - 5_000
        assert math.isclose(result, expected, abs_tol=1)

    def test_taper_reaches_zero(self):
        """PA is zero at £100k + 2×PA."""
        zero_income = TAPER_THRESHOLD + 2 * PERSONAL_ALLOWANCE
        assert effective_personal_allowance(zero_income, PERSONAL_ALLOWANCE) == 0.0

    def test_taper_cannot_go_negative(self):
        """PA is floored at zero for very high incomes."""
        assert effective_personal_allowance(500_000, PERSONAL_ALLOWANCE) == 0.0


# ── compute_tax ───────────────────────────────────────────────────────────────

class TestComputeTax:
    def test_below_personal_allowance(self):
        """No tax is due on income at or below the personal allowance."""
        assert compute_tax(PERSONAL_ALLOWANCE, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT) == 0.0
        assert compute_tax(10_000, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT) == 0.0

    def test_zero_income(self):
        assert compute_tax(0, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT) == 0.0

    def test_basic_rate_only(self):
        """Income of £22,570 → taxable £10,000 → tax £2,000 at 20 %."""
        income = PERSONAL_ALLOWANCE + 10_000
        expected = 10_000 * TAX_RATES["basic"]
        result = compute_tax(income, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert math.isclose(result, expected, abs_tol=1e-6)

    def test_higher_rate_kicks_in(self):
        """Income just above the basic-rate limit should attract higher-rate tax."""
        income = BASIC_RATE_LIMIT + 10_000  # £60,270
        basic_taxable = BASIC_RATE_LIMIT - PERSONAL_ALLOWANCE
        expected = (basic_taxable * TAX_RATES["basic"]
                    + 10_000 * TAX_RATES["higher"])
        result = compute_tax(income, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert math.isclose(result, expected, abs_tol=1e-6)

    def test_additional_rate_kicks_in(self):
        """Income above £125,140 should attract the additional (45 %) rate."""
        income = HIGHER_RATE_LIMIT + 20_000  # £145,140
        # At this income the personal allowance has fully tapered to zero
        basic_band  = BASIC_RATE_LIMIT  - PERSONAL_ALLOWANCE
        higher_band = HIGHER_RATE_LIMIT - BASIC_RATE_LIMIT
        # eff_pa = 0 at this income, so taxable = income
        # but basic_band and higher_band are computed from the passed pa
        result = compute_tax(income, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert result > 0

    def test_higher_income_pays_more_tax(self):
        """A higher income should always result in at least as much tax."""
        tax_low  = compute_tax(30_000, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        tax_high = compute_tax(80_000, PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert tax_high > tax_low

    def test_frozen_vs_uprated_threshold(self):
        """Raising the personal allowance should reduce tax for the same income."""
        income = 40_000
        tax_frozen  = compute_tax(income, 12_570, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        tax_uprated = compute_tax(income, 14_000, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert tax_uprated < tax_frozen


# ── total_revenue ─────────────────────────────────────────────────────────────

class TestTotalRevenue:
    def test_returns_positive_value(self):
        rev = total_revenue(PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert rev > 0

    def test_plausible_magnitude(self):
        """Revenue should be in the rough range of £200bn–£300bn for 2024/25."""
        rev = total_revenue(PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert 100 < rev < 400

    def test_higher_wage_scale_raises_revenue(self):
        """Wage growth increases total taxable income and hence revenue."""
        rev_base  = total_revenue(PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT, income_scale=1.0)
        rev_grown = total_revenue(PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT, income_scale=1.2)
        assert rev_grown > rev_base

    def test_higher_pa_reduces_revenue(self):
        """Raising the personal allowance reduces total revenue (all else equal)."""
        rev_low_pa  = total_revenue(12_570, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        rev_high_pa = total_revenue(15_000, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT)
        assert rev_high_pa < rev_low_pa

    def test_frozen_exceeds_wage_uprated(self):
        """Frozen thresholds should raise more revenue than wage-uprated thresholds."""
        wage_scale = 1.04
        rev_frozen = total_revenue(
            PERSONAL_ALLOWANCE, BASIC_RATE_LIMIT, HIGHER_RATE_LIMIT, income_scale=wage_scale
        )
        rev_uprated = total_revenue(
            PERSONAL_ALLOWANCE * wage_scale,
            BASIC_RATE_LIMIT   * wage_scale,
            HIGHER_RATE_LIMIT  * wage_scale,
            income_scale=wage_scale,
        )
        assert rev_frozen > rev_uprated


# ── build_scenarios ───────────────────────────────────────────────────────────

class TestBuildScenarios:
    def setup_method(self):
        self.df = build_scenarios()

    def test_returns_dataframe(self):
        assert isinstance(self.df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        assert len(self.df) == PROJECTION_YEARS + 1  # base year + projection years

    def test_required_columns_present(self):
        expected_cols = {
            "Tax Year",
            "Frozen Thresholds (£bn)",
            "CPI-Uprated (£bn)",
            "Wage-Growth-Uprated (£bn)",
            "RPI-Uprated (£bn)"
        }
        assert expected_cols.issubset(set(self.df.columns))

    def test_base_year_label(self):
        assert self.df.iloc[0]["Tax Year"] == "2024/25"

    def test_final_year_label(self):
        assert self.df.iloc[-1]["Tax Year"] == "2029/30"

    def test_frozen_exceeds_cpi_uprated(self):
        """Fiscal drag: frozen thresholds raise more revenue than CPI-uprated ones."""
        for _, row in self.df.iterrows():
            assert row["Frozen Thresholds (£bn)"] >= row["CPI-Uprated (£bn)"], (
                f"Failed for {row['Tax Year']}"
            )
    def test_frozen_exceeds_rpi_uprated(self):
        """Fiscal drag: frozen thresholds raise more revenue than RPI-uprated ones."""
        for _, row in self.df.iterrows():
            assert row["Frozen Thresholds (£bn)"] >= row["RPI-Uprated (£bn)"], (
                f"Failed for {row['Tax Year']}"
            )

    def test_cpi_uprated_exceeds_wage_growth_uprated(self):
        """CPI uprating raises more revenue than wage uprating (cpi < wage growth)."""
        for _, row in self.df.iterrows():
            assert row["CPI-Uprated (£bn)"] >= row["Wage-Growth-Uprated (£bn)"], (
                f"Failed for {row['Tax Year']}"
            )
    def test_rpi_uprated_exceeds_wage_growth_uprated(self):
        """RPI uprating raises more revenue than wage uprating (rpi < wage growth)."""
        for _, row in self.df.iterrows():
            assert row["RPI-Uprated (£bn)"] >= row["Wage-Growth-Uprated (£bn)"], (
                f"Failed for {row['Tax Year']}"
            )
    def test_revenue_grows_over_time_for_frozen_scenario(self):
        """Frozen thresholds + wage growth → revenue increases each year."""
        revenues = self.df["Frozen Thresholds (£bn)"].tolist()
        assert all(revenues[i] <= revenues[i + 1] for i in range(len(revenues) - 1))


class TestBuildScenariosRPI:
    def setup_method(self):
        self.df = build_scenarios()

    def test_rpi_spending_baseline_column_present(self):
        """RPI Spending Baseline column should be present in the output."""
        assert "RPI Spending Baseline (£bn)" in self.df.columns

    def test_rpi_spending_baseline_equals_frozen_at_base_year(self):
        """At t=0 (base year) the RPI spending baseline equals the frozen-threshold revenue."""
        base_row = self.df.iloc[0]
        assert base_row["RPI Spending Baseline (£bn)"] == base_row["Frozen Thresholds (£bn)"]

    def test_rpi_spending_baseline_grows_over_time(self):
        """RPI spending baseline should increase each year."""
        values = self.df["RPI Spending Baseline (£bn)"].tolist()
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_custom_rates_accepted(self):
        """build_scenarios() should accept custom rpi/cpi/wage rate arguments."""
        df_custom = build_scenarios(rpi_rate=0.04, cpi_rate=0.02, wage_rate=0.05)
        assert isinstance(df_custom, pd.DataFrame)
        assert len(df_custom) == PROJECTION_YEARS + 1

    def test_annual_rpi_constant_positive(self):
        """ANNUAL_RPI should be a positive float."""
        assert isinstance(ANNUAL_RPI, float)
        assert ANNUAL_RPI > 0
