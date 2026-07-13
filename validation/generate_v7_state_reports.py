#!/usr/bin/env python3
"""Generate state-level v7 diagnostic and scenario report plots.

Uses existing CSV outputs:
1) fit assessment: observed vs posterior fit intervals from v7_all_states_full_annual_fit.csv
2) baseline projections: deterministic baseline incidence/diagnoses/PrEP through 2036
3) scenario comparisons: paired uncertainty endpoint deltas vs baseline at 2036

No model simulation is run by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_FIT_PATH = Path(
    "/Users/meibinchen/Documents/GitHub/EEE-SD-Model/sem_sd_1225_refactor/notebooks/transition_bayes_v7_pilot_outputs/v7_all_states_full_annual_fit.csv"
)
DEFAULT_SCENARIO_DIR = Path("outputs/v7_scenario_results_2036")


def pretty_indicator(x: str) -> str:
    return {
        "incidence": "Incidence",
        "diagnoses": "Diagnoses",
        "prep_on_count": "PrEP Count",
        "undiagnosed": "Undiagnosed",
    }.get(x, x.replace("_", " ").title())


def load_inputs(fit_path: Path, scenario_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()
    traj_path = scenario_dir / "v7_deterministic_scenario_trajectories_2017_2036.csv"
    impact_path = scenario_dir / "v7_uncertainty_scenario_endpoint_2036_state_vs_baseline.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing baseline/scenario trajectory file: {traj_path}")
    if not impact_path.exists():
        raise FileNotFoundError(f"Missing paired uncertainty scenario impact file: {impact_path}")
    traj = pd.read_csv(traj_path)
    impact = pd.read_csv(impact_path)
    return fit, traj, impact


def plot_fit_assessment(axs, fit: pd.DataFrame, state: str) -> None:
    if fit.empty:
        for ax in axs:
            ax.text(0.5, 0.5, "Fit CSV not found", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
        return

    d = fit[fit["state"].astype(str).eq(state)].sort_values("year")
    specs = [
        ("Incidence fit", "inc_observed", "inc_median", "inc_p05", "inc_p95", "#1d3557"),
        ("Diagnoses fit", "dx_observed", "dx_median", "dx_p05", "dx_p95", "#e63946"),
    ]
    for ax, (title, obs, med, lo, hi, color) in zip(axs, specs):
        if d.empty or med not in d:
            ax.text(0.5, 0.5, f"No fit data for {state}", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        x = d["year"].to_numpy()
        ax.fill_between(x, d[lo].to_numpy(), d[hi].to_numpy(), color=color, alpha=0.18, label="90% interval")
        ax.plot(x, d[med].to_numpy(), color=color, linewidth=2.2, label="fit median")
        ax.scatter(x, d[obs].to_numpy(), color="black", s=28, label="observed", zorder=3)
        ax.axvline(2020, color="0.55", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)


def plot_baseline_projection(axs, traj: pd.DataFrame, state: str) -> None:
    d = traj[(traj["state"].astype(str).eq(state)) & (traj["scenario"].eq("baseline"))].copy()
    specs = [
        ("Baseline incidence", "incidence", "#1d3557"),
        ("Baseline diagnoses", "diagnoses", "#e63946"),
        ("Baseline PrEP count", "prep_on_count", "#457b9d"),
    ]
    for ax, (title, indicator, color) in zip(axs, specs):
        s = d[d["indicator"].eq(indicator)].sort_values("year")
        if s.empty:
            ax.text(0.5, 0.5, f"No {indicator} baseline data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        ax.plot(s["year"], s["value"], color=color, linewidth=2.4)
        ax.axvline(2023, color="0.5", linestyle=":", linewidth=1, label="last CDC input")
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Count")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)


def plot_scenario_endpoint(axs, impact: pd.DataFrame, state: str, endpoint_year: int) -> None:
    d = impact[(impact["state"].astype(str).eq(state)) & (impact["year"].eq(endpoint_year))].copy()
    specs = [
        ("Incidence impact", "incidence", "reduction", "#2a9d8f"),
        ("Diagnoses impact", "diagnoses", "reduction", "#2a9d8f"),
        ("PrEP impact", "prep_on_count", "increase", "#457b9d"),
    ]
    for ax, (title, indicator, direction_label, color) in zip(axs, specs):
        s = d[d["indicator"].eq(indicator)].copy()
        if s.empty:
            ax.text(0.5, 0.5, f"No {indicator} impact data", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        # For lower-is-better indicators, convert negative delta to positive benefit.
        if indicator in {"incidence", "diagnoses", "undiagnosed"}:
            s["benefit_median"] = -s["delta_median"]
            s["benefit_lo"] = -s["delta_p95"]
            s["benefit_hi"] = -s["delta_p05"]
        else:
            s["benefit_median"] = s["delta_median"]
            s["benefit_lo"] = s["delta_p05"]
            s["benefit_hi"] = s["delta_p95"]

        s = s.sort_values("benefit_median", ascending=True)
        y = np.arange(len(s))
        x = s["benefit_median"].to_numpy()
        xerr = np.vstack([
            x - s["benefit_lo"].to_numpy(),
            s["benefit_hi"].to_numpy() - x,
        ])
        colors = np.where(x >= 0, color, "#e76f51")
        ax.barh(y, x, color=colors, alpha=0.82)
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.25", elinewidth=1.1, capsize=2)
        ax.axvline(0, color="0.2", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s["scenario_label"].tolist(), fontsize=7)
        ax.set_title(f"{title}: {endpoint_year}")
        ax.set_xlabel(f"{pretty_indicator(indicator)} {direction_label} vs baseline")
        ax.grid(axis="x", alpha=0.2)
        for yi, (_, row) in enumerate(s.iterrows()):
            ax.text(row["benefit_median"], yi, f" Pr={row['prob_beneficial']:.2f}", va="center", fontsize=7)


def write_state_summary_csv(impact: pd.DataFrame, output_dir: Path, endpoint_year: int) -> Path:
    keep = impact[impact["year"].eq(endpoint_year)].copy()
    keep["benefit_median"] = np.where(
        keep["indicator"].isin(["incidence", "diagnoses", "undiagnosed"]),
        -keep["delta_median"],
        keep["delta_median"],
    )
    keep["benefit_p05"] = np.where(
        keep["indicator"].isin(["incidence", "diagnoses", "undiagnosed"]),
        -keep["delta_p95"],
        keep["delta_p05"],
    )
    keep["benefit_p95"] = np.where(
        keep["indicator"].isin(["incidence", "diagnoses", "undiagnosed"]),
        -keep["delta_p05"],
        keep["delta_p95"],
    )
    cols = [
        "state", "scenario", "scenario_label", "indicator", "year",
        "benefit_median", "benefit_p05", "benefit_p95", "prob_beneficial",
        "baseline_median", "scenario_median", "delta_median", "pct_delta_median",
    ]
    path = output_dir / f"v7_state_level_scenario_impact_summary_{endpoint_year}.csv"
    keep[cols].sort_values(["state", "indicator", "benefit_median"], ascending=[True, True, False]).to_csv(path, index=False)
    return path


def make_report_for_state(state: str, fit: pd.DataFrame, traj: pd.DataFrame, impact: pd.DataFrame, output_dir: Path, endpoint_year: int) -> Path:
    fig = plt.figure(figsize=(15, 18), constrained_layout=True)
    subfigs = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.5])
    fig.suptitle(f"{state}: v7 model diagnostics and scenario impacts", fontsize=18, fontweight="bold")

    axs_fit = subfigs[0].subplots(1, 2)
    subfigs[0].suptitle("1. Fit assessment", fontsize=13, fontweight="bold")
    plot_fit_assessment(np.ravel(axs_fit), fit, state)

    axs_base = subfigs[1].subplots(1, 3)
    subfigs[1].suptitle("2. Baseline deterministic projections", fontsize=13, fontweight="bold")
    plot_baseline_projection(np.ravel(axs_base), traj, state)

    axs_scen = subfigs[2].subplots(1, 3)
    subfigs[2].suptitle("3. Scenario endpoint impacts vs baseline (paired uncertainty)", fontsize=13, fontweight="bold")
    plot_scenario_endpoint(np.ravel(axs_scen), impact, state, endpoint_year)

    path = output_dir / f"v7_state_report_{state}_{endpoint_year}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def make_pdf(states: list[str], fit: pd.DataFrame, traj: pd.DataFrame, impact: pd.DataFrame, output_dir: Path, endpoint_year: int) -> Path:
    path = output_dir / f"v7_state_reports_{endpoint_year}.pdf"
    with PdfPages(path) as pdf:
        for state in states:
            fig = plt.figure(figsize=(15, 18), constrained_layout=True)
            subfigs = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 1.5])
            fig.suptitle(f"{state}: v7 model diagnostics and scenario impacts", fontsize=18, fontweight="bold")
            axs_fit = subfigs[0].subplots(1, 2)
            subfigs[0].suptitle("1. Fit assessment", fontsize=13, fontweight="bold")
            plot_fit_assessment(np.ravel(axs_fit), fit, state)
            axs_base = subfigs[1].subplots(1, 3)
            subfigs[1].suptitle("2. Baseline deterministic projections", fontsize=13, fontweight="bold")
            plot_baseline_projection(np.ravel(axs_base), traj, state)
            axs_scen = subfigs[2].subplots(1, 3)
            subfigs[2].suptitle("3. Scenario endpoint impacts vs baseline (paired uncertainty)", fontsize=13, fontweight="bold")
            plot_scenario_endpoint(np.ravel(axs_scen), impact, state, endpoint_year)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate state-level v7 report plots from existing outputs")
    parser.add_argument("--states", nargs="*", default=["CA", "TX", "NY", "MS"])
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--fit-path", type=Path, default=DEFAULT_FIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v7_state_reports"))
    parser.add_argument("--endpoint-year", type=int, default=2036)
    parser.add_argument("--pdf", action="store_true", help="Also write a multi-page PDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit, traj, impact = load_inputs(args.fit_path, args.scenario_dir)
    summary_path = write_state_summary_csv(impact, args.output_dir, args.endpoint_year)
    print(f"saved {summary_path.resolve()}")
    for state in args.states:
        path = make_report_for_state(state, fit, traj, impact, args.output_dir, args.endpoint_year)
        print(f"saved {path.resolve()}")
    if args.pdf:
        path = make_pdf(args.states, fit, traj, impact, args.output_dir, args.endpoint_year)
        print(f"saved {path.resolve()}")


if __name__ == "__main__":
    main()
