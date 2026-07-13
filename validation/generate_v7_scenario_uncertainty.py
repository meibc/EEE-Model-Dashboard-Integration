#!/usr/bin/env python3
"""Generate paired uncertainty scenario impacts vs baseline for v7 dashboard integration.

This script answers: under posterior/SEM uncertainty, how much does each scenario
change outcomes relative to baseline?

Key design choice:
- baseline and scenario use the same sampled SEM and CDC draw indices.
- impact is summarized from paired differences:
      delta_draw = scenario_draw - baseline_draw

This is preferable to subtracting separately summarized scenario and baseline
intervals because shared parameter uncertainty is preserved within each draw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RuntimeConfig
from input import load_uncertainty_inputs
from prediction.joint import UncertaintyRunner


SCENARIO_CODEBOOK = {
    "s1_reduce_ahs": {
        "state_codes": ["reduce_ahs"],
        "relationship_codes": [],
        "label": "Reduce anticipated healthcare stigma",
    },
    "s2_reduce_gss": {
        "state_codes": ["reduce_gss"],
        "relationship_codes": [],
        "label": "Reduce general social stigma",
    },
    "s3_reduce_family_stigma": {
        "state_codes": ["reduce_family_stigma"],
        "relationship_codes": [],
        "label": "Reduce family stigma",
    },
    "s4_increase_seehcp": {
        "state_codes": ["increase_seehcp"],
        "relationship_codes": [],
        "label": "Increase healthcare contact",
    },
    "s5_reduce_risk": {
        "state_codes": ["reduce_risk"],
        "relationship_codes": [],
        "label": "Reduce risk behavior",
    },
    "s6_weaken_stigma_to_care": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_care"],
        "label": "Weaken stigma → care pathway",
    },
    "s7_weaken_stigma_to_prep": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_prep"],
        "label": "Weaken stigma → PrEP pathway",
    },
    "s8_weaken_stigma_to_hivtest": {
        "state_codes": [],
        "relationship_codes": ["weaken_stigma_to_hivtest"],
        "label": "Weaken stigma → HIV test pathway",
    },
    "s9_combined_stigma_package": {
        "state_codes": ["reduce_ahs", "reduce_gss", "reduce_family_stigma"],
        "relationship_codes": [
            "weaken_stigma_to_care",
            "weaken_stigma_to_prep",
            "weaken_stigma_to_hivtest",
        ],
        "label": "Combined stigma package",
    },
}

# lower = lower value is beneficial; higher = higher value is beneficial.
INDICATORS = {
    "incidence": "lower",
    "diagnoses": "lower",
    "prep_on_count": "higher",
    "undiagnosed": "lower",
}


def summarize(vals: np.ndarray, prefix: str) -> dict[str, float]:
    vals = np.asarray(vals, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_p025": float(np.quantile(vals, 0.025)),
        f"{prefix}_p05": float(np.quantile(vals, 0.05)),
        f"{prefix}_median": float(np.quantile(vals, 0.5)),
        f"{prefix}_p95": float(np.quantile(vals, 0.95)),
        f"{prefix}_p975": float(np.quantile(vals, 0.975)),
    }


def build_runner(
    sem_loader,
    cdc_loader,
    units,
    model_years: np.ndarray,
    state_codes: list[str] | None = None,
    relationship_codes: list[str] | None = None,
) -> UncertaintyRunner:
    return UncertaintyRunner(
        sem_loader=sem_loader,
        cdc_params_loader=cdc_loader,
        units=units,
        model_years=model_years,
        state_intervention_codes=list(state_codes or []),
        relationship_intervention_codes=list(relationship_codes or []),
    )


def extract_endpoint(sample, year_idx: int) -> dict[str, float]:
    out = sample.cdc_output
    return {
        "incidence": float(out.incidence[year_idx]),
        "diagnoses": float(out.diagnosed[year_idx]),
        "prep_on_count": float(out.prep_on_count[year_idx]),
        "undiagnosed": float(out.undiagnosed[year_idx]),
    }


def compute_scenario_impacts(
    *,
    standardized_input: Path,
    output_dir: Path,
    endpoint_year: int,
    n_samples: int,
    seed: int,
    unit_ids: list[str] | None = None,
) -> tuple[Path, Path]:
    cfg = RuntimeConfig(
        mode="uncertainty",
        scenario_mode="baseline",
        standardized_input_path=standardized_input,
        target_end_year=endpoint_year,
        n_samples=n_samples,
        seed=seed,
        show_progress=False,
        unit_ids=unit_ids,
    )
    sem_loader, units, cdc_loader, model_years = load_uncertainty_inputs(cfg)
    model_years = np.asarray(model_years, dtype=int)
    if endpoint_year not in set(model_years.tolist()):
        raise ValueError(f"endpoint_year={endpoint_year} not in model_years={model_years.tolist()}")
    year_idx = int(np.where(model_years == endpoint_year)[0][0])

    available_units = [uid for uid in sem_loader.geo_names if uid in units and uid in cdc_loader.geo_names]
    if unit_ids:
        requested = set(unit_ids)
        available_units = [uid for uid in available_units if uid in requested]
    if not available_units:
        raise ValueError("No valid units available for scenario uncertainty run")

    base_runner = build_runner(sem_loader, cdc_loader, units, model_years)

    # Paired draw indices by state, matching the existing UncertaintyRunner.run_all seed convention.
    draw_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, uid in enumerate(available_units):
        rng = np.random.default_rng(seed + i)
        draw_indices[uid] = (
            rng.choice(base_runner.S_sem, size=n_samples, replace=True),
            rng.choice(base_runner.S_cdc, size=n_samples, replace=True),
        )

    print(f"Computing baseline paired draws: {len(available_units)} states x {n_samples} samples", flush=True)
    baseline = {uid: {ind: np.empty(n_samples, dtype=float) for ind in INDICATORS} for uid in available_units}
    for uid in available_units:
        idx_sem, idx_cdc = draw_indices[uid]
        for j, (s_sem, s_cdc) in enumerate(zip(idx_sem, idx_cdc)):
            vals = extract_endpoint(base_runner.predict_sample(uid, int(s_sem), int(s_cdc)), year_idx)
            for indicator, value in vals.items():
                baseline[uid][indicator][j] = value

    state_rows = []
    aggregate_rows = []

    for scenario, spec in SCENARIO_CODEBOOK.items():
        print(f"Computing {scenario}: {spec['label']}", flush=True)
        runner = build_runner(
            sem_loader,
            cdc_loader,
            units,
            model_years,
            state_codes=spec["state_codes"],
            relationship_codes=spec["relationship_codes"],
        )
        scenario_vals = {uid: {ind: np.empty(n_samples, dtype=float) for ind in INDICATORS} for uid in available_units}
        for uid in available_units:
            idx_sem, idx_cdc = draw_indices[uid]
            for j, (s_sem, s_cdc) in enumerate(zip(idx_sem, idx_cdc)):
                vals = extract_endpoint(runner.predict_sample(uid, int(s_sem), int(s_cdc)), year_idx)
                for indicator, value in vals.items():
                    scenario_vals[uid][indicator][j] = value

        for indicator, direction in INDICATORS.items():
            base_agg = np.sum([baseline[uid][indicator] for uid in available_units], axis=0)
            scen_agg = np.sum([scenario_vals[uid][indicator] for uid in available_units], axis=0)
            delta_agg = scen_agg - base_agg
            pct_agg = delta_agg / np.maximum(base_agg, 1e-12) * 100
            beneficial = delta_agg < 0 if direction == "lower" else delta_agg > 0
            row = {
                "scenario": scenario,
                "scenario_label": spec["label"],
                "year": endpoint_year,
                "indicator": indicator,
                "n_samples": n_samples,
                "beneficial_direction": direction,
                "prob_beneficial": float(np.mean(beneficial)),
            }
            row.update(summarize(base_agg, "baseline"))
            row.update(summarize(scen_agg, "scenario"))
            row.update(summarize(delta_agg, "delta"))
            row.update(summarize(pct_agg, "pct_delta"))
            aggregate_rows.append(row)

            for uid in available_units:
                base = baseline[uid][indicator]
                scen = scenario_vals[uid][indicator]
                delta = scen - base
                pct = delta / np.maximum(base, 1e-12) * 100
                beneficial = delta < 0 if direction == "lower" else delta > 0
                srow = {
                    "scenario": scenario,
                    "scenario_label": spec["label"],
                    "state": uid,
                    "year": endpoint_year,
                    "indicator": indicator,
                    "n_samples": n_samples,
                    "beneficial_direction": direction,
                    "prob_beneficial": float(np.mean(beneficial)),
                }
                srow.update(summarize(base, "baseline"))
                srow.update(summarize(scen, "scenario"))
                srow.update(summarize(delta, "delta"))
                srow.update(summarize(pct, "pct_delta"))
                state_rows.append(srow)

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"v7_uncertainty_scenario_endpoint_{endpoint_year}_state_vs_baseline.csv"
    aggregate_path = output_dir / f"v7_uncertainty_scenario_endpoint_{endpoint_year}_aggregate_vs_baseline.csv"
    pd.DataFrame(state_rows).to_csv(state_path, index=False)
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)
    return state_path, aggregate_path


def make_example_plot(aggregate_csv: Path, output_path: Path, indicator: str = "incidence") -> Path:
    """Create a simple aggregate scenario-impact plot from an existing paired-delta CSV."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(aggregate_csv)
    d = df[df["indicator"].eq(indicator)].copy()
    if d.empty:
        raise ValueError(f"No rows found for indicator={indicator!r} in {aggregate_csv}")

    # For lower-is-better outcomes, plot reduction as positive benefit.
    if indicator in {"incidence", "diagnoses", "undiagnosed"}:
        d["benefit_median"] = -d["delta_median"]
        d["benefit_lo"] = -d["delta_p95"]
        d["benefit_hi"] = -d["delta_p05"]
        x_label = f"{indicator.replace('_', ' ').title()} reduction vs baseline in {int(d['year'].iloc[0])}"
    else:
        d["benefit_median"] = d["delta_median"]
        d["benefit_lo"] = d["delta_p05"]
        d["benefit_hi"] = d["delta_p95"]
        x_label = f"{indicator.replace('_', ' ').title()} increase vs baseline in {int(d['year'].iloc[0])}"

    d = d.sort_values("benefit_median", ascending=True)
    y = np.arange(len(d))
    x = d["benefit_median"].to_numpy()
    xerr = np.vstack([x - d["benefit_lo"].to_numpy(), d["benefit_hi"].to_numpy() - x])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = np.where(x >= 0, "#2a9d8f", "#e76f51")
    ax.barh(y, x, color=colors, alpha=0.82)
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="0.25", elinewidth=1.4, capsize=3)
    ax.axvline(0, color="0.2", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["scenario_label"].tolist(), fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_title("Paired uncertainty scenario impact vs baseline")
    ax.grid(axis="x", alpha=0.22)

    for yi, (_, row) in enumerate(d.iterrows()):
        ax.text(
            row["benefit_median"],
            yi,
            f"  Pr={row['prob_beneficial']:.2f}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paired v7 scenario uncertainty impacts vs baseline")
    parser.add_argument("--standardized-input", type=Path, default=Path("standardized_input_v7.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v7_scenario_results_2036"))
    parser.add_argument("--endpoint-year", type=int, default=2036)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--units", nargs="*", default=None)
    parser.add_argument("--skip-compute", action="store_true", help="Only make example plot from existing aggregate CSV")
    parser.add_argument("--make-example-plot", action="store_true", help="Create an example aggregate impact plot")
    parser.add_argument("--plot-indicator", default="incidence", choices=list(INDICATORS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_path = args.output_dir / f"v7_uncertainty_scenario_endpoint_{args.endpoint_year}_aggregate_vs_baseline.csv"
    if not args.skip_compute:
        state_path, aggregate_path = compute_scenario_impacts(
            standardized_input=args.standardized_input,
            output_dir=args.output_dir,
            endpoint_year=args.endpoint_year,
            n_samples=args.n_samples,
            seed=args.seed,
            unit_ids=args.units,
        )
        print(f"saved {state_path.resolve()}")
        print(f"saved {aggregate_path.resolve()}")

    if args.make_example_plot:
        plot_path = args.output_dir / f"v7_scenario_impact_{args.plot_indicator}_{args.endpoint_year}.png"
        make_example_plot(aggregate_path, plot_path, indicator=args.plot_indicator)
        print(f"saved {plot_path.resolve()}")


if __name__ == "__main__":
    main()
