from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from config import RuntimeConfig
from plotting import save_run_plots
from runner import run_prediction


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference-only runtime for EEE-SD model")
    p.add_argument("--mode", choices=["deterministic", "uncertainty"], default="deterministic")
    p.add_argument("--scenario-mode", choices=["baseline", "intervention"], default="baseline")
    p.add_argument("--target-end-year", type=int, default=2036)
    p.add_argument("--units", nargs="*", default=None)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save", type=str, default=None)
    p.add_argument(
        "--standardized-input",
        type=str,
        default="standardized_input_v8.npz",
        help="Path to consolidated standardized input (.npz)",
    )
    p.add_argument("--state-codes", nargs="*", default=[])
    p.add_argument("--relationship-codes", nargs="*", default=[])
    p.add_argument("--plot", action="store_true", help="Save sanity plots for selected units")
    p.add_argument("--plot-dir", type=str, default="plots", help="Directory to save plot PNGs")
    p.add_argument("--plot-units", nargs="*", default=None, help="Units to plot (default: run units)")
    p.add_argument(
        "--plot-compare-baseline",
        action="store_true",
        help="Overlay the opposite scenario (baseline/intervention) on the same plots",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = RuntimeConfig(
        mode=args.mode,
        scenario_mode=args.scenario_mode,
        target_end_year=args.target_end_year,
        unit_ids=args.units,
        n_samples=args.n_samples,
        seed=args.seed,
        standardized_input_path=Path(args.standardized_input),
        state_intervention_codes=args.state_codes,
        relationship_intervention_codes=args.relationship_codes,
        save_output_path=Path(args.save) if args.save else None,
    )

    out = run_prediction(cfg)
    plot_paths = []
    if args.plot:
        comparison_out = None
        comparison_label = None
        if args.plot_compare_baseline:
            compare_cfg = deepcopy(cfg)
            if cfg.scenario_mode == "intervention":
                compare_cfg.scenario_mode = "baseline"
                compare_cfg.state_intervention_codes = []
                compare_cfg.relationship_intervention_codes = []
                comparison_label = "baseline"
            else:
                compare_cfg.scenario_mode = "intervention"
                comparison_label = "intervention"
            compare_cfg.save_output_path = None
            comparison_out = run_prediction(compare_cfg)

        plot_paths = save_run_plots(
            cfg=cfg,
            out=out,
            plot_dir=Path(args.plot_dir),
            unit_ids=args.plot_units or cfg.unit_ids,
            comparison_out=comparison_out,
            comparison_label=comparison_label,
        )

    if cfg.mode == "deterministic":
        print(f"Deterministic run complete: {len(out.results)} units")
    else:
        print(f"Uncertainty run complete: {len(out.results)} units")
    if plot_paths:
        print(f"Plots saved: {len(plot_paths)} files in {Path(args.plot_dir)}")


if __name__ == "__main__":
    main()
