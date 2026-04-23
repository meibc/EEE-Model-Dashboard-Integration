from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from alignment import extend_years
from input.standardized_runtime import StandardizedBundle


def _to_unit_map(sem_output: Any) -> dict[str, Any]:
    return {u.id: u for u in sem_output.inputs.units}


def _plot_forecast_region(ax, start_year: float, end_year: float) -> None:
    ax.axvline(start_year, color="0.45", linestyle=":", linewidth=1.2)
    ax.axvspan(start_year, end_year, color="#eef6ff", alpha=0.45)


def _stack_metric(samples: list[Any], metric: str) -> np.ndarray:
    return np.asarray([getattr(s.cdc_output, metric) for s in samples], dtype=float)


def _get_raw_series(unit: Any, raw_name: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    vals = unit.get_cdc(raw_name) if raw_name is not None else None
    if vals is None or unit.cdc_years is None:
        return None, None

    ry = np.asarray(unit.cdc_years, dtype=float)
    rv = np.asarray(vals, dtype=float)
    if rv.size != ry.size:
        return None, None
    return ry, rv


def _plot_sem_var(
    ax,
    *,
    title: str,
    var_name: str,
    color: str,
    unit: Any,
    sem_years: np.ndarray,
    v_names: list[str],
    sem_med: np.ndarray,
    sem_lo: np.ndarray | None,
    sem_hi: np.ndarray | None,
    comparison_sem: np.ndarray | None,
    comparison_label: str | None = None,
) -> None:
    if var_name not in v_names or var_name not in unit.amis_names:
        ax.text(0.5, 0.5, f"{var_name} unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        return

    idx = v_names.index(var_name)
    y = np.asarray(sem_med[idx], dtype=float)
    ys = np.asarray(extend_years(sem_years, y.shape[0]), dtype=float)

    ax.plot(ys, y, color=color, linewidth=2.0, label="prediction")
    if sem_lo is not None and sem_hi is not None:
        lo = np.asarray(sem_lo[idx], dtype=float)
        hi = np.asarray(sem_hi[idx], dtype=float)
        ax.fill_between(ys, lo, hi, color=color, alpha=0.16, linewidth=0, label="prediction 5-95%")

    if comparison_sem is not None:
        by = np.asarray(comparison_sem[idx], dtype=float)
        bys = np.asarray(extend_years(sem_years, by.shape[0]), dtype=float)
        label = comparison_label or "comparison"
        ax.plot(bys, by, color=color, linestyle="--", linewidth=1.4, alpha=0.9, label=label)

    obs_years = np.asarray(unit.amis_years, dtype=float)
    obs = np.asarray(unit.get_amis(var_name), dtype=float)
    ax.scatter(obs_years, obs, color=color, s=24, alpha=0.9, label="raw")

    _plot_forecast_region(ax, float(np.max(obs_years)), float(np.max(ys)))
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)


def _plot_cdc_var(
    ax,
    *,
    title: str,
    color: str,
    years: np.ndarray,
    y_med: np.ndarray,
    y_lo: np.ndarray | None,
    y_hi: np.ndarray | None,
    comparison_med: np.ndarray | None,
    comparison_label: str | None,
    raw_years: np.ndarray | None,
    raw_values: np.ndarray | None,
    raw_label: str,
) -> None:
    years = np.asarray(years, dtype=float)
    y_med = np.asarray(y_med, dtype=float)

    ax.plot(years, y_med, color=color, linewidth=2.0, label="prediction")
    if y_lo is not None and y_hi is not None:
        ax.fill_between(years, np.asarray(y_lo, dtype=float), np.asarray(y_hi, dtype=float), color=color, alpha=0.16, linewidth=0, label="prediction 5-95%")

    if comparison_med is not None:
        label = comparison_label or "comparison"
        ax.plot(
            years,
            np.asarray(comparison_med, dtype=float),
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Output Count")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

    legend_handles, legend_labels = ax.get_legend_handles_labels()

    if raw_years is not None and raw_values is not None:
        ln = ax.plot(
            np.asarray(raw_years, dtype=float),
            np.asarray(raw_values, dtype=float),
            color="#6d597a",
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            alpha=0.85,
            label=raw_label,
            linestyle=":",
        )[0]
        legend_handles.append(ln)
        legend_labels.append(raw_label)

        forecast_start = float(np.max(raw_years))
    else:
        forecast_start = float(np.min(years))

    _plot_forecast_region(ax, forecast_start, float(np.max(years)))

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="best", fontsize=8)


def save_run_plots(
    cfg: Any,
    out: Any,
    plot_dir: str | Path,
    unit_ids: list[str] | None = None,
    comparison_out: Any | None = None,
    comparison_label: str | None = None,
) -> list[Path]:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sem_output = StandardizedBundle(Path(cfg.standardized_input_path)).build_sem_output()
    units = _to_unit_map(sem_output)

    available = list(out.results.keys())
    selected = available if unit_ids is None else [u for u in unit_ids if u in out.results]
    if not selected:
        return []

    paths: list[Path] = []
    is_unc = hasattr(next(iter(out.results.values())), "samples")

    sem_specs = [
        ("SEM: Testing", cfg.hivtest_var, "#2a9d8f"),
        ("SEM: PrEP Use", cfg.prep_var, "#e76f51"),
        ("SEM: Risk Behavior", cfg.risk_var, "#264653"),
    ]
    cdc_specs = [
        ("CDC: Incidence", "incidence", "#1d3557", "Estimated HIV incidence (MSM)", "incidence raw"),
        ("CDC: Diagnosed", "diagnosed", "#e63946", "HIV diagnoses", "diagnosed raw"),
        ("CDC: PrEP On", "prep_on_count", "#457b9d", "PrEP", "PrEP raw"),
    ]

    for uid in selected:
        fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
        ax = axes.ravel()

        unit = units[uid]
        sem_years = np.asarray(out.sem_years if hasattr(out, "sem_years") else sem_output.predictions.ts, dtype=float)

        if is_unc:
            unc_res = out.results[uid]
            sem_stack = np.asarray([s.sem_trajectory for s in unc_res.samples], dtype=float)
            sem_q05, sem_med, sem_q95 = np.quantile(sem_stack, [0.05, 0.5, 0.95], axis=0)

            comparison_sem = None
            comparison_unc = None
            if comparison_out is not None and uid in comparison_out.results:
                comparison_unc = comparison_out.results[uid]
                comparison_sem = np.quantile(
                    np.asarray([s.sem_trajectory for s in comparison_unc.samples], dtype=float),
                    0.5,
                    axis=0,
                )

            for i, (title, var_name, color) in enumerate(sem_specs):
                _plot_sem_var(
                    ax[i],
                    title=title,
                    var_name=var_name,
                    color=color,
                    unit=unit,
                    sem_years=sem_years,
                    v_names=out.v_names,
                    sem_med=sem_med,
                    sem_lo=sem_q05,
                    sem_hi=sem_q95,
                    comparison_sem=comparison_sem,
                    comparison_label=comparison_label,
                )

            cdc_years = np.asarray(unc_res.years, dtype=float)
            for j, (title, metric, color, raw_name, raw_label) in enumerate(cdc_specs, start=3):
                arr = _stack_metric(unc_res.samples, metric)
                q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95], axis=0)

                comp = None
                if comparison_unc is not None:
                    comp = np.quantile(_stack_metric(comparison_unc.samples, metric), 0.5, axis=0)

                ry, rv = _get_raw_series(unit, raw_name)

                _plot_cdc_var(
                    ax[j],
                    title=title,
                    color=color,
                    years=cdc_years,
                    y_med=q50,
                    y_lo=q05,
                    y_hi=q95,
                    comparison_med=comp,
                    comparison_label=comparison_label,
                    raw_years=ry,
                    raw_values=rv,
                    raw_label=raw_label,
                )

            mode_tag = "uncertainty"

        else:
            res = out.results[uid]
            sem_med = np.asarray(res.sem_trajectory, dtype=float)

            comparison_sem = None
            comparison_cdc = None
            if comparison_out is not None and uid in comparison_out.results:
                comparison_sem = np.asarray(comparison_out.results[uid].sem_trajectory, dtype=float)
                comparison_cdc = comparison_out.results[uid].cdc_output

            for i, (title, var_name, color) in enumerate(sem_specs):
                _plot_sem_var(
                    ax[i],
                    title=title,
                    var_name=var_name,
                    color=color,
                    unit=unit,
                    sem_years=sem_years,
                    v_names=out.v_names,
                    sem_med=sem_med,
                    sem_lo=None,
                    sem_hi=None,
                    comparison_sem=comparison_sem,
                    comparison_label=comparison_label,
                )

            cdc_output = res.cdc_output
            cdc_years = np.asarray(cdc_output.years, dtype=float)
            for j, (title, metric, color, raw_name, raw_label) in enumerate(cdc_specs, start=3):
                y = np.asarray(getattr(cdc_output, metric), dtype=float)
                comp = None if comparison_cdc is None else np.asarray(getattr(comparison_cdc, metric), dtype=float)

                ry, rv = _get_raw_series(unit, raw_name)

                _plot_cdc_var(
                    ax[j],
                    title=title,
                    color=color,
                    years=cdc_years,
                    y_med=y,
                    y_lo=None,
                    y_hi=None,
                    comparison_med=comp,
                    comparison_label=comparison_label,
                    raw_years=ry,
                    raw_values=rv,
                    raw_label=raw_label,
                )

            mode_tag = "deterministic"

        fig.suptitle(f"{uid} | mode={mode_tag} | scenario={cfg.scenario_mode}")
        if comparison_out is not None:
            path = plot_dir / f"{uid}_{cfg.mode}_comparison.png"
        else:
            path = plot_dir / f"{uid}_{cfg.mode}_{cfg.scenario_mode}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths
