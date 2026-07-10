from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from input.standardized_runtime import StandardizedBundle
from output import load


def _find_result_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*.pkl") if ".git" not in p.parts],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _is_uncertainty_output(obj: Any) -> bool:
    return hasattr(obj, "results") and bool(obj.results) and hasattr(next(iter(obj.results.values())), "samples")


def _plot_series(ax, years: np.ndarray, values: np.ndarray, label: str, color: str) -> None:
    ax.plot(years, values, linewidth=2.8, label=label, color=color, solid_capstyle="round")


def _style_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.9)
    ax.grid(axis="x", alpha=0.08, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.3)
    ax.spines["bottom"].set_alpha(0.3)
    ax.tick_params(axis="both", labelsize=9)


def _safe_get_raw_sem(unit: Any, var_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        vals = unit.get_amis(var_name)
    except Exception:
        return None, None
    if vals is None:
        return None, None
    y = np.asarray(vals, dtype=float)
    x = np.asarray(unit.amis_years, dtype=float)
    if x.size != y.size:
        return None, None
    return x, y


def _safe_get_raw_cdc(unit: Any, raw_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        vals = unit.get_cdc(raw_name)
    except Exception:
        return None, None
    if vals is None:
        return None, None
    y = np.asarray(vals, dtype=float)
    x = np.asarray(unit.cdc_years, dtype=float)
    if x.size != y.size:
        return None, None
    return x, y


def _sem_years_for_output(out: Any, sem_len: int, cdc_len: int) -> np.ndarray:
    if hasattr(out, "sem_years"):
        ys = np.asarray(out.sem_years, dtype=float)
        if ys.size == sem_len:
            return ys
    if hasattr(out, "years"):
        ys = np.asarray(out.years, dtype=float)
        if ys.size == sem_len:
            return ys
        if ys.size == cdc_len and sem_len != cdc_len:
            # Fallback when only CDC years are available in output metadata.
            return np.linspace(float(ys[0]), float(ys[-1]), num=sem_len)
    return np.arange(sem_len, dtype=float)


def _render_sem_uncertainty_panel(
    out: Any,
    unit_result: Any,
    sem_indices: list[int],
    sem_labels: list[str],
    raw_unit: Any | None,
) -> None:
    st.subheader("SEM trajectories (median with 5-95%)")
    sem_stack = np.asarray([s.sem_trajectory for s in unit_result.samples], dtype=float)
    sem_len = sem_stack.shape[2]
    cdc_len = len(np.asarray(unit_result.years, dtype=float))
    sem_years = _sem_years_for_output(out, sem_len=sem_len, cdc_len=cdc_len)

    fig, axes = plt.subplots(len(sem_indices), 1, figsize=(11, 3.6 * len(sem_indices)), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfd")
    if len(sem_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(sem_indices):
        q05, q50, q95 = np.quantile(sem_stack[:, idx, :], [0.05, 0.5, 0.95], axis=0)
        color = plt.cm.Set2(i % 8)
        _plot_series(axes[i], sem_years, q50, f"{sem_labels[i]} median", color)
        axes[i].fill_between(sem_years, q05, q95, color=color, alpha=0.2, label="5-95%")
        if raw_unit is not None:
            rx, ry = _safe_get_raw_sem(raw_unit, sem_labels[i])
            if rx is not None and ry is not None:
                axes[i].scatter(rx, ry, color=color, s=28, alpha=0.95, edgecolor="white", linewidth=0.5, label="observed")
        _style_axes(axes[i], sem_labels[i], "Value")
        axes[i].legend(loc="best")

    st.pyplot(fig)


def _render_cdc_uncertainty_panel(unit_result: Any, raw_unit: Any | None) -> None:
    st.subheader("CDC predictions (median with 5-95%)")
    metrics = [
        ("incidence", "Incidence", "#1d3557", "Estimated HIV incidence (MSM)"),
        ("diagnosed", "Diagnosed", "#e63946", "HIV diagnoses"),
        ("prep_on_count", "PrEP On", "#457b9d", "PrEP"),
    ]
    years = np.asarray(unit_result.years, dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfd")
    for i, (metric, title, color, raw_name) in enumerate(metrics):
        arr = np.asarray([getattr(s.cdc_output, metric) for s in unit_result.samples], dtype=float)
        q05, q50, q95 = np.quantile(arr, [0.05, 0.5, 0.95], axis=0)
        _plot_series(axes[i], years, q50, f"{title} median", color)
        axes[i].fill_between(years, q05, q95, color=color, alpha=0.2, label="5-95%")
        if raw_unit is not None:
            rx, ry = _safe_get_raw_cdc(raw_unit, raw_name)
            if rx is not None and ry is not None:
                axes[i].plot(rx, ry, color="#6d597a", marker="o", markersize=4, linewidth=1.5, linestyle=":", alpha=0.92, label="observed")
        _style_axes(axes[i], title, "Count")
        axes[i].legend(loc="best")
    st.pyplot(fig)


def _render_sem_deterministic_panel(
    out: Any,
    unit_result: Any,
    sem_indices: list[int],
    sem_labels: list[str],
    raw_unit: Any | None,
) -> None:
    st.subheader("SEM trajectories")
    sem = np.asarray(unit_result.sem_trajectory, dtype=float)
    sem_years = _sem_years_for_output(out, sem_len=sem.shape[1], cdc_len=len(np.asarray(unit_result.cdc_output.years, dtype=float)))

    fig, axes = plt.subplots(len(sem_indices), 1, figsize=(11, 3.6 * len(sem_indices)), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfd")
    if len(sem_indices) == 1:
        axes = [axes]

    for i, idx in enumerate(sem_indices):
        color = plt.cm.Set2(i % 8)
        _plot_series(axes[i], sem_years, sem[idx], sem_labels[i], color)
        if raw_unit is not None:
            rx, ry = _safe_get_raw_sem(raw_unit, sem_labels[i])
            if rx is not None and ry is not None:
                axes[i].scatter(rx, ry, color=color, s=28, alpha=0.95, edgecolor="white", linewidth=0.5, label="observed")
        _style_axes(axes[i], sem_labels[i], "Value")
        axes[i].legend(loc="best")

    st.pyplot(fig)


def _render_cdc_deterministic_panel(unit_result: Any, raw_unit: Any | None) -> None:
    st.subheader("CDC predictions")
    cdc = unit_result.cdc_output
    years = np.asarray(cdc.years, dtype=float)

    metrics = [
        ("incidence", "Incidence", "#1d3557", "Estimated HIV incidence (MSM)"),
        ("diagnosed", "Diagnosed", "#e63946", "HIV diagnoses"),
        ("prep_on_count", "PrEP On", "#457b9d", "PrEP"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfd")
    for i, (metric, title, color, raw_name) in enumerate(metrics):
        y = np.asarray(getattr(cdc, metric), dtype=float)
        _plot_series(axes[i], years, y, title, color)
        if raw_unit is not None:
            rx, ry = _safe_get_raw_cdc(raw_unit, raw_name)
            if rx is not None and ry is not None:
                axes[i].plot(rx, ry, color="#6d597a", marker="o", markersize=4, linewidth=1.5, linestyle=":", alpha=0.92, label="observed")
        _style_axes(axes[i], title, "Count")
        axes[i].legend(loc="best")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="EEE Model Predictions", layout="wide")
    st.title("EEE Model Predictions by State")

    root = Path(".").resolve()
    discovered = _find_result_files(root)

    st.sidebar.header("Data source")
    source_mode = st.sidebar.radio("Choose result source", ["Auto-discovered .pkl", "Manual path"], index=0)

    selected_path: Path | None = None
    if source_mode == "Auto-discovered .pkl":
        if not discovered:
            st.warning("No .pkl result files found. Run your model with `--save output/your_results.pkl`.")
            return
        selected_label = st.sidebar.selectbox(
            "Result file",
            [str(p.relative_to(root)) for p in discovered],
            index=0,
        )
        selected_path = root / selected_label
    else:
        entered = st.sidebar.text_input("Path to .pkl", value="output/results.pkl")
        selected_path = Path(entered).expanduser()
        if not selected_path.is_absolute():
            selected_path = (root / selected_path).resolve()
        if not selected_path.exists():
            st.warning(f"File not found: {selected_path}")
            return

    try:
        out = load(selected_path)
    except Exception as exc:
        st.error(f"Could not load {selected_path}: {exc}")
        return

    st.caption(f"Loaded: `{selected_path}`")

    if not hasattr(out, "results") or not out.results:
        st.error("This file does not look like a supported model output with unit results.")
        return

    unit_ids = sorted(list(out.results.keys()))
    selected_unit = st.sidebar.selectbox("State / Unit", unit_ids, index=0)
    unit_result = out.results[selected_unit]
    standardized_path = st.sidebar.text_input("Observed data (.npz)", value="standardized_input.npz")
    raw_unit = None
    standardized_file = Path(standardized_path).expanduser()
    if not standardized_file.is_absolute():
        standardized_file = (root / standardized_file).resolve()
    if standardized_file.exists():
        try:
            units = StandardizedBundle(standardized_file).build_units()
            raw_unit = units.get(selected_unit)
        except Exception as exc:
            st.sidebar.warning(f"Could not load observed data: {exc}")
    else:
        st.sidebar.info("Observed data file not found; showing predictions only.")

    st.subheader(f"Selected: {selected_unit}")
    st.write(f"Mode: {'uncertainty' if _is_uncertainty_output(out) else 'deterministic'}")

    sem_names = list(getattr(out, "v_names", []))
    sem_default = sem_names[: min(3, len(sem_names))]
    selected_sem_names = st.sidebar.multiselect("SEM variables", sem_names, default=sem_default)
    if not selected_sem_names and sem_names:
        selected_sem_names = [sem_names[0]]
    sem_indices = [sem_names.index(name) for name in selected_sem_names]

    if _is_uncertainty_output(out):
        if sem_indices:
            _render_sem_uncertainty_panel(out, unit_result, sem_indices, selected_sem_names, raw_unit)
        _render_cdc_uncertainty_panel(unit_result, raw_unit)
    else:
        if sem_indices:
            _render_sem_deterministic_panel(out, unit_result, sem_indices, selected_sem_names, raw_unit)
        _render_cdc_deterministic_panel(unit_result, raw_unit)


if __name__ == "__main__":
    main()
