from __future__ import annotations

from config import RuntimeConfig
from input import load_deterministic_inputs, load_uncertainty_inputs
from output import save
from prediction.joint import run_joint, run_uncertainty


def _resolve_interventions(cfg: RuntimeConfig) -> tuple[list[str], list[str]]:
    if cfg.scenario_mode == "baseline":
        return [], []
    return list(cfg.state_intervention_codes), list(cfg.relationship_intervention_codes)


def _maybe_save(result, cfg: RuntimeConfig) -> None:
    if cfg.save_output_path is not None:
        save(result, cfg.save_output_path)


def run_deterministic(cfg: RuntimeConfig):
    cfg.validate()
    state_codes, rel_codes = _resolve_interventions(cfg)
    sem_output, units, cdc_loader, model_years = load_deterministic_inputs(cfg)

    result = run_joint(
        sem_output=sem_output,
        cdc_params_loader=cdc_loader,
        units=units,
        unit_ids=cfg.unit_ids,
        model_years=model_years,
        hivtest_var=cfg.hivtest_var,
        prep_var=cfg.prep_var,
        risk_var=cfg.risk_var,
        n_elig_var=cfg.n_elig_var,
        prevalence_var=cfg.prevalence_var,
        viral_suppression_var=cfg.viral_suppression_var,
        state_intervention_codes=state_codes,
        relationship_intervention_codes=rel_codes,
        intervention_duration_steps=cfg.intervention_duration_steps,
    )

    _maybe_save(result, cfg)

    return result


def run_uncertainty_mode(cfg: RuntimeConfig):
    cfg.validate()
    state_codes, rel_codes = _resolve_interventions(cfg)
    sem_loader, units, cdc_loader, model_years = load_uncertainty_inputs(cfg)

    result = run_uncertainty(
        sem_loader=sem_loader,
        cdc_params_loader=cdc_loader,
        units=units,
        unit_ids=cfg.unit_ids,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        show_progress=cfg.show_progress,
        model_years=model_years,
        hivtest_var=cfg.hivtest_var,
        prep_var=cfg.prep_var,
        risk_var=cfg.risk_var,
        n_elig_var=cfg.n_elig_var,
        prevalence_var=cfg.prevalence_var,
        viral_suppression_var=cfg.viral_suppression_var,
        state_intervention_codes=state_codes,
        relationship_intervention_codes=rel_codes,
        intervention_duration_steps=cfg.intervention_duration_steps,
    )

    _maybe_save(result, cfg)

    return result


def run_prediction(cfg: RuntimeConfig):
    """Unified entrypoint for inference-only runtime."""
    if cfg.mode == "deterministic":
        return run_deterministic(cfg)
    return run_uncertainty_mode(cfg)
