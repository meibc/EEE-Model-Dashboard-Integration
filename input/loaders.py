from __future__ import annotations

from config import RuntimeConfig
from input.standardized_runtime import (
    StandardizedBundle,
    StandardizedCDCParamsLoader,
    StandardizedSEMParamsLoader,
    resolve_model_years,
)


def load_deterministic_inputs(cfg: RuntimeConfig):
    bundle = StandardizedBundle(cfg.standardized_input_path)
    sem_output = bundle.build_sem_output()
    units = bundle.build_units()
    cdc_loader = StandardizedCDCParamsLoader(bundle)
    model_years = resolve_model_years(cdc_loader.years, cfg.target_end_year)
    return sem_output, units, cdc_loader, model_years


def load_uncertainty_inputs(cfg: RuntimeConfig):
    bundle = StandardizedBundle(cfg.standardized_input_path)
    units = bundle.build_units()
    sem_loader = StandardizedSEMParamsLoader(bundle)
    cdc_loader = StandardizedCDCParamsLoader(bundle)
    model_years = resolve_model_years(cdc_loader.years, cfg.target_end_year)
    return sem_loader, units, cdc_loader, model_years
