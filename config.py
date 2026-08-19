from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class RuntimeConfig:
    """Inference-only runtime config for dashboard integration."""

    mode: Literal["deterministic", "uncertainty"] = "deterministic"
    scenario_mode: Literal["baseline", "intervention"] = "baseline"

    # Standardized artifact
    standardized_input_path: Path = Path("standardized_input_v8.npz")

    # Scope
    target_end_year: int | None = 2036
    unit_ids: list[str] | None = None

    # Uncertainty settings
    n_samples: int = 1000
    seed: int = 123
    show_progress: bool = True

    # Variable mapping
    hivtest_var: str = "hivtest12"
    prep_var: str = "prep_used"
    risk_var: str = "risk_behavior"
    n_elig_var: str = "PrEP Eligible"
    prevalence_var: str = "Estimated HIV prevalence (MSM)"
    viral_suppression_var: str = "HIV viral suppression"

    # Interventions
    state_intervention_codes: list[str] = field(default_factory=list)
    relationship_intervention_codes: list[str] = field(default_factory=list)
    intervention_duration_steps: int = 3

    # Optional save path (if set, caller can persist result)
    save_output_path: Path | None = None

    def validate(self) -> None:
        if self.mode not in {"deterministic", "uncertainty"}:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.scenario_mode not in {"baseline", "intervention"}:
            raise ValueError(f"Invalid scenario_mode: {self.scenario_mode}")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if not Path(self.standardized_input_path).exists():
            raise FileNotFoundError(f"Missing standardized input: {self.standardized_input_path}")
