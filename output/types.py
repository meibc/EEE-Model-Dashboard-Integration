from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CDCInputs:
    years: np.ndarray
    tau: np.ndarray
    prep_on: np.ndarray
    N_elig: np.ndarray
    risk_behavior: np.ndarray
    no_vs: np.ndarray


@dataclass
class CDCOutput:
    unit_id: str
    years: np.ndarray
    prep_on_count: np.ndarray
    incidence: np.ndarray
    diagnosed: np.ndarray
    undiagnosed: np.ndarray


@dataclass
class JointResult:
    unit_id: str
    sem_trajectory: np.ndarray
    cdc_inputs: CDCInputs
    cdc_output: CDCOutput


@dataclass
class JointOutput:
    results: dict[str, JointResult]
    sem_years: np.ndarray
    cdc_years: np.ndarray
    v_names: list[str]


@dataclass
class UncertaintySample:
    sem_idx: int
    cdc_idx: int
    sem_trajectory: np.ndarray
    cdc_output: CDCOutput
    cdc_inputs: CDCInputs | None = None


@dataclass
class UncertaintyResult:
    unit_id: str
    samples: list[UncertaintySample]
    years: np.ndarray


@dataclass
class UncertaintyOutput:
    results: dict[str, UncertaintyResult]
    years: np.ndarray
    v_names: list[str]


@dataclass
class RunOutput:
    """Nominal SEM output container used for typing/documentation only."""

    inputs: Any
    fit: Any
    predictions: Any

@dataclass
class PreparedData:
    """Inputs for estimation."""
    units: list
    sign_mask: np.ndarray
    SigmaY: np.ndarray
    M: np.ndarray
    ts: np.ndarray
    v_names: list[str]
    ts_cdc: np.ndarray | None = None
    cdc_names: list[str] | None = None

