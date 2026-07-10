from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CDCParams:
    """CDC / transition parameters for a single geography.

    Compatibility notes:
    - kdx is the old-compatible name for the v7 diagnosis/testing scale kappa.
    - U0 is the runtime initial undiagnosed stock U[0].
    - risk0 is the baseline risk reference used in the v7 incidence equation.
    - post_multiplier is applied to diagnosis probability from 2021 onward.
    """

    beta: float
    alpha: float
    kdx: float
    U0: float
    kappa_prep: float
    risk0: float = 1.0
    post_multiplier: float = 1.0
