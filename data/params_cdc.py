from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CDCParams:
    """CDC parameters for a single geography."""

    beta: float
    alpha: float
    kdx: float
    U0: float
    kappa_prep: float
