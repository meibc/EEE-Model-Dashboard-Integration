from __future__ import annotations

import numpy as np

from data.params_cdc import CDCParams
from output.types import CDCInputs, CDCOutput


class CDCPredictor:
    def __init__(self, params: CDCParams):
        self.params = params

    def predict(self, inputs: CDCInputs, unit_id: str) -> CDCOutput:
        p = self.params
        T = len(inputs.years)

        prep_on_count = p.kappa_prep * inputs.prep_on * inputs.N_elig
        prep_off_count = inputs.N_elig - prep_on_count

        n_elig_safe = np.maximum(inputs.N_elig, 1e-12)
        incidence = p.beta * ((prep_off_count / n_elig_safe) + (p.alpha * inputs.risk_behavior)) * inputs.no_vs
        delta = 1 - np.exp(-p.kdx * inputs.tau)

        undiagnosed = np.zeros(T)
        diagnosed = np.zeros(T)

        undiagnosed[0] = p.U0
        diagnosed[0] = np.maximum(0, p.U0 * delta[0])

        for t in range(1, T):
            undiagnosed[t] = np.maximum(0, undiagnosed[t - 1] + incidence[t - 1] - diagnosed[t - 1])
            diagnosed[t] = np.maximum(0, undiagnosed[t] * delta[t])

        return CDCOutput(
            unit_id=unit_id,
            years=inputs.years,
            prep_on_count=prep_on_count,
            incidence=incidence,
            diagnosed=diagnosed,
            undiagnosed=undiagnosed,
        )
