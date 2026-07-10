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

        # v7 incidence equation.  inputs.prep_on is the SEM prep_used
        # proportion; prep_off is therefore a proportion, not a count.
        prep_off = np.clip(1 - inputs.prep_on, 1e-6, 1)
        risk0 = max(float(p.risk0), 0.05)
        risk_ratio = np.clip(inputs.risk_behavior / risk0, 0.5, 3.0)
        incidence = p.beta * inputs.no_vs * prep_off * np.power(risk_ratio, p.alpha)

        # v7 diagnosis equation.  The CDCInputs field is still named `tau`
        # for compatibility, but contains direct hivtest12 / p_test.
        post_indicator = (np.asarray(inputs.years, dtype=int) >= 2021).astype(float)
        diagnosis_multiplier = 1 + post_indicator * (p.post_multiplier - 1)
        delta = 1 - np.exp(-p.kdx * inputs.tau * diagnosis_multiplier)

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
