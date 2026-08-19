from __future__ import annotations

import numpy as np

from prediction.codebooks import INTERVENTION_CODEBOOK, REL_CODEBOOK


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class StateIntervention:
    def __init__(self, var_idx, start_t, end_t, delta, mode="linear", effect="probability_relative_to_baseline"):
        self.var_idx = var_idx
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode
        self.effect = effect

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0
        frac = (t - self.start_t) / max(self.end_t - self.start_t, 1)
        if self.mode == "step":
            return 1.0
        if self.mode == "sigmoid":
            return 1 / (1 + np.exp(-10 * (frac - 0.5)))
        return float(frac)

    def apply(self, X_next, X_prev, t, predictor, baseline_p=None):
        if self.effect != "probability_relative_to_baseline":
            raise ValueError(f"Unknown state intervention effect: {self.effect}")
        if baseline_p is None:
            raise ValueError("probability_relative_to_baseline requires baseline_p")

        X_new = X_next.copy()
        scale = self._scale(t)
        ref_logit = predictor.reference_logits
        if ref_logit is None:
            ref_logit = np.zeros(X_new.shape[0], dtype=float)

        p_current = predictor._inverse_x(X_new)[self.var_idx]
        multiplier = max(0.0, 1.0 + self.delta)
        p_target = np.clip(float(baseline_p[self.var_idx]) * multiplier, 1e-6, 1 - 1e-6)
        p_new = np.clip((1 - scale) * float(p_current) + scale * p_target, 1e-6, 1 - 1e-6)
        X_new[self.var_idx] = _logit(p_new) - ref_logit[self.var_idx]
        return X_new


class RelationshipIntervention:
    def __init__(self, i: int, j: int, start_t: int, end_t: int, delta: float, mode: str = "linear", effect: str = "attenuate"):
        self.i = i
        self.j = j
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode
        self.effect = effect

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0
        return float((t - self.start_t) / max(self.end_t - self.start_t, 1))

    def apply(self, J_base, t):
        J_new = J_base.copy()
        scale = self._scale(t)
        if self.effect == "attenuate":
            attenuation = np.clip(self.delta * scale, 0.0, 1.0)
            J_new[self.i, self.j] = J_base[self.i, self.j] * (1 - attenuation)
        elif self.effect == "multiplicative":
            J_new[self.i, self.j] = J_base[self.i, self.j] * (1 + self.delta * scale)
        else:
            raise ValueError(f"Unknown relationship intervention effect: {self.effect}")
        return J_new


def _intervention_window(unit, sem_years, duration_steps=3):
    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))
    end_t = min(start_t + int(duration_steps), len(sem_years) - 1)
    return start_t, end_t


def build_state_interventions(unit, sem_years, v_names, codes, duration_steps=3, state_codebook=None):
    codebook = INTERVENTION_CODEBOOK if state_codebook is None else state_codebook
    start_t, end_t = _intervention_window(unit, sem_years, duration_steps)
    if start_t >= len(sem_years):
        return []
    interventions = []
    for code in codes:
        spec = codebook.get(code)
        if spec is None or spec["var"] not in v_names:
            continue
        interventions.append(
            StateIntervention(
                var_idx=v_names.index(spec["var"]),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                effect=spec["effect"],
            )
        )
    return interventions


def build_relationship_interventions(unit, sem_years, v_names, codes, duration_steps=3, rel_codebook=None):
    codebook = REL_CODEBOOK if rel_codebook is None else rel_codebook
    start_t, end_t = _intervention_window(unit, sem_years, duration_steps)
    if start_t >= len(sem_years):
        return []
    interventions = []
    for code in codes:
        spec = codebook.get(code)
        if spec is None:
            continue
        src = spec["from"]
        tgt = spec["to"]
        if src not in v_names or tgt not in v_names:
            continue
        interventions.append(
            RelationshipIntervention(
                i=v_names.index(tgt),
                j=v_names.index(src),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                effect=spec["effect"],
            )
        )
    return interventions
