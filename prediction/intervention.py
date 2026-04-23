from __future__ import annotations

import numpy as np

from prediction.codebooks import INTERVENTION_CODEBOOK, REL_CODEBOOK


class StateIntervention:
    def __init__(self, var_idx, start_t, end_t, delta, mode="linear", space="logit"):
        self.var_idx = var_idx
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode
        self.space = space

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0

        span = max(self.end_t - self.start_t, 1)
        frac = (t - self.start_t) / span

        if self.mode == "linear":
            return frac
        if self.mode == "step":
            return 1.0
        if self.mode == "sigmoid":
            return 1 / (1 + np.exp(-10 * (frac - 0.5)))
        return frac

    def apply(self, X_next, X_prev, t, transforms=None):
        X_new = X_next.copy()

        if callable(self.delta):
            adj = self.delta(t, X_new, X_prev)
        else:
            adj = self.delta * self._scale(t)

        if self.space == "logit":
            X_new[self.var_idx] += adj
        elif self.space == "prob":
            if transforms is None:
                raise ValueError("Transforms required for prob-space intervention")

            y = transforms.inverse_logit(X_new[self.var_idx])
            y = np.clip(y + adj, 1e-6, 1 - 1e-6)
            X_new[self.var_idx] = transforms.logit(y)

        return X_new


class RelationshipIntervention:
    def __init__(self, i: int, j: int, start_t: int, end_t: int, delta: float, mode: str = "linear"):
        self.i = i
        self.j = j
        self.start_t = start_t
        self.end_t = end_t
        self.delta = delta
        self.mode = mode

    def _scale(self, t):
        if t < self.start_t:
            return 0.0
        if t >= self.end_t:
            return 1.0

        span = max(self.end_t - self.start_t, 1)
        frac = (t - self.start_t) / span

        if self.mode == "linear":
            return frac
        if self.mode == "step":
            return 1.0
        return frac

    def apply(self, J_base, t):
        J_new = J_base.copy()
        J_new[self.i, self.j] += self.delta * self._scale(t)
        return J_new


def build_state_interventions(unit, sem_years, v_names, codes, duration_steps=3, state_codebook=None):
    codebook = INTERVENTION_CODEBOOK if state_codebook is None else state_codebook

    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))
    if start_t >= len(sem_years):
        return []

    end_t = min(start_t + duration_steps, len(sem_years) - 1)
    interventions = []

    for code in codes:
        spec = codebook.get(code)
        if spec is None:
            continue

        var = spec["var"]
        if var not in v_names:
            continue

        interventions.append(
            StateIntervention(
                var_idx=v_names.index(var),
                start_t=start_t,
                end_t=end_t,
                delta=spec["delta"],
                mode=spec.get("mode", "linear"),
                space=spec.get("space", "logit"),
            )
        )

    return interventions


def build_relationship_interventions(unit, sem_years, v_names, codes, duration_steps=3, rel_codebook=None):
    codebook = REL_CODEBOOK if rel_codebook is None else rel_codebook

    last_obs_year = int(unit.amis_years[-1])
    start_t = int(np.searchsorted(sem_years, last_obs_year, side="right"))
    if start_t >= len(sem_years):
        return []

    end_t = min(start_t + duration_steps, len(sem_years) - 1)
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
            )
        )

    return interventions
