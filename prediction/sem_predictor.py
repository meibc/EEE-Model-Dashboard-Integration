from __future__ import annotations

import numpy as np

from prediction.transforms import Transforms


class Predictor:
    def __init__(self):
        self.transforms = Transforms()

    def predict_next(self, J, X_current, u):
        return J @ X_current + u

    def predict_trajectory(
        self,
        J,
        X0,
        u,
        n_steps: int,
        state_interventions=None,
        rel_interventions=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        m = X0.shape[0]
        Xpred = np.zeros((m, n_steps))
        Xpred[:, 0] = X0

        J_base = J.copy()

        for t in range(1, n_steps):
            J_t = J_base.copy()
            if rel_interventions:
                for iv in rel_interventions:
                    J_t = iv.apply(J_t, t)

            X_next = J_t @ Xpred[:, t - 1] + u

            if state_interventions:
                for iv in state_interventions:
                    X_next = iv.apply(X_next, Xpred[:, t - 1], t, transforms=self.transforms)

            Xpred[:, t] = X_next

        Ypred = self.transforms.inverse_logit(Xpred)
        return Ypred, Xpred
