from __future__ import annotations

import numpy as np


class Transforms:
    @staticmethod
    def logit(p: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        p_clipped = np.clip(p, epsilon, 1 - epsilon)
        return np.log(p_clipped / (1 - p_clipped))

    @staticmethod
    def inverse_logit(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


def hazard_proxy(hivtest: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(hivtest, dtype=float), 1e-9, 1 - 1e-9)
    return -np.log1p(-p)
