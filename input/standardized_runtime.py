from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data.params_cdc import CDCParams
from data.params_sem import SEMParams
from data.unit import Unit


@dataclass
class StandardizedBundle:
    path: Path
    _cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is None:
            raw = np.load(self.path, allow_pickle=True)
            self._cache = {k: raw[k] for k in raw.files}
        return self._cache

    @property
    def geo_ids(self) -> list[str]:
        return list(np.asarray(self._load()["geo_ids"]).tolist())

    @property
    def model_years(self) -> np.ndarray:
        return np.asarray(self._load()["model_years"], dtype=int)

    @property
    def sem_obs_years(self) -> np.ndarray:
        return np.asarray(self._load()["sem_obs_years"], dtype=int)

    @property
    def sem_pred_years(self) -> np.ndarray:
        return np.asarray(self._load()["sem_pred_years"], dtype=int)

    @property
    def sem_v_names(self) -> list[str]:
        return list(np.asarray(self._load()["sem_v_names"]).tolist())

    @property
    def cdc_raw_names(self) -> list[str]:
        return list(np.asarray(self._load()["cdc_raw_names"]).tolist())

    @property
    def cdc_native_years(self) -> np.ndarray:
        d = self._load()
        if "cdc_native_years" in d:
            return np.asarray(d["cdc_native_years"], dtype=int)
        return self.model_years

    def build_units(self) -> dict[str, Unit]:
        d = self._load()
        geos = self.geo_ids
        amis_years = self.sem_obs_years
        amis_names = self.sem_v_names
        cdc_years = self.cdc_native_years
        cdc_names = self.cdc_raw_names
        cdc_key = "cdc_raw_native" if "cdc_raw_native" in d else "cdc_raw"

        units: dict[str, Unit] = {}
        for i, geo in enumerate(geos):
            units[geo] = Unit(
                id=geo,
                kind="state",
                amis_years=np.asarray(amis_years, dtype=float),
                amis_values=np.asarray(d["sem_obs"][i], dtype=float),
                amis_names=list(amis_names),
                cdc_years=np.asarray(cdc_years, dtype=float),
                cdc_values=np.asarray(d[cdc_key][i], dtype=float),
                cdc_names=list(cdc_names),
            )
        return units

    def build_sem_output(self):
        d = self._load()
        geos = self.geo_ids

        units = self.build_units()
        reference_probs = np.asarray(d["sem_reference_probs"], dtype=float)
        drift = np.asarray(d["sem_fit_drift"], dtype=float)
        fit_results = {
            geo: SimpleNamespace(
                J=np.asarray(d["sem_fit_J_last"][i], dtype=float),
                drift=np.asarray(drift[i], dtype=float),
                reference_probs=reference_probs,
            )
            for i, geo in enumerate(geos)
        }
        pred_results = {
            geo: SimpleNamespace(Ypred_trajectory=np.asarray(d["sem_pred"][i], dtype=float))
            for i, geo in enumerate(geos)
        }

        return SimpleNamespace(
            inputs=SimpleNamespace(
                units=list(units.values()),
                v_names=list(self.sem_v_names),
                ts=np.asarray(self.sem_obs_years, dtype=int),
            ),
            fit=SimpleNamespace(results=fit_results),
            predictions=SimpleNamespace(
                results=pred_results,
                v_names=list(self.sem_v_names),
                ts=np.asarray(self.sem_pred_years, dtype=int),
            ),
        )


class StandardizedSEMParamsLoader:
    def __init__(self, bundle: StandardizedBundle):
        self.bundle = bundle
        self._data = bundle._load()

    @property
    def geo_names(self) -> list[str]:
        return self.bundle.geo_ids

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self._data["sem_J_samples"]).shape[0])

    @property
    def v_names(self) -> list[str]:
        return self.bundle.sem_v_names

    @property
    def ts(self) -> np.ndarray:
        return self.bundle.sem_pred_years

    def _idx(self, unit_id: str) -> int:
        try:
            return self.geo_names.index(unit_id)
        except ValueError as exc:
            raise KeyError(f"Unknown unit_id: {unit_id}") from exc

    def load_point_estimates(self, unit_id: str) -> SEMParams:
        i = self._idx(unit_id)
        j = np.asarray(self._data["sem_J_samples"][:, i, :, :], dtype=float).mean(axis=0)
        drift = np.asarray(self._data["sem_drift_samples"][:, i, :], dtype=float).mean(axis=0)
        reference_probs = np.asarray(self._data["sem_reference_probs"], dtype=float)
        return SEMParams(J=j, drift=drift, reference_probs=reference_probs)

    def load_sample(self, sample_idx: int, unit_id: str) -> SEMParams:
        i = self._idx(unit_id)
        j = np.asarray(self._data["sem_J_samples"][sample_idx, i, :, :], dtype=float)
        drift = np.asarray(self._data["sem_drift_samples"][sample_idx, i, :], dtype=float)
        reference_probs = np.asarray(self._data["sem_reference_probs"], dtype=float)
        return SEMParams(J=j, drift=drift, reference_probs=reference_probs)


class StandardizedCDCParamsLoader:
    def __init__(self, bundle: StandardizedBundle):
        self.bundle = bundle
        self._data = bundle._load()

    @property
    def geo_names(self) -> list[str]:
        return self.bundle.geo_ids

    @property
    def years(self) -> np.ndarray:
        return self.bundle.model_years

    @property
    def n_samples(self) -> int:
        return int(np.asarray(self._data["cdc_beta"]).shape[0])

    def _idx(self, unit_id: str) -> int:
        try:
            return self.geo_names.index(unit_id)
        except ValueError as exc:
            raise KeyError(f"Unknown unit_id: {unit_id}") from exc

    def load_point_estimates(self, unit_id: str) -> CDCParams:
        i = self._idx(unit_id)
        risk0 = float(np.asarray(self._data.get("cdc_risk0", np.ones(len(self.geo_names)))[i], dtype=float))
        post = self._data.get("cdc_post_multiplier")
        post_multiplier = 1.0 if post is None else float(np.asarray(post[:, i], dtype=float).mean())
        return CDCParams(
            beta=float(np.asarray(self._data["cdc_beta"][:, i], dtype=float).mean()),
            alpha=float(np.asarray(self._data["cdc_alpha"][:, i], dtype=float).mean()),
            kdx=float(np.asarray(self._data["cdc_kdx"][:, i], dtype=float).mean()),
            U0=float(np.asarray(self._data["cdc_U0"][:, i], dtype=float).mean()),
            kappa_prep=float(np.asarray(self._data["cdc_kappa_prep"][i], dtype=float)),
            risk0=risk0,
            post_multiplier=post_multiplier,
        )

    def load_sample(self, sample_idx: int, unit_id: str) -> CDCParams:
        i = self._idx(unit_id)
        risk0 = float(np.asarray(self._data.get("cdc_risk0", np.ones(len(self.geo_names)))[i], dtype=float))
        post = self._data.get("cdc_post_multiplier")
        post_multiplier = 1.0 if post is None else float(np.asarray(post[sample_idx, i], dtype=float))
        return CDCParams(
            beta=float(np.asarray(self._data["cdc_beta"][sample_idx, i], dtype=float)),
            alpha=float(np.asarray(self._data["cdc_alpha"][sample_idx, i], dtype=float)),
            kdx=float(np.asarray(self._data["cdc_kdx"][sample_idx, i], dtype=float)),
            U0=float(np.asarray(self._data["cdc_U0"][sample_idx, i], dtype=float)),
            kappa_prep=float(np.asarray(self._data["cdc_kappa_prep"][i], dtype=float)),
            risk0=risk0,
            post_multiplier=post_multiplier,
        )


def resolve_model_years(bundle_years: np.ndarray, target_end_year: int | None) -> np.ndarray:
    years = np.asarray(bundle_years, dtype=int)
    if target_end_year is None:
        return years
    end = int(target_end_year)
    if end > int(years[-1]):
        raise ValueError(
            f"target_end_year={end} exceeds standardized_input max year {int(years[-1])}. "
            "Rebuild standardized_input.npz with a later target_end_year."
        )
    return years[years <= end]
