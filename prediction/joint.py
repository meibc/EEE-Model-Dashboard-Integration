from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

from alignment import build_cdc_inputs_from_sem, extend_to_end_year, extend_years
from data.unit import Unit
from output.types import (
    CDCInputs,
    JointOutput,
    JointResult,
    RunOutput,
    UncertaintyOutput,
    UncertaintyResult,
    UncertaintySample,
)
from prediction.epi_predictor import CDCPredictor
from prediction.intervention import (
    build_relationship_interventions,
    build_state_interventions,
)
from prediction.sem_predictor import Predictor
from prediction.transforms import Transforms


class JointRunner:
    def __init__(
        self,
        sem_output: RunOutput,
        cdc_params_loader: Any,
        units: dict[str, Unit],
        model_years: np.ndarray | None = None,
        state_intervention_codes: list[str] | None = None,
        relationship_intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 3,
        hivtest_var: str = "hivtest12",
        prep_var: str = "prep_used",
        risk_var: str = "risk_behavior",
        n_elig_var: str = "PrEP Eligible",
        prevalence_var: str = "Estimated HIV prevalence (MSM)",
        viral_suppression_var: str = "HIV viral suppression",
    ):
        self.sem_output = sem_output
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.prevalence_var = prevalence_var
        self.viral_suppression_var = viral_suppression_var
        self.state_intervention_codes = list(state_intervention_codes or [])
        self.relationship_intervention_codes = list(relationship_intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.predictor = Predictor()
        self.transforms = Transforms()

        self._v_names = sem_output.predictions.v_names
        self.model_years = (
            np.asarray(model_years, dtype=int)
            if model_years is not None
            else np.asarray(sem_output.predictions.ts, dtype=int)
        )
        self._sem_years = extend_to_end_year(
            sem_output.predictions.ts,
            target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None,
        )
        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        self._risk_idx = self._v_names.index(risk_var)
        self._intervention_cache: dict[str, tuple[list, list]] = {}

    def _get_interventions(self, unit_id: str) -> tuple[list, list]:
        if unit_id in self._intervention_cache:
            return self._intervention_cache[unit_id]

        unit = self.units[unit_id]
        state_iv = build_state_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.state_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        rel_iv = build_relationship_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.relationship_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        self._intervention_cache[unit_id] = (state_iv, rel_iv)
        return state_iv, rel_iv

    def _build_cdc_inputs(self, unit_id: str, sem_traj: np.ndarray) -> CDCInputs:
        sem_years = extend_years(self._sem_years, sem_traj.shape[1])
        tau, prep_on, n_elig, risk_behavior, no_vs = build_cdc_inputs_from_sem(
            sem_traj=sem_traj,
            unit=self.units[unit_id],
            hivtest_idx=self._hivtest_idx,
            prep_idx=self._prep_idx,
            risk_idx=self._risk_idx,
            sem_years=sem_years,
            model_years=self.model_years,
            n_elig_var=self.n_elig_var,
            prevalence_var=self.prevalence_var,
            viral_suppression_var=self.viral_suppression_var,
        )
        return CDCInputs(
            years=self.model_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
            risk_behavior=risk_behavior,
            no_vs=no_vs,
        )

    def _build_sem_trajectory(self, unit_id: str) -> np.ndarray:
        unit = self.units[unit_id]
        fit = self.sem_output.fit.results[unit_id]
        J_fit = np.asarray(fit.J, dtype=float)
        J = J_fit[:, :, -1] if J_fit.ndim == 3 else J_fit

        reference_probs = getattr(fit, "reference_probs", None)
        predictor = Predictor(reference_probs=reference_probs)
        ref_logits = predictor.reference_logits
        if ref_logits is None:
            ref_logits = np.zeros(J.shape[0], dtype=float)

        y0 = np.asarray(unit.amis_values[:, 0], dtype=float)
        x0 = self.transforms.logit(y0) - ref_logits
        u = np.zeros(J.shape[0], dtype=float) if getattr(fit, "drift", None) is None else np.asarray(fit.drift, dtype=float)

        # Always roll out on the full SEM horizon used for CDC alignment/model years.
        # Using legacy prediction length can truncate trajectory and nullify interventions.
        n_steps = len(self._sem_years)

        baseline_probs, _ = predictor.predict_trajectory(J, x0, u, n_steps)
        state_iv, rel_iv = self._get_interventions(unit_id)

        ypred, _ = predictor.predict_trajectory(
            J,
            x0,
            u,
            n_steps,
            state_interventions=state_iv,
            rel_interventions=rel_iv,
            baseline_probs=baseline_probs,
        )
        return ypred

    def predict(self, unit_id: str) -> JointResult:
        use_stored_pred = (
            self.sem_output.predictions is not None
            and unit_id in self.sem_output.predictions.results
            and not self.state_intervention_codes
            and not self.relationship_intervention_codes
        )

        if use_stored_pred:
            stored = np.asarray(self.sem_output.predictions.results[unit_id].Ypred_trajectory, dtype=float)
            # Recompute if stored predictions do not span the runtime SEM horizon.
            if stored.shape[1] >= len(self._sem_years):
                sem_traj = stored[:, : len(self._sem_years)]
            else:
                sem_traj = self._build_sem_trajectory(unit_id)
        else:
            sem_traj = self._build_sem_trajectory(unit_id)

        cdc_inputs = self._build_cdc_inputs(unit_id, sem_traj)

        cdc_params = self.cdc_loader.load_point_estimates(unit_id)
        cdc_output = CDCPredictor(cdc_params).predict(cdc_inputs, unit_id)

        return JointResult(
            unit_id=unit_id,
            sem_trajectory=sem_traj,
            cdc_inputs=cdc_inputs,
            cdc_output=cdc_output,
        )

    def run(self, unit_ids: list[str] | None = None) -> JointOutput:
        if unit_ids is None:
            if self.sem_output.predictions is not None:
                unit_ids = list(self.sem_output.predictions.results.keys())
            else:
                unit_ids = list(self.sem_output.fit.results.keys())

        available = set(self.cdc_loader.geo_names)
        valid_ids = [uid for uid in unit_ids if uid in available]

        results = {uid: self.predict(uid) for uid in valid_ids}

        return JointOutput(
            results=results,
            sem_years=self._sem_years,
            cdc_years=self.model_years,
            v_names=self._v_names,
        )


class UncertaintyRunner:
    def __init__(
        self,
        sem_loader: Any,
        cdc_params_loader: Any,
        units: dict[str, Unit],
        model_years: np.ndarray | None = None,
        state_intervention_codes: list[str] | None = None,
        relationship_intervention_codes: list[str] | None = None,
        intervention_duration_steps: int = 3,
        v_names: list[str] | None = None,
        hivtest_var: str = "hivtest12",
        prep_var: str = "prep_used",
        risk_var: str = "risk_behavior",
        n_elig_var: str = "PrEP Eligible",
        prevalence_var: str = "Estimated HIV prevalence (MSM)",
        viral_suppression_var: str = "HIV viral suppression",
    ):
        self.sem_loader = sem_loader
        self.cdc_loader = cdc_params_loader
        self.units = units
        self.hivtest_var = hivtest_var
        self.prep_var = prep_var
        self.risk_var = risk_var
        self.n_elig_var = n_elig_var
        self.prevalence_var = prevalence_var
        self.viral_suppression_var = viral_suppression_var
        self.state_intervention_codes = list(state_intervention_codes or [])
        self.relationship_intervention_codes = list(relationship_intervention_codes or [])
        self.intervention_duration_steps = int(intervention_duration_steps)
        self.predictor = Predictor()
        self.transforms = Transforms()

        self._v_names = list(v_names if v_names is not None else sem_loader.v_names)
        self.model_years = np.asarray(
            model_years if model_years is not None else sem_loader.ts,
            dtype=int,
        )
        self._sem_years = extend_to_end_year(
            sem_loader.ts,
            target_end_year=int(self.model_years[-1]) if self.model_years.size > 0 else None,
        )
        self.S_sem = sem_loader.n_samples
        self._unit_order = list(sem_loader.geo_names)

        self._hivtest_idx = self._v_names.index(hivtest_var)
        self._prep_idx = self._v_names.index(prep_var)
        self._risk_idx = self._v_names.index(risk_var)
        self.S_cdc = cdc_params_loader.n_samples
        self._intervention_cache: dict[str, tuple[list, list]] = {}

    def _get_interventions(self, unit_id: str) -> tuple[list, list]:
        if unit_id in self._intervention_cache:
            return self._intervention_cache[unit_id]

        unit = self.units[unit_id]
        state_iv = build_state_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.state_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )

        rel_iv = build_relationship_interventions(
            unit,
            self._sem_years,
            self._v_names,
            codes=self.relationship_intervention_codes,
            duration_steps=self.intervention_duration_steps,
        )
        self._intervention_cache[unit_id] = (state_iv, rel_iv)
        return state_iv, rel_iv

    def _build_cdc_inputs(self, unit_id: str, sem_traj: np.ndarray) -> CDCInputs:
        sem_years = extend_years(self._sem_years, sem_traj.shape[1])
        tau, prep_on, n_elig, risk_behavior, no_vs = build_cdc_inputs_from_sem(
            sem_traj=sem_traj,
            unit=self.units[unit_id],
            hivtest_idx=self._hivtest_idx,
            prep_idx=self._prep_idx,
            risk_idx=self._risk_idx,
            sem_years=sem_years,
            model_years=self.model_years,
            n_elig_var=self.n_elig_var,
            prevalence_var=self.prevalence_var,
            viral_suppression_var=self.viral_suppression_var,
        )
        return CDCInputs(
            years=self.model_years,
            tau=tau,
            prep_on=prep_on,
            N_elig=n_elig,
            risk_behavior=risk_behavior,
            no_vs=no_vs,
        )

    def _build_sem_trajectory(self, unit_id: str, sem_idx: int) -> np.ndarray:
        sem_params = self.sem_loader.load_sample(sem_idx, unit_id)
        J = np.asarray(sem_params.J, dtype=float)

        unit = self.units[unit_id]
        predictor = Predictor(reference_probs=sem_params.reference_probs)
        ref_logits = predictor.reference_logits
        if ref_logits is None:
            ref_logits = np.zeros(J.shape[0], dtype=float)

        y0 = np.asarray(unit.amis_values[:, 0], dtype=float)
        x0 = self.transforms.logit(y0) - ref_logits

        u = np.zeros(J.shape[0], dtype=float) if sem_params.drift is None else np.asarray(sem_params.drift, dtype=float)
        n_steps = len(self._sem_years)

        baseline_probs, _ = predictor.predict_trajectory(J, x0, u, n_steps)
        state_iv, rel_iv = self._get_interventions(unit_id)

        ypred, _ = predictor.predict_trajectory(
            J,
            x0,
            u,
            n_steps,
            state_interventions=state_iv,
            rel_interventions=rel_iv,
            baseline_probs=baseline_probs,
        )

        return ypred

    def predict_sample(self, unit_id: str, sem_idx: int, cdc_idx: int) -> UncertaintySample:
        sem_traj = self._build_sem_trajectory(unit_id, sem_idx)
        cdc_inputs = self._build_cdc_inputs(unit_id, sem_traj)

        cdc_params = self.cdc_loader.load_sample(cdc_idx, unit_id)
        cdc_output = CDCPredictor(cdc_params).predict(cdc_inputs, unit_id)

        return UncertaintySample(
            sem_idx=sem_idx,
            cdc_idx=cdc_idx,
            sem_trajectory=sem_traj,
            cdc_output=cdc_output,
            cdc_inputs=cdc_inputs,
        )

    def run(
        self,
        unit_id: str,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> UncertaintyResult:
        rng = np.random.default_rng(seed)

        idx_sem = rng.choice(self.S_sem, size=n_samples, replace=True)
        idx_cdc = rng.choice(self.S_cdc, size=n_samples, replace=True)

        samples = []
        iterator = zip(idx_sem, idx_cdc)
        if show_progress:
            iterator = tqdm(iterator, total=n_samples, desc=f"MC {unit_id}")

        for s_sem, s_cdc in iterator:
            samples.append(self.predict_sample(unit_id, int(s_sem), int(s_cdc)))

        return UncertaintyResult(unit_id=unit_id, samples=samples, years=self.model_years)

    def run_all(
        self,
        unit_ids: list[str] | None = None,
        n_samples: int = 1000,
        seed: int = 123,
        show_progress: bool = True,
    ) -> UncertaintyOutput:
        if unit_ids is None:
            unit_ids = list(self._unit_order)

        available_cdc = set(self.cdc_loader.geo_names)
        available_units = set(self.units.keys())
        available_sem = set(self._unit_order)
        unit_ids = [
            uid for uid in unit_ids if uid in available_cdc and uid in available_units and uid in available_sem
        ]

        results = {}
        for i, uid in enumerate(unit_ids):
            results[uid] = self.run(uid, n_samples, seed + i, show_progress)

        return UncertaintyOutput(results=results, years=self.model_years, v_names=self._v_names)


def run_joint(
    sem_output: RunOutput,
    cdc_params_loader: Any,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    **kwargs,
) -> JointOutput:
    runner = JointRunner(sem_output, cdc_params_loader, units, **kwargs)
    return runner.run(unit_ids)


def run_uncertainty(
    sem_loader: Any,
    cdc_params_loader: Any,
    units: dict[str, Unit],
    unit_ids: list[str] | None = None,
    n_samples: int = 1000,
    seed: int = 123,
    show_progress: bool = True,
    **kwargs,
) -> UncertaintyOutput:
    runner = UncertaintyRunner(sem_loader, cdc_params_loader, units, **kwargs)
    return runner.run_all(unit_ids, n_samples, seed, show_progress=show_progress)
