# EEE Prediction Runtime

Inference runtime for SEM -> CDC projections using a single standardized input file.

## Overview

This repo runs forward prediction for:
- SEM trajectories (testing, PrEP use, risk behavior, and related SEM variables)
- CDC outputs (incidence, diagnosed, undiagnosed, prep-on count)

It supports:
- deterministic runs
- uncertainty runs (Monte Carlo sampling)
- intervention scenarios (state and relationship intervention codebooks)
- sanity plotting

## Runtime Flow: Inputs -> Model -> Outputs

### 1. Input

The runtime expects one file:
- `standardized_input.npz`

No other raw model artifact files are required at runtime.

### 2. Model

Main execution path:
- `cli.py` parses arguments
- `runner.py` dispatches deterministic or uncertainty mode
- `input/standardized_runtime.py` reads `standardized_input.npz`
- `prediction/joint.py` runs SEM + CDC coupling
- `prediction/intervention.py` applies intervention effects when enabled

### 3. Output

Runtime returns in-memory dataclasses from `output/types.py`:
- deterministic: `JointOutput`
- uncertainty: `UncertaintyOutput`

Optional:
- `--save <path>` writes pickle output
- `--plot` writes PNGs

## Project Structure

- `cli.py`: command-line entrypoint
- `config.py`: runtime configuration and validation
- `runner.py`: orchestration (`run_prediction`)
- `alignment.py`: year alignment and CDC input derivation utilities
- `input/`
- `input/loaders.py`: standardized-only loading adapters
- `input/standardized_runtime.py`: bundle schema readers/loaders
- `prediction/`
- `prediction/joint.py`: deterministic + uncertainty runners
- `prediction/intervention.py`: intervention application logic
- `prediction/codebooks.py`: intervention code definitions
- `prediction/epi_predictor.py`: CDC prediction equations
- `output/types.py`: output dataclasses
- `output/io.py`: save/load helpers
- `plotting.py`: six-panel plotting per unit
- `standardized_input.npz`: required runtime input

## `standardized_input.npz` Schema

Think of the file as 5 blocks:
- index metadata (what each axis means)
- year axes (what timelines exist)
- SEM observed + SEM model parameters
- CDC raw input series
- CDC posterior parameter samples

Axis symbols used below:
- `G`: number of geographies
- `M`: number of SEM variables
- `K`: number of CDC raw variables
- `S_sem`: number of SEM posterior samples
- `S_cdc`: number of CDC posterior samples
- `T_sem_obs`: SEM observed years
- `T_sem_pred`: SEM prediction years
- `T_cdc_obs`: observed CDC years
- `T_model`: CDC model years used for simulation

### Index Metadata

- `schema_version` `(1,)`: schema/version tag
- `geo_ids` `(G,)`: geography IDs (for example `NY`, `CA`, ...)
- `sem_v_names` `(M,)`: SEM variable names (for example `hivtest12`, `prep_used`)
- `cdc_raw_names` `(K,)`: CDC raw variable names (for example `HIV diagnoses`, `PrEP`)

### Year Axes

- `sem_obs_years` `(T_sem_obs,)`: years where SEM observations exist
- `sem_pred_years` `(T_sem_pred,)`: years for SEM trajectory rollout
- `cdc_native_years` `(T_cdc_obs,)`: native observed CDC years
- `model_years` `(T_model,)`: CDC simulation years (can extend beyond native years)

### SEM Block

- `sem_obs` `(G, M, T_sem_obs)`: observed SEM values
- `sem_pred` `(G, M, T_sem_pred)`: baseline SEM trajectory values
- `sem_fit_J_last` `(G, M, M)`: fitted SEM matrix used for deterministic rollout
- `sem_J_samples` `(S_sem, G, M, M)`: SEM posterior samples for uncertainty mode

### CDC Raw Block

- `cdc_raw_native` `(G, K, T_cdc_obs)`: raw CDC series on observed years (used for plotting observed points and forecast split)
- `cdc_raw` `(G, K, T_model)`: CDC raw series aligned to model years (used in model computations)

### CDC Posterior Block

- `cdc_beta` `(S_cdc, G)`: posterior samples for `beta`
- `cdc_alpha` `(S_cdc, G)`: posterior samples for `alpha`
- `cdc_kdx` `(S_cdc, G)`: posterior samples for `kdx`
- `cdc_U0` `(S_cdc, G)`: posterior samples for `U0`
- `cdc_kappa_prep` `(G,)`: per-geo PrEP scaling constant

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Commands

All commands assume `standardized_input.npz` exists at repo root.

Deterministic baseline:

```bash
python -m cli --mode deterministic --scenario-mode baseline --units NY
```

Deterministic intervention:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs \
  --relationship-codes weaken_stigma_to_hivtest
```

Uncertainty:

```bash
python -m cli --mode uncertainty --scenario-mode baseline --units NY --n-samples 500 --seed 123
```

Use a custom standardized file path:

```bash
python -m cli --standardized-input /path/to/standardized_input.npz --mode deterministic --units NY
```

## Plotting

Baseline plot:

```bash
python -m cli --mode deterministic --scenario-mode baseline --units NY --plot --plot-dir plots
```

Intervention with baseline overlay:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs \
  --relationship-codes weaken_stigma_to_hivtest \
  --plot --plot-compare-baseline --plot-dir plots
```

Uncertainty plot:

```bash
python -m cli --mode uncertainty --units NY --n-samples 100 --plot --plot-dir plots
```

Each figure has 6 panels:
- SEM testing
- SEM prep use
- SEM risk behavior
- CDC incidence
- CDC diagnosed
- CDC prep on

Output filenames:
- no comparison: `plots/<UNIT>_<mode>_<scenario>.png`
- comparison: `plots/<UNIT>_<mode>_comparison.png`

## Interventions

Intervention definitions live in:
- `prediction/codebooks.py`

State codes apply variable-level shifts.
Relationship codes modify SEM coupling terms.

If a code references a variable not present in `sem_v_names`, that intervention is skipped.

## Troubleshooting

Missing input file:
- error: `Missing standardized input: ...`
- fix: place `standardized_input.npz` at repo root or pass `--standardized-input <path>`

Target year too large:
- error: `target_end_year=... exceeds standardized_input max year ...`
- fix: lower `--target-end-year` or use a standardized file with longer horizon

Matplotlib cache warning:
- harmless in restricted environments; plots still save successfully
