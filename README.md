# System Dynamics Prediction Model for StigmaScope

This model captures socio-behavioral and stigma-related dynamics that shape HIV epidemiological outcomes for men who have sex with men (MSM).

This repository contains the prediction runtime that will be integrated into the [StigmaScope.com](https://stigmascope.com) dashboard for the Ending the HIV Epidemic initiative.

## Overview
### Model Diagram

![System dynamics model diagram](EEE_slide_model.png)

The runtime produces forward projections for all variables in the socio-behavioral model (SEM) and population dynamics model (Epi).

### Socio-behavioral variables (proportion in population)

- anticipated healthcare stigma
- general social stigma
- family stigma
- seeing a healthcare provider annually
- sexual orientation disclosure to a healthcare provider
- risk behavior
- use of PrEP in the past year
- HIV testing in the past year

### Population dynamics outputs (counts)

- annual PrEP use
- HIV diagnoses
- estimated HIV incidence

The runtime reads prepared standardized inputs and parameters, and supports deterministic and uncertainty prediction for user-specified forecast years under baseline and intervention scenarios.

## Runtime Flow: Inputs -> Model -> Outputs

### 1. Input

Required file:
- `standardized_input.npz`

Details on the file structure are below. 

### 2. Model

Execution path:
- `cli.py` parses arguments
- `runner.py` dispatches deterministic or uncertainty mode
- `input/standardized_runtime.py` reads `standardized_input.npz`
- `prediction/joint.py` runs SEM + CDC coupling
- `prediction/intervention.py` applies intervention effects when enabled

### 3. Output

Runtime returns in-memory dataclasses from `output/types.py`:
- deterministic: `JointOutput`
- uncertainty: `UncertaintyOutput`

Optional outputs:
- `--save <path>` writes pickle output
- `--plot` writes PNGs

## Project Structure

### Primary Runtime Files
- `standardized_input.npz`: required runtime input
- `cli.py`: command-line entrypoint
- `config.py`: runtime configuration and validation
- `runner.py`: orchestration (`run_prediction`)
- `alignment.py`: year alignment and CDC input derivation utilities
- `plotting.py`: six-panel plotting per unit
- `requirements.txt`: dependencies

### Source Tree

```text
.
├── cli.py
├── config.py
├── runner.py
├── alignment.py
├── plotting.py
├── requirements.txt
├── standardized_input.npz
├── input/
│   ├── __init__.py
│   ├── loaders.py
│   └── standardized_runtime.py
├── prediction/
│   ├── __init__.py
│   ├── codebooks.py
│   ├── epi_predictor.py
│   ├── intervention.py
│   ├── joint.py
│   ├── sem_predictor.py
│   └── transforms.py
├── output/
│   ├── __init__.py
│   ├── io.py
│   └── types.py
└── data/
    ├── __init__.py
    ├── params_cdc.py
    ├── params_sem.py
    └── unit.py
```

## `standardized_input.npz` Schema

### Schema Blocks
- index metadata (what each axis means)
- year axes (what timelines exist)
- SEM observed (for plotting)
- SEM parameters (for prediction)
- CDC raw input series (for plotting)
- CDC posterior parameter samples (for uncertainty prediction)

### Axis Symbols
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

- Assumption: `standardized_input.npz` exists at repo root.

### CLI Options

- `--standardized-input <path>`: path to standardized input file (default: `standardized_input.npz`)
- `--mode {deterministic,uncertainty}`: run type
- `--scenario-mode {baseline,intervention}`: scenario to run
- `--target-end-year <year>`: truncate forecast horizon to this year (must be <= max year in input)
- `--units <id1 id2 ...>`: run only selected geographies (for example `NY CA`)
- `--n-samples <int>`: Monte Carlo sample count (uncertainty mode)
- `--seed <int>`: random seed (uncertainty mode)
- `--state-codes <code1 code2 ...>`: state intervention codes (intervention scenario)
- `--relationship-codes <code1 code2 ...>`: relationship intervention codes (intervention scenario)
- `--save <path>`: save runtime output as pickle
- `--plot`: generate plots
- `--plot-dir <path>`: directory for generated plots (default: `plots`)
- `--plot-units <id1 id2 ...>`: plot subset of units (defaults to run units)
- `--plot-compare-baseline`: overlay opposite scenario in plots

### Examples

Deterministic baseline run:

```bash
python -m cli --mode deterministic --scenario-mode baseline --units NY
```

Deterministic intervention run:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs \
  --relationship-codes weaken_stigma_to_hivtest
```

Uncertainty run:

```bash
python -m cli --mode uncertainty --scenario-mode baseline --units NY --n-samples 500 --seed 123
```

Custom standardized input path:

```bash
python -m cli --standardized-input /path/to/standardized_input.npz --mode deterministic --units NY
```

## Plotting

### Baseline Plot

```bash
python -m cli --mode deterministic --scenario-mode baseline --units NY --plot --plot-dir plots
```

### Intervention With Baseline Overlay

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs \
  --relationship-codes weaken_stigma_to_hivtest \
  --plot --plot-compare-baseline --plot-dir plots
```

### Uncertainty Plot

```bash
python -m cli --mode uncertainty --units NY --n-samples 100 --plot --plot-dir plots
```

### Plot Contents

- Each figure has 6 panels:
- SEM testing
- SEM prep use
- SEM risk behavior
- CDC incidence
- CDC diagnosed
- CDC prep on

### Output Filenames
- no comparison: `plots/<UNIT>_<mode>_<scenario>.png`
- comparison: `plots/<UNIT>_<mode>_comparison.png`

## Interventions

Definition file:
- `prediction/codebooks.py`

State codes apply variable-level shifts.
Relationship codes modify SEM coupling terms.

If a code references a variable not present in `sem_v_names`, that intervention is skipped.

### Intervention Examples

State intervention only:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs
```

Relationship intervention only:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --relationship-codes weaken_stigma_to_hivtest
```

Combined state + relationship interventions:

```bash
python -m cli \
  --mode deterministic \
  --scenario-mode intervention \
  --units NY \
  --state-codes reduce_ahs increase_seehcp \
  --relationship-codes weaken_stigma_to_hivtest
```

Intervention in uncertainty mode:

```bash
python -m cli \
  --mode uncertainty \
  --scenario-mode intervention \
  --units NY \
  --n-samples 500 \
  --seed 123 \
  --state-codes reduce_ahs \
  --relationship-codes weaken_stigma_to_hivtest
```

## Troubleshooting

### Missing Input File
- error: `Missing standardized input: ...`
- fix: place `standardized_input.npz` at repo root or pass `--standardized-input <path>`

### Target Year Too Large
- error: `target_end_year=... exceeds standardized_input max year ...`
- fix: lower `--target-end-year` or use a standardized file with longer horizon

### Matplotlib Cache Warning
- harmless in restricted environments; plots still save successfully
