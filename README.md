# HGSM reviewer code package

This repository contains the code associated with the manuscript:

**Physics-Guided Surrogate Adaptation and Generation for Accelerator Beam Dynamics**

The package is organised to make the workflow easier for editors and reviewers to inspect. The original uploaded scripts are preserved in `original_uploaded_scripts/`, and reviewer-facing copies with clearer names are provided in `scripts/`.

## Important reviewer note

This package contains the uploaded Python scripts and documentation, but the current upload does **not** include all data/checkpoint assets needed to reproduce every result end-to-end. The missing assets are listed in `MISSING_ASSETS.md`.

In particular, the BO and PPO optimisation scripts require `model_single.py` and `single_runner.py`, which were not included in the uploaded files. These must be added before submitting the code to reviewers if the optimisation results are part of the manuscript.

## Directory structure

```text
hgsm_reviewer_code/
├── README.md
├── MANIFEST.md
├── RUN_ORDER.md
├── REVIEWER_NOTES.md
├── MISSING_ASSETS.md
├── CODE_AVAILABILITY_STATEMENT_TEMPLATE.md
├── requirements.txt
├── environment.yml
├── check_package.py
├── models_first.py
├── models_second.py
├── scripts/
│   ├── 01_sobol_sampling_full.py
│   ├── 02_sobol_sampling_shifted_validation.py
│   ├── 03_run_parallel_astra_simulations.py
│   ├── 04_make_train_val_test_hdf5.py
│   ├── 05_make_calibration_hdf5.py
│   ├── 06_train_hypernet_mean_baseline.py
│   ├── 07_train_hypernet_uq.py
│   ├── 08_continue_hypernet_uq_training.py
│   ├── 09_train_quantile_baseline.py
│   ├── 10_train_single_baseline.py
│   ├── 11_evaluate_hypernet_validation_test.py
│   ├── 12_few_shot_latent_calibration.py
│   ├── 13_generate_shifted_systems.py
│   ├── 14_bo_transfer_studies.py
│   └── 15_ppo_transfer_studies.py
└── original_uploaded_scripts/
```

## Main workflow

A typical full workflow is:

1. Generate Sobol samples for simulation inputs.
2. Run ASTRA simulations to produce raw `.pkl` simulation outputs.
3. Convert simulation outputs into HDF5 training, validation, test, and calibration datasets.
4. Train the HyperNet/HGSM model and baseline models.
5. Select the model using validation metrics and evaluate once on the held-out test set.
6. Perform few-shot latent calibration.
7. Generate shifted surrogate systems by physics-guided latent traversal.
8. Run BO and PPO transfer studies on the generated surrogate systems.

See `RUN_ORDER.md` for script-level details.

## Installation

Create an environment using either `requirements.txt` or `environment.yml`.

```bash
conda env create -f environment.yml
conda activate hgsm-review
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The ASTRA simulation step also requires the ASTRA executable and input files, which are not included in this package.

## Quick package check

The following command checks that the provided Python files are syntactically valid and reports whether expected data/checkpoint assets are present:

```bash
python check_package.py
```

This check does not run training, ASTRA simulation, BO, or PPO experiments.

## Notes on class names

The class name `HypyerNet` is retained to preserve compatibility with existing checkpoint files and scripts. It appears to be a historical spelling of `HyperNet`.


## Refactored reviewer-facing training script

The mean-prediction HyperGAN training script has been refactored for readability and reviewer inspection:

```bash
python scripts/06_train_hypernet_mean_baseline.py --path-dir . --epochs 1000 --embedding-dim 100 --beta 50 --seed 0
```

The refactored script preserves the original workflow while using clearer configuration, safer device handling, deterministic seeding, and a structured trainer class. See `REFACTORING_NOTES.md` for details.
