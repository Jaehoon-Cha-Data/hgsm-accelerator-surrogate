# HGSM reviewer code package

This repository contains the code associated with the manuscript:

**Physics-Guided Surrogate Adaptation and Generation for Accelerator Beam Dynamics**


## Directory structure

```text
hgsm_reviewer_code/
├── README.md
├── requirements.txt
├── models_first.py
├── models_second.py
├── scripts/
    ├── 01_train_hypernet_mean_baseline.py
    ├── 02_train_hypernet_uq.py
    ├── 03_continue_hypernet_uq_training.py
    ├── 04_train_quantile_baseline.py
    ├── 05_train_single_baseline.py
    ├── 06_hgsm_model_inference.py
    ├── 07_few_shot_latent_calibration.py
    ├── 08_generate_shifted_systems.py
    ├── 09_bo_transfer_studies.py
    └── 10_ppo_transfer_studies.py

```

## Main workflow

A typical full workflow is:

1. Train the HyperNet/HGSM model and baseline models.
2. Select the model using validation metrics and evaluate once on the held-out test set.
3. Perform few-shot latent calibration.
4. Generate shifted surrogate systems by physics-guided latent traversal.
5. Run BO and PPO transfer studies on the generated surrogate systems.


## Installation

Create an environment using either `requirements.txt` or `environment.yml`.

```bash
conda env create -f environment.yml
conda activate hgsm-review
```
