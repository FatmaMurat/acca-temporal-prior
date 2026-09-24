
# Temporal Prior-Guided Segmentation of Pulmonary Lesions in Longitudinal CT

Code and analysis outputs for the study *"Temporal Prior-Guided Segmentation of 
Pulmonary Lesions in Longitudinal CT: Input-Level versus Prompt-Level Prior 
Injection"*.

## Overview

In longitudinal CT follow-up the lesion delineation from the previous timepoint
is routinely available, yet rarely used by automatic segmentation methods. This
work converts the reference mask at t−1 into a soft spatial prior (exponential
distance encoding, tau = 15 mm) and injects it alongside the image at t. The
previous image itself is not required. Two injection pathways are compared: a
fourth input channel at full resolution, and the pretrained SAM2 mask-prompt
encoder on the stride-16 grid. Experiments cover 244 consecutive t−1→t pairs
from 136 patients with patient-level 5-fold cross-validation.

## Repository structure

    scripts/    analysis pipeline, run in numerical order (00 → 11)
    metrics/    result files underlying the tables and figures of the paper

## Pipeline

| Script | Purpose |
|---|---|
| `00_env_probe.py` | environment and dependency check |
| `01_prior_cache_and_baseline.py` | prior construction, alignment, copy-forward baseline |
| `02_make_colab_bundle.py` | packaging of images, masks and priors for training |
| `03_model.py` | architectures (SAM2 / EVA-02 backbones, FPN decoder, losses) |
| `04_train.py` | training loop, 5-fold cross-validation |
| `05_analyze.py` | per-pair evaluation |
| `06_ablation_compare.py` | paired comparison and subgroup analysis |
| `07_sam2_setup.py` | SAM2 checkpoint and configuration setup |
| `08_ensemble.py` | ensemble and oracle upper bound |
| `09_prompt_verify.py` | verification of the mask-prompt pathway |
| `10_growth_bland_altman.py` | area-based growth and categorical change analysis |
| `11_nnunet_baseline.py` | nnU-Net baseline with the same two channels |

`temporal_pair_dataloader_v2.py` is the data loader used by `04_train.py`.
The `_v2` suffix is a remnant of development; it is the only version.

## Result files

`metrics/` contains the per-pair and per-fold outputs from which every table and
figure in the paper can be reproduced, including the main comparison
(`tablo1.csv`), paired tests (`tablo1b.csv`), subgroup analyses (`tablo1c.csv`),
size strata (`boyut_strata.csv`), ensemble and oracle results
(`ensemble_pairs.csv`), growth analysis (`growth_sam2_prior.csv`), per-arm
training logs (`metrics_*.csv`) and the nnU-Net per-fold results
(`nnunet_prior/`).

## Paths and environment

The scripts were developed on Google Colab and contain absolute paths for that
environment. Before running elsewhere, update the `RUNS_DIR` and `CACHE_ROOT`
variables at the top of the relevant scripts.

Dependencies are listed in `requirements.txt`. SAM2 is installed separately from
its official repository (see `07_sam2_setup.py`).

## Data and model weights

The anonymised dataset and trained model weights are not included in this
repository and are available from the corresponding author on reasonable
request.

## Citation

[Citation details will be added upon publication.]

## License

MIT — see `LICENSE`.
