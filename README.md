# Replication materials for "Validating open-source machine translation for quantitative text analysis"

This repository contains the replication data and code for the paper "Validating open-source machine translation for quantitative text analysis" by Hauke Licht, Ronja Sczepanski, Moritz Laurer, and Ayjeren Bekmuratovna forthcoming in _Political Science Research and Methods_ (PSRM).

The replication materials are also available in the PSRM Harvard Dataverse: https://doi.org/10.7910/DVN/8UOOEW

Please address any **questions** you have about the replication materials at the corresponding author Hauke Licht [hauke.licht@ubik.ac.at](mailto:hauke.licht@ubik.ac.at).

## Replication files

This repository contains the code and data for replicating the results reported in the paper and supporting materials.
Please refer to the R script [`run_replication.R`](./`run_replication.R`) for a **reproducibility run** of all analyses** (described in the next section).

The replication materials are organized in the following folders:

- `code/`: Contains the code for all experiments and analyses.
- `data/`: Contains the data used all experiments and analyses and quantitative results produced.
- `paper/`: Contains the figures and tables reported in the paper and the supporting materials.

### Code

The **code** in this repository is organized into the following folders:

- [`code/01-data_preparation/`](./`code/01-data_preparation/`): R scripts for preparing raw and labeled text data.
- [`code/02-machine_translation/`](./`code/02-machine_translation/`): Python scripts for machine translation of the text data.
- [`code/03-topic_modeling/`](./`code/03-topic_modeling/`): R scripts for fitting topic models to the text data.
- [`code/04-classifier_finetuning/`](./`code/04-classifier_finetuning/`): Python scripts for fine-tuning classifiers on the text data.
- [`code/05-analyses/`](./`code/05-analyses/`): R scripts for analyzing the results of topic modeling and classifier fine-tuning studies, i.e., Studies I and II. 

### Data

The **data** in this repository is organized into the following folders:

- `data/exdata/`: External data required for our analyses but not publicly available otherwise.
- `data/datasets`: prepared topic modeling and classifier fine-tuning datasets (based on code in [`code/01-data_preparation/`](./`code/01-data_preparation/`)), including texts' translations with open-source machine translation models (based on code in [`code/02-machine_translation/`](./`code/02-machine_translation/`))
- `data/results/`: topic modeling and classifier fine-tuning results

### Tables and figures

All **tables and figures** produced in our analyses and reported in the paper and supporting materials can be found in `paper/figures/` and `paper/tables/`, respectively.

## Reproducibility run

All data analysis for generating the numbers, tables, and figures reported in the paper and supporting materials were conducted in R.
The script [`run_replication.R`](./`run_replication.R`) implements a complete reproducibility run and logs the results to `replication_run.log`.

### Scope of reproducibility

The replication run ([`run_replication.R`](./`run_replication.R`)) focuses exclusively on the **main analyses**, i.e., the scripts in [`code/05-analyses/`](./`code/05-analyses/`). 
Data preparation ([`code/01-data_preparation/`](./`code/01-data_preparation/`)), topic model fitting ([`code/03-topic_modeling/`](./`code/03-topic_modeling/`)) and transformer classifier fine-tuning ([`code/04-classifier_finetuning/`](./`code/04-classifier_finetuning/`)) are *not* part of the reproducibility run because they require HPC/GPU infrastructure and long compute times (see Section [Compute times](#compute-times) below).
Pre-computed results from these steps are provided in `data/results/` so that the analysis scripts can be run without re-running them.

### Numerical reproducibility

The analysis scripts in [`code/05-analyses/`](./`code/05-analyses/`) read pre-computed results from `data/results/` and produce deterministic outputs. Running [`run_replication.R`](./`run_replication.R`) should therefore reproduce all numbers, tables, and figures in the paper exactly.

Results produced by earlier pipeline steps are subject to the following sources of non-determinism:

- **Machine translation** ([`code/02-machine_translation/`](./`code/02-machine_translation/`)): outputs are deterministic given the same model version and hardware, but floating-point non-determinism across GPU architectures may cause minor numerical differences.
- **Topic model fitting** ([`code/03-topic_modeling/`](./`code/03-topic_modeling/`)): STM models are fit with random initialisation controlled by random number generation state ("seed"); nevertheless, results may differ slightly across runs even with the same seed on different hardware.
- **Classifier fine-tuning** ([`code/04-classifier_finetuning/`](./`code/04-classifier_finetuning/`)): transformer fine-tuning on GPUs is non-deterministic by default; exact metric values may differ slightly across runs and GPU types.

### Estimated run times

- **Main analyses** (running [`run_replication.R`](./`run_replication.R`) on a consumer-grade laptop): approximately 15 minutes.
- **Full replication**:
  - Machine translation: see Table 1 in [000_README.pdf](./000_README.pdf)
  - Topic model fitting: approx. 20 minutes with 40 CPU cores on HPC cluster.
  - Classifier fine-tuning: see Tables 2f. in [000_README.pdf](./000_README.pdf)

## Computer setup

### Software requirements

### R 

We used R for 

1. raw and labeled text data preparation (see [`code/01-data_preparation`](./`code/01-data_preparation`)),
2. topics model fitting (see [`code/03-topic_modeling`](./`code/03-topic_modeling`)), and
3. analysis of our topic modeling and classifier fine-tuning studies' results  (see [`code/05-analyses`](./`code/05-analyses`)).

We used R 4.2.0 for the original research.
Analyses were successfully reproduced with R 4.5.2. 

We use the R package manager `renv` to manage R package dependencies.
The `renv` setup and [`replication_r_requirements.txt`](./`replication_r_requirements.txt`) cover only the packages required for the **main analyses** ([`code/05-analyses/`](./`code/05-analyses/`)). 
R packages needed for data preparation and topic model fitting are listed separately in [`r_requirements.txt`](./`r_requirements.txt`).

To install all required R packages, open R and run:

```R
install.packages("devtools")
devtools::install_version("renv", version = "1.0.10")
renv::restore()
```

A detailed list of R package dependencies and versions can be found in file [`replication_r_requirements.txt`](./`replication_r_requirements.txt`)

To also replicate the R-based data preparation and topic modeling fitting, please see the additional dependencies in [`r_requirements.txt`](./`r_requirements.txt`).

### Python 

We used Python for 

1. machine translation (see [`code/02-machine_translation/`](./`code/02-machine_translation/`)), and
2. classifier fine-tuning (see [`code/04-classifier_finetuning/`](./`code/04-classifier_finetuning/`))
3. BERTScore-based translation similarity analysis (see `code/05-analyses/translation_*/`)

We used Python 3.10 and `pip` and the python `venv` module to manage python package dependencies.
Specifically, we have kept packages used for machine translation and classifier fine-tuning separate to avoid version conflicts.

To create the virtual environments, open the Terminal app (macOS) or command line (Linux) and run:

```bash
cd code/02-machine_translation
chmod +x setup_venv.sh
./setup_venv.sh

cd ../03-classifier_finetuning
chmod +x setup_venv.sh
./setup_venv.sh
```

### Hardware requirements

All analyses in code/05-analyses/ can be run on a consumer-grade laptop with CPU processors.
For machine translation, topic model fitting, and classifier fine-tuning, we used the ETH Zurich's EULER HPC cluster via SLURM.
The necessary R, python, and SLURM job scripts can be found in folders [`code/02-machine_translation/`](./`code/02-machine_translation/`), [`code/03-topic_modeling/`](./`code/03-topic_modeling/`), and [`code/04-classifier_finetuning/`](./`code/04-classifier_finetuning/`), respectively.
