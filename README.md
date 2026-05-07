<div align="center">

<img src="img/logo.png" alt="Project logo" width="180"/>

# EEG Emotion Recognition

*A deep-learning pipeline for five-class emotion recognition (Happy, Sad, Fear, Disgust, Neutral) from EEG signals using 1D and 2D Convolutional Neural Networks on the SEED-V dataset.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MNE](https://img.shields.io/badge/MNE-1.7-2D4F8E)](https://mne.tools/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.13-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![h5py](https://img.shields.io/badge/h5py-3.11-FFA500)](https://www.h5py.org/)

</div>

---

## Overview

This repository contains the implementation of an **EEG-based emotion recognition** system,
developed as the final project for the *Intelligenza Artificiale e Sistemi (IAS)* course
at the University of Cagliari.

The pipeline ingests raw electroencephalographic (EEG) recordings from the
[SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html) dataset, segments them into
fixed-length temporal windows, and trains two convolutional neural network architectures
to classify five distinct emotional states:

| Label | Emotion  |
|:-----:|:---------|
|   0   | Disgust  |
|   1   | Fear     |
|   2   | Sad      |
|   3   | Neutral  |
|   4   | Happy    |

Two complementary architectures are compared:

- **1D-CNN** &mdash; convolves along the temporal axis of each EEG channel, treating the
  signal as a time series.
- **2D-CNN** &mdash; convolves over the joint spatial&ndash;temporal matrix
  (channels &times; time samples), exploiting inter-channel structure.

The two models are evaluated and compared on accuracy and confusion matrices.

## Repository Structure

```
IAS_EEG/
├── img/                          # Logo and figures
├── data/
│   ├── raw/                      # SEED-V raw recordings (.cnt) and metadata
│   └── processed/                # Generated artefacts (HDF5, CSV)
├── notebooks/
│   ├── Load_cnt_file_with_mne.ipynb   # Tutorial: loading .cnt with MNE
│   ├── Preprocessing.ipynb            # Segmentation + HDF5 dataset generation
│   └── Models.ipynb                   # CNN definition, training, evaluation
├── docs/                         # Project brief and reference survey (PDF)
├── environment.yaml              # Conda environment specification
├── LICENSE                       # Apache License 2.0
└── README.md
```

## Dataset

The pipeline uses a portion of the **SEED-V** dataset (BCMI Lab, Shanghai Jiao Tong
University), comprising 9 EEG recordings from **3 subjects** across **3 sessions** each.
Each session presents 15 video stimuli covering the five target emotions.

| Property                    | Value                                              |
|-----------------------------|----------------------------------------------------|
| Subjects                    | 3                                                  |
| Sessions per subject        | 3                                                  |
| Stimuli per session         | 15                                                 |
| Sampling frequency          | 1000 Hz                                            |
| EEG channels (after dropping `M1, M2, VEO, HEO`) | 62                            |
| Segment length              | 1.5 s (1500 samples)                               |
| Total segments extracted    | 14 691                                             |
| Train / Validation / Test   | 70 % / 15 % / 15 % (stratified)                    |

The raw data must be downloaded separately and placed under `data/raw/`. The
processed HDF5 archive (`eeg_dataset_v2.h5`, shape `(N, 62, 1500)`) and the
band-power feature CSV (`dataset_v3_features.csv`) are produced by the
preprocessing notebook.

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- ~10 GB of free disk space for the raw EEG recordings

### Installation

```bash
git clone <repository-url>
cd IAS_EEG
conda env create -f environment.yaml
conda activate eeg-env
```

### Data Placement

Place the 9 SEED-V `.cnt` files together with `Scores.xlsx`,
`emotion_label_and_stimuli_order.xlsx`, and `trial_start_end_timestamp.txt`
inside `data/raw/`.

## Usage

The notebooks are designed to be executed from the `notebooks/` directory; all
data paths are expressed relative to it (`../data/raw`, `../data/processed`).

1. **Preprocessing** &mdash; run [`notebooks/Preprocessing.ipynb`](notebooks/Preprocessing.ipynb).
   Loads the raw `.cnt` files, drops non-EEG channels, segments each video
   stimulus into 1.5 s windows, and serialises the result to
   `data/processed/eeg_dataset_v2.h5`. A second cell extracts log-PSD band-power
   features and saves them to `data/processed/dataset_v3_features.csv`.

2. **Modelling** &mdash; run [`notebooks/Models.ipynb`](notebooks/Models.ipynb).
   Loads the HDF5 dataset, performs a stratified 70/15/15 split, trains the
   CNN architecture, and reports test accuracy and confusion matrix.

## Results

Results (accuracies, confusion matrices, training curves) will be reported
here once the experiments are finalised.

## License

This project is released under the **Apache License 2.0**. See the
[LICENSE](LICENSE) file for the full text.

## Authors

| Author        | Email                                                                    |
|---------------|--------------------------------------------------------------------------|
| D. Ermellino  | [d.ermellino1@studenti.unica.it](mailto:d.ermellino1@studenti.unica.it)  |
| G. Dasara     | [g.dasara2@studenti.unica.it](mailto:g.dasara2@studenti.unica.it)        |
| L. Agus       | [l.agus22@studenti.unica.it](mailto:l.agus22@studenti.unica.it)          |

*Department of Mathematics and Computer Science &mdash; University of Cagliari*

## Acknowledgements

- The [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html) dataset is provided by the
  BCMI Lab, Shanghai Jiao Tong University.
- EEG I/O and pre-processing rely on the [MNE-Python](https://mne.tools/) library.
