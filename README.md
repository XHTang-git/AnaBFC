# AnaBFC: Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision

[cite_start]This repository contains the official PyTorch implementation of the paper **"Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision"** (Medical Image Analysis, 2026)[cite: 2, 7]. 

## 📖 Overview

[cite_start]Bias field correction is particularly challenging in whole-body MRI, where acquisition-related intensity inhomogeneity and organ-dependent anatomical contrast often vary at similar spatial scales across a large field of view[cite: 14]. [cite_start]Conventional methods based on local homogeneity can reduce intensity inhomogeneity while also weakening tissue contrast and lesion conspicuity[cite: 15].

[cite_start]We present **AnaBFC**, an anatomy-aware framework for bias field correction in whole-body MRI trained with weak anatomical supervision. 
* [cite_start]**Training:** AnaBFC uses coarse tissue masks to define where intensity consistency should be encouraged, regularizes the estimated bias field with smoothness and unit-mean constraints, and leverages unlabeled scans through consistency regularization[cite: 17]. [cite_start]It also introduces *Anatomy-aware Contrast Calibration (AnaCC)* to discourage the bias estimate from absorbing normal anatomical contrast[cite: 18].
* [cite_start]**Inference:** At inference, AnaBFC requires no anatomical masks and takes only the input MRI volume.

## 📂 Repository Structure

The project is organized as follows:

* `data/`: Contains dataloaders and data processing scripts for the MRI volumes.
* `model/`: Defines the AnaBFC network architectures, including the Bias Field Correction Module (BFCM) and the Anatomy-aware Contrast Calibration (AnaCC) network.
* `option/`: Contains configuration files (e.g., hyperparameters, paths) for training and testing setups.
* `util/`: Contains utility functions, including metrics (SSIM, CV computation) and logging tools.
* `train.py`: The main script for training the AnaBFC model.
* `test.py`: The main script for running inference/testing on new MRI volumes.

## 🛠️ Requirements

The code has been tested under the following environment:
* Python 3.8
* PyTorch (with CUDA 12.6 support)
* Visdom (for real-time training visualization)

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/2018013129-WBFC.git](https://github.com/yourusername/2018013129-WBFC.git)
   cd 2018013129-WBFC
