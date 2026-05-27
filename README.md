# AnaBFC: Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision

This repository contains the official PyTorch implementation of the paper **"Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision"**. 

## 📖 Overview

Bias field correction is particularly challenging in whole-body MRI, where acquisition-related intensity inhomogeneity and organ-dependent anatomical contrast often vary at similar spatial scales across a large field of view. Conventional methods based on local homogeneity can reduce intensity inhomogeneity while also weakening tissue contrast and lesion conspicuity.

We present **AnaBFC**, an anatomy-aware framework for bias field correction in whole-body MRI trained with weak anatomical supervision. 
* **Training:** AnaBFC uses coarse tissue masks to define where intensity consistency should be encouraged, regularizes the estimated bias field with smoothness and unit-mean constraints, and leverages unlabeled scans through consistency regularization. It also introduces *Anatomy-aware Contrast Calibration (AnaCC)* to discourage the bias estimate from absorbing normal anatomical contrast.
* **Inference:** At inference, AnaBFC requires no anatomical masks and takes only the input MRI volume.

## 📂 Repository Structure

The project is organized as follows:

* `data/`: Contains dataloaders and data processing scripts for the MRI volumes.
* `model/`: Defines the AnaBFC network architectures.
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
