# AnaBFC: Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision

Bias Field Correction for Whole-Body MRI with Weak Anatomical Supervision.

## Overview

Bias field correction in whole-body MRI is difficult because MRI artifacts and normal organ contrast often look similar across a large field of view[cite: 1]. Conventional methods struggle to tell them apart, sometimes removing important anatomical details.

To address this, we introduce **AnaBFC**, an anatomy-aware framework that separates acquisition artifacts from normal tissue signals[cite: 1]. Using weak anatomical supervision, AnaBFC successfully removes intensity inhomogeneity while preserving essential anatomical structures and image contrast.


## Repository Structure

* `data/`: Dataloaders and processing scripts.
* `model/`: Network architectures for the AnaBFC framework.
* `option/`: Configuration files (hyperparameters, paths, etc.).
* `util/`: Utility functions, including metrics and logging.
* `train.py`: Main script for model training.
* `test.py`: Main script for inference and evaluation.

## Requirements

* Python 3.8
* PyTorch (CUDA support recommended)
* Visdom
