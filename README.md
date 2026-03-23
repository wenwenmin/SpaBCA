# SpaBCA: Inferring Super-Resolved Gene Expression by Integrating Histology Images and Spatial Transcriptomics with Bi-directional Cross-Attention

## Introduction

Spatial transcriptomics (ST) enables genome-wide gene expression profiling with spatial context, yet its limited resolution restricts the characterization of fine-grained tissue heterogeneity. Imaging-based methods achieve high spatial resolution but are limited in gene coverage, while sequencing-based platforms provide whole-transcriptome profiling but suffer from coarse spatial resolution.

To address this challenge, we propose **SpaBCA**, a multimodal framework for inferring **super-resolved spatial gene expression** by integrating histology images with spot-level ST data. Specifically, gene expression is enhanced via spatial interpolation, while histology images are encoded to extract morphological features. A unified fusion network models cross-modal interactions, where **self-attention refines histological representations** and **bidirectional cross-attention (BCA)** captures morphology–gene dependencies. In addition, a **channel attention (CA)** mechanism further enhances feature representation, and a **multiple instance learning (MIL)** strategy ensures consistency with observed measurements.

Extensive experiments demonstrate that SpaBCA consistently outperforms state-of-the-art methods, achieving significant improvements in RMSE, SSIM, and PCC across multiple datasets. Moreover, SpaBCA enhances spatial patterns and facilitates biologically meaningful downstream analysis, highlighting its potential for advancing spatial transcriptomics research.

---

## Overview

<p align="center">
  <img src="figures/overview.png" width="900">
</p>

---

## Requirements

All experiments were conducted on an NVIDIA GPU (e.g., RTX 3090).

```bash
conda create -n SpaBCA python=3.10
conda activate SpaBCA
pip install -r requirements.txt
```

## Data

We evaluate SpaBCA on multiple publicly available spatial transcriptomics datasets, including:

- **Xenium human breast cancer dataset**  
- **Visium HD human breast cancer dataset**  
- **Visium HD mouse brain dataset**  


The datasets can be downloaded from the following sources:

- Xenium: https://www.10xgenomics.com/products/xenium-in-situ  
- Visium HD: https://www.10xgenomics.com/datasets/visium-hd  


---

## Data Preprocessing

This project provides end-to-end preprocessing pipelines for different spatial transcriptomics platforms:

- `Preprocessing_Visium_HD.py` — preprocessing workflow for Visium HD data  
- `Preprocessing_Xenium.py` — preprocessing workflow for Xenium data  

These scripts generate the required inputs for training and inference, including:

- processed histology images  
- spatial coordinates  
- tissue masks  
- gene expression matrices  

---

## Pre-trained Foundation Model

We adopt **UNI**, a general-purpose foundation model for histology feature extraction.

Before running SpaBCA, please request access to the pretrained weights:

👉 https://huggingface.co/mahmoodlab/UNI

## Training and Inference

The SpaBCA pipeline consists of four stages:

1. Data preprocessing  
2. High-resolution gene construction  
3. Histology feature extraction  
4. SpaBCA inference  

After preprocessing, the main steps are as follows:

### Step 1: High-resolution gene construction
```bash
python High-Density_Gene_Expression_Construction.py --directory data/
```
### Step 2: Histology feature extraction
```bash
python Histological_Feature_Extraction.py --directory data/ --login YOUR_KEY
```
### Step 3: SpaBCA inference
```bash
python SpaBCA_demo.py --directory data/ --epochs 500 --n-states 5
```

## Results

SpaBCA achieves superior performance across multiple datasets:

- **RMSE ↓**
- **SSIM ↑**
- **PCC ↑**

Compared with previous state-of-the-art methods (e.g., HISTEX), SpaBCA provides:

- lower prediction error  
- improved structural similarity  
- stronger correlation with ground truth  

---

## Contact

If you have any questions, please contact:

- jiahui_ding2025@163.com


---

## Paper

This repository contains the implementation of:

**"SpaBCA: A Cross-Modal Fusion Framework for Super-Resolved Spatial Transcriptomics via Morphological-Genetic Synergy"**

*(Submitted to ISBRA 2026)*

