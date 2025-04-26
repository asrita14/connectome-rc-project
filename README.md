# Connectome-RC-Project
# Neuroinformatics Project

This repository contains projects completed as part of coursework for the **Neuroinformatics** class (Spring 2025). The work focuses on applying **connectome-based architectures** in **reservoir computing** frameworks, exploring biological network structures to improve machine learning performance on cognitive tasks.

## Project Overview

The primary objectives of these projects are to:

- Implement and evaluate **Spiking Neural Networks (SNNs)**, **Memristive Switching Networks (MSS)**, and **Echo State Networks (ESNs)**.
- Integrate **empirical connectome data** (e.g., Human and Drosophila) into reservoir models.
- Analyze the impact of network structure on computational performance, criticality, and learning capacity.
- Visualize network dynamics using **PCA projections**, **spike raster plots**, and **weight evolution** plots.
- Perform **cross-species analysis** to compare the role of network motifs in task performance.

Experiments include tasks such as:

- **Perceptual Decision Making** (NeuroGym)
- **Memory Capacity Testing** (Reservoir Computing Benchmark)

## Repository Structure
```text
├── code/                  # Python scripts and Jupyter notebooks
│   ├── snn_models/
│   ├── mss_models/
│   ├── esn_models/
│   └── utils/
├── data/                  # Connectome files (external links if too large)
├── figures/               # Visualizations (PCA plots, spike rasters, etc.)
├── results/               # Model evaluation results
├── report/                # Project reports, slides, supplementary material
└── README.md              # This file
```
## Setup Instructions
```
pip install --upgrade wheel==0.38.4 setuptools==65.5.1 "pip<24.1"
git clone https://github.com/netneurolab/conn2res.git
cd conn2res
pip install .
cd ..
git clone -b v0.0.1 https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```
## Acknowledgments

-conn2res toolbox: for connectome-based reservoir modeling.
-NeuroGym: for cognitive task datasets.
-Neuroinformatics course instructors: for guidance and feedback.
