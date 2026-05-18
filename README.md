# Satellite Optical Brightness Model

Code and data accompanying the paper **"Satellite Optical Brightness"** (2023)  
by Forrest Fankhauser, J. Anthony Tyson, and Jacob Askari.

📄 **Paper:** [arXiv:2305.11123](https://arxiv.org/abs/2305.11123) | [AJ 166 59 (2023)](https://doi.org/10.3847/1538-3881/ace047)

---

## Overview

As mega-constellations of low-Earth orbit satellites grow in number, their optical brightness has become a significant concern for ground-based astronomy. This repository contains the data and Jupyter notebooks used to build and evaluate brightness models for **OneWeb** and **Starlink** satellites.

The models calculate the apparent magnitude of a satellite as seen by a ground-based observer, accounting for:

- **Direct solar illumination** of the satellite body and solar panels
- **Indirect illumination** from sunlight reflected off Earth's surface
- **BRDF (Bidirectional Reflectance Distribution Function)** fitting for satellite chassis and solar panel components
- **Observer geometry** — satellite position, solar phase angle, and observer location

A key finding of the underlying research is that Earthshine (indirect reflected light from Earth's surface) contributes meaningfully to apparent satellite brightness, especially during civil twilight.

---



## Repository Structure

```
Brightnessmodel/
├── OneWeb(MMT-9)/               # MMT telescope observations and modeling for 4 OneWeb satellites
│   ├── sat/                     # Raw MMT-9 observation data for each satellite
│   ├── sat_results/             # Satellite position data from sat-checker
│   ├── 165/                     # Output plots for OneWeb-0165
│   ├── 187/                     # Output plots for OneWeb-0187
│   ├── 208/                     # Output plots for OneWeb-0208
│   ├── 209/                     # Output plots for OneWeb-0209
│   ├── OneWeb_0165.ipynb        # Brightness modeling notebook for OneWeb-0165
│   ├── OneWeb_0187.ipynb        # Brightness modeling notebook for OneWeb-0187
│   ├── OneWeb_0208.ipynb        # Brightness modeling notebook for OneWeb-0208
│   └── OneWeb_0209.ipynb        # Brightness modeling notebook for OneWeb-0209
├── Oneweb_Overall/              # Overall OneWeb brightness analysis
│   ├── sat/                     # Raw observation data
│   ├── sat_results/             # Satellite position data from sat-checker
│   └── *.ipynb / *.png          # Analysis notebooks and output figures
├── analysis/                    # General analysis notebooks and utilities
├── requirements.txt             # Python dependencies
├── README.md
└── TODO.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Dependencies

| Package | Purpose |
|---|---|
| `lumos-sat` | Satellite brightness modeling (BRDF, geometry, flux calculation) |
| `pandas` | Data loading and manipulation |
| `ipykernel` | Jupyter notebook kernel |

---

## Data

The `data/` directory contains a database of brightness observations of OneWeb and Starlink satellites collected across a range of solar phase angles. These observations are used to:

1. Fit empirical BRDF parameters for satellite components (chassis and solar panels)
2. Validate predicted brightness against measured apparent magnitudes

---

## Methods

The brightness modeling pipeline follows the methodology described in the paper:

1. **BRDF estimation** — Reflective properties of satellite components are fit to multi-angle observational data.
2. **Flux integration** — Scattering contributions from all satellite surfaces are integrated to produce an angular brightness pattern.
3. **Earthshine correction** — An improved model of Earth surface reflectance (derived from aircraft data) is applied to account for indirect illumination.
4. **Apparent magnitude calculation** — The full model predicts observed brightness as a function of satellite position and observer location.

---

