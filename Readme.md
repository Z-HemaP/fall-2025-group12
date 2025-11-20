# Multi-Armed Bandit Simulation Project

## Overview

- Simulates multiple multi-armed bandit algorithms for research and experimentation.
- Supports stationary, nonstationary/drifting, Gaussian, and Bernoulli environments.
- Designed for comparing performance, reproducibility, and flexibility.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Modules](#modules)
    - [env.py](#envpy)
    - [model.py](#modelpy)
    - [plot.py](#plotpy)
    - [main.py](#mainpy)
- [Data and Outputs](#data-and-outputs)
- [Customization](#customization)
## Directory Structure
fall-2025-group12/
│
├── demo/                # Demo scripts and notebooks
├── presentation/        # Presentation slides and materials
├── reports/             # Project reports/documentation
├── research_paper/      # Supporting research documentation
├── Results/             # Output files (.csv, plots, etc)
│   ├── all_data.csv
│   ├── all_matrices.csv
│   ├── all_rewards.csv
│   ├── Data Average Plot.jpg
├── src/
│   ├── component/
│   │   ├── env.py       # Synthetic environment generation
│   │   ├── model.py     # Bandit models/algorithms
│   │   ├── plot.py      # Plotting functions
│   ├── main/
│   │   ├── main.py      # Experiment runner
│   ├── tests/           # Test cases
│
└── Readme.md            # Project documentation


## Installation & Requirements

- Requires Python 3.8+
- Needed packages:
    - `numpy`
    - `pandas`
    - `matplotlib`
- Install all dependencies at once:
    ```
    pip install -r requirements.txt
    ```

## Usage

- Run the main experiment script:
    ```
    python src/main/main.py
    ```
- Results are automatically saved in the `Results` folder.

## Modules

### env.py

- Generates bandit environments for simulations.
- Supports:
    - Gaussian environments
    - Bernoulli environments
    - Nonstationary (drifting) setups
- Configurable parameters:
    - Number of arms, means/probabilities, std deviations, observations, categories, etc.

### model.py

- Contains multi-armed bandit algorithm implementations.
- Algorithms included:
    - NonStationaryEpsilonGreedy
    - UCB (Upper Confidence Bound)
    - SlidingWindowUCB
    - CUSUMUCB
    - EXP3S
- Each model:
    - Can be configured independently
    - Offers training, arm selection, and logging

### plot.py

- Visualizes experiment data and results.
- Provides:
    - Violin plots for distributions
    - Rolling average and cumulative reward charts
    - Model comparison figures

### main.py

- Central experiment runner and results aggregator.
- Steps:
    - Sets up random seeds and parameters
    - Collects and saves run data
    - Creates visual summaries of results

## Data and Outputs

- Output files:
    - `all_data.csv`, `all_rewards.csv`, `all_matrices.csv`: Tabular experiment results
    - `Data Average Plot.jpg`: Visual summary of average model performance

## Customization

- Adjust parameters in `main.py` and `env.py` to:
    - Change number of arms, users, documents, and observations
    - Experiment with different reward structures and algorithm settings
    - Tune model hyperparameters as desired





