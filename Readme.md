


# Multi-Armed Bandit Simulation Project

## Overview
Simulates and compares multiple multi-armed bandit algorithms for research and experimentation on drifting news‑recommendation–style environments.  
Supports stationary and nonstationary/drifting setups with Gaussian and Bernoulli rewards, with an emphasis on reproducibility through fixed seeds, repeated experiments, and saved result matrices. 

## Table of Contents
- Overview
- Directory Structure
- Installation & Requirements
- Usage
- Modules
  - env.py
  - model.py
  - plot_ci.py
  - runner_lib.py
  - main.py
  - results.py
- Data and Outputs
- Customization

## Directory Structure

```bash fall-2025-group12/
│
├── demo/ # Demo GIF, video, and overview README
├── presentation/ # Capstone presentation slides and materials
├── reports/ # LaTeX report sources and compiled PDFs
├── research_paper/ # Supporting research documentation
├── Results/ # Saved experiment outputs and figures
│ ├── all_models_average_ci.png
│ ├── all_models_cumulative_ci.png
│ ├── model_average_all_ci.png
│ ├── model_cumulative_all_ci.png
│ ├── CUSUMUCB/
│ ├── EXP3S/
│ ├── NonStationaryEpsilonGreedy/
│ ├── SlidingWindowUCB/
│ └── UCB/
├── src/
│ ├── component/
│ │ ├── env.py # Synthetic environment generation
│ │ ├── model.py # Bandit algorithms
│ │ ├── plot_ci.py # Plotting with confidence intervals
│ │ └── runner_lib.py # Multi‑experiment runner
│ ├── main/
│ │ ├── main.py # High‑level experiment script
│ │ └── results.py # Aggregate and plot saved results
│ └── tests/ # Test scripts for env/model/main
│
└── requirements.txt # Python dependencies
```
## Installation & Requirements
- Python 3.8+.
- Required packages:
  - `numpy`
  - `pandas`
  - `matplotlib` 

Install all dependencies from the repository root:
- Install all dependencies at once:
    ```
    pip install -r requirements.txt
    ```


## Usage

### Run full experiment pipeline

    ```
    python -m src.main.main
    ```
This sets the project root, configures environment type (e.g., drifting), number of documents, observations, and experiments, then runs all models and saves outputs under `Results/`. 

### Regenerate plots from saved results
```
python -m src.main.results
```
This loads saved `.npy` arrays for each model, aggregates them across experiments, and produces average and cumulative performance plots with confidence intervals. 

## Modules

### env.py
- Generates synthetic bandit environments.  
- Supports:
  - Gaussian reward environments
  - Bernoulli reward environments
  - Custom drifting environment with evolving document popularity/quality and user preferences  
- Key parameters: environment type, number of arms/documents/users, observations, means/probabilities, standard deviations, categories, and drift hyperparameters (`alpha`, `a`, `b`, etc.). 

### model.py
- Implements multi‑armed bandit algorithms:
  - `NonStationaryEpsilonGreedy`
  - `UCB`
  - `SlidingWindowUCB`
  - `CUSUMUCB`
  - `EXP3S` 
- Each model provides methods for training, arm selection, logging rewards, saving basic statistics, and resetting state. 

### plot_ci.py
- Visualization utilities:
  - Rolling average rewards per arm and model with 95% confidence intervals.
  - Cumulative rewards per arm and model with confidence intervals.
  - Per‑model average and cumulative reward/selection plots across experiments. 

### runner_lib.py
- `runnerlib` class orchestrates experiments:
  - Creates the chosen environment using `create_environment`.
  - Initializes each model with its hyperparameters.
  - Runs training for multiple random seeds.
  - Saves environment data, realized rewards, and action matrices as `.npy` files into model‑specific folders in `Results/`. 

### main.py
- Top‑level experiment script:
  - Sets `projectroot`, adds it to `sys.path`, and defines global settings such as `num_experiments`, `ndocs`, `observations`, `epsilon`, and environment type.
  - Instantiates `runnerlib` and calls `run_all_models()`. 

### results.py
- Post‑processing script:
  - Loads all `*data.npy`, `*rewards.npy`, and `*matrix.npy` files for each model.
  - Stacks them into experiment × time × arms arrays.
  - Calls plotting functions in `plot_ci.py` to generate cross‑model and per‑model figures saved in `Results/`. 

## Data and Outputs
- Per‑experiment files (for each model in `Results/<ModelName>/`):
  - `*_data.npy` – reward matrices used in training.
  - `*_rewards.npy` – time‑series rewards.
  - `*_matrix.npy` – arm‑selection indicator matrices. 
- Aggregate figures in `Results/`:
  - `all_models_average_ci.png`
  - `all_models_cumulative_ci.png`
  - `model_average_all_ci.png`
  - `model_cumulative_all_ci.png` 

## Customization
- Modify high‑level experimental settings (number of experiments, environment, horizon length, epsilon, etc.) in `src/main/main.py`. 
- Adjust environment structure and drift parameters via keyword arguments to `create_environment` in `env.py`. 
- Tune algorithm‑specific hyperparameters (e.g., epsilon and learning rate, sliding‑window size, CUSUM thresholds, EXP3S gamma/alpha) in `runner_lib.py` or when instantiating models.
