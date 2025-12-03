# src/ Package Overview

The `src/` directory contains all Python source code for generating environments, running multi‑armed bandit algorithms, orchestrating experiments, and visualizing results.

## Structure

```bash fall-2025-group12/src/

src/
├── component/
│ ├── env.py
│ ├── model.py
│ ├── plot_ci.py
│ └── runner_lib.py
├── main/
│ ├── main.py
│ └── results.py
└── tests/

```


## component/

### env.py
- Builds synthetic bandit environments.
- Supports Gaussian, Bernoulli, and custom drifting setups.
- Main entry point: a function that returns a rewards matrix with shape `(n_steps, n_arms)` given parameters such as environment type, number of documents/arms, observations, categories, and drift hyperparameters.

### model.py
- Implements all bandit algorithms used in the project:
  - `NonStationaryEpsilonGreedy`
  - `UCB`
  - `SlidingWindowUCB`
  - `CUSUMUCB`
  - `EXP3S`
- Each class:
  - Is initialized with `n_arms` and algorithm‑specific hyperparameters.
  - Exposes a `train(data)` method that:
    - Interacts with the environment data over time.
    - Returns a summary table, a reward history, and a matrix of chosen arms.

### plot_ci.py
- Contains plotting utilities for experiment analysis:
  - Average reward over time with confidence intervals for all models.
  - Cumulative reward over time with confidence intervals.
  - Per‑model average and cumulative reward/selection plots.
- Functions consume stacked arrays of shape `(num_experiments, n_steps, n_arms)` and save or show Matplotlib figures.

### runner_lib.py
- Provides the `runner_lib` (or similarly named) class to orchestrate experiments.
- Responsibilities:
  - Create the chosen environment using `env.py`.
  - Initialize each bandit model from `model.py` with appropriate hyperparameters.
  - Run multiple experiments over different random seeds.
  - Save environment data, rewards, and action matrices as `.npy` files in the top‑level `Results/` directory, organized per model.

## main/

### main.py
- High‑level script to run the full experiment pipeline.
- Typical steps:
  - Set the project root and add it to `sys.path`.
  - Configure experiment settings (environment type, number of documents, number of experiments, horizon length, epsilon, etc.).
  - Instantiate `runner_lib` and call its method to run all models and produce raw result files under `Results/`.

### results.py
- Post‑processing and visualization script.
- Loads all `.npy` files produced by the experiments from the `Results/` directory.
- Stacks them into arrays per model and calls functions in `plot_ci.py` to:
  - Generate cross‑model average and cumulative reward plots.
  - Generate per‑model average and cumulative plots.
- Saves the final figures back into the `Results/` directory.

## tests/

- Placeholder for unit or integration tests related to:
  - Environment generation.
  - Model behavior.
  - Main/runner pipelines.
- Can be extended with automated checks to ensure reproducibility and correctness as the project evolves.
