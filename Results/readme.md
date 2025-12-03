## Results

The `Results/` directory stores all outputs produced by running the bandit experiments and post‑processing scripts.

### Layout

```bash fall-2025-group12/Results/

Results/
├── NonStationaryEpsilonGreedy/ # Per-experiment .npy files for this model
├── SlidingWindowUCB/
├── CUSUMUCB/
├── EXP3S/
├── all_models_average_ci.png
├── all_models_cumulative_ci.png
├── model_average_all_ci.png
└── model_cumulative_all_ci.png
```

Each model subfolder (e.g., `Results/UCB/`) contains one set of files per random seed:

- `*_data.npy`   – Environment reward matrix used for that run (`n_steps × n_arms`).
- `*_rewards.npy` – Sequence of rewards obtained by the model over time.
- `*_matrix.npy`  – Arm‑selection indicator matrix over time (`n_steps × n_arms`).

These files are written when you run the main experiment pipeline (via `runner_lib` from `main.py`).

### Aggregate figures

After running `src/main/results.py`, the following summary plots are created in `Results/`:

- `all_models_average_ci.png`  
  - Mean rolling average reward over time for each model, with confidence intervals.

- `all_models_cumulative_ci.png`  
  - Mean cumulative reward over time for each model, with confidence intervals.

- `model_average_all_ci.png`  
  - For each model, plots of average reward and (optionally) selection statistics across experiments.

- `model_cumulative_all_ci.png`  
  - For each model, cumulative rewards and selections across experiments.

These figures let you visually compare how quickly and how well each algorithm adapts to the drifting environment.

### How to regenerate Results

1. Run all experiments (creates the per‑model `*_data.npy`, `*_rewards.npy`, `*_matrix.npy` files):

```
python -m src.main.main
```


2. From the project root, generate the aggregate plots:

```
python -m src.main.results
```

Both commands will populate/update the `Results/` directory and overwrite the PNG files if they already exist.
