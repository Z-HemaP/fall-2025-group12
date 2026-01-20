from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from demo.model import NonStationaryEpsilonGreedy , UCB, SlidingWindowUCB, CUSUMUCB, EXP3S
from demo.env import *
from demo.plot_ci import *
from demo.runner_lib import runner_lib


N_ARM = None

# ---------------------------------------------------------------------
# 1. Load logged bandit data and build reward matrix
# ---------------------------------------------------------------------

def load_bandit_matrix(
    csv_path: Path,
    max_rows: Optional[int] = None,
    reward_col: str = "reward",
    category_col: str = "category",

):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # All unique categories
    all_categories = sorted(df[category_col].unique())

    # Respect N_ARM
    if (N_ARM is None) or (N_ARM >= len(all_categories)):
        categories = all_categories
    else:
        categories = all_categories[:N_ARM]
        df = df[df[category_col].isin(categories)]

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_arms = len(categories)
    T = len(df)

    print(f"Rows: {T}")
    print(f"Arms: {n_arms}")
    print(f"Categories: {categories}")

    data = np.zeros((T, n_arms), dtype=float)
    cat_series = df[category_col].to_numpy()
    rew_series = df[reward_col].to_numpy()

    for t in range(T):
        a = cat_to_idx[cat_series[t]]
        data[t, a] = rew_series[t]

    return data, categories


# ---------------------------------------------------------------------
# Helper for multiprocessing
# ---------------------------------------------------------------------

def _run_one_experiment(name, ModelClass, data, n_arms, seed, epsilon):
    """
    Helper to run one (model, seed) experiment.
    Returns (seed, rewards_list, rewards_matrix).
    """

    if name == "NonStationaryEpsilonGreedy":
        model = ModelClass(
            n_arms=n_arms, epsilon=epsilon, alpha=0.1, random_seed=seed
        )
    elif name == "UCB":
        model = ModelClass(n_arms=n_arms, random_seed=seed)
    elif name == "SlidingWindowUCB":
        model = ModelClass(n_arms=n_arms, window=200, random_seed=seed)
    elif name == "CUSUMUCB":
        model = ModelClass(
            n_arms=n_arms,
            epsilon=0.05,
            h=10.0,
            u_init=0.5,
            alpha=0.2,
            random_seed=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    table_str, rewards_list, rewards_matrix = model.train(data)
    return seed, rewards_list, rewards_matrix


# ---------------------------------------------------------------------
# 2. Run a model multiple times (NO TQDM)
# ---------------------------------------------------------------------

def run_single_model(
    name: str,
    ModelClass,
    data: np.ndarray,
    n_arms: int,
    num_experiments: int = 10,
    epsilon: float = 0.05,
):
    T = data.shape[0]
    rewards_all = np.zeros((num_experiments, T))
    matrices_all = np.zeros((num_experiments, T, n_arms))

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 10_000, num_experiments)

    print(f"Running {name} for {num_experiments} seeds...")

    for i, seed in enumerate(seeds):

        if name == "NonStationaryEpsilonGreedy":
            model = ModelClass(
                n_arms=n_arms, epsilon=epsilon, alpha=0.1, random_seed=seed
            )
        elif name == "UCB":
            model = ModelClass(n_arms=n_arms, random_seed=seed)
        elif name == "SlidingWindowUCB":
            model = ModelClass(n_arms=n_arms, window=200, random_seed=seed)
        elif name == "CUSUMUCB":
            model = ModelClass(
                n_arms=n_arms,
                epsilon=0.05,
                h=10.0,
                u_init=0.5,
                alpha=0.2,
                random_seed=seed,
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        table_str, rewards_list, rewards_matrix = model.train(data)

        rewards_all[i] = np.asarray(rewards_list)
        matrices_all[i] = np.asarray(rewards_matrix)

    return rewards_all, matrices_all

# ---------------------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------------------

def main():

    project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12/demo"
    os.chdir(project_root)

    data_path = os.path.join(project_root,"bandit_dataset_10k_time.csv")

    results_dir = os.path.join(project_root,"results")
    

    data, categories = load_bandit_matrix(data_path, max_rows=10000)
    n_arms = len(categories)

    model_classes = {
        "NonStationaryEpsilonGreedy": NonStationaryEpsilonGreedy,
        "UCB": UCB,
        "SlidingWindowUCB": SlidingWindowUCB,
        "CUSUMUCB": CUSUMUCB,
    }

    num_experiments = 50
    all_model_rewards = {}
    all_model_matrices = {}

    for name, cls in model_classes.items():

        print(f"\n=== Running model: {name} ===")

        rewards_all, matrices_all = run_single_model(
            name=name,
            ModelClass=cls,
            data=data,
            n_arms=n_arms,
            num_experiments=num_experiments,
        )

        all_model_rewards[name] = rewards_all
        all_model_matrices[name] = matrices_all

        np.save(results_dir / f"{name}_rewards.npy", rewards_all)
        np.save(results_dir / f"{name}_matrices.npy", matrices_all)

    print("\nGenerating plots...")

    model_average_plot_all_models_with_ci(
        all_model_rewards, all_model_matrices, alpha=0.2
    )
    plt.savefig(results_dir / "bandit_logged_avg_reward.png", dpi=300, bbox_inches="tight")
    plt.close()

    model_cumulative_plot_all_models_with_ci(
        all_model_rewards, all_model_matrices, alpha=0.2
    )
    plt.savefig(results_dir / "bandit_logged_cum_reward.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Done! Plots saved in /results")


if __name__ == "__main__":
    main()
