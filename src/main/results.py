#%%
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.component.plot_ci import *


results_dir = os.path.join(project_root, "Results")
# model_names = ['EXP3S']
model_names = ["NonStationaryEpsilonGreedy", "SlidingWindowUCB", "CUSUMUCB", "EXP3S"]

all_model_data = {}
all_model_rewards = {}
all_model_matrices = {}

for model_name in model_names:
    model_dir = os.path.join(results_dir, model_name)
    data_arrays = []
    rewards_arrays = []
    matrix_arrays = []
    for fname in sorted(os.listdir(model_dir)):
        if fname.endswith("_data.npy"):
            arr = np.load(os.path.join(model_dir, fname))
            # arr shape: (n_steps, n_arms)
            data_arrays.append(arr)
        elif fname.endswith("_rewards.npy"):
            arr = np.load(os.path.join(model_dir, fname))
            # arr shape: (n_steps, n_arms)
            rewards_arrays.append(arr)
        elif fname.endswith("_matrix.npy"):
            arr = np.load(os.path.join(model_dir, fname))
            # arr shape: (n_steps, n_arms)
            matrix_arrays.append(arr)
    if data_arrays:
        # shape: (num_experiments, n_steps, n_arms)
        all_model_data[model_name] = np.stack(data_arrays, axis=0)
    if rewards_arrays:
        # shape: (num_experiments, n_steps, n_arms)
        all_model_rewards[model_name] = np.stack(rewards_arrays, axis=0)
    if matrix_arrays:
        # shape: (num_experiments, n_steps, n_arms)
        all_model_matrices[model_name] = np.stack(matrix_arrays, axis=0)

# %%
plot_all_models_average_with_ci(all_model_data, arm_means_dict=None, alpha=0.2)
plt.savefig(os.path.join(results_dir, "all_models_average_ci.png"),
            dpi=200, bbox_inches="tight")
plt.show()

plot_all_models_cumulative_with_ci(all_model_data, arm_means_dict=None, alpha=0.2)
plt.savefig(os.path.join(results_dir, "all_models_cumulative_ci.png"),
            dpi=200, bbox_inches="tight")
plt.show()

model_average_plot_all_models_with_ci(all_model_rewards, all_model_matrices, alpha=0.15)
plt.savefig(os.path.join(results_dir, "model_average_all_ci.png"),
            dpi=200, bbox_inches="tight")
plt.show()

model_cumulative_plot_all_models_with_ci(all_model_rewards, all_model_matrices, alpha=0.15)
plt.savefig(os.path.join(results_dir, "model_cumulative_all_ci.png"),
            dpi=200, bbox_inches="tight")
plt.show()
# %%
