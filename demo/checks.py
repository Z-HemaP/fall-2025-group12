#%%
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12/demo"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from plot_ci import *


results_dir = os.path.join(project_root, "results")
# model_names = ['EXP3S']
model_names = ["EXP3S"]#, "SlidingWindowUCB", "CUSUMUCB", "EXP3S"]

all_model_data = {}
all_model_rewards = {}
all_model_matrices = {}

for model_name in model_names:
    model_dir = os.path.join(results_dir)
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
