# %%
import os
import sys

project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.component.model import NonStationaryEpsilonGreedy, UCB, SlidingWindowUCB, CUSUMUCB, EXP3S
from src.component.env import *
from src.component.plot import *
import warnings
import csv
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
num_experiments = 50
np.random.seed(42)
seeds = np.random.choice(np.arange(100, dtype=int),
                         size=num_experiments,
                         replace=False)
environment = "drifting"
n_docs = 2  # number of documents (arms)

epsilon = 0.05
observations = 10000

arm_means = None

# %%
# Dictionary to store results for each model
model_results = {
    'NonStationaryEpsilonGreedy': {'all_data': [], 'all_rewards': [], 'all_matrices': []},
    # 'UCB': {'all_data': [], 'all_rewards': [], 'all_matrices': []},
    'SlidingWindowUCB': {'all_data': [], 'all_rewards': [], 'all_matrices': []},
    'CUSUMUCB': {'all_data': [], 'all_rewards': [], 'all_matrices': []},
    'EXP3S': {'all_data': [], 'all_rewards': [], 'all_matrices': []}
}

# Run experiments for all models
for i, curr_seed in enumerate(seeds):
    print(f"Running experiment {i + 1}/{num_experiments} with seed {curr_seed}")

    # Create environment once per seed
    data = create_environment(
        env=environment,
        random_seed=curr_seed,
        categories=["sports", "politics"],
        alpha=0.01,
        a=2,
        b=5,
        n_documents=n_docs,
        n_users=1,
        observations=observations
    )

    # Define all models
    models = {
        'NonStationaryEpsilonGreedy': NonStationaryEpsilonGreedy(n_arms=n_docs, epsilon=epsilon, alpha=0.001, random_seed=curr_seed),
        # 'UCB': UCB(n_arms=n_docs, random_seed=curr_seed),
        'SlidingWindowUCB': SlidingWindowUCB(n_arms=n_docs, window=50, random_seed=curr_seed),
        'CUSUMUCB': CUSUMUCB(n_arms = n_docs, epsilon=0.2, h=10.0,u_init=0.5, alpha = 0.2, random_seed=curr_seed),
        'EXP3S': EXP3S(n_arms=n_docs, gamma=0.07, alpha=0.1, random_seed=curr_seed)
    }

    # Train each model on the same data
    for model_name, model in models.items():
        table, rewards, matrix = model.train(data)

        model_results[model_name]['all_data'].append(data)
        model_results[model_name]['all_rewards'].append(rewards)
        model_results[model_name]['all_matrices'].append(matrix)

# %%
# Save results for each model
results_dir = os.path.join(project_root, "Results")
os.makedirs(results_dir, exist_ok=True)

for model_name in model_results.keys():
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save data
    outfile_path = os.path.join(model_dir, "all_data.csv")
    with open(outfile_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(model_results[model_name]['all_data'])

    outfile_path = os.path.join(model_dir, "all_rewards.csv")
    with open(outfile_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(model_results[model_name]['all_rewards'])

    outfile_path = os.path.join(model_dir, "all_matrices.csv")
    with open(outfile_path, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(model_results[model_name]['all_matrices'])

#%%

