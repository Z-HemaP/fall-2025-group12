
#%%
import os
import sys
project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.component.model import NonStationaryEpsilonGreedy , UCB, SlidingWindowUCB, CUMSUMUCB, EXP3S
from src.component.env import *
from src.component.plot import *
import warnings
import csv

warnings.filterwarnings("ignore")
num_experiments = 10
np.random.seed(42)          
seeds = np.random.randint(0,100,size = num_experiments) 
environment = "drifting"
n_docs = 2  # number of documents (arms)

# random_seed = 100
epsilon = 0.05
observations = 10000



arm_means = None

#%%
all_data = []
all_rewards = []
all_matrices = []

for i, curr_seed in enumerate(seeds):
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
    # model = NonStationaryEpsilonGreedy(n_arms=n_docs, epsilon=epsilon, alpha=0.001, random_seed=curr_seed)
    # model = UCB(n_arms=n_docs, random_seed=curr_seed)
    # model = SlidingWindowUCB(n_arms=n_docs, window=50, random_seed=curr_seed)
    # model = CUSUMUCB(n_arms=n_docs, h=10,epsilon=0.2, M=5, random_seed=curr_seed)
    model = EXP3S(n_arms=n_docs, gamma=0.07, alpha=0.1, random_seed=curr_seed)

    table, rewards, matrix = model.train(data)
    
    all_data.append(data)
    all_rewards.append(rewards)
    all_matrices.append(matrix)


#%%
results_dir = os.path.join(project_root, "Results")
os.makedirs(results_dir, exist_ok=True)
#%%
# Use absolute path for file writing
outfile_path = os.path.join(results_dir, "all_data.csv")
with open(outfile_path, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(all_data)

outfile_path = os.path.join(results_dir, "all_rewards.csv")
with open(outfile_path, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(all_rewards)

outfile_path = os.path.join(results_dir, "all_matrices.csv")
with open(outfile_path, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(all_matrices)

#%%
combined_data = np.vstack(all_data)

violinplot_environment(combined_data, arm_means=None)

#%%

for i in range(num_experiments):
    model_average_plot(all_data[i], all_rewards[i], all_matrices[i], arm_means, top=None, alpha=0.2)

# ax.set_xlabel("Time Steps")
# ax.set_ylabel("Average Reward / Selection Rate")
# ax.set_title("Model Performance Over 50 Seeds")
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()    


#%%

fig, ax = plt.subplots(figsize=(12, 8))
for i in range(num_experiments):
    show_legend = (i == 0)
    data_average_plot(all_data[i], arm_means, ax=ax,legend=show_legend)

# ax.set_xlabel("Time Steps")
# ax.set_ylabel("Average Reward / Selection Rate")
# ax.set_title("Model Performance Over 50 Seeds")
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle("Data Average Plot", fontsize=16)
fig.tight_layout()
plt.show()    
    
#%%

import numpy as np
all_data_array = np.array(all_data)  # shape: (num_experiments, num_steps, num_arms)

fig, ax = plt.subplots(figsize=(12, 8))
data_average_plot_with_ci(all_data_array, arm_means=arm_means, ax=ax)
fig.suptitle("Data Average Plot with Confidence Intervals", fontsize=16)
plt.tight_layout()
outfile_path = os.path.join(results_dir, "Data Average Plot.jpg")

plt.savefig(outfile_path, dpi=300, bbox_inches='tight')
plt.show()


#%%
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(num_experiments):
    show_legend = (i == 0)
    data_cumulative_plot(all_data[i], arm_means, ax=ax, legend=show_legend)

# ax.set_xlabel("Time Steps")
# ax.set_ylabel("Average Reward / Selection Rate")
# ax.set_title("Model Performance Over 50 Seeds")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.suptitle("Data Cumulative Plot", fontsize=16)
fig.tight_layout()
plt.show()  
# %%
plt.figure()
for i, (data, rewards, matrix) in enumerate(zip(all_data, all_rewards, all_matrices)):
    show_legend = (i == 0)
    model_cumulative_plot(data, rewards, matrix, arm_means, legend = show_legend)
plt.title("Model Cumulative Plot")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# %%
