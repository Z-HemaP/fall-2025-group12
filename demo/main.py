
#%%
import os
import sys
from pathlib import Path
import numpy as np

project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from demo.model import NonStationaryEpsilonGreedy , UCB, SlidingWindowUCB, CUSUMUCB, EXP3S
from demo.env import *
from demo.plot_ci import *
from demo.runner_lib import runner_lib
import warnings
import csv

warnings.filterwarnings("ignore")
num_experiments = 50
np.random.seed(42)          
seeds = np.random.choice(np.arange(100, dtype=int),
                         size=num_experiments,
                         replace=False)
environment = "drifting"
n_docs = 2  # number of documents (arms)

# random_seed = 100
epsilon = 0.05
observations = 10000



arm_means = None

#%%

'''
run models and save results
'''

runner = runner_lib(
    project_root=project_root,
    environment="drifting",
    n_docs=2,
    num_experiments=num_experiments,
    observations=10000,
    epsilon=0.05
)
runner.run_all_models()



#%%

'''plot results'''




