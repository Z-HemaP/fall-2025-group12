import os
import numpy as np
from src.component.model import NonStationaryEpsilonGreedy, UCB, SlidingWindowUCB, CUSUMUCB, EXP3S
from src.component.env import create_environment

class runner_lib:
    def __init__(self, project_root, environment, n_docs, num_experiments, observations, epsilon=0.05):
        self.project_root = project_root
        self.environment = environment
        self.n_docs = n_docs
        self.num_experiments = num_experiments
        self.observations = observations
        self.epsilon = epsilon
        
        self.results_dir = os.path.join(self.project_root, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        np.random.seed(42)
        self.seeds = np.random.choice(np.arange(100, dtype=int), size=num_experiments, replace=False)
        
        self.models = {
            "NonStationaryEpsilonGreedy": NonStationaryEpsilonGreedy,
            "UCB": UCB,
            "SlidingWindowUCB": SlidingWindowUCB,
            "CUSUMUCB": CUSUMUCB,
            "EXP3S": EXP3S
        }

    def run_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not recognized.")
        
        model_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        ModelClass = self.models[model_name]

        for i, curr_seed in enumerate(self.seeds):
            print(f"Running {model_name}, experiment {i+1}/{self.num_experiments} with seed {curr_seed}...")
            data = create_environment(
                env=self.environment,
                random_seed=curr_seed,
                categories=["sports", "politics"],
                alpha=0.01,
                a=2,
                b=5,
                n_documents=self.n_docs,
                n_users=1,
                observations=self.observations
            )

            # Initialize model with appropriate parameters (adjust as needed)
            if model_name == "NonStationaryEpsilonGreedy":
                model = ModelClass(n_arms=self.n_docs, epsilon=self.epsilon, alpha=0.001, random_seed=curr_seed)
            elif model_name == "UCB":
                model = ModelClass(n_arms=self.n_docs, random_seed=curr_seed)
            elif model_name == "SlidingWindowUCB":
                model = ModelClass(n_arms=self.n_docs, window=50, random_seed=curr_seed)
            elif model_name == "CUSUMUCB":
                model = ModelClass(n_arms=self.n_docs, epsilon=0.2, h=10.0, u_init=0.5, alpha=0.2, random_seed=curr_seed)
            elif model_name == "EXP3S":
                model = ModelClass(n_arms=self.n_docs, gamma=0.07, alpha=0.1, random_seed=curr_seed)
            else:
                raise ValueError(f"Unknown model {model_name}")

            table, rewards, matrix = model.train(data)

            # Save results in model-specific folder
            filename_base = f"{model_name}_seed{curr_seed}"
            np.save(os.path.join(model_dir, f"{filename_base}_data.npy"), data)
            np.save(os.path.join(model_dir, f"{filename_base}_rewards.npy"), rewards)
            np.save(os.path.join(model_dir, f"{filename_base}_matrix.npy"), matrix)
            print(f"Saved results for {filename_base} in folder {model_name}")

    def run_all_models(self):
        for model_name in self.models.keys():
            self.run_model(model_name)
