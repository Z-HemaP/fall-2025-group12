Multi-Armed Bandit Algorithms (Fall 2025 Group 12)
This project implements and analyzes several algorithms for multi-armed bandit problems in nonstationary and adversarial environments. It includes code to generate synthetic environments, run different models, and visualize results such as average rewards and cumulative performance.

Structure


reports/, research_paper/ – Written reports and supporting research documents.

Results/ – Experimental results, including data CSV files and plots:

all_data.csv, all_rewards.csv, all_matrices.csv – Experiment outputs.

Data Average Plot.jpg – Matplotlib plot visualizing model performance.

src/ – Main source code:

component/

env.py – Functions to generate synthetic environments (Gaussian, Bernoulli, drifting).​

model.py – Implementations of bandit algorithms:

Non-Stationary Epsilon-Greedy

UCB and Sliding Window UCB

CUSUM-UCB

EXP3.S (switch-adaptive).​

plot.py – Functions for plotting reward distributions, rolling averages, and cumulative plots.​

main/

main.py – Main experiment runner script. Configures and executes batches of bandit experiments, saves data to CSV, generates plots.​

tests/ – Unit test and utility files.

Usage
Install required dependencies (e.g., numpy, pandas, matplotlib).

Run experiments via main.py in src/main/:

Configure experiment options (number of arms, observations, environment type) within main.py.

Outputs are saved in the Results directory as CSV and image files.

Explore/visualize results with plot functions in plot.py.

Algorithms
NonStationaryEpsilonGreedy – Uses exponential recency-weighted reward updates.

UCB & SlidingWindowUCB – Upper-Confidence Bound variants; SlidingWindow applies recent stats for nonstationarity.

CUSUMUCB – Detects changes via CUSUM, resetting stats when a change occurs.

EXP3S – Adversarial/stationary bandit model with switch-adaptive updates.

Data & Experiments
Synthetic environments are designed in env.py and support customizable arms, reward means (Gaussian/Bernoulli), and drift properties.

Experiments compare average/cumulative performance across models and document how algorithms adapt to changing reward distributions.