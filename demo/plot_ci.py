import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np

def plot_all_models_average_with_ci(all_model_data, arm_means_dict=None, alpha=0.3):
    """
    Overlaid mean rolling average plots with confidence intervals, one line per arm per model.
    
    Args:
        all_model_data: dict of model name → numpy array (num_experiments, n_steps, n_arms)
        arm_means_dict: dict of model name → list of arrays for true means per environment (optional)
        alpha: transparency for CI shading
    """
    plt.figure(figsize=(14, 8))
    for m, (model_name, all_data) in enumerate(all_model_data.items()):
        if m == 0:
            num_experiments, num_steps, num_arms = all_data.shape
            time_steps = np.arange(1, num_steps + 1)
            arm_means = arm_means_dict.get(model_name) if arm_means_dict else None
            for j in range(num_arms):
                rolling_avgs = np.cumsum(all_data[:, :, j], axis=1) / time_steps
                mean_rolling_avg = np.mean(rolling_avgs, axis=0)
                stderr_rolling_avg = np.std(rolling_avgs, axis=0) / np.sqrt(num_experiments)
                ci_upper = mean_rolling_avg + 1.96 * stderr_rolling_avg
                ci_lower = mean_rolling_avg - 1.96 * stderr_rolling_avg

                color = f"C{j + m*num_arms}"
                plt.plot(time_steps, mean_rolling_avg, color=color, 
                        label=f"Arm {j+1}", linewidth=2)
                plt.fill_between(time_steps, ci_lower, ci_upper, color=color, alpha=alpha)
                
                if arm_means is not None:
                    if len(arm_means) == 1:
                        plt.axhline(y=arm_means[0][j], color=color, linestyle="--", alpha=0.7)
                    else:
                        samples_per_env = num_steps // len(arm_means)
                        for env_idx, means in enumerate(arm_means):
                            start_x = env_idx * samples_per_env
                            end_x = (env_idx + 1) * samples_per_env
                            plt.plot([start_x, end_x], [means[j], means[j]], color=color, linestyle="--", alpha=0.7)

    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward")
    plt.title("All Models: Data Average with Confidence Intervals")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

def plot_all_models_cumulative_with_ci(all_model_data, arm_means_dict=None, alpha=0.3):
    """
    Overlaid mean cumulative reward plots with confidence intervals, one line per arm per model.
    
    Args:
        all_model_data: dict of model name → numpy array (num_experiments, n_steps, n_arms)
        arm_means_dict: dict of model name → list of arrays for true means per environment (optional)
        alpha: transparency for CI shading
    """
    plt.figure(figsize=(14, 8))
    for m, (model_name, all_data) in enumerate(all_model_data.items()):
        if m == 0:
            num_experiments, num_steps, num_arms = all_data.shape
            time_steps = np.arange(1, num_steps + 1)
            arm_means = arm_means_dict.get(model_name) if arm_means_dict else None
            for j in range(num_arms):
                cumsums = np.cumsum(all_data[:, :, j], axis=1)
                mean_cumsum = np.mean(cumsums, axis=0)
                stderr_cumsum = np.std(cumsums, axis=0) / np.sqrt(num_experiments)
                ci_upper = mean_cumsum + 1.96 * stderr_cumsum
                ci_lower = mean_cumsum - 1.96 * stderr_cumsum

                color = f"C{j + m*num_arms}"
                plt.plot(time_steps, mean_cumsum, color=color, 
                        label=f"Arm {j+1}", linewidth=2)
                plt.fill_between(time_steps, ci_lower, ci_upper, color=color, alpha=alpha)
                
                if arm_means is not None:
                    if len(arm_means) == 1:
                        expected_cumsum = arm_means[0][j] * time_steps
                        plt.plot(time_steps, expected_cumsum, color=color, linestyle="--", alpha=0.7)
                    else:
                        samples_per_env = num_steps // len(arm_means)
                        for env_idx, means in enumerate(arm_means):
                            start_idx = env_idx * samples_per_env
                            end_idx = (env_idx + 1) * samples_per_env
                            if env_idx == 0:
                                expected_cumsum = means[j] * np.arange(1, samples_per_env + 1)
                            else:
                                prev_cumsum = means[j] * samples_per_env * env_idx
                                expected_cumsum = prev_cumsum + means[j] * np.arange(1, samples_per_env + 1)
                            plt.plot(np.arange(start_idx, end_idx), expected_cumsum, color=color, linestyle="--", alpha=0.7)

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Reward")
    plt.title("All Models: Cumulative Reward with Confidence Intervals")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()



def model_average_plot_all_models_with_ci(all_model_rewards,
                                          all_model_matrices,
                                          alpha=0.15):
    """
    Single figure: for each model, plot mean rolling average of rewards and
    arm selection frequency with confidence intervals across experiments.
    
    all_model_rewards: dict[model_name] -> (num_experiments, n_steps)
    all_model_matrices: dict[model_name] -> (num_experiments, n_steps, n_arms)
    """
    if not all_model_rewards or not all_model_matrices:
        print("No data to plot.")
        return

    plt.figure(figsize=(14, 8))

    linestyles = ["-", "--", "-.", ":"]
    model_colors = {m: f"C{i}" for i, m in enumerate(all_model_rewards.keys())}

    for model_name, rewards in all_model_rewards.items():
        if model_name not in all_model_matrices:
            continue

        actions = all_model_matrices[model_name]
        num_exp, n_steps = rewards.shape
        _, _, n_arms = actions.shape
        time_steps = np.arange(1, n_steps + 1)
        base_color = model_colors[model_name]

        # Rolling average of actual rewards with CI
        rolling = np.cumsum(rewards, axis=1) / time_steps
        mean_roll = rolling.mean(axis=0)
        stderr_roll = rolling.std(axis=0) / np.sqrt(num_exp)
        ci_up = mean_roll + 1.96 * stderr_roll
        ci_lo = mean_roll - 1.96 * stderr_roll

        plt.plot(
            time_steps,
            mean_roll,
            color=base_color,
            linewidth=2.5,
            label=f"{model_name} - Actual Reward",
        )
        plt.fill_between(time_steps, ci_lo, ci_up, color=base_color, alpha=alpha)

        # # Arm selection frequencies with CI
        # for j in range(n_arms):
        #     cums = np.cumsum(actions[:, :, j], axis=1)
        #     freq = cums / time_steps
        #     mean_freq = freq.mean(axis=0)
        #     stderr_freq = freq.std(axis=0) / np.sqrt(num_exp)
        #     ci_up_f = mean_freq + 1.96 * stderr_freq
        #     ci_lo_f = mean_freq - 1.96 * stderr_freq

        #     ls = linestyles[j % len(linestyles)]
        #     plt.plot(
        #         time_steps,
        #         mean_freq,
        #         color='black',
        #         linestyle=ls,
        #         linewidth=1.5,
        #         label=f"{model_name} - Arm {j+1} Sel. Rate",
        #     )
        #     plt.fill_between(time_steps, ci_lo_f, ci_up_f,
        #                      color=base_color, alpha=alpha * 0.7)

    plt.xlabel("Time Steps")
    plt.ylabel("Average Reward / Selection Rate")
    plt.title("All Models - Average Performance with Confidence Intervals")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


def model_cumulative_plot_all_models_with_ci(all_model_rewards,
                                             all_model_matrices,
                                             alpha=0.15):
    """
    Single figure: for each model, plot mean cumulative rewards and
    cumulative selections per arm with confidence intervals across experiments.
    
    all_model_rewards: dict[model_name] -> (num_experiments, n_steps)
    all_model_matrices: dict[model_name] -> (num_experiments, n_steps, n_arms)
    """
    if not all_model_rewards or not all_model_matrices:
        print("No data to plot.")
        return

    plt.figure(figsize=(14, 8))

    linestyles = ["-", "--", "-.", ":"]
    model_colors = {m: f"C{i}" for i, m in enumerate(all_model_rewards.keys())}

    for model_name, rewards in all_model_rewards.items():
        if model_name not in all_model_matrices:
            continue

        actions = all_model_matrices[model_name]
        num_exp, n_steps = rewards.shape
        _, _, n_arms = actions.shape
        time_steps = np.arange(1, n_steps + 1)
        base_color = model_colors[model_name]

        # Cumulative rewards with CI
        cum_rewards = np.cumsum(rewards, axis=1)
        mean_cum = cum_rewards.mean(axis=0)
        stderr_cum = cum_rewards.std(axis=0) / np.sqrt(num_exp)
        ci_up = mean_cum + 1.96 * stderr_cum
        ci_lo = mean_cum - 1.96 * stderr_cum

        plt.plot(
            time_steps,
            mean_cum,
            color=base_color,
            linewidth=2.5,
            label=f"{model_name} - Cum. Reward",
        )
        plt.fill_between(time_steps, ci_lo, ci_up, color=base_color, alpha=alpha)

        # # Cumulative selections per arm with CI
        # for j in range(n_arms):
        #     cums = np.cumsum(actions[:, :, j], axis=1)
        #     mean_sel = cums.mean(axis=0)
        #     stderr_sel = cums.std(axis=0) / np.sqrt(num_exp)
        #     ci_up_s = mean_sel + 1.96 * stderr_sel
        #     ci_lo_s = mean_sel - 1.96 * stderr_sel

        #     ls = linestyles[j % len(linestyles)]
        #     plt.plot(
        #         time_steps,
        #         mean_sel,
        #         color= 'black',
        #         linestyle=ls,
        #         linewidth=0.5,
        #         label=f"{model_name} - Arm {j+1} Sel.",
        #     )
        #     plt.fill_between(time_steps, ci_lo_s, ci_up_s,
        #                      color=base_color, alpha=0.03,zorder=2,)

    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Rewards / Selections")
    plt.title("All Models - Cumulative Performance with Confidence Intervals")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()
