"""Plotting utilities for experiments."""

import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(results, save_path="ibrl_comparison.png"):
    """
    Plot comparison of all agents across environments.
    
    Args:
        results: Results dictionary from compare_all
        save_path: Path to save figure
    """
    # 5 environments (rewards) + 5 convergence plots + 1 credal width = 11 subplots
    # Layout: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    agent_types = ["classical", "bayesian", "ib"]
    colors = {"classical": "blue", "bayesian": "green", "ib": "red"}
    
    env_types = ["bandit", "newcomb", "twin_pd", "misspecified", "wasserstein"]
    env_names = {
        "bandit": "Bandit",
        "newcomb": "Newcomb",
        "twin_pd": "Twin PD",
        "misspecified": "Misspecified",
        "wasserstein": "Wasserstein"
    }
    
    # Row 1: Reward plots for all 5 environments
    reward_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)]
    
    for idx, env_type in enumerate(env_types):
        row, col = reward_positions[idx]
        ax = axes[row, col]
        
        for agent_type in agent_types:
            all_rewards = [r["rewards"] for r in results[env_type][agent_type]]
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            
            # Moving average
            window = 50
            if len(mean_rewards) >= window:
                smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
                x = np.arange(len(smoothed))
                
                ax.plot(x, smoothed, label=agent_type.capitalize(), 
                       color=colors[agent_type], linewidth=2)
                ax.fill_between(x, smoothed - std_rewards[:len(smoothed)], 
                               smoothed + std_rewards[:len(smoothed)], 
                               alpha=0.2, color=colors[agent_type])
        
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Reward" if env_type == "bandit" else "Reward ($)", fontsize=9)
        ax.set_title(f"{env_names[env_type]} Environment", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Row 2-3: Convergence plots for all 5 environments
    convergence_positions = [(1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]
    
    for idx, env_type in enumerate(env_types):
        row, col = convergence_positions[idx]
        ax = axes[row, col]
        
        if env_type == "bandit":
            # For bandit, show cumulative reward
            for agent_type in agent_types:
                all_rewards = [r["rewards"] for r in results[env_type][agent_type]]
                mean_rewards = np.mean(all_rewards, axis=0)
                cumulative = np.cumsum(mean_rewards)
                
                ax.plot(cumulative, label=agent_type.capitalize(), 
                       color=colors[agent_type], linewidth=2)
            
            ax.set_ylabel("Cumulative Reward", fontsize=9)
            ax.set_title("Cumulative Performance", fontsize=10)
        else:
            # For policy-dependent envs, show one-boxing rate
            for agent_type in agent_types:
                all_actions = [r["actions"] for r in results[env_type][agent_type]]
                mean_actions = np.mean(all_actions, axis=0)
                
                window = 50
                if len(mean_actions) >= window:
                    smoothed = np.convolve(mean_actions, np.ones(window)/window, mode='valid')
                    x = np.arange(len(smoothed))
                    
                    # Plot one-boxing rate (1 - action, since 0=one-box, 1=two-box)
                    ax.plot(x, 1 - smoothed, label=agent_type.capitalize(), 
                           color=colors[agent_type], linewidth=2)
            
            ax.set_ylabel("One-Boxing Rate", fontsize=9)
            ax.set_title(f"Policy Convergence ({env_names[env_type]})", fontsize=10)
            ax.set_ylim([-0.05, 1.05])
        
        ax.set_xlabel("Episode", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Credal interval convergence (bottom right)
    ax = axes[2, 2]
    
    # Use Newcomb IB results for credal width
    ib_results = results["newcomb"]["ib"]
    all_widths = [r["credal_widths"] for r in ib_results if r["credal_widths"] is not None]
    
    if all_widths:
        mean_widths = np.mean(all_widths, axis=0)
        std_widths = np.std(all_widths, axis=0)
        x = np.arange(len(mean_widths))
        
        ax.plot(x, mean_widths, color="red", linewidth=2, label="IB Agent")
        ax.fill_between(x, mean_widths - std_widths, mean_widths + std_widths, 
                        alpha=0.3, color="red")
    
    ax.set_xlabel("Episode", fontsize=9)
    ax.set_ylabel("Interval Width", fontsize=9)
    ax.set_title("Credal Interval Convergence (IB Agent)", fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Hide unused subplot
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()
