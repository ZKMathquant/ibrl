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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    agent_types = ["classical", "bayesian", "ib"]
    colors = {"classical": "blue", "bayesian": "green", "ib": "red"}
    
    # Bandit rewards
    ax = axes[0, 0]
    for agent_type in agent_types:
        all_rewards = [r["rewards"] for r in results["bandit"][agent_type]]
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        
        # Moving average
        window = 50
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed))
        
        ax.plot(x, smoothed, label=agent_type.capitalize(), color=colors[agent_type])
        ax.fill_between(x, smoothed - std_rewards[:len(smoothed)], 
                        smoothed + std_rewards[:len(smoothed)], 
                        alpha=0.2, color=colors[agent_type])
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Bandit Environment")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Newcomb rewards
    ax = axes[0, 1]
    for agent_type in agent_types:
        all_rewards = [r["rewards"] for r in results["newcomb"][agent_type]]
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        
        window = 50
        smoothed = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed))
        
        ax.plot(x, smoothed, label=agent_type.capitalize(), color=colors[agent_type])
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward ($)")
    ax.set_title("Newcomb's Problem")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Action distribution (Newcomb)
    ax = axes[1, 0]
    for agent_type in agent_types:
        all_actions = [r["actions"] for r in results["newcomb"][agent_type]]
        mean_actions = np.mean(all_actions, axis=0)
        
        window = 50
        smoothed = np.convolve(mean_actions, np.ones(window)/window, mode='valid')
        x = np.arange(len(smoothed))
        
        ax.plot(x, 1 - smoothed, label=agent_type.capitalize(), color=colors[agent_type])
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("One-Boxing Rate")
    ax.set_title("Policy Convergence (Newcomb)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # Credal interval width (IB only)
    ax = axes[1, 1]
    ib_results = results["newcomb"]["ib"]
    all_widths = [r["credal_widths"] for r in ib_results if r["credal_widths"] is not None]
    
    if all_widths:
        mean_widths = np.mean(all_widths, axis=0)
        std_widths = np.std(all_widths, axis=0)
        x = np.arange(len(mean_widths))
        
        ax.plot(x, mean_widths, color="red", linewidth=2)
        ax.fill_between(x, mean_widths - std_widths, mean_widths + std_widths, 
                        alpha=0.3, color="red")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Interval Width")
    ax.set_title("Credal Interval Convergence (IB Agent)")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()
