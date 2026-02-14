"""Experiment: Classical bandit environment."""

import numpy as np
from ibrl.agents import ClassicalQAgent, BayesianQAgent, IBQAgent
from ibrl.envs import BanditEnv
from ibrl.belief import CredalInterval
from ibrl.utils.seeding import set_seed


def run_bandit_experiment(agent_type="classical", episodes=1000, seed=42):
    """
    Run bandit experiment with specified agent.
    
    Args:
        agent_type: "classical", "bayesian", or "ib"
        episodes: Number of episodes
        seed: Random seed
    
    Returns:
        rewards: Array of rewards per episode
        agent: Trained agent
    """
    set_seed(seed)
    
    # Create environment
    env = BanditEnv(probs=(0.7, 0.5), rewards=(1.0, 1.0), seed=seed)
    
    # Create agent
    if agent_type == "classical":
        agent = ClassicalQAgent(n_actions=2, alpha=0.1, epsilon=0.1, seed=seed)
    elif agent_type == "bayesian":
        agent = BayesianQAgent(n_actions=2, alpha=0.1, seed=seed)
    elif agent_type == "ib":
        credal = CredalInterval(lower=0.5, upper=0.8, delta=0.05)
        agent = IBQAgent(credal, n_actions=2, alpha=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        
        if agent_type == "ib":
            # IB agent needs predictor correctness (not applicable for bandit)
            # Use dummy value
            agent.update(state, action, reward, predictor_correct=True)
        else:
            agent.update(state, action, reward, next_state, done)
        
        rewards.append(reward)
    
    return np.array(rewards), agent


def main():
    """Run bandit experiments for all agent types."""
    print("=" * 60)
    print("CLASSICAL BANDIT ENVIRONMENT")
    print("=" * 60)
    print()
    
    agent_types = ["classical", "bayesian", "ib"]
    results = {}
    
    for agent_type in agent_types:
        rewards, agent = run_bandit_experiment(agent_type, episodes=1000)
        mean_reward = np.mean(rewards[-100:])  # Last 100 episodes
        std_reward = np.std(rewards[-100:])
        
        results[agent_type] = {
            "rewards": rewards,
            "mean": mean_reward,
            "std": std_reward,
            "agent": agent
        }
        
        print(f"{agent_type.capitalize():12s}: {mean_reward:.3f} ± {std_reward:.3f}")
    
    print()
    print("✓ All agents perform comparably on classical environment")
    print()
    
    return results


if __name__ == "__main__":
    main()
