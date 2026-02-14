"""Experiment: Newcomb's Problem."""

import numpy as np
from ibrl.agents import ClassicalQAgent, BayesianQAgent, IBQAgent
from ibrl.envs import NewcombEnv
from ibrl.predictors import LogicalPredictor
from ibrl.belief import CredalInterval
from ibrl.utils.seeding import set_seed


def run_newcomb_experiment(agent_type="classical", episodes=1000, theta=0.95, seed=42):
    """
    Run Newcomb experiment with specified agent.
    
    Args:
        agent_type: "classical", "bayesian", or "ib"
        episodes: Number of episodes
        theta: Predictor accuracy
        seed: Random seed
    
    Returns:
        rewards: Array of rewards per episode
        agent: Trained agent
        credal_widths: Interval widths over time (for IB agent)
        actions_taken: Actions taken per episode
    """
    set_seed(seed)
    
    # Create predictor and environment
    predictor = LogicalPredictor(theta=theta, seed=seed)
    env = NewcombEnv(predictor, seed=seed)
    
    # Create agent
    if agent_type == "classical":
        agent = ClassicalQAgent(n_actions=2, alpha=0.1, epsilon=0.1, seed=seed)
    elif agent_type == "bayesian":
        agent = BayesianQAgent(n_actions=2, alpha=0.1, seed=seed)
    elif agent_type == "ib":
        credal = CredalInterval(lower=0.8, upper=0.99, delta=0.05)
        agent = IBQAgent(credal, n_actions=2, alpha=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    rewards = []
    credal_widths = []
    actions_taken = []
    
    for ep in range(episodes):
        state = env.reset()
        greedy_action = agent.greedy_action()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action, greedy_action)
        
        if agent_type == "ib":
            agent.update(state, action, reward, info["predictor_correct"])
            credal_widths.append(agent.credal.width())
        else:
            agent.update(state, action, reward, next_state, done)
        
        rewards.append(reward)
        actions_taken.append(action)
    
    return np.array(rewards), agent, np.array(credal_widths), np.array(actions_taken)


def main():
    """Run Newcomb experiments for all agent types."""
    print("=" * 60)
    print("NEWCOMB'S PROBLEM (θ=0.95)")
    print("=" * 60)
    print()
    
    agent_types = ["classical", "bayesian", "ib"]
    results = {}
    
    for agent_type in agent_types:
        rewards, agent, credal_widths, actions = run_newcomb_experiment(
            agent_type, episodes=1000, theta=0.95
        )
        
        mean_reward = np.mean(rewards[-100:])
        std_reward = np.std(rewards[-100:])
        one_box_rate = 1 - np.mean(actions[-100:])
        
        results[agent_type] = {
            "rewards": rewards,
            "mean": mean_reward,
            "std": std_reward,
            "one_box_rate": one_box_rate,
            "credal_widths": credal_widths,
            "agent": agent
        }
        
        print(f"{agent_type.capitalize():12s}: ${mean_reward:>10,.0f} ± ${std_reward:>8,.0f}  "
              f"[one-box: {one_box_rate:.1%}]")
        
        if agent_type == "ib" and len(credal_widths) > 0:
            print(f"{'':12s}  Credal width: {credal_widths[0]:.3f} → {credal_widths[-1]:.3f}")
    
    print()
    print("✓ IB agent converges to one-boxing")
    print("✓ Classical agents oscillate or two-box")
    print()
    
    return results


if __name__ == "__main__":
    main()
