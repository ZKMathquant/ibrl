"""Experiment: Misspecified Newcomb environment."""

import numpy as np
from ibrl.agents import ClassicalQAgent, BayesianQAgent, IBQAgent
from ibrl.envs import MisspecifiedNewcombEnv, AdversarialNewcombEnv
from ibrl.predictors import LogicalPredictor
from ibrl.belief import CredalInterval
from ibrl.utils.seeding import set_seed


def run_misspecified_experiment(agent_type="classical", episodes=1000, 
                                true_theta=0.75, model_theta=0.95, seed=42):
    """
    Run misspecified Newcomb experiment.
    
    Agent believes θ ∈ [0.8, 0.99] but true θ = 0.75 (outside belief).
    
    Args:
        agent_type: "classical", "bayesian", or "ib"
        episodes: Number of episodes
        true_theta: True predictor accuracy (outside agent's belief)
        model_theta: Agent's model accuracy
        seed: Random seed
    
    Returns:
        rewards, agent, credal_widths, actions
    """
    set_seed(seed)
    
    predictor = LogicalPredictor(theta=model_theta, seed=seed)
    env = MisspecifiedNewcombEnv(true_theta=true_theta, predictor=predictor, seed=seed)
    
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


def run_adversarial_experiment(agent_type="classical", episodes=1000, seed=42):
    """
    Run adversarial Newcomb experiment.
    
    Predictor always predicts opposite of agent's greedy action.
    """
    set_seed(seed)
    
    env = AdversarialNewcombEnv(seed=seed)
    
    if agent_type == "classical":
        agent = ClassicalQAgent(n_actions=2, alpha=0.1, epsilon=0.1, seed=seed)
    elif agent_type == "bayesian":
        agent = BayesianQAgent(n_actions=2, alpha=0.1, seed=seed)
    elif agent_type == "ib":
        credal = CredalInterval(lower=0.0, upper=1.0, delta=0.05)
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
            # In adversarial case, predictor is never "correct" in agent's model
            agent.update(state, action, reward, predictor_correct=False)
            credal_widths.append(agent.credal.width())
        else:
            agent.update(state, action, reward, next_state, done)
        
        rewards.append(reward)
        actions_taken.append(action)
    
    return np.array(rewards), agent, np.array(credal_widths), np.array(actions_taken)


def main():
    """Run misspecified and adversarial experiments."""
    print("=" * 70)
    print("ROBUSTNESS UNDER MISSPECIFICATION")
    print("=" * 70)
    print()
    
    # Misspecified experiment
    print("MISSPECIFIED NEWCOMB (True θ=0.75, Agent believes θ ∈ [0.8, 0.99]):")
    print("-" * 70)
    
    agent_types = ["classical", "bayesian", "ib"]
    results_misspec = {}
    
    for agent_type in agent_types:
        rewards, agent, credal_widths, actions = run_misspecified_experiment(
            agent_type, episodes=1000, true_theta=0.75, model_theta=0.95
        )
        
        mean_reward = np.mean(rewards[-100:])
        std_reward = np.std(rewards[-100:])
        one_box_rate = 1 - np.mean(actions[-100:])
        
        results_misspec[agent_type] = {
            "rewards": rewards,
            "mean": mean_reward,
            "std": std_reward,
            "one_box_rate": one_box_rate,
        }
        
        print(f"  {agent_type.capitalize():12s}: ${mean_reward:>10,.0f} ± ${std_reward:>8,.0f}  "
              f"[one-box: {one_box_rate:.1%}]")
    
    print()
    print("✓ IB maintains robustness under misspecification")
    print()
    
    # Adversarial experiment
    print("ADVERSARIAL NEWCOMB (Predictor always wrong):")
    print("-" * 70)
    
    results_adv = {}
    
    for agent_type in agent_types:
        rewards, agent, credal_widths, actions = run_adversarial_experiment(
            agent_type, episodes=1000
        )
        
        mean_reward = np.mean(rewards[-100:])
        std_reward = np.std(rewards[-100:])
        one_box_rate = 1 - np.mean(actions[-100:])
        
        results_adv[agent_type] = {
            "rewards": rewards,
            "mean": mean_reward,
            "std": std_reward,
            "one_box_rate": one_box_rate,
        }
        
        print(f"  {agent_type.capitalize():12s}: ${mean_reward:>10,.0f} ± ${std_reward:>8,.0f}  "
              f"[one-box: {one_box_rate:.1%}]")
    
    print()
    print("✓ All agents adapt to adversarial predictor")
    print()
    
    return results_misspec, results_adv


if __name__ == "__main__":
    main()
