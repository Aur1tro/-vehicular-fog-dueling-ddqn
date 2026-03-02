"""
run.py — Quick command-line entry point to train all four agents.

Usage:
    cd "D:\\Python Projects\\ML Project"
    python run.py

This script trains DQN, Double DQN, Dueling DQN, and Dueling DDQN on the
vehicular fog environment and produces comparison plots.  For interactive
exploration use notebooks/experiment.ipynb instead.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config import Config
from environment.vehicular_env import VehicularFogEnv
from agents.dqn import DQNAgent
from agents.ddqn import DDQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.dueling_ddqn import DuelingDDQNAgent
from training.trainer import Trainer
from utils.plotting import plot_rewards, plot_convergence


def main() -> None:
    cfg = Config()
    cfg.seed_everything()

    print(f"Device: {cfg.device}")
    print(f"Vehicles: {cfg.num_vehicles}, Episodes: {cfg.num_episodes}")
    print(f"Action dim: {cfg.num_vehicles * cfg.num_lambda_levels}")
    print("=" * 60)

    # Create environment (shared config)
    env = VehicularFogEnv(cfg)
    state_dim = env.state_dim
    action_dim = env.action_dim

    # Instantiate all four agents
    agents = [
        DQNAgent(state_dim, action_dim, cfg),
        DDQNAgent(state_dim, action_dim, cfg),
        DuelingDQNAgent(state_dim, action_dim, cfg),
        DuelingDDQNAgent(state_dim, action_dim, cfg),
    ]

    histories = {}

    for agent in agents:
        print(f"\n{'='*60}")
        print(f"Training: {agent.name}")
        print(f"{'='*60}")

        # Each agent gets its own fresh environment & trainer
        cfg.seed_everything()  # reset seeds for fair comparison
        env = VehicularFogEnv(cfg)
        trainer = Trainer(agent, env, cfg)
        history = trainer.train(verbose=True)
        histories[agent.name] = history

        # Quick evaluation
        eval_results = trainer.evaluate(num_episodes=20)
        print(
            f"  → Eval: mean_reward={eval_results['mean_reward']:.3f}, "
            f"mean_aoi={eval_results['mean_aoi']:.3f}"
        )

    # ── Plots ────────────────────────────────────────────────────────────
    print("\nGenerating plots…")
    plot_rewards(histories, save_path="reward_comparison.png")
    plot_convergence(histories, metric="episode_avg_aoi", save_path="convergence_aoi.png")
    print("Done! Plots saved to reward_comparison.png and convergence_aoi.png")


if __name__ == "__main__":
    main()

