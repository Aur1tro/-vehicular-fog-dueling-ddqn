"""
trainer.py — Generic training loop for any DQN-family agent.

The Trainer class orchestrates:
    1. Episode rollouts  (env.reset → env.step loop)
    2. ε-greedy exploration with exponential decay
    3. Experience replay  (store transitions, sample mini-batches)
    4. Agent gradient updates
    5. Target network synchronisation (hard or soft)
    6. Logging of per-episode metrics (reward, AoI)

It is agent-agnostic — any object with the interface:
    select_action(state, epsilon) → int
    train_step(s, a, r, s', d)   → float
    update_target(soft, tau)
can be plugged in.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.vehicular_env import VehicularFogEnv
from agents.replay_buffer import ReplayBuffer
from utils.config import Config


class Trainer:
    """Runs the training loop and collects metrics.

    Parameters
    ----------
    agent : object
        A DQN-family agent (DQN, DDQN, Dueling DQN, Dueling DDQN).
    env : VehicularFogEnv
        The vehicular fog network environment.
    config : Config
        Hyper-parameters & simulation settings.
    """

    def __init__(
        self,
        agent: Any,
        env: VehicularFogEnv,
        config: Optional[Config] = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.cfg = config or Config()

        self.buffer = ReplayBuffer(
            capacity=self.cfg.replay_buffer_size,
            seed=self.cfg.seed,
        )

        # Exploration schedule
        self.epsilon = self.cfg.epsilon_start

    # ==================================================================
    #  Main training loop
    # ==================================================================

    def train(self, num_episodes: Optional[int] = None, verbose: bool = True) -> Dict[str, List[float]]:
        """Train the agent for a number of episodes.

        Returns
        -------
        history : dict
            Keys: 'episode_rewards', 'episode_avg_aoi', 'episode_losses',
                  'epsilons'
        """
        num_episodes = num_episodes or self.cfg.num_episodes

        history: Dict[str, List[float]] = {
            "episode_rewards": [],
            "episode_avg_aoi": [],
            "episode_losses": [],
            "epsilons": [],
        }

        for ep in range(1, num_episodes + 1):
            state = self.env.reset()
            ep_reward = 0.0
            ep_loss = 0.0
            ep_aoi_sum = 0.0
            loss_count = 0
            step_count = 0

            done = False
            while not done:
                # ── 1. Select action ─────────────────────────────────────
                action = self.agent.select_action(state, self.epsilon)

                # ── 2. Environment step ──────────────────────────────────
                next_state, reward, done, info = self.env.step(action)

                # ── 3. Store transition ──────────────────────────────────
                self.buffer.push(state, action, reward, next_state, done)

                # ── 4. Learn from replay buffer ──────────────────────────
                if len(self.buffer) >= self.cfg.batch_size:
                    s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
                    loss = self.agent.train_step(s, a, r, s2, d)
                    ep_loss += loss
                    loss_count += 1

                    # Soft target update every step (if enabled)
                    if self.cfg.use_soft_update:
                        self.agent.update_target(soft=True, tau=self.cfg.tau)

                # Accumulate metrics
                ep_reward += reward
                ep_aoi_sum += info.get("avg_aoi", 0.0)
                step_count += 1

                state = next_state

            # ── End of episode bookkeeping ───────────────────────────────

            # Hard target update (if not using soft updates)
            if not self.cfg.use_soft_update and ep % self.cfg.target_update_freq == 0:
                self.agent.update_target(soft=False)

            # Decay epsilon
            self.epsilon = max(
                self.cfg.epsilon_end,
                self.epsilon * self.cfg.epsilon_decay,
            )

            # Record metrics
            avg_loss = ep_loss / max(loss_count, 1)
            avg_aoi = ep_aoi_sum / max(step_count, 1)

            history["episode_rewards"].append(ep_reward)
            history["episode_avg_aoi"].append(avg_aoi)
            history["episode_losses"].append(avg_loss)
            history["epsilons"].append(self.epsilon)

            if verbose and ep % 50 == 0:
                print(
                    f"[{self.agent.name}] Episode {ep:4d}/{num_episodes} | "
                    f"Reward: {ep_reward:8.2f} | "
                    f"Avg AoI: {avg_aoi:6.3f} | "
                    f"Loss: {avg_loss:8.5f} | "
                    f"ε: {self.epsilon:.4f}"
                )

        return history

    # ==================================================================
    #  Evaluate (no exploration, no training)
    # ==================================================================

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Run the trained agent greedily and report average metrics.

        Returns
        -------
        results : dict
            'mean_reward', 'mean_aoi'
        """
        rewards = []
        aois = []

        for _ in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0.0
            ep_aoi = 0.0
            steps = 0
            done = False

            while not done:
                action = self.agent.select_action(state, epsilon=0.0)
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_aoi += info.get("avg_aoi", 0.0)
                steps += 1

            rewards.append(ep_reward)
            aois.append(ep_aoi / max(steps, 1))

        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_aoi": float(np.mean(aois)),
        }

