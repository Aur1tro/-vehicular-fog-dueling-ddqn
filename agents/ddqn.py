"""
ddqn.py — Double Deep Q-Network (Double DQN) agent.

The key insight of Double DQN
------------------------------
Standard DQN uses the *same* network to both **select** and **evaluate**
the next-state action, which causes systematic over-estimation of Q-values:

    target_DQN = r + γ · max_a' Q_target(s', a')
                              ↑ same network picks & evaluates

Double DQN decouples these two steps:

    a* = argmax_a' Q_online(s', a')      ← online net *selects*
    target = r + γ · Q_target(s', a*)     ← target net *evaluates*

This simple change significantly reduces over-estimation bias and leads
to more stable training and better final policies.

Reference
---------
van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning",
AAAI 2016.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.config import Config
from agents.dqn import QNetwork  # reuse the same FC architecture


class DDQNAgent:
    """Double DQN — decouples action selection from evaluation."""

    name = "Double DQN"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[Config] = None,
    ) -> None:
        self.cfg = config or Config()
        self.device = self.cfg.device
        self.action_dim = action_dim
        self.gamma = self.cfg.gamma

        # Online and target networks (same architecture as vanilla DQN)
        self.online_net = QNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)
        self.loss_fn = nn.MSELoss()

    # ── Action selection (identical to DQN) ──────────────────────────────

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Training step — Double DQN target ────────────────────────────────

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """One gradient update using the Double-DQN target.

        The critical difference from vanilla DQN is here:
            a*     = argmax_a' Q_online(s', a')   ← selection
            target = r + γ · Q_target(s', a*)      ← evaluation
        """
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a)
        q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # ── DOUBLE DQN: online net selects, target net evaluates ──
            #   Step 1: online network picks the best action for s'
            best_actions = self.online_net(s2).argmax(dim=1, keepdim=True)  # (B,1)
            #   Step 2: target network evaluates Q at that action
            q_next = self.target_net(s2).gather(1, best_actions).squeeze(1)  # (B,)
            target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_sa, target)

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        return loss.item()

    # ── Target update ────────────────────────────────────────────────────

    def update_target(self, soft: bool = False, tau: float = 0.005) -> None:
        if soft:
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())

