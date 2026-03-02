"""
dqn.py — Vanilla Deep Q-Network (DQN) agent.

Architecture
------------
A simple multi-layer perceptron that maps state → Q(s, a) for all actions:

    Input(state_dim) → FC(hidden) → ReLU → FC(hidden) → ReLU → FC(action_dim)

Training rule (standard DQN)
----------------------------
    target = r + γ · max_a' Q_target(s', a')          if not done
           = r                                          if done

    loss   = MSE( Q_online(s, a),  target )

Key components:
    • Online network   — updated every gradient step
    • Target network   — periodically copied from online (hard) or
                         Polyak-averaged (soft) to stabilise training
    • ε-greedy policy  — balances exploration vs exploitation

Reference
---------
Mnih et al., "Human-level control through deep reinforcement learning",
Nature 2015.
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


# ═══════════════════════════════════════════════════════════════════════
#  Q-Network
# ═══════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """Fully-connected Q-network.

    Maps observation s → Q(s, ·) ∈ ℝ^|A|.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state tensor → Q-values for all actions."""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
#  DQN Agent
# ═══════════════════════════════════════════════════════════════════════

class DQNAgent:
    """Vanilla DQN with experience replay and target network."""

    name = "DQN"

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

        # Online and target networks
        self.online_net = QNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        # Optimiser
        self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """ε-greedy action selection.

        With probability ε choose a random action (exploration);
        otherwise pick argmax_a Q_online(s, a) (exploitation).
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Training step ────────────────────────────────────────────────────

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Perform one gradient update on a mini-batch.

        Parameters
        ----------
        states, actions, rewards, next_states, dones :
            Arrays from the replay buffer, shapes (B, …).

        Returns
        -------
        loss : float
            Scalar loss value for logging.
        """
        # Convert to tensors
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        # ── Current Q(s, a) ──
        q_values = self.online_net(s)                        # (B, |A|)
        q_sa = q_values.gather(1, a.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Target: r + γ · max_a' Q_target(s', a') ──
        with torch.no_grad():
            q_next = self.target_net(s2)                     # (B, |A|)
            q_next_max = q_next.max(dim=1)[0]                # (B,)
            target = r + self.gamma * q_next_max * (1.0 - d)

        loss = self.loss_fn(q_sa, target)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        return loss.item()

    # ── Target network update ────────────────────────────────────────────

    def update_target(self, soft: bool = False, tau: float = 0.005) -> None:
        """Synchronise the target network with the online network.

        Parameters
        ----------
        soft : bool
            If True, apply Polyak averaging: θ_t ← τ·θ + (1−τ)·θ_t
            If False, hard copy.
        tau : float
            Polyak coefficient (only used when soft=True).
        """
        if soft:
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())

