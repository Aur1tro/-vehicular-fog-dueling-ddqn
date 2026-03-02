"""
dueling_dqn.py — Dueling DQN agent.

Dueling Architecture
---------------------
Instead of a single stream that outputs Q(s, a), the network splits into
two parallel streams after shared feature layers:

    Shared features ──┬── Value stream   → V(s)        (scalar)
                      └── Advantage stream → A(s, a)   (one per action)

These are combined to produce Q-values:

    Q(s, a) = V(s) + [ A(s, a) − mean_a'( A(s, a') ) ]

Why subtract the mean of A?
    • It makes V(s) identifiable — without the subtraction, V and A could
      absorb arbitrary constants.
    • The mean-centering forces A to represent *relative* advantages of
      each action, while V captures the *absolute* state value.

Why is this better?
    • Many states don't require distinguishing between actions (e.g., when
      all vehicles are out of range).  The value stream can learn this
      efficiently without having to learn all Q(s,a) independently.
    • Better generalisation across actions → faster convergence.

This file uses the standard (non-double) DQN target:
    target = r + γ · max_a' Q_target(s', a')

Reference
---------
Wang et al., "Dueling Network Architectures for Deep Reinforcement
Learning", ICML 2016.
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
#  Dueling Q-Network
# ═══════════════════════════════════════════════════════════════════════

class DuelingQNetwork(nn.Module):
    """Dueling architecture: shared backbone → Value + Advantage streams.

    Forward pass:
        features = backbone(s)
        V        = value_head(features)           ∈ ℝ
        A        = advantage_head(features)       ∈ ℝ^|A|
        Q        = V + (A − A.mean())
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: scalar V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a) for each action
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, ·) via the dueling combination formula."""
        features = self.backbone(x)                     # (B, hidden)
        value = self.value_head(features)                # (B, 1)
        advantage = self.advantage_head(features)        # (B, |A|)

        # Q(s,a) = V(s) + [ A(s,a) − mean_a( A(s,·) ) ]
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ═══════════════════════════════════════════════════════════════════════
#  Dueling DQN Agent
# ═══════════════════════════════════════════════════════════════════════

class DuelingDQNAgent:
    """Dueling DQN with standard (non-double) target."""

    name = "Dueling DQN"

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

        self.online_net = DuelingQNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, self.cfg.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)
        self.loss_fn = nn.MSELoss()

    # ── Action selection ─────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Training step — standard DQN target ──────────────────────────────

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Standard DQN target (not double)
            q_next_max = self.target_net(s2).max(dim=1)[0]
            target = r + self.gamma * q_next_max * (1.0 - d)

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

