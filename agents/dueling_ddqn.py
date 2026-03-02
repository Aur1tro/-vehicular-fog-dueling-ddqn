"""
dueling_ddqn.py — Dueling Double DQN agent (the paper's proposed method).

This agent combines the **best of both worlds**:

1. **Dueling architecture** (Wang et al., 2016)
   — Separate Value V(s) and Advantage A(s,a) streams
   — Q(s,a) = V(s) + [A(s,a) − mean(A(s,·))]
   — Better generalisation when many actions have similar values

2. **Double Q-learning** (van Hasselt et al., 2016)
   — Online network *selects* the best next action
   — Target network *evaluates* Q at that action
   — Reduces over-estimation bias of standard DQN

Combined target:
    a*     = argmax_a' Q_online(s', a')
    target = r + γ · Q_target(s', a*)

Where Q_online and Q_target both use the dueling architecture internally.

Why Dueling DDQN outperforms the other three variants for AoI optimisation
---------------------------------------------------------------------------
• In vehicular fog networks many state–action pairs have similar Q-values
  (e.g., when most vehicles are out of range).  The dueling decomposition
  lets the agent learn V(s) — the inherent "goodness" of the traffic/fog
  state — separately from the relative advantage of each allocation action.
  This yields faster credit assignment.

• Double Q-learning prevents the agent from over-estimating the benefit of
  aggressive λ allocations, producing more conservative and reliable
  offloading policies that keep AoI below the threshold consistently.

Reference
---------
"Optimizing AoI in Mobility-Based Vehicular Fog Networks:
 A Dueling-DDQN Approach"
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
from agents.dueling_dqn import DuelingQNetwork  # reuse the dueling architecture


class DuelingDDQNAgent:
    """Dueling Double DQN — the proposed approach in the paper."""

    name = "Dueling DDQN"

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

        # Both networks use the Dueling architecture
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

    # ── Training step — DOUBLE target + DUELING architecture ─────────────

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """Gradient update using the Dueling Double DQN target.

        Combines:
            • Dueling Q-network (implicit in self.online_net / target_net)
            • Double Q-learning target selection/evaluation split
        """
        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a) — produced by dueling decomposition
        q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # ── DOUBLE: online selects, target evaluates ──
            best_actions = self.online_net(s2).argmax(dim=1, keepdim=True)
            q_next = self.target_net(s2).gather(1, best_actions).squeeze(1)
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

