"""
replay_buffer.py — Experience Replay Buffer for off-policy RL.

Stores transitions (s, a, r, s', done) and supports uniform random sampling.

Why experience replay?
----------------------
• Breaks temporal correlations between consecutive samples, stabilising
  training of neural-network-based Q-functions.
• Enables mini-batch SGD which is more sample-efficient than online
  one-step updates.

Reference
---------
Mnih et al., "Human-level control through deep reinforcement learning",
Nature 2015.
"""

import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """Fixed-size FIFO buffer with uniform sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    seed : int
        Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 50_000, seed: int = 42) -> None:
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.RandomState(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch.

        Returns
        -------
        states      : np.ndarray, shape (B, state_dim)
        actions     : np.ndarray, shape (B,)   — int64
        rewards     : np.ndarray, shape (B,)   — float32
        next_states : np.ndarray, shape (B, state_dim)
        dones       : np.ndarray, shape (B,)   — float32 (0 or 1)
        """
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

