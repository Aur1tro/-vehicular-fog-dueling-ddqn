"""
config.py — Central configuration for the Vehicular Fog Network simulation.

All tunable hyperparameters are defined here in a single dataclass so that
experiments are reproducible and easy to modify.  Every parameter is annotated
with its physical / RL meaning.

Reference
---------
"Optimizing AoI in Mobility-Based Vehicular Fog Networks:
 A Dueling-DDQN Approach"
"""

from dataclasses import dataclass, field
import torch
import numpy as np
import random


@dataclass
class Config:
    # ── Simulation / Network Parameters ──────────────────────────────────

    num_vehicles: int = 10
    """N — number of vehicles (task generators) in the network."""

    num_rsus: int = 1
    """Number of Road-Side Units (fog nodes) serving the vehicles."""

    rsu_positions: list = field(default_factory=lambda: [[500.0, 500.0]])
    """(x, y) coordinates of each RSU on the 2-D plane (metres)."""

    area_width: float = 1000.0
    """Width of the rectangular simulation area (metres)."""

    area_height: float = 1000.0
    """Height of the rectangular simulation area (metres)."""

    coverage_radius: float = 300.0
    """d_max — maximum communication range (metres).
    A vehicle can offload a task only if its Euclidean distance to the RSU
    is ≤ d_max."""

    mu: float = 5.0
    """μ — RSU service rate (tasks / time-slot).
    Upper bound on the aggregate arrival rate the RSU can handle."""

    epsilon_error: float = 0.1
    """ε — packet error probability (0 ≤ ε < 1).
    Models lossy wireless links; appears in the AoI denominator as (1-ε)."""

    aoi_threshold: float = 10.0
    """Δ_threshold — maximum tolerable AoI for any vehicle.
    Exceeding this incurs a penalty in the reward signal."""

    # ── Mobility Model ───────────────────────────────────────────────────

    vehicle_speed: float = 15.0
    """v — constant speed of every vehicle (m / time-slot)."""

    dt: float = 1.0
    """Duration of one simulation time-step (seconds)."""

    # ── Discrete Action Space ────────────────────────────────────────────

    num_lambda_levels: int = 5
    """K — number of discrete arrival-rate levels the agent can assign
    to each vehicle.  The actual λ values are linspace(0, μ, K) so that
    the service constraint λ ≤ μ is always satisfied."""

    # ── Episode / Time Parameters ────────────────────────────────────────

    max_steps_per_episode: int = 100
    """T — number of time-steps in one episode."""

    num_episodes: int = 1000
    """Total training episodes."""

    # ── RL Hyper-Parameters ──────────────────────────────────────────────

    learning_rate: float = 1e-3
    """Adam optimiser learning rate."""

    gamma: float = 0.99
    """Discount factor γ for future rewards."""

    batch_size: int = 64
    """Mini-batch size sampled from the replay buffer."""

    replay_buffer_size: int = 50_000
    """Maximum capacity of the experience-replay buffer."""

    epsilon_start: float = 1.0
    """Initial exploration rate for ε-greedy policy."""

    epsilon_end: float = 0.01
    """Minimum exploration rate after decay."""

    epsilon_decay: float = 0.995
    """Multiplicative decay applied to ε after every episode."""

    target_update_freq: int = 10
    """Number of episodes between hard target-network syncs.
    (Set to 1 and use tau for soft updates if preferred.)"""

    tau: float = 0.005
    """Polyak averaging coefficient for soft target updates.
    θ_target ← τ·θ_online + (1−τ)·θ_target"""

    use_soft_update: bool = False
    """If True, use soft (Polyak) updates every step instead of periodic
    hard copies."""

    # ── Reproducibility ──────────────────────────────────────────────────

    seed: int = 42
    """Global random seed (Python, NumPy, PyTorch, CUDA)."""

    # ── Hidden layer sizes (shared across all agents for fair comparison) ─

    hidden_dim: int = 128
    """Width of every hidden fully-connected layer in the Q-networks."""

    # ── Derived / computed at runtime ────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Automatically use GPU when available."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def lambda_levels(self) -> np.ndarray:
        """Discrete λ values the agent can choose, shape (K,).
        Linearly spaced from a small positive value up to μ so that
        λ ≤ μ is guaranteed.  Level 0 means 'do not serve'."""
        return np.linspace(0.0, self.mu, self.num_lambda_levels)

    # ── Seed everything ──────────────────────────────────────────────────

    def seed_everything(self) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

