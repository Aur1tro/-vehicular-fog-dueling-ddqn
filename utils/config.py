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
    # Parameters calibrated to match the ICC paper's reported AoI ≈ 0.85.
    # Key insight: the 1/((1-ε)λ) term in the AoI formula dominates when
    # per-vehicle λ is small.  We need μ large enough so that λ_i = μ/N_active
    # keeps AoI in the sub-1 range.

    num_vehicles: int = 40
    """N — number of vehicles (task generators) in the network.
    With N=40 and μ=60, each active vehicle gets λ ≈ μ/N = 1.5,
    yielding AoI(1.5, 60, 0.05) ≈ 0.72.  The key invariant is
    μ/N ≈ 1.5 so that per-vehicle AoI stays in the 0.7–1.0 range."""

    num_rsus: int = 1
    """Number of Road-Side Units (fog nodes) serving the vehicles."""

    rsu_positions: list = field(default_factory=lambda: [[500.0, 500.0]])
    """(x, y) coordinates of each RSU on the 2-D plane (metres).
    Centred in the simulation area for symmetric coverage."""

    area_width: float = 1000.0
    """Width of the rectangular simulation area (metres).
    Scaled up for 40 vehicles to avoid overcrowding."""

    area_height: float = 1000.0
    """Height of the rectangular simulation area (metres)."""

    coverage_radius: float = 500.0
    """d_max — maximum communication range (metres).
    Covers ~78 % of the 1000×1000 area so most vehicles participate."""

    mu: float = 60.0
    """μ — RSU aggregate service rate (tasks / time-slot).
    Scaled with N: μ/N = 60/40 = 1.5 per vehicle.
    AoI(1.5, 60, 0.05) ≈ 0.72.  This keeps AoI in the paper's
    target range of ≈ 0.85 ± 0.1."""

    epsilon_error: float = 0.05
    """ε — packet error probability (0 ≤ ε < 1).
    Paper uses small ε (0.01–0.1).  0.05 is a realistic mid-point."""

    aoi_threshold: float = 2.0
    """Δ_threshold — maximum tolerable AoI for any vehicle.
    Set relative to achievable AoI so violations are meaningful.
    Optimal AoI ≈ 0.85 → threshold at ~2× gives room for exploration."""

    # ── Mobility Model ───────────────────────────────────────────────────

    vehicle_speed: float = 10.0
    """v — constant speed of every vehicle (m / time-slot).
    Lower speed → vehicles stay in coverage longer → more stable λ."""

    dt: float = 1.0
    """Duration of one simulation time-step (seconds)."""

    # ── Discrete Action Space ────────────────────────────────────────────

    num_lambda_levels: int = 10
    """K — number of discrete arrival-rate levels the agent can assign
    to each vehicle.  Higher K gives finer-grained λ control.
    λ values are linspace(λ_min, μ/N_min, K) where λ_min > 0."""

    # ── Episode / Time Parameters ────────────────────────────────────────

    max_steps_per_episode: int = 100
    """T — number of time-steps in one episode.  100 steps provides
    sufficient signal per episode while keeping training fast on CPU."""

    num_episodes: int = 1000
    """Total training episodes."""

    # ── RL Hyper-Parameters ──────────────────────────────────────────────

    learning_rate: float = 5e-4
    """Adam optimiser learning rate.  Slightly lower for stability."""

    gamma: float = 0.99
    """Discount factor γ for future rewards."""

    batch_size: int = 128
    """Mini-batch size sampled from the replay buffer.
    Larger batch for N=40 to reduce gradient variance."""

    replay_buffer_size: int = 100_000
    """Maximum capacity of the experience-replay buffer."""

    epsilon_start: float = 1.0
    """Initial exploration rate for ε-greedy policy."""

    epsilon_end: float = 0.01
    """Minimum exploration rate after decay."""

    epsilon_decay: float = 0.990
    """Multiplicative decay applied to ε after every episode.
    0.990 → ε reaches 0.01 after ~460 episodes."""

    target_update_freq: int = 5
    """Number of episodes between hard target-network syncs."""

    tau: float = 0.005
    """Polyak averaging coefficient for soft target updates.
    θ_target ← τ·θ_online + (1−τ)·θ_target"""

    use_soft_update: bool = True
    """If True, use soft (Polyak) updates every step instead of periodic
    hard copies.  Soft updates provide smoother target evolution."""

    # ── Reproducibility ──────────────────────────────────────────────────

    seed: int = 42
    """Global random seed (Python, NumPy, PyTorch, CUDA)."""

    # ── Hidden layer sizes (shared across all agents for fair comparison) ─

    hidden_dim: int = 256
    """Width of every hidden fully-connected layer in the Q-networks.
    Increased for N=40: state_dim = 3×40 = 120 inputs need wider layers."""

    # ── Derived / computed at runtime ────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Automatically use GPU when available."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def lambda_levels(self) -> np.ndarray:
        """Discrete λ values the agent can choose per vehicle, shape (K,).
        Starts from a small positive value (not zero — zero λ → infinite AoI)
        up to a per-vehicle maximum of μ / 2.  This ensures the total
        allocation across N vehicles can sum to at most N·(μ/2) which is
        then clipped to μ in the environment."""
        lam_min = 0.1  # small positive to avoid 1/λ blow-up
        lam_max = self.mu / max(self.num_vehicles // 2, 1)
        return np.linspace(lam_min, lam_max, self.num_lambda_levels)

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

