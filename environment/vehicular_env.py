"""
vehicular_env.py — OpenAI-Gym-style environment for the Vehicular Fog Network.

Redesigned to match the ICC paper:
"Optimizing AoI in Mobility-Based Vehicular Fog Networks: A Dueling-DDQN Approach"

KEY DESIGN (matches the paper's formulation)
---------------------------------------------
The RL agent selects a *rate-allocation level* for all active vehicles at
each time-step.  The level determines the target per-vehicle λ; the
environment then enforces the aggregate service constraint Σλ_i ≤ μ by
proportional scaling.

Action space: K discrete levels (e.g., K=10).
  - Each level maps to a target per-vehicle λ from the palette.
  - Active vehicles (within coverage) all receive that λ.
  - If total λ exceeds μ, all rates are scaled down proportionally.

This keeps the action space small (K=10) so the agent can learn effectively,
while still capturing the key trade-off: higher λ → lower AoI but risk
violating the service constraint when many vehicles are active.

State  (observation)
-----
A flat vector of size 3·N containing, for each vehicle:
    • normalised distance to the RSU  (d_i / d_max, clipped to [0, 2])
    • participation flag              (1 if d_i ≤ d_max, else 0)
    • normalised current AoI          (Δ_i / Δ_threshold, clipped to [0, 2])

Reward
------
    r = -avg_AoI + bonus_if_below_threshold - penalty_for_violations

Higher (less negative) is better — encourages the agent to minimise AoI.

Reference
---------
"Optimizing AoI in Mobility-Based Vehicular Fog Networks:
 A Dueling-DDQN Approach"
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any, Optional

from environment.mobility import MobilityModel
from environment.aoi_model import compute_aoi, compute_average_aoi, check_threshold

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.config import Config


class VehicularFogEnv:
    """Gym-style environment for AoI optimisation in vehicular fog networks.

    Follows the standard interface:
        state              = env.reset()
        state, r, done, info = env.step(action)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()

        # ── Mobility sub-model ───────────────────────────────────────────
        self.mobility = MobilityModel(
            num_vehicles=self.cfg.num_vehicles,
            speed=self.cfg.vehicle_speed,
            dt=self.cfg.dt,
            width=self.cfg.area_width,
            height=self.cfg.area_height,
            seed=self.cfg.seed,
        )

        # RSU location (use the first RSU for the single-RSU case)
        self.rsu_pos = np.array(self.cfg.rsu_positions[0])

        # Discrete λ palette: shape (K,)
        self.lambda_levels = self.cfg.lambda_levels

        # ── Action & observation dimensions ──────────────────────────────
        self.K = self.cfg.num_lambda_levels
        self.N = self.cfg.num_vehicles

        # Action space: K levels — agent picks a target λ applied to all
        # active vehicles, subject to the Σλ ≤ μ constraint
        self.action_dim = self.K
        self.state_dim = self.N * 3  # 3 features per vehicle

        # ── Per-vehicle AoI tracking ─────────────────────────────────────
        self.current_aoi = np.zeros(self.N)
        self.current_lambdas = np.zeros(self.N)

        # ── Episode bookkeeping ──────────────────────────────────────────
        self.current_step = 0
        self.done = False

    # ==================================================================
    #  reset()
    # ==================================================================

    def reset(self) -> np.ndarray:
        """Reset the environment to an initial state.

        Returns
        -------
        state : np.ndarray, shape (state_dim,)
        """
        self.mobility.reset()
        self.current_step = 0
        self.done = False

        # Initial AoI — moderate starting value
        self.current_aoi = np.ones(self.N) * 1.0

        # Compute initial distances & participation
        distances = self.mobility.distances_to(self.rsu_pos)
        participation = (distances <= self.cfg.coverage_radius).astype(float)

        # Give participating vehicles a fair baseline λ = μ / n_active
        n_active = max(int(participation.sum()), 1)
        base_lambda = self.cfg.mu / n_active
        self.current_lambdas = participation * base_lambda

        # Compute initial AoI per vehicle
        for i in range(self.N):
            if participation[i] and self.current_lambdas[i] > 0:
                self.current_aoi[i] = compute_aoi(
                    self.current_lambdas[i], self.cfg.mu, self.cfg.epsilon_error
                )
            else:
                self.current_aoi[i] = self.cfg.aoi_threshold

        return self._build_state(distances, participation)

    # ==================================================================
    #  step(action)
    # ==================================================================

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time-step.

        Parameters
        ----------
        action : int
            Index into [0, K): selects a λ-level for all active vehicles.

        Returns
        -------
        state  : np.ndarray   — next observation
        reward : float         — shaped reward (higher = better)
        done   : bool          — True when episode ends
        info   : dict          — diagnostics (avg_aoi, per-vehicle AoI, etc.)
        """
        self.current_step += 1

        # ── 1. Decode the action ─────────────────────────────────────────
        target_lambda = self.lambda_levels[min(action, self.K - 1)]

        # ── 2. Move vehicles ─────────────────────────────────────────────
        self.mobility.step()
        distances = self.mobility.distances_to(self.rsu_pos)
        participation = (distances <= self.cfg.coverage_radius).astype(float)

        # ── 3. Assign λ values ───────────────────────────────────────────
        n_active = max(int(participation.sum()), 1)

        # Each active vehicle gets the target λ from the chosen level
        self.current_lambdas = participation * target_lambda

        # Enforce service constraint: Σλ_i ≤ μ
        total_lambda = self.current_lambdas.sum()
        if total_lambda > self.cfg.mu and total_lambda > 0:
            self.current_lambdas *= self.cfg.mu / total_lambda

        # Ensure a minimum λ for active vehicles to avoid 1/λ blow-up
        for i in range(self.N):
            if participation[i] and self.current_lambdas[i] < 0.05:
                self.current_lambdas[i] = 0.05

        # ── 4. Compute AoI for each vehicle ──────────────────────────────
        threshold_violations = 0
        for i in range(self.N):
            if participation[i] and self.current_lambdas[i] > 0:
                self.current_aoi[i] = compute_aoi(
                    self.current_lambdas[i], self.cfg.mu, self.cfg.epsilon_error
                )
            else:
                # Vehicle out of range → AoI grows (stale information)
                self.current_aoi[i] = min(
                    self.current_aoi[i] + 0.5,
                    self.cfg.aoi_threshold * 2
                )

            if not check_threshold(self.current_aoi[i], self.cfg.aoi_threshold):
                threshold_violations += 1

        # ── 5. Compute reward ────────────────────────────────────────────
        avg_aoi = compute_average_aoi(
            self.current_lambdas, self.cfg.mu, self.cfg.epsilon_error,
            participation_mask=participation.astype(bool),
        )

        # Primary reward: negative AoI (agent minimises AoI)
        reward = -avg_aoi

        # Bonus for staying below threshold
        if avg_aoi < self.cfg.aoi_threshold:
            reward += 1.0

        # Small penalty for per-vehicle threshold violations
        reward -= 0.05 * threshold_violations

        # ── 6. Check termination ─────────────────────────────────────────
        self.done = self.current_step >= self.cfg.max_steps_per_episode

        # ── 7. Build next state ──────────────────────────────────────────
        state = self._build_state(distances, participation)

        info = {
            "avg_aoi": avg_aoi,
            "per_vehicle_aoi": self.current_aoi.copy(),
            "threshold_violations": threshold_violations,
            "n_active": int(participation.sum()),
            "lambdas": self.current_lambdas.copy(),
        }

        return state, reward, self.done, info

    # ==================================================================
    #  Internal helpers
    # ==================================================================

    def _build_state(
        self, distances: np.ndarray, participation: np.ndarray
    ) -> np.ndarray:
        """Construct a flat observation vector of size 3·N.

        Features per vehicle (i):
            [0] normalised distance  d_i / d_max  (clipped to [0, 2])
            [1] participation flag   1{d_i ≤ d_max}
            [2] normalised AoI       clip(Δ_i / Δ_thresh, 0, 2)
        """
        norm_dist = np.clip(distances / self.cfg.coverage_radius, 0.0, 2.0)
        norm_aoi = np.clip(self.current_aoi / self.cfg.aoi_threshold, 0.0, 2.0)

        state = np.stack([norm_dist, participation, norm_aoi], axis=1).flatten()
        return state.astype(np.float32)

    # ── Convenience properties ───────────────────────────────────────────

    @property
    def observation_space_dim(self) -> int:
        return self.state_dim

    @property
    def action_space_dim(self) -> int:
        return self.action_dim

