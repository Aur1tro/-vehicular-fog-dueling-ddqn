"""
vehicular_env.py — OpenAI-Gym-style environment for the Vehicular Fog Network.

The environment simulates N vehicles that move according to a constant-speed
mobility model and offload tasks to one RSU.  The RL agent decides how to
allocate the offloading rate λ to each vehicle at every time-step.

State  (observation)
-----
A flat vector of size 3·N containing, for each vehicle:
    • normalised distance to the RSU  (d_i / d_max)
    • participation flag              (1 if d_i ≤ d_max, else 0)
    • current AoI                     (Δ_i / Δ_threshold, clipped to [0,1])

Action
------
A single discrete integer in [0, K^N_active) that encodes a *combination*
of λ-levels for participating vehicles.

To keep the action space tractable (exponential blow-up with N), we use a
simplified scheme:

    action ∈ {0, 1, …, K·N − 1}

Interpretation:
    vehicle_index = action // K
    lambda_level  = action %  K

The selected vehicle gets the chosen λ; all other participating vehicles
receive a baseline rate (e.g., μ / |S|) to avoid infinite AoI.

Reward
------
    r = 1 / avg_AoI

Higher is better — encourages the agent to minimise average AoI.
A small penalty is added if any vehicle exceeds Δ_threshold.

Episode termination
-------------------
After T = max_steps_per_episode time-steps.

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
        self.lambda_levels = self.cfg.lambda_levels  # 0 … μ

        # ── Action & observation dimensions ──────────────────────────────
        self.K = self.cfg.num_lambda_levels
        self.N = self.cfg.num_vehicles
        self.action_dim = self.N * self.K          # flat discrete space
        self.state_dim = self.N * 3                # 3 features per vehicle

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

        # Initial AoI is high (no information received yet)
        self.current_aoi = np.full(self.N, self.cfg.aoi_threshold)

        # Compute initial distances & participation
        distances = self.mobility.distances_to(self.rsu_pos)
        participation = (distances <= self.cfg.coverage_radius).astype(float)

        # Give participating vehicles a fair baseline λ
        n_active = max(int(participation.sum()), 1)
        base_lambda = self.cfg.mu / n_active
        self.current_lambdas = participation * base_lambda

        # Compute initial AoI per vehicle
        for i in range(self.N):
            if participation[i]:
                self.current_aoi[i] = compute_aoi(
                    self.current_lambdas[i], self.cfg.mu, self.cfg.epsilon_error
                )
            else:
                self.current_aoi[i] = self.cfg.aoi_threshold  # out of range

        return self._build_state(distances, participation)

    # ==================================================================
    #  step(action)
    # ==================================================================

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time-step.

        Parameters
        ----------
        action : int
            Index into the flat action space [0, N·K).
            Encodes *which vehicle* and *which λ level*.

        Returns
        -------
        state  : np.ndarray   — next observation
        reward : float         — 1 / avg_AoI (higher = better)
        done   : bool          — True when episode ends
        info   : dict          — diagnostics (avg_aoi, per-vehicle AoI, etc.)
        """
        self.current_step += 1

        # ── 1. Decode the action ─────────────────────────────────────────
        vehicle_idx = action // self.K
        lambda_level = action % self.K

        # Safety: clamp vehicle index
        vehicle_idx = min(vehicle_idx, self.N - 1)

        # ── 2. Move vehicles ─────────────────────────────────────────────
        self.mobility.step()
        distances = self.mobility.distances_to(self.rsu_pos)
        participation = (distances <= self.cfg.coverage_radius).astype(float)

        # ── 3. Assign λ values ───────────────────────────────────────────
        n_active = max(int(participation.sum()), 1)
        # Baseline: spread RSU capacity equally among active vehicles
        base_lambda = self.cfg.mu / (n_active + 1)  # +1 reserves room for agent's pick
        self.current_lambdas = participation * base_lambda

        # Agent's chosen vehicle gets the specific λ (if in range)
        chosen_lambda = self.lambda_levels[lambda_level]
        if participation[vehicle_idx]:
            self.current_lambdas[vehicle_idx] = chosen_lambda

        # Enforce service constraint: total λ ≤ μ per RSU
        total_lambda = self.current_lambdas.sum()
        if total_lambda > self.cfg.mu and total_lambda > 0:
            # Scale down proportionally
            self.current_lambdas *= self.cfg.mu / total_lambda

        # ── 4. Compute AoI for each vehicle ──────────────────────────────
        threshold_violations = 0
        for i in range(self.N):
            if participation[i] and self.current_lambdas[i] > 0:
                self.current_aoi[i] = compute_aoi(
                    self.current_lambdas[i], self.cfg.mu, self.cfg.epsilon_error
                )
            else:
                # Vehicle out of range or λ=0 → AoI grows linearly
                self.current_aoi[i] = min(
                    self.current_aoi[i] + 1.0, self.cfg.aoi_threshold * 2
                )

            if not check_threshold(self.current_aoi[i], self.cfg.aoi_threshold):
                threshold_violations += 1

        # ── 5. Compute reward ────────────────────────────────────────────
        avg_aoi = compute_average_aoi(
            self.current_lambdas, self.cfg.mu, self.cfg.epsilon_error,
            participation_mask=participation.astype(bool),
        )

        # Reward: inverse of average AoI (higher reward ↔ lower AoI)
        reward = 1.0 / max(avg_aoi, 1e-6)

        # Penalise threshold violations to steer the agent toward feasibility
        reward -= 0.1 * threshold_violations

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
            [0] normalised distance  d_i / d_max
            [1] participation flag   1{d_i ≤ d_max}
            [2] normalised AoI       clip(Δ_i / Δ_thresh, 0, 1)
        """
        norm_dist = distances / self.cfg.coverage_radius  # may exceed 1
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

