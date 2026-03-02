"""
mobility.py — Constant-Speed Mobility Model for Vehicular Fog Networks.

Each vehicle moves at a fixed speed *v* in a direction θ that is drawn
uniformly at random at the start of each episode (and optionally
re-randomised when the vehicle hits a boundary).

Position update equations (per time-step dt):
    x(t+1) = x(t) + v · dt · cos(θ)
    y(t+1) = y(t) + v · dt · sin(θ)

When a vehicle exits the rectangular area [0, W] × [0, H] it "bounces"
by reflecting its heading angle, keeping it inside the coverage region.

Reference
---------
Section III-A of
"Optimizing AoI in Mobility-Based Vehicular Fog Networks:
 A Dueling-DDQN Approach"
"""

import numpy as np
from typing import Tuple


class MobilityModel:
    """Simulates N vehicles moving at constant speed on a 2-D plane.

    Attributes
    ----------
    num_vehicles : int
        Number of vehicles.
    speed : float
        Constant speed v (m/time-step).
    dt : float
        Duration of one time-step (seconds).
    width, height : float
        Rectangular area dimensions (metres).
    positions : np.ndarray, shape (N, 2)
        Current (x, y) of each vehicle.
    headings : np.ndarray, shape (N,)
        Current heading angle θ (radians) of each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        speed: float,
        dt: float,
        width: float,
        height: float,
        seed: int = 42,
    ) -> None:
        self.num_vehicles = num_vehicles
        self.speed = speed
        self.dt = dt
        self.width = width
        self.height = height
        self.rng = np.random.RandomState(seed)

        # Will be populated by reset()
        self.positions: np.ndarray = np.zeros((num_vehicles, 2))
        self.headings: np.ndarray = np.zeros(num_vehicles)

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Place vehicles at random positions with random headings.

        Returns
        -------
        positions : np.ndarray, shape (N, 2)
        """
        self.positions = self.rng.uniform(
            low=[0.0, 0.0],
            high=[self.width, self.height],
            size=(self.num_vehicles, 2),
        )
        # θ ∈ [0, 2π)
        self.headings = self.rng.uniform(0, 2 * np.pi, size=self.num_vehicles)
        return self.positions.copy()

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self) -> np.ndarray:
        """Advance all vehicles by one time-step.

        Position update:
            x += v · dt · cos(θ)
            y += v · dt · sin(θ)

        Boundary handling: reflection (heading component is negated when
        the vehicle crosses a wall).

        Returns
        -------
        positions : np.ndarray, shape (N, 2)
        """
        # Displacement
        dx = self.speed * self.dt * np.cos(self.headings)
        dy = self.speed * self.dt * np.sin(self.headings)

        self.positions[:, 0] += dx
        self.positions[:, 1] += dy

        # ── Reflect off boundaries ──
        self._reflect_boundaries()

        return self.positions.copy()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _reflect_boundaries(self) -> None:
        """Reflect vehicles that have exited the area.

        If x < 0 or x > width, negate the x-component of heading
        (θ → π − θ) and clamp position.  Similarly for y.
        """
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        # Left / right walls
        out_left = x < 0
        out_right = x > self.width
        if np.any(out_left | out_right):
            self.headings[out_left | out_right] = (
                np.pi - self.headings[out_left | out_right]
            )
            x[out_left] = -x[out_left]
            x[out_right] = 2 * self.width - x[out_right]

        # Bottom / top walls
        out_bottom = y < 0
        out_top = y > self.height
        if np.any(out_bottom | out_top):
            self.headings[out_bottom | out_top] = (
                -self.headings[out_bottom | out_top]
            )
            y[out_bottom] = -y[out_bottom]
            y[out_top] = 2 * self.height - y[out_top]

        # Keep heading in [0, 2π)
        self.headings = self.headings % (2 * np.pi)

        self.positions[:, 0] = x
        self.positions[:, 1] = y

    # ── Utility ──────────────────────────────────────────────────────────

    def distances_to(self, point: np.ndarray) -> np.ndarray:
        """Euclidean distance from every vehicle to a given point.

        Parameters
        ----------
        point : array-like, shape (2,)
            (x, y) of the reference point (e.g. RSU location).

        Returns
        -------
        dists : np.ndarray, shape (N,)
        """
        return np.linalg.norm(self.positions - np.asarray(point), axis=1)

