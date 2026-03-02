"""
aoi_model.py — Age-of-Information (AoI) computation for Vehicular Fog Networks.

The AoI measures the *freshness* of the information that has been delivered
from a vehicle (source) to the RSU (destination) via the fog network.

Closed-form AoI for vehicle i at RSU h
---------------------------------------
    Δ_{i,h} =   1 / ((1 − ε) · λ)
               + 1 / ((1 − ε) · μ)
               + λ  / (μ · (λ + μ))

Where:
    λ  — task arrival / offloading rate assigned to vehicle i
    μ  — RSU service rate (processing capacity)
    ε  — packet error probability on the wireless link

Constraints:
    1. Service constraint:   λ ≤ μ   (RSU must be able to handle the load)
    2. Threshold constraint: Δ_{i,h} ≤ Δ_threshold

Average AoI across all *participating* vehicles (those within d_max):
    Δ_avg = (1 / |S|) · Σ_{i ∈ S} Δ_{i,h}

Reference
---------
Eq. (6)-(8) of
"Optimizing AoI in Mobility-Based Vehicular Fog Networks:
 A Dueling-DDQN Approach"
"""

import numpy as np
from typing import Optional


def compute_aoi(
    lambda_i: float,
    mu: float,
    epsilon: float,
) -> float:
    """Compute the Age-of-Information for a single vehicle–RSU pair.

    Parameters
    ----------
    lambda_i : float
        Task offloading rate assigned to vehicle i (tasks / time-slot).
        Must satisfy 0 < λ ≤ μ.
    mu : float
        RSU processing / service rate (tasks / time-slot).
    epsilon : float
        Packet error probability, 0 ≤ ε < 1.

    Returns
    -------
    aoi : float
        Δ_{i,h} — instantaneous AoI for this vehicle.
        Returns a large penalty value if λ is invalid.

    Mathematical derivation
    -----------------------
    The three terms correspond to:
      • 1/((1−ε)λ)        — average inter-arrival time, inflated by errors
      • 1/((1−ε)μ)        — average service time, inflated by errors
      • λ/(μ(λ+μ))        — queueing delay correction factor
    """
    # Guard against degenerate cases
    if lambda_i <= 0:
        # Vehicle is not being served → AoI is effectively infinite;
        # return a large but finite penalty so gradients stay bounded.
        return 1000.0

    if lambda_i > mu:
        # Violates service constraint → penalise heavily
        return 1000.0

    one_minus_eps = 1.0 - epsilon  # (1 − ε)

    term1 = 1.0 / (one_minus_eps * lambda_i)   # inter-arrival component
    term2 = 1.0 / (one_minus_eps * mu)          # service component
    term3 = lambda_i / (mu * (lambda_i + mu))   # queueing correction

    aoi = term1 + term2 + term3
    return aoi


def compute_average_aoi(
    lambdas: np.ndarray,
    mu: float,
    epsilon: float,
    participation_mask: Optional[np.ndarray] = None,
) -> float:
    """Compute the average AoI across participating vehicles.

    Parameters
    ----------
    lambdas : np.ndarray, shape (N,)
        Offloading rates assigned to each vehicle.
    mu : float
        RSU service rate.
    epsilon : float
        Error probability.
    participation_mask : np.ndarray of bool, shape (N,), optional
        True for vehicles within coverage radius.  If None, all vehicles
        are considered.

    Returns
    -------
    avg_aoi : float
        Mean AoI over participating vehicles.
    """
    if participation_mask is None:
        participation_mask = np.ones(len(lambdas), dtype=bool)

    active = np.where(participation_mask)[0]
    if len(active) == 0:
        # No vehicle in range — worst case
        return 1000.0

    aoi_values = np.array(
        [compute_aoi(lambdas[i], mu, epsilon) for i in active]
    )
    return float(np.mean(aoi_values))


def check_threshold(aoi: float, threshold: float) -> bool:
    """Return True if the AoI is within the acceptable threshold."""
    return aoi <= threshold

