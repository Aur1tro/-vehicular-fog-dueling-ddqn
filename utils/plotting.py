"""
plotting.py — Visualisation utilities for training results.

Provides three publication-ready plot functions:

1. plot_rewards       — Reward vs episode for all four agents (overlaid)
2. plot_aoi_vs_vehicles — Average AoI as a function of # vehicles
3. plot_convergence   — Smoothed reward curves to compare convergence speed

All functions accept dictionaries keyed by agent name so they are
algorithm-agnostic.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


# ── Styling defaults ─────────────────────────────────────────────────────
COLORS = {
    "DQN": "#1f77b4",
    "Double DQN": "#ff7f0e",
    "Dueling DQN": "#2ca02c",
    "Dueling DDQN": "#d62728",
}

LINE_STYLES = {
    "DQN": "-",
    "Double DQN": "--",
    "Dueling DQN": "-.",
    "Dueling DDQN": "-",
}


def _smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Simple moving-average smoother for noisy training curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ═══════════════════════════════════════════════════════════════════════
#  1. Reward vs Episode
# ═══════════════════════════════════════════════════════════════════════

def plot_rewards(
    histories: Dict[str, Dict[str, List[float]]],
    window: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Overlay smoothed reward curves for multiple agents.

    Parameters
    ----------
    histories : dict
        {agent_name: {"episode_rewards": [...], ...}}
    window : int
        Moving-average window size.
    save_path : str, optional
        If given, save the figure to this path.
    """
    plt.figure(figsize=(10, 6))

    for name, hist in histories.items():
        rewards = hist["episode_rewards"]
        smoothed = _smooth(rewards, window)
        color = COLORS.get(name, None)
        ls = LINE_STYLES.get(name, "-")
        plt.plot(smoothed, label=name, color=color, linestyle=ls, linewidth=1.5)

    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Cumulative Reward", fontsize=13)
    plt.title("Reward vs Episode — DQN Variant Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  2. AoI vs Number of Vehicles
# ═══════════════════════════════════════════════════════════════════════

def plot_aoi_vs_vehicles(
    results: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
) -> None:
    """Bar / line chart of average AoI for varying number of vehicles.

    Parameters
    ----------
    results : dict
        {agent_name: {num_vehicles: mean_aoi, ...}}
    """
    plt.figure(figsize=(10, 6))

    for name, data in results.items():
        xs = sorted(data.keys())
        ys = [data[x] for x in xs]
        color = COLORS.get(name, None)
        ls = LINE_STYLES.get(name, "-")
        plt.plot(xs, ys, marker="o", label=name, color=color, linestyle=ls, linewidth=1.5)

    plt.xlabel("Number of Vehicles (N)", fontsize=13)
    plt.ylabel("Average AoI", fontsize=13)
    plt.title("Average AoI vs Number of Vehicles", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
#  3. Convergence Comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_convergence(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "episode_avg_aoi",
    window: int = 30,
    save_path: Optional[str] = None,
) -> None:
    """Compare convergence speed across algorithms.

    By default plots smoothed average AoI per episode.
    """
    plt.figure(figsize=(10, 6))

    ylabel_map = {
        "episode_avg_aoi": "Average AoI",
        "episode_rewards": "Cumulative Reward",
        "episode_losses": "Loss",
    }

    for name, hist in histories.items():
        values = hist.get(metric, [])
        smoothed = _smooth(values, window)
        color = COLORS.get(name, None)
        ls = LINE_STYLES.get(name, "-")
        plt.plot(smoothed, label=name, color=color, linestyle=ls, linewidth=1.5)

    plt.xlabel("Episode", fontsize=13)
    plt.ylabel(ylabel_map.get(metric, metric), fontsize=13)
    plt.title(f"Convergence Comparison — {ylabel_map.get(metric, metric)}", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

