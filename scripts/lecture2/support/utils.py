"""On-demand reports and summary plots; no simulation/control policy."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared.common import OUT_DIR

OUT = Path(OUT_DIR) / "lecture2"


def save_report(name, data):
    """Write JSON, converting NumPy arrays/scalars without losing numeric values."""

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (name + ".json")
    path.write_text(json.dumps(data, indent=2, default=encode, allow_nan=False))
    print("Report:", path)
    return path


def _save_figure(fig, filename):
    """Use the same output directory and resolution for all summary figures."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def plot_admittance(log, stiffness):
    """Plot the Y components of applied/external/filtered force and reference/actual pose."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    time = log[:, 0]
    for column, label in ((2, "Applied Fy (world)"), (5, "Compensated sensor Fy"), (8, "Filtered Fy")):
        axes[0].plot(time, log[:, column], label=label)
    axes[0].set_ylabel("Force [N]")
    axes[1].plot(log[:, 0], log[:, 11] * 1000, label="Virtual reference y")
    axes[1].plot(log[:, 0], log[:, 14] * 1000, label="Actual EEF y")
    if stiffness:
        axes[1].axhline(8 / stiffness * 1000, ls=":", color="k", label="Static F/K (8 N)")
    axes[1].set_ylabel("Displacement [mm]")
    axes[1].set_xlabel("Simulation time [s]")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    _save_figure(fig, "admittance.png")


def plot_polishing(log, filename="polishing_admittance.png", curved=False, motion_start=None):
    """Compare desired/measured normal force and the controller's position correction."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(log[:, 0], log[:, 1], "--", label="Desired contact force", color="darkorange")
    axes[0].plot(log[:, 0], log[:, 3], label="Measured normal force (filtered)", color="seagreen")
    axes[0].set_ylabel("Contact force [N]")
    axes[0].legend()
    axes[1].plot(
        log[:, 0],
        1000 * log[:, 4],
        color="royalblue",
        label="Inward normal compensation delta" if curved else "Downward compensation delta",
    )
    axes[1].set_ylabel("Compensation [mm]")
    axes[1].set_xlabel("Simulation time [s]")
    axes[1].legend()
    start = 4.0 if curved and motion_start is None else motion_start
    for ax in axes:
        ax.grid(alpha=0.25)
        if start is not None:
            ax.axvline(start, color="grey", ls=":", lw=1)
    fig.suptitle(
        ("Curved-surface " if curved else "") + "Force-error admittance: M delta_ddot + B delta_dot = Fd - Fm (K = 0)"
    )
    _save_figure(fig, filename)
