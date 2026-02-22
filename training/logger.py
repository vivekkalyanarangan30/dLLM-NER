"""Lightweight training logger: CSV file + live Colab-friendly plots.

No external dependencies beyond matplotlib (already in requirements.txt).
Designed for Colab — call plot_training_log() in a cell at any time to see
curves, even while training is still running.

CSV format (appended per event):
    step,epoch,train_loss,val_loss,learning_rate,elapsed_sec
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt


# Use non-interactive backend by default; Colab overrides this.
matplotlib.use("Agg")

LOG_FILENAME = "training_log.csv"
FIELDNAMES = ["step", "epoch", "train_loss", "val_loss", "learning_rate", "elapsed_sec"]


class TrainingLogger:
    """Append-only CSV logger. Safe across Colab restarts (append mode)."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, LOG_FILENAME)

        # Write header only if file doesn't exist or is empty
        write_header = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
        if write_header:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log_train(self, step: int, epoch: int, train_loss: float,
                  learning_rate: float, elapsed_sec: float):
        """Log a training step."""
        self._append({
            "step": step,
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": "",
            "learning_rate": f"{learning_rate:.2e}",
            "elapsed_sec": f"{elapsed_sec:.1f}",
        })

    def log_val(self, step: int, epoch: int, val_loss: float, elapsed_sec: float):
        """Log a validation result."""
        self._append({
            "step": step,
            "epoch": epoch,
            "train_loss": "",
            "val_loss": f"{val_loss:.6f}",
            "learning_rate": "",
            "elapsed_sec": f"{elapsed_sec:.1f}",
        })

    def _append(self, row: dict):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow(row)


def read_training_log(log_path: str) -> dict:
    """Read a training_log.csv and return parsed columns.

    Returns
    -------
    dict with keys: "steps", "train_losses", "val_steps", "val_losses",
    "learning_rates", "elapsed"
    """
    steps, train_losses = [], []
    val_steps, val_losses = [], []
    learning_rates = []
    elapsed = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            if row["train_loss"]:
                steps.append(step)
                train_losses.append(float(row["train_loss"]))
                learning_rates.append(float(row["learning_rate"]))
                elapsed.append(float(row["elapsed_sec"]))
            if row["val_loss"]:
                val_steps.append(step)
                val_losses.append(float(row["val_loss"]))

    return {
        "steps": steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "elapsed": elapsed,
    }


def plot_training_log(
    log_path: str,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot training and validation loss curves from a CSV log file.

    Call this in a Colab cell at any time (even mid-training) to see progress::

        from training.logger import plot_training_log
        plot_training_log("/content/drive/MyDrive/dLLM-NER/checkpoints/training_log.csv")

    Parameters
    ----------
    log_path : str
        Path to training_log.csv.
    save_path : str, optional
        If provided, save the figure to this path (PNG).
    show : bool
        Whether to call plt.show() (set False for non-interactive).
    """
    data = read_training_log(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Loss curves ---
    ax = axes[0]
    if data["steps"]:
        ax.plot(data["steps"], data["train_losses"], alpha=0.4, color="#2196F3",
                linewidth=0.8, label="Train (raw)")
        # Smoothed train loss (exponential moving average)
        if len(data["train_losses"]) > 10:
            smoothed = _ema(data["train_losses"], alpha=0.05)
            ax.plot(data["steps"], smoothed, color="#1565C0",
                    linewidth=2, label="Train (smoothed)")
    if data["val_steps"]:
        ax.plot(data["val_steps"], data["val_losses"], "o-", color="#F44336",
                linewidth=2, markersize=5, label="Validation")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Learning rate ---
    ax = axes[1]
    if data["steps"]:
        ax.plot(data["steps"], data["learning_rates"], color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-4, -4))

    # --- Throughput ---
    ax = axes[2]
    if len(data["elapsed"]) > 1:
        # Steps per second (local rate)
        rate = []
        for i in range(1, len(data["elapsed"])):
            dt = data["elapsed"][i] - data["elapsed"][i - 1]
            ds = data["steps"][i] - data["steps"][i - 1]
            if dt > 0:
                rate.append(ds / dt)
            else:
                rate.append(0.0)
        ax.plot(data["steps"][1:], rate, color="#FF9800", linewidth=1, alpha=0.5)
        if len(rate) > 10:
            smoothed_rate = _ema(rate, alpha=0.1)
            ax.plot(data["steps"][1:], smoothed_rate, color="#E65100", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Steps/sec")
    ax.set_title("Training Throughput")
    ax.grid(True, alpha=0.3)

    # Add summary text
    if data["train_losses"]:
        latest_train = data["train_losses"][-1]
        latest_step = data["steps"][-1]
        elapsed_hrs = data["elapsed"][-1] / 3600 if data["elapsed"] else 0
        summary = f"Step {latest_step} | Train: {latest_train:.4f}"
        if data["val_losses"]:
            summary += f" | Val: {data['val_losses'][-1]:.4f} (best: {min(data['val_losses']):.4f})"
        summary += f" | {elapsed_hrs:.1f}h elapsed"
        fig.suptitle(summary, fontsize=11, y=1.02)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _ema(values: list, alpha: float = 0.1) -> list:
    """Exponential moving average for smoothing."""
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result
