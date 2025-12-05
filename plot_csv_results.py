#!/usr/bin/env python3
"""
Quick utility to visualize Monte-Carlo statistics saved by CollaborativeGame.

By default it reads `Scenario_1_mc_stats.csv` and produces a PNG with
error-bar plots for distance (with optional min/max bands), acceleration,
and scenario duration grouped by the `alpha` column in the CSV.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(csv_path: Path) -> np.ndarray:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.ndim == 0:
        data = data.reshape(1)  # keep shapes consistent when file has one row
    return data


def _iter_alphas(data: np.ndarray) -> Iterable[Tuple[float, np.ndarray]]:
    for alpha in np.unique(data["alpha"]):
        mask = data["alpha"] == alpha
        yield alpha, np.sort(data[mask], order="mc_run")


def _has_distance_range(data: np.ndarray) -> bool:
    names = set(data.dtype.names or [])
    return {"distance_max"}.issubset(names)


def _filter_valid_rows(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Drop rows where metrics were marked invalid (-1 sentinel).
    Returns (filtered_data, num_dropped).
    """
    has_dist_range = _has_distance_range(data)
    valid_mask = (
        (data["distance_mean"] >= 0)
        & (data["distance_std"] >= 0)
        & ((data["distance_max"] >= 0) if has_dist_range else True)
        & (data["a1_acc_mean"] >= 0)
        & (data["a1_acc_std"] >= 0)
        & (data["a2_acc_mean"] >= 0)
        & (data["a2_acc_std"] >= 0)
        & (data["scenario_time_std"] >= 0)
    )
    filtered = data[valid_mask]
    return filtered, int((~valid_mask).sum())

def _failure_stats_by_alpha(raw: np.ndarray, filtered: np.ndarray):
    """
    Compute failure counts/percentages per alpha using raw vs. filtered data.
    """
    stats = []
    for alpha in np.unique(raw["alpha"]):
        total = int((raw["alpha"] == alpha).sum())
        valid = int((filtered["alpha"] == alpha).sum()) if filtered.size else 0
        failed = total - valid
        pct_failed = 0.0 if total == 0 else 100.0 * failed / total
        stats.append(
            {
                "alpha": float(alpha),
                "total": total,
                "failed": failed,
                "pct_failed": pct_failed,
            }
        )
    return stats


def summarize_by_alpha(data: np.ndarray):
    """
    Compute simple means across MC runs for each alpha.
    """
    summaries = []
    has_range = _has_distance_range(data)
    for alpha, grp in _iter_alphas(data):
        summaries.append(
            {
                "alpha": alpha,
                "n": len(grp),
                "distance_mean": float(np.mean(grp["distance_mean"])),
                "distance_std": float(np.mean(grp["distance_std"])),
                "distance_max": float(np.mean(grp["distance_max"])) if has_range else None,
                "a1_acc_mean": float(np.mean(grp["a1_acc_mean"])),
                "a1_acc_std": float(np.mean(grp["a1_acc_std"])),
                "a2_acc_mean": float(np.mean(grp["a2_acc_mean"])),
                "a2_acc_std": float(np.mean(grp["a2_acc_std"])),
                "scenario_time": float(np.mean(grp["scenario_time"])),
                "scenario_time_std": float(np.mean(grp["scenario_time_std"])),
            }
        )
    return summaries


def mean_max_distance_by_alpha(data: np.ndarray):
    """
    Return list of dicts with mean of the maximum distance per alpha.
    """
    if not _has_distance_range(data):
        return []
    results = []
    for alpha, grp in _iter_alphas(data):
        results.append({"alpha": float(alpha), "dist_max_mean": float(np.mean(grp["distance_max"]))})
    return results


def _plot_metric(ax, alpha_groups, mean_key: str, std_key: str, ylabel: str):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_groups)))
    for color, (alpha, grp) in zip(colors, alpha_groups):
        ax.errorbar(
            grp["mc_run"],
            grp[mean_key],
            yerr=grp[std_key],
            fmt="o-",
            capsize=3,
            label=f"alpha={alpha:.2f}",
            color=color,
        )
    ax.set_xlabel("MC run")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def _plot_distance(ax, alpha_groups, has_range: bool):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alpha_groups)))
    for color, (alpha, grp) in zip(colors, alpha_groups):
        mc = grp["mc_run"]
        if has_range:
            ax.fill_between(mc, grp["distance_max"], color=color, alpha=0.15)
        ax.errorbar(
            mc,
            grp["distance_mean"],
            yerr=grp["distance_std"],
            fmt="o-",
            capsize=3,
            label=f"alpha={alpha:.2f}",
            color=color,
        )
    ax.set_xlabel("MC run")
    ylabel = "Distance [m]" if not has_range else "Distance [m] (mean ± std, min/max band)"
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def plot_results(data: np.ndarray, csv_path: Path, output: Path | None = None, show: bool = False) -> Path:
    alpha_groups = list(_iter_alphas(data))
    if not alpha_groups:
        raise ValueError(f"No rows found in {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Stats from {csv_path.name}")

    has_range = _has_distance_range(data)
    _plot_distance(axes[0], alpha_groups, has_range)
    _plot_metric(axes[1], alpha_groups, "a1_acc_mean", "a1_acc_std", "Human accel [m/s^2]")

    # Overlay robot accel on a second y-axis for clarity.
    ax2 = axes[1].twinx()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(alpha_groups)))
    for color, (alpha, grp) in zip(colors, alpha_groups):
        ax2.errorbar(
            grp["mc_run"],
            grp["a2_acc_mean"],
            yerr=grp["a2_acc_std"],
            fmt="s--",
            capsize=3,
            label=f"robot alpha={alpha:.2f}",
            color=color,
        )
    ax2.set_ylabel("Robot accel [m/s^2]")
    ax2.grid(False)
    ax2.legend(loc="lower right")

    _plot_metric(axes[2], alpha_groups, "scenario_time", "scenario_time_std", "Scenario time [s]")

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    output_path = output or csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    os.system("clear")
    parser = argparse.ArgumentParser(description="Plot results from *_mc_stats.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("Scenario_2_mc_stats.csv"),
        help="Path to the mc_stats CSV produced by CollaborativeGame.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to save the plot (defaults to <csv>.png next to the CSV).",
    )
    parser.add_argument("--show", default=True, help="Display the interactive window in addition to saving.")
    args = parser.parse_args()

    data_raw = _load_csv(args.csv)
    data, dropped = _filter_valid_rows(data_raw)
    if dropped:
        print(f"Ignoring {dropped} infeasible run(s) marked with -1 stats.")
    if data.size == 0:
        raise SystemExit("No valid rows remain after filtering infeasible runs.")

    fail_stats = _failure_stats_by_alpha(data_raw, data)
    print("Failed run percentage per alpha:")
    for fs in fail_stats:
        print(
            f"  alpha={fs['alpha']:.3f}: failed {fs['failed']}/{fs['total']} "
            f"({fs['pct_failed']:.1f}%)"
        )

    has_range = _has_distance_range(data)
    if not has_range:
        print("No distance_min/distance_max columns found; plotting distance mean/std only.")

    dist_max_stats = mean_max_distance_by_alpha(data)
    if dist_max_stats:
        print("Mean max distance per alpha:")
        for item in dist_max_stats:
            print(f"  alpha={item['alpha']:.3f}: dist_max_mean={item['dist_max_mean']:.3f}")

    print("MC means per alpha:")
    for summary in summarize_by_alpha(data):
        parts = [
            f"alpha={summary['alpha']:.3f} (n={summary['n']}):",
            f"dist_mean={summary['distance_mean']:.3f}",
            f"dist_std_mean={summary['distance_std']:.3f}",
        ]
        if summary.get("distance_min") is not None:
            parts.append(f"dist_min_mean={summary['distance_min']:.3f}")
        if summary.get("distance_max") is not None:
            parts.append(f"dist_max_mean={summary['distance_max']:.3f}")
        parts.extend(
            [
                f"a1_acc_mean={summary['a1_acc_mean']:.3f}",
                f"a1_acc_std_mean={summary['a1_acc_std']:.3f}",
                f"a2_acc_mean={summary['a2_acc_mean']:.3f}",
                f"a2_acc_std_mean={summary['a2_acc_std']:.3f}",
                f"scenario_time={summary['scenario_time']:.3f}",
                f"scenario_time_std_mean={summary['scenario_time_std']:.3f}",
            ]
        )
        print(" ".join(parts))

    output_path = plot_results(data, args.csv, output=args.out, show=args.show)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
