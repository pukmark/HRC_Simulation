#!/usr/bin/env python3
"""
Quick utility to visualize Monte-Carlo statistics saved by CollaborativeGame.

By default it reads `Scenario_1_mc_stats.csv` and produces a PNG with
error-bar plots for distance (with optional min/max bands), acceleration,
and scenario duration grouped by the `alpha` column (and `beta` when present).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(csv_path: Path) -> np.ndarray:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = data.reshape(1)  # keep shapes consistent when file has one row
    return data


def _iter_groups(data: np.ndarray) -> Iterable[Tuple[Tuple[float, float | None], np.ndarray]]:
    names = set(data.dtype.names or [])
    has_beta = "beta" in names
    if has_beta:
        alphas = np.unique(data["alpha"])
        betas = np.unique(data["beta"])
        for alpha in alphas:
            for beta in betas:
                mask = (data["alpha"] == alpha) & (data["beta"] == beta)
                if not np.any(mask):
                    continue
                yield (float(alpha), float(beta)), np.sort(data[mask], order="mc_run")
    else:
        for alpha in np.unique(data["alpha"]):
            mask = data["alpha"] == alpha
            yield (float(alpha), None), np.sort(data[mask], order="mc_run")


def _has_distance_range(data: np.ndarray) -> bool:
    names = set(data.dtype.names or [])
    return {"distance_max"}.issubset(names)


def _filter_valid_rows(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Drop rows where metrics were marked invalid (-1 sentinel).
    Returns (filtered_data, num_dropped).
    """
    has_dist_range = _has_distance_range(data)
    names = set(data.dtype.names or [])
    has_confidence = {"confidence_mean", "confidence_std"}.issubset(names)

    valid_mask = (
         (data["distance_std"] >= 0)
        # & (data["distance_mean"] >= 0)
        & ((data["distance_max"] >= 0) if has_dist_range else True)
        & (data["a1_acc_mean"] >= 0)
        & (data["a1_acc_std"] >= 0)
        & (data["a2_acc_mean"] >= 0)
        & (data["a2_acc_std"] >= 0)
        & (data["scenario_time_std"] >= 0)
        & ((data["confidence_mean"] >= 0) if has_confidence else True)
        & ((data["confidence_std"] >= 0) if has_confidence else True)
    )
    filtered = data[valid_mask]
    return filtered, int((~valid_mask).sum())

def _failure_stats_by_group(raw: np.ndarray, filtered: np.ndarray):
    """
    Compute failure counts/percentages per (alpha, beta) group using raw vs. filtered data.
    """
    stats = []
    for (alpha, beta), _ in _iter_groups(raw):
        if beta is None:
            raw_mask = raw["alpha"] == alpha
            filt_mask = filtered["alpha"] == alpha if filtered.size else np.zeros(0, dtype=bool)
        else:
            raw_mask = (raw["alpha"] == alpha) & (raw["beta"] == beta)
            filt_mask = (
                (filtered["alpha"] == alpha) & (filtered["beta"] == beta)
                if filtered.size
                else np.zeros(0, dtype=bool)
            )
        total = int(raw_mask.sum())
        valid = int(filt_mask.sum()) if filtered.size else 0
        failed = total - valid
        pct_failed = 0.0 if total == 0 else 100.0 * failed / total
        stats.append(
            {
                "alpha": float(alpha),
                "beta": float(beta) if beta is not None else None,
                "total": total,
                "failed": failed,
                "pct_failed": pct_failed,
            }
        )
    return stats


def summarize_by_group(data: np.ndarray):
    """
    Compute simple means across MC runs for each (alpha, beta) group.
    """
    summaries = []
    names = set(data.dtype.names or [])
    has_range = _has_distance_range(data)
    has_confidence = {"confidence_mean", "confidence_std"}.issubset(names)
    for (alpha, beta), grp in _iter_groups(data):
        summaries.append(
            {
                "alpha": alpha,
                "beta": beta,
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
                "confidence_mean": float(np.mean(grp["confidence_mean"])) if has_confidence else None,
                "confidence_std": float(np.mean(grp["confidence_std"])) if has_confidence else None,
            }
        )
    return summaries


def mean_max_distance_by_group(data: np.ndarray):
    """
    Return list of dicts with mean of the maximum distance per group.
    """
    if not _has_distance_range(data):
        return []
    results = []
    for (alpha, beta), grp in _iter_groups(data):
        results.append(
            {
                "alpha": float(alpha),
                "beta": float(beta) if beta is not None else None,
                "dist_max_mean": float(np.mean(grp["distance_max"])),
            }
        )
    return results


def _group_label(alpha: float, beta: float | None) -> str:
    if beta is None:
        return f"alpha={alpha:.2f}"
    return f"alpha={alpha:.2f}, beta={beta:.2f}"


def _plot_metric(ax, groups, mean_key: str, std_key: str, ylabel: str):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
    for color, ((alpha, beta), grp) in zip(colors, groups):
        ax.errorbar(
            grp["mc_run"],
            grp[mean_key],
            yerr=grp[std_key],
            fmt="o-",
            capsize=3,
            label=_group_label(alpha, beta),
            color=color,
        )
    ax.set_xlabel("MC run")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def _distance_error_series(grp: np.ndarray):
    names = set(grp.dtype.names or [])
    if "d_init" in names:
        d_nom = float(np.mean(grp["d_init"]))
        return grp["distance_mean"] - d_nom, grp["distance_std"], d_nom
    return grp["distance_mean"], grp["distance_std"], None


def _plot_distance(ax, groups, has_range: bool):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
    for color, ((alpha, beta), grp) in zip(colors, groups):
        mc = grp["mc_run"]
        if has_range:
            ax.fill_between(mc, grp["distance_max"], color=color, alpha=0.15)
        dist_mean, dist_std, d_nom = _distance_error_series(grp)
        ax.errorbar(
            mc,
            dist_mean,
            yerr=dist_std,
            fmt="o-",
            capsize=3,
            label=_group_label(alpha, beta),
            color=color,
        )
    ax.set_xlabel("MC run")
    if d_nom is None:
        ylabel = "Distance [m]" if not has_range else "Distance [m] (mean ± std, min/max band)"
    else:
        ylabel = "Distance error [m] from d_init" if not has_range else "Distance error [m] (mean ± std, min/max band)"
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()


def _plot_summary_stats(groups, csv_path: Path, output: Path | None = None, show: bool = False) -> Path:
    labels = [_group_label(alpha, beta) for (alpha, beta), _ in groups]
    x = np.arange(len(groups))

    def stats(metric: str):
        means = [float(np.mean(grp[metric])) for _, grp in groups]
        stds = [float(np.std(grp[metric])) for _, grp in groups]
        return means, stds

    names = set(groups[0][1].dtype.names or [])
    has_confidence = {"confidence_mean", "confidence_std"}.issubset(names)

    fig_rows = 3 if has_confidence else 2
    fig, axes = plt.subplots(fig_rows, 2, figsize=(13, 10 if has_confidence else 8))
    fig.suptitle(f"Per-scenario mean/std from {csv_path.name}")

    metric_axes = [
        (axes[0, 0], "distance_mean", "Distance [m]"),
        (axes[0, 1], "a1_acc_mean", "Human accel [m/s^2]"),
        (axes[1, 0], "a2_acc_mean", "Robot accel [m/s^2]"),
        (axes[1, 1], "scenario_time", "Scenario time [s]"),
    ]
    if has_confidence:
        metric_axes.append((axes[2, 0], "confidence_mean", "Confidence"))

    for ax, metric, ylabel in metric_axes:
        means, stds = stats(metric)
        if metric == "distance_mean":
            if "d_init" in names:
                d_nom = [float(np.mean(grp["d_init"])) for _, grp in groups]
                means = [m - d for m, d in zip(means, d_nom)]
                ylabel = "Distance error [m] from d_init"
                ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    if has_confidence:
        axes[2, 1].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    output_path = output or csv_path.with_name(f"{csv_path.stem}_summary.png")
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_results(data: np.ndarray, csv_path: Path, output: Path | None = None, show: bool = False) -> Path:
    groups = list(_iter_groups(data))
    if not groups:
        raise ValueError(f"No rows found in {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Stats from {csv_path.name}")

    has_range = _has_distance_range(data)
    _plot_distance(axes[0], groups, has_range)
    _plot_metric(axes[1], groups, "a1_acc_mean", "a1_acc_std", "Human accel [m/s^2]")

    # Overlay robot accel on a second y-axis for clarity.
    ax2 = axes[1].twinx()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(groups)))
    for color, ((alpha, beta), grp) in zip(colors, groups):
        ax2.errorbar(
            grp["mc_run"],
            grp["a2_acc_mean"],
            yerr=grp["a2_acc_std"],
            fmt="s--",
            capsize=3,
            label=f"robot {_group_label(alpha, beta)}",
            color=color,
        )
    ax2.set_ylabel("Robot accel [m/s^2]")
    ax2.grid(False)
    ax2.legend(loc="lower right")

    _plot_metric(axes[2], groups, "scenario_time", "scenario_time_std", "Scenario time [s]")

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
        default=Path("Scenario_1_mc_stats.csv"),
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

    # Log scenario-level parameters if present in the CSV.
    param_keys = [
        "N",
        "dt_Solver",
        "dt_Sim",
        "a1_acc_limit",
        "Human_PreDefined_Traj",
        "Human_RandomWalk_Traj",
        "d_init",
        "Tf",
        "pp_theta",
        "pp_factor",
        "Nmc",
        "theta_des",
        "x1_init_x",
        "x1_init_y",
        "x2_init_x",
        "x2_init_y",
        "x1_des_x",
        "x1_des_y",
        "num_obstacles",
    ]
    names = set(data_raw.dtype.names or [])
    present_params = [k for k in param_keys if k in names]
    if present_params:
        print("Scenario parameters from CSV:")
        for key in present_params:
            values = np.unique(data_raw[key])
            if values.size == 1:
                print(f"  {key}={values[0]}")
            else:
                joined = ", ".join(str(v) for v in values[:6])
                suffix = "" if values.size <= 6 else ", ..."
                print(f"  {key} has {values.size} values: {joined}{suffix}")

    fail_stats = _failure_stats_by_group(data_raw, data)
    print("Failed run percentage per group:")
    for fs in fail_stats:
        label = _group_label(fs["alpha"], fs["beta"])
        print(f"  {label}: failed {fs['failed']}/{fs['total']} ({fs['pct_failed']:.1f}%)")

    has_range = _has_distance_range(data)
    if not has_range:
        print("No distance_min/distance_max columns found; plotting distance mean/std only.")

    dist_max_stats = mean_max_distance_by_group(data)
    if dist_max_stats:
        print("Mean max distance per group:")
        for item in dist_max_stats:
            label = _group_label(item["alpha"], item["beta"])
            print(f"  {label}: dist_max_mean={item['dist_max_mean']:.3f}")

    print("MC means per group:")
    for summary in summarize_by_group(data):
        label = _group_label(summary["alpha"], summary["beta"])
        parts = [
            f"{label} (n={summary['n']}):",
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
        if summary.get("confidence_mean") is not None:
            parts.append(f"confidence_mean={summary['confidence_mean']:.3f}")
        if summary.get("confidence_std") is not None:
            parts.append(f"confidence_std_mean={summary['confidence_std']:.3f}")
        print(" ".join(parts))

    output_path = plot_results(data, args.csv, output=args.out, show=args.show)
    summary_groups = list(_iter_groups(data))
    summary_path = _plot_summary_stats(summary_groups, csv_path=args.csv, show=args.show)
    print(f"Saved plot to {output_path}")
    print(f"Saved summary plot to {summary_path}")


if __name__ == "__main__":
    main()
