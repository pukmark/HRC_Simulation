import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


@dataclass
class Scenario:
    name: str
    x1_init: np.ndarray
    x2_init: np.ndarray
    x1_des: np.ndarray
    theta_des: float
    obstacles: List[Dict[str, np.ndarray]]
    Nmc: int = 1


def limited_cmd(v: np.ndarray, v_max: float) -> np.ndarray:
    if np.linalg.norm(v) > v_max:
        v = v / np.linalg.norm(v) * v_max
    return v


def capture_frame_agg(fig, canvas, w_target: int, h_target: int) -> np.ndarray:
    # Render with Agg (stable, off-screen)
    canvas.draw()
    buf, (w_cur, h_cur) = canvas.print_to_buffer()  # returns RGBA bytes
    frame = np.frombuffer(buf, dtype=np.uint8).reshape(h_cur, w_cur, 4)[:, :, :3]  # RGBA->RGB

    # Enforce exact size (pad or crop deterministically; no tiling)
    if (h_cur, w_cur) != (h_target, w_target):
        out = np.zeros((h_target, w_target, 3), dtype=np.uint8)
        hh = min(h_cur, h_target)
        ww = min(w_cur, w_target)
        out[:hh, :ww, :] = frame[:hh, :ww, :]
        frame = out
    return frame


def random_scenarios(
    num: int = 4,
    seed: int | None = None,
    workspace: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 10.0), (-5.0, 10.0)),
    min_separation: float = 2.5,
    max_separation: float = 3.5,
    obstacle_prob: float = 0.7,
    max_obstacles: int = 1,
) -> List[Scenario]:
    """
    Generate a list of randomized scenarios with reasonable spacing/obstacle placement.
    """
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    for i in range(num):
        base = np.array([[rng.uniform(*workspace[0]), rng.uniform(*workspace[1])]])
        sep_dist = rng.uniform(min_separation, max_separation)
        sep_angle = rng.uniform(0, 2 * math.pi)
        sep_vec = sep_dist * np.array([[math.cos(sep_angle), math.sin(sep_angle)]])

        x1_init = base
        x2_init = base + sep_vec

        x2_init[0, 0] = np.clip(x2_init[0, 0], *workspace[0])
        x2_init[0, 1] = np.clip(x2_init[0, 1], *workspace[1])

        x1_des = base + rng.uniform(2.0, 4.0, size=(1, 2)) * rng.choice([-1, 1], size=(1, 2))
        x1_des[0, 0] = np.clip(x1_des[0, 0], *workspace[0])
        x1_des[0, 1] = np.clip(x1_des[0, 1], *workspace[1])

        theta_des = rng.uniform(0, 2 * math.pi)

        obstacles: List[Dict[str, np.ndarray]] = []
        if rng.random() < obstacle_prob:
            n_obs = rng.integers(1, max_obstacles + 1)
            for _ in range(n_obs):
                pos = np.array([[rng.uniform(*workspace[0]), rng.uniform(*workspace[1])]])
                diam = float(rng.uniform(0.3, 0.7))
                obstacles.append({"Pos": pos, "diam": diam})

        scenarios.append(
            Scenario(
                name=f"Random_{i}",
                x1_init=x1_init,
                x2_init=x2_init,
                x1_des=x1_des,
                theta_des=theta_des,
                obstacles=obstacles,
            )
        )
    return scenarios


def init_plot_context(
    scenario: Scenario,
    game_sol,
    x1_init: np.ndarray,
    x2_init: np.ndarray,
    x1_des: np.ndarray,
    x2_des: np.ndarray,
    tf: float,
) -> Dict:
    fig_in = 6.3
    dpi = 160
    fig = plt.figure(figsize=(2 * fig_in, fig_in), dpi=dpi, constrained_layout=False)
    ax_xy = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, fig=fig)
    ax_xy.plot(x1_init[0, 0], x1_init[0, 1], "gs", markersize=6, label="Human Start")
    ax_xy.plot(x2_init[0, 0], x2_init[0, 1], "bs", markersize=6, label="Robot Start")
    ax_xy.plot(x1_des[0, 0], x1_des[0, 1], "go", markersize=6, label="Human Target")
    ax_xy.plot(x2_des[0, 0], x2_des[0, 1], "bo", markersize=6, label="Robot Target")
    p1_plot = ax_xy.plot([], [], "g^", markersize=6)[0]
    p2_plot = ax_xy.plot([], [], "b^", markersize=6)[0]
    p12_line = ax_xy.plot([], [], "r-", linewidth=3)[0]
    p12_line_pred = [ax_xy.plot([], [], "r:", linewidth=3)[0] for _ in range(4)]
    p12_line_hist = [ax_xy.plot([], [], "r:", linewidth=3)[0] for _ in range(20)]
    p1_pred = ax_xy.plot([], [], "g--", linewidth=3)[0]
    p2_pred = ax_xy.plot([], [], "b--", linewidth=3)[0]
    p1_pred_positive = ax_xy.plot([], [], "g--", linewidth=3)[0]
    p1_pred_negative = ax_xy.plot([], [], "g:", linewidth=3)[0]
    p2_pred_positive = ax_xy.plot([], [], "k--", linewidth=1)[0]
    p2_pred_negative = ax_xy.plot([], [], "k:", linewidth=1)[0]
    p1_hist = ax_xy.plot([], [], "g-", linewidth=3)[0]
    p2_hist = ax_xy.plot([], [], "b-", linewidth=3)[0]
    tgt1_plot = ax_xy.plot([], [], "go", markersize=6)[0]
    tgt2_plot = ax_xy.plot([], [], "bo", markersize=6)[0]
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("X Position [m]")
    ax_xy.set_ylabel("Y Position [m]")
    ax_xy.set_xlim(game_sol.xlim)
    ax_xy.set_ylim(game_sol.ylim)
    ax_xy.grid(True)
    ax_xy.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    for obstacle in scenario.obstacles:
        x = obstacle["diam"] / 2 * np.cos(np.linspace(0, 2 * np.pi, 100))
        y = obstacle["diam"] / 2 * np.sin(np.linspace(0, 2 * np.pi, 100))
        ax_xy.plot(obstacle["Pos"][0, 0] + x, obstacle["Pos"][0, 1] + y, "k-", linewidth=3)

    ax_vel = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1, fig=fig)
    p1_vel = ax_vel.plot([], [], "gs", markersize=6)[0]
    p2_vel = ax_vel.plot([], [], "bs", markersize=6)[0]
    p1_vel_pred = ax_vel.plot([], [], "g-.", markersize=3)[0]
    p2_vel_pred = ax_vel.plot([], [], "b-.", markersize=3)[0]
    p1_vel_hist = ax_vel.plot([], [], "g-", markersize=3)[0]
    p2_vel_hist = ax_vel.plot([], [], "b-", markersize=3)[0]
    ax_vel.set_xlim([0, tf])
    ax_vel.set_ylim([0, max(game_sol.v1_max, game_sol.v2_max) + 1])
    ax_vel.plot([0, tf], [game_sol.v1_max, game_sol.v1_max], "g:", markersize=3)
    ax_vel.plot([0, tf], [game_sol.v2_max, game_sol.v2_max], "b:", markersize=3)
    ax_vel.grid(True)
    ax_vel.set_title("Velocity [m/s]")

    ax_acc = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1, fig=fig)
    p1_acc = ax_acc.plot([], [], "gs", markersize=6)[0]
    p2_acc = ax_acc.plot([], [], "bs", markersize=6)[0]
    p1_acc_pred = ax_acc.plot([], [], "g-.", markersize=3)[0]
    p2_acc_pred = ax_acc.plot([], [], "b-.", markersize=3)[0]
    p1_acc_hist = ax_acc.plot([], [], "g-", markersize=3)[0]
    p2_acc_hist = ax_acc.plot([], [], "b-", markersize=3)[0]
    ax_acc.set_xlim([0, tf])
    ax_acc.set_ylim([0, max(game_sol.a1_max, game_sol.a2_max) + 1])
    ax_acc.plot([0, tf], [game_sol.a1_max, game_sol.a1_max], "g:", markersize=3)
    ax_acc.plot([0, tf], [game_sol.a2_max, game_sol.a2_max], "b:", markersize=3)
    ax_acc.grid(True)
    ax_acc.set_title("Acceleration [m/s^2]")

    ax_dist = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1, fig=fig)
    p12_dist = ax_dist.plot([], [], "rs", markersize=6)[0]
    p12_dist_pred = ax_dist.plot([], [], "b-.", markersize=3)[0]
    p12_dist_hist = ax_dist.plot([], [], "b-", markersize=3)[0]
    ax_dist.plot([0, tf], [game_sol.d_min, game_sol.d_min], "b:", markersize=3)
    ax_dist.plot([0, tf], [game_sol.d_max, game_sol.d_max], "b:", markersize=3)
    ax_dist.set_xlim([0, tf])
    ax_dist.set_ylim([game_sol.d_min - 0.1, game_sol.d_max + 0.1])
    ax_dist.grid(True)
    ax_dist.set_xlabel("Time [Sec]")
    ax_dist.set_title("Distance [m]")

    ax_sigL = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1, fig=fig)
    p_sigL = [ax_sigL.plot([], [], "ks", markersize=3)[0] for _ in range(game_sol.sol.sigL.shape[1])]
    ax_sigL.set_xlim([-1.0, max(1.0, game_sol.sol.sigL.shape[0])])
    ax_sigL.set_ylim([-0.1, 1.1])
    ax_sigL.grid(True)
    ax_sigL.set_xlabel("Prediction Step [-]")
    ax_sigL.set_ylabel("sigL Multipliers")

    ax_sigOA = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1, fig=fig)
    p_sigOA = [ax_sigOA.plot([], [], "ks", markersize=3)[0] for _ in range(game_sol.sol.sigOA.shape[1])]
    ax_sigOA.set_xlim([-1.0, max(1.0, game_sol.sol.sigOA.shape[0])])
    ax_sigOA.set_ylim([-0.1, 1.1])
    ax_sigOA.grid(True)
    ax_sigOA.set_xlabel("Prediction Step [-]")
    ax_sigOA.set_ylabel("sigOA Multipliers")

    canvas = FigureCanvas(fig)
    w_target = int(2 * fig_in * dpi)
    h_target = int(fig_in * dpi)

    return {
        "Frames": [],
        "fig": fig,
        "canvas": canvas,
        "w_target": w_target,
        "h_target": h_target,
        "ax_xy": ax_xy,
        "ax_vel": ax_vel,
        "ax_acc": ax_acc,
        "ax_dist": ax_dist,
        "ax_sigL": ax_sigL,
        "ax_sigOA": ax_sigOA,
        "p1_plot": p1_plot,
        "p2_plot": p2_plot,
        "p12_line": p12_line,
        "p12_line_pred": p12_line_pred,
        "p12_line_hist": p12_line_hist,
        "p1_pred": p1_pred,
        "p2_pred": p2_pred,
        "p1_pred_positive": p1_pred_positive,
        "p1_pred_negative": p1_pred_negative,
        "p2_pred_positive": p2_pred_positive,
        "p2_pred_negative": p2_pred_negative,
        "p1_hist": p1_hist,
        "p2_hist": p2_hist,
        "tgt1_plot": tgt1_plot,
        "tgt2_plot": tgt2_plot,
        "p1_vel": p1_vel,
        "p2_vel": p2_vel,
        "p1_vel_pred": p1_vel_pred,
        "p2_vel_pred": p2_vel_pred,
        "p1_vel_hist": p1_vel_hist,
        "p2_vel_hist": p2_vel_hist,
        "p1_acc": p1_acc,
        "p2_acc": p2_acc,
        "p1_acc_pred": p1_acc_pred,
        "p2_acc_pred": p2_acc_pred,
        "p1_acc_hist": p1_acc_hist,
        "p2_acc_hist": p2_acc_hist,
        "p12_dist": p12_dist,
        "p12_dist_pred": p12_dist_pred,
        "p12_dist_hist": p12_dist_hist,
        "p_sigL": p_sigL,
        "p_sigOA": p_sigOA,
    }


def update_plot_context(
    plot_context: Dict,
    t: float,
    x1_state: np.ndarray,
    x2_state: np.ndarray,
    v1_state: np.ndarray,
    v2_state: np.ndarray,
    a1_cmd: np.ndarray,
    a2_cmd: np.ndarray,
    x1_hist: np.ndarray,
    x2_hist: np.ndarray,
    v1_hist: np.ndarray,
    v2_hist: np.ndarray,
    a1_hist: np.ndarray,
    a2_hist: np.ndarray,
    t_hist: np.ndarray,
    game_sol,
    solver_type: str,
    scenario: Scenario,
    n_mc: int,
    x1_des: np.ndarray,
    x2_des: np.ndarray,
    dt_solver: float,
    dt_sim: float,
    max_payload_penetration: float,
) -> None:
    fig = plot_context["fig"]
    canvas = plot_context["canvas"]
    w_target = plot_context["w_target"]
    h_target = plot_context["h_target"]
    ax_xy = plot_context["ax_xy"]
    ax_vel = plot_context["ax_vel"]
    ax_acc = plot_context["ax_acc"]
    ax_dist = plot_context["ax_dist"]
    ax_sigL = plot_context["ax_sigL"]
    ax_sigOA = plot_context["ax_sigOA"]
    p1_plot = plot_context["p1_plot"]
    p2_plot = plot_context["p2_plot"]
    p12_line = plot_context["p12_line"]
    p12_line_pred = plot_context["p12_line_pred"]
    p12_line_hist = plot_context["p12_line_hist"]
    p1_pred = plot_context["p1_pred"]
    p2_pred = plot_context["p2_pred"]
    p1_pred_positive = plot_context["p1_pred_positive"]
    p1_pred_negative = plot_context["p1_pred_negative"]
    p2_pred_positive = plot_context["p2_pred_positive"]
    p2_pred_negative = plot_context["p2_pred_negative"]
    p1_hist = plot_context["p1_hist"]
    p2_hist = plot_context["p2_hist"]
    tgt1_plot = plot_context["tgt1_plot"]
    tgt2_plot = plot_context["tgt2_plot"]
    p1_vel = plot_context["p1_vel"]
    p2_vel = plot_context["p2_vel"]
    p1_vel_pred = plot_context["p1_vel_pred"]
    p2_vel_pred = plot_context["p2_vel_pred"]
    p1_vel_hist = plot_context["p1_vel_hist"]
    p2_vel_hist = plot_context["p2_vel_hist"]
    p1_acc = plot_context["p1_acc"]
    p2_acc = plot_context["p2_acc"]
    p1_acc_pred = plot_context["p1_acc_pred"]
    p2_acc_pred = plot_context["p2_acc_pred"]
    p1_acc_hist = plot_context["p1_acc_hist"]
    p2_acc_hist = plot_context["p2_acc_hist"]
    p12_dist = plot_context["p12_dist"]
    p12_dist_pred = plot_context["p12_dist_pred"]
    p12_dist_hist = plot_context["p12_dist_hist"]
    p_sigL = plot_context["p_sigL"]
    p_sigOA = plot_context["p_sigOA"]
    frames = plot_context["Frames"]

    p1_plot.set_data([x1_state[0, 0]], [x1_state[0, 1]])
    p2_plot.set_data([x2_state[0, 0]], [x2_state[0, 1]])
    p12_line.set_data([x1_state[0, 0], x2_state[0, 0]], [x1_state[0, 1], x2_state[0, 1]])
    indx = np.linspace(0, game_sol.N, 1 + len(p12_line_pred))
    for i in range(len(p12_line_pred)):
        i_indx = int(indx[i + 1])
        p12_line_pred[i].set_data(
            [game_sol.sol.x1[i_indx, 0], game_sol.sol.x2[i_indx, 0]],
            [game_sol.sol.x1[i_indx, 1], game_sol.sol.x2[i_indx, 1]],
        )
    p1_pred.set_data(game_sol.sol.x1[:, 0], game_sol.sol.x1[:, 1])
    p2_pred.set_data(game_sol.sol.x2[:, 0], game_sol.sol.x2[:, 1])

    p1_hist.set_data(x1_hist[:, 0], x1_hist[:, 1])
    p2_hist.set_data(x2_hist[:, 0], x2_hist[:, 1])
    tgt1_plot.set_data([x1_des[0, 0]], [x1_des[0, 1]])
    tgt2_plot.set_data([x2_des[0, 0]], [x2_des[0, 1]])
    ax_xy.set_title(
    rf"Time: {t:3.3f} [s], "
    rf"$\alpha$ = {game_sol.sol.alpha:.3f}, "
    rf"$\beta$ = {game_sol.sol.beta:.3f}, "
    rf"Confidence = {game_sol.sol.confidence:.3f}, "
    rf"Penetration = {max_payload_penetration:.3f}"
)
    ax_xy.legend(ncol=4, loc="upper center")

    t_pred = np.linspace(game_sol.sol.time - dt_solver, t - dt_solver + game_sol.N * dt_solver, game_sol.N + 1)
    p1_vel.set_data([t], [np.linalg.norm(v1_state)])
    p2_vel.set_data([t], [np.linalg.norm(v2_state)])
    p1_vel_pred.set_data(t_pred, np.linalg.norm(game_sol.sol.v1, axis=1))
    p2_vel_pred.set_data(t_pred, np.linalg.norm(game_sol.sol.v2, axis=1))
    p1_vel_hist.set_data(t_hist, np.linalg.norm(v1_hist, axis=1))
    p2_vel_hist.set_data(t_hist, np.linalg.norm(v2_hist, axis=1))

    p1_acc.set_data([t], [np.linalg.norm(a1_cmd)])
    p2_acc.set_data([t], [np.linalg.norm(a2_cmd)])
    p1_acc_pred.set_data(t_pred[:-1], np.linalg.norm(game_sol.sol.a1, axis=1))
    p2_acc_pred.set_data(t_pred[:-1], np.linalg.norm(game_sol.sol.a2, axis=1))
    p1_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a1_hist, axis=1))
    p2_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a2_hist, axis=1))

    p12_dist.set_data([t], [np.linalg.norm(x1_state - x2_state)])
    p12_dist_pred.set_data(t_pred, np.linalg.norm(game_sol.sol.x1 - game_sol.sol.x2, axis=1))
    p12_dist_hist.set_data(t_hist[:, 0], np.linalg.norm(x1_hist - x2_hist, axis=1))
    if solver_type == "DG":
        for i in range(len(p_sigL)):
            p_sigL[i].set_data(
                [np.linspace(0, game_sol.sol.sigL.shape[0] - 1, game_sol.sol.sigL.shape[0])],
                [game_sol.sol.sigL[:, i]],
            )
        ax_sigL.set_ylim([-0.1, np.max([1.0, 1.1 * np.max(game_sol.sol.sigL)])])
        if scenario.obstacles:
            for i in range(len(p_sigOA)):
                p_sigOA[i].set_data(
                    [np.linspace(0, game_sol.sol.sigOA.shape[0] - 1, game_sol.sol.sigOA.shape[0])],
                    [game_sol.sol.sigOA[:, i]],
                )
            ax_sigOA.set_ylim([-0.1, np.max([1.0, 1.1 * np.max(game_sol.sol.sigOA)])])

    step = max(1, int(dt_solver / dt_sim))
    if t_hist.shape[0] % (10 * step) == 0:
        k = t_hist.shape[0] // (10 * step) - 1
        p12_line_hist[k].set_data([x1_hist[-1, 0], x2_hist[-1, 0]], [x1_hist[-1, 1], x2_hist[-1, 1]])

    plt.pause(0.1)

    fig.canvas.draw()
    if n_mc == 0:
        frame = capture_frame_agg(fig, canvas, w_target, h_target)
        frames.append(frame)


def finalize_plot_context(
    plot_context: Dict,
    scenario: Scenario,
    ialpha: int,
    alpha: float,
    beta: float,
    hist_col: List[str],
    t_hist: np.ndarray,
    v1_hist: np.ndarray,
    v2_hist: np.ndarray,
    a1_hist: np.ndarray,
    a2_hist: np.ndarray,
    x1_hist: np.ndarray,
    x2_hist: np.ndarray,
) -> None:
    fig = plot_context["fig"]
    ax_xy = plot_context["ax_xy"]
    ax_vel = plot_context["ax_vel"]
    ax_acc = plot_context["ax_acc"]
    ax_dist = plot_context["ax_dist"]
    p12_line_pred = plot_context["p12_line_pred"]
    p12_line_hist = plot_context["p12_line_hist"]
    p1_pred = plot_context["p1_pred"]
    p2_pred = plot_context["p2_pred"]
    p1_hist = plot_context["p1_hist"]
    p2_hist = plot_context["p2_hist"]
    p1_vel = plot_context["p1_vel"]
    p2_vel = plot_context["p2_vel"]
    p1_vel_pred = plot_context["p1_vel_pred"]
    p2_vel_pred = plot_context["p2_vel_pred"]
    p1_vel_hist = plot_context["p1_vel_hist"]
    p2_vel_hist = plot_context["p2_vel_hist"]
    p1_acc_pred = plot_context["p1_acc_pred"]
    p2_acc_pred = plot_context["p2_acc_pred"]
    p1_acc = plot_context["p1_acc"]
    p2_acc = plot_context["p2_acc"]
    p12_dist_pred = plot_context["p12_dist_pred"]

    for line in p12_line_pred:
        line.set_data([], [])
    for line in p12_line_hist:
        line.set_data([], [])

    if fig is not None:
        fig.savefig(f"{scenario.name}_traj_final.png")

    ax_xy.plot(x1_hist[:, 0], x1_hist[:, 1], "--", color=hist_col[ialpha], linewidth=2, label=rf"$(\alpha, \beta)=({alpha}, {beta})$")
    ax_xy.plot(x2_hist[:, 0], x2_hist[:, 1], "-", color=hist_col[ialpha], linewidth=2)
    ax_xy.legend(ncol=4, loc="upper center")
    ax_xy.set_title("Trajectories")
    if ialpha == 0:
        ax_vel.plot(t_hist, np.linalg.norm(v1_hist, axis=1), "--", color=hist_col[ialpha], linewidth=2, label="Human")
        ax_vel.plot(t_hist, np.linalg.norm(v2_hist, axis=1), "-", color=hist_col[ialpha], linewidth=2, label="Robot")
    else:
        ax_vel.plot(t_hist, np.linalg.norm(v1_hist, axis=1), "--", color=hist_col[ialpha], linewidth=2)
        ax_vel.plot(t_hist, np.linalg.norm(v2_hist, axis=1), "-", color=hist_col[ialpha], linewidth=2)

    p1_vel.set_data([], [])
    p2_vel.set_data([], [])
    p1_vel_pred.set_data([], [])
    p2_vel_pred.set_data([], [])
    ax_vel.legend()
    ax_acc.plot(t_hist[:-1], np.linalg.norm(a1_hist, axis=1), "--", color=hist_col[ialpha], linewidth=2)
    ax_acc.plot(t_hist[:-1], np.linalg.norm(a2_hist, axis=1), "-", color=hist_col[ialpha], linewidth=2)
    ax_dist.plot(t_hist, np.linalg.norm(x1_hist - x2_hist, axis=1), "-", color=hist_col[ialpha], linewidth=2)
    p1_pred.set_data([], [])
    p2_pred.set_data([], [])
    p1_hist.set_data([], [])
    p2_hist.set_data([], [])
    p1_vel_pred.set_data([], [])
    p2_vel_pred.set_data([], [])
    p1_vel_hist.set_data([], [])
    p2_vel_hist.set_data([], [])
    p1_vel.set_data([], [])
    p2_vel.set_data([], [])
    p1_acc_pred.set_data([], [])
    p2_acc_pred.set_data([], [])
    p1_acc.set_data([], [])
    p2_acc.set_data([], [])
    p12_dist_pred.set_data([], [])

    for k in range(9, np.shape(x1_hist)[0] - 1, 10):
        ax_xy.plot([x1_hist[k, 0], x2_hist[k, 0]], [x1_hist[k, 1], x2_hist[k, 1]], ":", color=hist_col[ialpha], linewidth=1)
    if fig is not None:
        fig.savefig(f"{scenario.name}_final.png")


def write_frames_to_mp4(plot_context: Dict, scenario_name: str) -> None:
    if not plot_context:
        return
    frames = plot_context.get("Frames", [])
    fig = plot_context.get("fig")
    if fig is not None and frames:
        writer = imageio.get_writer(
            f"{scenario_name}.mp4",
            fps=3,
            codec="libx264",
            quality=8,
            macro_block_size=16,
        )
        w_target = plot_context["w_target"]
        h_target = plot_context["h_target"]
        for frame in frames:
            assert frame.shape == (h_target, w_target, 3), f"Bad frame shape {frame.shape}"
            writer.append_data(frame)
        writer.close()
    if fig is not None:
        plt.close(fig)
