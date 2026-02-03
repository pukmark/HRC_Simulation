#!/usr/bin/env python3
import os
os.system('clear')

import numpy as np
from scipy.interpolate import interp1d
import math
from dataclasses import dataclass

import CollaborativeGameSolver as GameSolver
import CollaborativeGameGeneralSolver as GeneralGameSolver

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
import sys



Tf = 15.0
alpha_vec = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
beta_vec = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
dalpha, dbeta = 0.1, 0.1

N = 10
dt_Solver = 0.1
dt_Sim = 0.1
a1_acc_limit = 10.0
Human_PreDefined_Traj = True
Human_RandomWalk_Traj = False


@dataclass
class Scenario:
    name: str
    x1_init: np.ndarray
    x2_init: np.ndarray
    x1_des: np.ndarray
    theta_des: float
    obstacles: List[Dict[str, np.ndarray]]
    Nmc: int = 1

# Define the initial conditions
Scenarios: List[Scenario] =[
# Scenarios.append({'Name': 'Scenario_1', 'x1_init': np.array([[5, 1]]), 'x2_init':np.array([[4, -3]]), 'x1_des': np.array([[3, 8]]), 'theta_des':np.deg2rad(0), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_1_WithObs', 'x1_init': np.array([[5, 1]]), 'x2_init':np.array([[4, -3]]), 'x1_des': np.array([[3, 8]]), 'theta_des':np.deg2rad(0), 'Obs': [{'Pos': np.array([[7,4]]), "diam": 0.5}]})
# Scenarios.append({'Name': 'Scenario_1_switch', 'x1_init': np.array([[4, -3]]), 'x2_init':np.array([[5,1]]), 'x1_des': np.array([[3, 8]]), 'theta_des':np.deg2rad(0), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_2', 'x1_init': np.array([[4, -3]]), 'x2_init':np.array([[4, 0]]), 'x1_des': np.array([[7, 5]]), 'theta_des':np.deg2rad(180), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_3', 'x1_init': np.array([[4, -3]]), 'x2_init':np.array([[6, -5]]), 'x1_des': np.array([[7, 5]]), 'theta_des':np.deg2rad(180), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_3_WithObs', 'x1_init': np.array([[4, -3]]), 'x2_init':np.array([[6, -5]]), 'x1_des': np.array([[7, 5]]), 'theta_des':np.deg2rad(180), 'Obs': Obstcles})
# Scenarios.append({'Name': 'Scenario_4', 'x1_init': np.array([[6, -5]]), 'x2_init':np.array([[4, -3]]), 'x1_des': np.array([[7, 5]]), 'theta_des':np.deg2rad(180), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_4_WithObs', 'x1_init': np.array([[6, -5]]), 'x2_init':np.array([[4, -3]]), 'x1_des': np.array([[7, 5]]), 'theta_des':np.deg2rad(180), 'Obs': Obstcles})
# Scenarios.append({'Name': 'Scenario_5', 'x1_init': np.array([[5, 2]]), 'x2_init':np.array([[2, 2]]), 'x1_des': np.array([[5, 8]]), 'theta_des':np.deg2rad(0), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_6', 'x1_init': np.array([[5, 0]]), 'x2_init':np.array([[2, 0]]), 'x1_des': np.array([[6, 0]]), 'theta_des':np.deg2rad(0), 'Obs': []})
# Scenarios.append({'Name': 'Scenario_6_Switch_WithObs', 'x1_init': np.array([[0, 0]]), 'x2_init':np.array([[3, 0]]), 'x1_des': np.array([[7, 4]]), 'theta_des':np.deg2rad(240), 'Obs': [{'Pos': np.array([[4.5,2]]), "diam": 0.5}]})

# Scenario(name="Scenario_1", x1_init=np.array([[-4, 2.0]]), x2_init=np.array([[-4.0+6*np.cos(np.deg2rad(225)), 2.0+6*np.sin(np.deg2rad(225))]]), x1_des=np.array([[6.0, -2.0]]), theta_des=np.deg2rad(90.0), obstacles=[], Nmc=1),
# Scenario(name="Scenario_2", x1_init=np.array( [[-1.5, 6.5]]), x2_init=np.array([[-1.5+3*np.cos(np.deg2rad(225)), 6.5+3*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.0, 4.0]]), "diam": 1.0}], Nmc=5),
Scenario(name="Scenario_2", x1_init=np.array( [[-1.5, 6.5]]), x2_init=np.array([[-1.5+7*np.cos(np.deg2rad(225)), 6.5+7*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, -.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.5, 0.0]]), "diam": 3.0}], Nmc=50),
# Scenario(name="Scenario_3", x1_init=np.array([[-1.5, 6.5]]), x2_init=np.array([[-1.5+3*np.cos(np.deg2rad(225)), 6.5+3*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.0, 4.0]]), "diam": 1.0}]),
# Scenario(name="Scenario_6_WithObs", x1_init=np.array([[3.0, 0.0]]), x2_init=np.array([[0.0, 0.0]]), x1_des=np.array([[7.0, 0.0]]), theta_des=np.deg2rad(60.0), obstacles=[{"Pos": np.array([[6.0, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_6_Switch_WithObs", x1_init=np.array([[0.0, 0.0]]), x2_init=np.array([[3.0, 0.0]]), x1_des=np.array([[7.0, 4.0]]), theta_des=np.deg2rad(240.0), obstacles=[{"Pos": np.array([[4.5, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_7", x1_init=np.array([[2.0, -1.0]]), x2_init=np.array([[5.0, -1.0]]), x1_des=np.array([[2.0, 8.0]]), theta_des=np.deg2rad(0.0), obstacles=[{"Pos": np.array([[3.0, 4.0]]), "diam": 0.5}]),
]

def LimitedCmd(v: np.array, v_max: float):
    if np.linalg.norm(v) > v_max:
        v = v / np.linalg.norm(v) * v_max

    return v

def capture_frame_agg(fig, canvas, w_target, h_target):
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
    seed: int = None,
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
        # Sample a base point and a separation vector
        base = np.array(
            [
                [rng.uniform(*workspace[0]), rng.uniform(*workspace[1])]
            ]
        )
        sep_dist = rng.uniform(min_separation, max_separation)
        sep_angle = rng.uniform(0, 2 * math.pi)
        sep_vec = sep_dist * np.array([[math.cos(sep_angle), math.sin(sep_angle)]])

        x1_init = base
        x2_init = base + sep_vec

        # Clamp into workspace
        x2_init[0, 0] = np.clip(x2_init[0, 0], *workspace[0])
        x2_init[0, 1] = np.clip(x2_init[0, 1], *workspace[1])

        # Sample goal near top/right-ish to keep motion meaningful
        x1_des = base + rng.uniform(2.0, 4.0, size=(1, 2)) * rng.choice([-1, 1], size=(1, 2))
        x1_des[0, 0] = np.clip(x1_des[0, 0], *workspace[0])
        x1_des[0, 1] = np.clip(x1_des[0, 1], *workspace[1])

        theta_des = rng.uniform(0, 2 * math.pi)

        obstacles: List[Dict[str, np.ndarray]] = []
        if rng.random() < obstacle_prob:
            n_obs = rng.integers(1, max_obstacles + 1)
            for _ in range(n_obs):
                pos = np.array(
                    [
                        [
                            rng.uniform(*workspace[0]),
                            rng.uniform(*workspace[1]),
                        ]
                    ]
                )
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

RT_Plot = True

hist_col = ['c','m','k','y','r']

Obstcles = []
Obstcles.append({'Pos': np.array([[6,0]]), "diam": 0.5})

Scenarios = Scenarios + random_scenarios(num=0)

pp_theta = np.deg2rad(180.0)
# pp_theta = np.atan2( (Scenarios[0].x1_des[0,1]-Scenarios[0].x1_init[0,1]), (Scenarios[0].x1_des[0,0]-Scenarios[0].x1_init[0,0]) )
pp_factor = 0.5

def run_single_mc(
    Scenario: Scenario,
    alpha: float,
    beta: float,
    ialpha: int,
    n_mc: int,
    SolverType: str,
    plot_context: Dict | None = None,
):
    # Some environments mark stdout as non-blocking in subprocesses; force blocking to avoid spurious errors.
    try:
        os.set_blocking(sys.stdout.fileno(), True)
    except (AttributeError, OSError, ValueError):
        pass

    np.random.seed(100 + n_mc)
    x1_init, x2_init, x1_des, theta_des = Scenario.x1_init, Scenario.x2_init, Scenario.x1_des, Scenario.theta_des
    Obstcles = Scenario.obstacles

    d_init = np.linalg.norm(x1_init-x2_init)
    x2_des = x1_des + d_init * np.array([np.cos(theta_des), np.sin(theta_des)])

    GameSol = GameSolver.CollaborativeGame(N=N, dt=dt_Solver, d=d_init, Obstcles=Scenario.obstacles)
    GeneralGameSol = GeneralGameSolver.CollaborativeGame(N=N, dt=dt_Solver, d=d_init, Obstcles=Scenario.obstacles)

    rt_plot = RT_Plot and plot_context is not None
    if rt_plot and plot_context.get('fig') is None:
        plot_context.setdefault('Frames', [])
        FIG_INCHES = 6.3
        DPI = 160
        fig = plt.figure(figsize=(2*FIG_INCHES, FIG_INCHES), dpi=DPI, constrained_layout=False)
        ax_xy = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2, fig=fig)
        ax_xy.plot(x1_init[0, 0], x1_init[0, 1], 'gs', markersize=6, label='Human Start')
        ax_xy.plot(x2_init[0, 0], x2_init[0, 1], 'bs', markersize=6, label='Robot Start')
        ax_xy.plot(x1_des[0, 0], x1_des[0, 1], 'go', markersize=6, label='Human Target')
        ax_xy.plot(x2_des[0, 0], x2_des[0, 1], 'bo', markersize=6, label='Robot Target')
        p1_plot = ax_xy.plot([], [], 'g^', markersize=6)[0]
        p2_plot = ax_xy.plot([], [], 'b^', markersize=6)[0]
        p12_line = ax_xy.plot([], [], 'r-', linewidth=3)[0]
        p12_line_pred = []
        for _ in range(4):
            p12_line_pred.append(ax_xy.plot([], [], 'r:', linewidth=3)[0])
        p12_line_hist = []
        for _ in range(20):
            p12_line_hist.append(ax_xy.plot([], [], 'r:', linewidth=3)[0])
        p1_pred = ax_xy.plot([], [], 'g--', linewidth=3)[0]
        p2_pred = ax_xy.plot([], [], 'b--', linewidth=3)[0]
        p1_pred_positive = ax_xy.plot([], [], 'g--', linewidth=3)[0]
        p1_pred_negative = ax_xy.plot([], [], 'g:', linewidth=3)[0]
        p2_pred_positive = ax_xy.plot([], [], 'k--', linewidth=1)[0]
        p2_pred_negative = ax_xy.plot([], [], 'k:', linewidth=1)[0]
        p1_hist = ax_xy.plot([], [], 'g-', linewidth=3)[0]
        p2_hist = ax_xy.plot([], [], 'b-', linewidth=3)[0]
        tgt1_plot = ax_xy.plot([], [], 'go', markersize=6)[0]
        tgt2_plot = ax_xy.plot([], [], 'bo', markersize=6)[0]
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel('X Position [m]')
        ax_xy.set_ylabel('Y Position [m]')
        xy_points = np.vstack((x1_init, x2_init, x1_des, x2_des))
        if Obstcles:
            obs_positions = np.vstack([obs['Pos'] for obs in Obstcles])
            xy_points = np.vstack((xy_points, obs_positions))
        ax_xy.set_xlim(GameSol.xlim)
        ax_xy.set_ylim(GameSol.ylim)
        ax_xy.grid(True)
        ax_xy.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05))
        for Obstcle in Obstcles:
            x, y = Obstcle['diam'] / 2 * np.cos(np.linspace(0, 2 * np.pi, 100)), Obstcle['diam'] / 2 * np.sin(np.linspace(0, 2 * np.pi, 100))
            ax_xy.plot(Obstcle['Pos'][0, 0] + x, Obstcle['Pos'][0, 1] + y, 'k-', linewidth=3)

        ax_vel = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1, fig=fig)
        p1_vel = ax_vel.plot([], [], 'gs', markersize=6)[0]
        p2_vel = ax_vel.plot([], [], 'bs', markersize=6)[0]
        p1_vel_pred = ax_vel.plot([], [], 'g-.', markersize=3)[0]
        p2_vel_pred = ax_vel.plot([], [], 'b-.', markersize=3)[0]
        p1_vel_hist = ax_vel.plot([], [], 'g-', markersize=3)[0]
        p2_vel_hist = ax_vel.plot([], [], 'b-', markersize=3)[0]
        ax_vel.set_xlim([0, Tf])
        ax_vel.set_ylim([0, max(GameSol.v1_max, GameSol.v2_max) + 1])
        ax_vel.plot([0, Tf], [GameSol.v1_max, GameSol.v1_max], 'g:', markersize=3)
        ax_vel.plot([0, Tf], [GameSol.v2_max, GameSol.v2_max], 'b:', markersize=3)
        ax_vel.grid(True)
        ax_vel.set_title('Velocity [m/s]')

        ax_acc = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1, fig=fig)
        p1_acc = ax_acc.plot([], [], 'gs', markersize=6)[0]
        p2_acc = ax_acc.plot([], [], 'bs', markersize=6)[0]
        p1_acc_pred = ax_acc.plot([], [], 'g-.', markersize=3)[0]
        p2_acc_pred = ax_acc.plot([], [], 'b-.', markersize=3)[0]
        p1_acc_hist = ax_acc.plot([], [], 'g-', markersize=3)[0]
        p2_acc_hist = ax_acc.plot([], [], 'b-', markersize=3)[0]
        ax_acc.set_xlim([0, Tf])
        ax_acc.set_ylim([0, max(GameSol.a1_max, GameSol.a2_max) + 1])
        ax_acc.plot([0, Tf], [GameSol.a1_max, GameSol.a1_max], 'g:', markersize=3)
        ax_acc.plot([0, Tf], [GameSol.a2_max, GameSol.a2_max], 'b:', markersize=3)
        ax_acc.grid(True)
        # ax_acc.legend()
        ax_acc.set_title('Acceleration [m/s^2]')

        ax_dist = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1, fig=fig)
        p12_dist = ax_dist.plot([], [], 'rs', markersize=6)[0]
        p12_dist_pred = ax_dist.plot([], [], 'b-.', markersize=3)[0]
        p12_dist_hist = ax_dist.plot([], [], 'b-', markersize=3)[0]
        ax_dist.plot([0, Tf], [GameSol.d_min, GameSol.d_min], 'b:', markersize=3)
        ax_dist.plot([0, Tf], [GameSol.d_max, GameSol.d_max], 'b:', markersize=3)
        ax_dist.set_xlim([0, Tf])
        ax_dist.set_ylim([GameSol.d_min - 0.1, GameSol.d_max + 0.1])
        ax_dist.grid(True)
        ax_dist.set_xlabel('Time [Sec]')
        ax_dist.set_title('Distance [m]')
        
        ax_sigL = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1, fig=fig)
        p_sigL = []
        for _ in range(GameSol.sol.sigL.shape[1]):
            p_sigL.append(ax_sigL.plot([], [], 'ks', markersize=3)[0])
        p_sigOA = []
        ax_sigL.set_xlim([-1.0, max(1.0, GameSol.sol.sigL.shape[0])])
        ax_sigL.set_ylim([-0.1, 1.1])
        ax_sigL.grid(True)
        ax_sigL.set_xlabel('Prediction Step [-]')
        ax_sigL.set_ylabel('sigL Multipliers')
        
        ax_sigOA = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1, fig=fig)
        p_sigOA = []
        for _ in range(GameSol.sol.sigOA.shape[1]):
            p_sigOA.append(ax_sigOA.plot([], [], 'ks', markersize=3)[0])
        ax_sigOA.set_xlim([-1.0, max(1.0, GameSol.sol.sigOA.shape[0])])
        ax_sigOA.set_ylim([-0.1, 1.1])
        ax_sigOA.grid(True)
        ax_sigOA.set_xlabel('Prediction Step [-]')
        ax_sigOA.set_ylabel('sigOA Multipliers')        

        canvas = FigureCanvas(fig)
        w_target = int(2*FIG_INCHES * DPI)
        h_target = int(FIG_INCHES * DPI)

        plot_context.update(
            {
                'fig': fig,
                'canvas': canvas,
                'w_target': w_target,
                'h_target': h_target,
                'ax_xy': ax_xy,
                'ax_vel': ax_vel,
                'ax_acc': ax_acc,
                'ax_dist': ax_dist,
                'ax_sigL': ax_sigL,
                'ax_sigOA': ax_sigOA,
                'p1_plot': p1_plot,
                'p2_plot': p2_plot,
                'p12_line': p12_line,
                'p12_line_pred': p12_line_pred,
                'p12_line_hist': p12_line_hist,
                'p1_pred': p1_pred,
                'p2_pred': p2_pred,
                'p1_pred_positive': p1_pred_positive,
                'p1_pred_negative': p1_pred_negative,
                'p2_pred_positive': p2_pred_positive,
                'p2_pred_negative': p2_pred_negative,
                'p1_hist': p1_hist,
                'p2_hist': p2_hist,
                'tgt1_plot': tgt1_plot,
                'tgt2_plot': tgt2_plot,
                'p1_vel': p1_vel,
                'p2_vel': p2_vel,
                'p1_vel_pred': p1_vel_pred,
                'p2_vel_pred': p2_vel_pred,
                'p1_vel_hist': p1_vel_hist,
                'p2_vel_hist': p2_vel_hist,
                'p1_acc': p1_acc,
                'p2_acc': p2_acc,
                'p1_acc_pred': p1_acc_pred,
                'p2_acc_pred': p2_acc_pred,
                'p1_acc_hist': p1_acc_hist,
                'p2_acc_hist': p2_acc_hist,
                'p12_dist': p12_dist,
                'p12_dist_pred': p12_dist_pred,
                'p12_dist_hist': p12_dist_hist,
                'p_sigL': p_sigL,
                'p_sigOA': p_sigOA,
            }
        )

    if rt_plot:
        fig = plot_context['fig']
        canvas = plot_context['canvas']
        w_target = plot_context['w_target']
        h_target = plot_context['h_target']
        ax_xy = plot_context['ax_xy']
        ax_vel = plot_context['ax_vel']
        ax_acc = plot_context['ax_acc']
        ax_dist = plot_context['ax_dist']
        ax_sigL = plot_context['ax_sigL']
        ax_sigOA = plot_context['ax_sigOA']
        p1_plot = plot_context['p1_plot']
        p2_plot = plot_context['p2_plot']
        p12_line = plot_context['p12_line']
        p12_line_pred = plot_context['p12_line_pred']
        p12_line_hist = plot_context['p12_line_hist']
        p1_pred = plot_context['p1_pred']
        p2_pred = plot_context['p2_pred']
        p1_pred_positive = plot_context['p1_pred_positive']
        p1_pred_negative = plot_context['p1_pred_negative']
        p2_pred_positive = plot_context['p2_pred_positive']
        p2_pred_negative = plot_context['p2_pred_negative']
        p1_hist = plot_context['p1_hist']
        p2_hist = plot_context['p2_hist']
        tgt1_plot = plot_context['tgt1_plot']
        tgt2_plot = plot_context['tgt2_plot']
        p1_vel = plot_context['p1_vel']
        p2_vel = plot_context['p2_vel']
        p1_vel_pred = plot_context['p1_vel_pred']
        p2_vel_pred = plot_context['p2_vel_pred']
        p1_vel_hist = plot_context['p1_vel_hist']
        p2_vel_hist = plot_context['p2_vel_hist']
        p1_acc = plot_context['p1_acc']
        p2_acc = plot_context['p2_acc']
        p1_acc_pred = plot_context['p1_acc_pred']
        p2_acc_pred = plot_context['p2_acc_pred']
        p1_acc_hist = plot_context['p1_acc_hist']
        p2_acc_hist = plot_context['p2_acc_hist']
        p12_dist = plot_context['p12_dist']
        p12_dist_pred = plot_context['p12_dist_pred']
        p12_dist_hist = plot_context['p12_dist_hist']
        p_sigL = plot_context['p_sigL']
        p_sigOA = plot_context['p_sigOA']
        Frames = plot_context['Frames']

    try:
        print(f"Scenario: {Scenario.name}, Alpha: {alpha}, MC: {n_mc+1}/{Scenario.Nmc}", flush=True)
    except (BlockingIOError, BrokenPipeError):
        # Skip logging if stdout is still non-blocking for any reason.
        pass
    t = 0.0
    t_hist = np.array([[t]])
    x1_hist, v1_hist, a1_hist = x1_init, np.zeros((1, 2)), np.zeros((0, 2))
    x2_hist, v2_hist, a2_hist = x2_init, np.zeros((1, 2)), np.zeros((0, 2))
    alpha_hist, beta_hist = np.array([[alpha]]), np.array([[beta]])
    confidence_hist = np.array([GameSol.sol.confidence], dtype=float)

    x1_state, x2_state = x1_init, x2_init
    v1_state, v2_state = np.zeros((1, 2)), np.zeros((1, 2))
    EndSimulation = False
    infeasible_run = False
    i_acc = 0
    GameSol.success = False
    GameSol.sol.time = -1.0
    while not EndSimulation:
        if t >= GameSol.sol.time + dt_Solver:
            if GameSol.success and SolverType == 'DG':
                z0 = GameSol.z0
                x1_guess, v1_guess, x2_guess, v2_guess = GameSol.calc_init_guess_from_input(x1_state, v1_state, x2_state, v2_state, GameSol.sol.a1[1:,:], GameSol.sol.a2[1:,:])
                z0[GameSol.indx_x1:GameSol.indx_x1 + N+1] = x1_guess[:, 0]
                z0[GameSol.indx_y1:GameSol.indx_y1 + N+1] = x1_guess[:, 1]
                z0[GameSol.indx_vx1:GameSol.indx_vx1 + N+1] = v1_guess[:, 0]
                z0[GameSol.indx_vy1:GameSol.indx_vy1 + N+1] = v1_guess[:, 1]
                z0[GameSol.indx_ax1:GameSol.indx_ax1 + N - 1] = GameSol.sol.a1[1:, 0]
                z0[GameSol.indx_ay1:GameSol.indx_ay1 + N - 1] = GameSol.sol.a1[1:, 1]
                z0[GameSol.indx_x2:GameSol.indx_x2 + N+1] = x2_guess[:, 0]
                z0[GameSol.indx_y2:GameSol.indx_y2 + N+1] = x2_guess[:, 1]
                z0[GameSol.indx_vx2:GameSol.indx_vx2 + N+1] = v2_guess[:, 0]
                z0[GameSol.indx_vy2:GameSol.indx_vy2 + N+1] = v2_guess[:, 1]
                z0[GameSol.indx_ax2:GameSol.indx_ax2 + N - 1] = GameSol.sol.a2[1:, 0]
                z0[GameSol.indx_ay2:GameSol.indx_ay2 + N - 1] = GameSol.sol.a2[1:, 1]
                
                Ntries = 3
                dalpha = (0.9-alpha)/Ntries
                dbeta = (0.9-beta)/Ntries
                for i_dalpha in range(0,Ntries+1):
                    GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha+dalpha*i_dalpha, beta+dbeta*i_dalpha, z0=z0, log=True)
                    if GameSol.success: break
                else:
                    Generalz0 = GeneralGameSol.z0
                    Generalz0[:z0.shape[0]] = z0
                    GeneralGameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, z0=Generalz0, log=True)
                if i_dalpha>0 and GameSol.success:
                    success = False
                    for i_dalpha_back in range(i_dalpha,-1,-1):
                        GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha + dalpha*i_dalpha_back, beta + dbeta*i_dalpha_back, log=True)
                        if GameSol.success: 
                            success = True
                    if success: 
                        GameSol.success = True
                    else:
                        GeneralGameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, log=True)
                
            if not GameSol.success and SolverType == 'DG':
                # x1_guess, v1_guess, a1_guess = GameSol.MPC_guess_human_calc(x1_state, v1_state, x1_des)
                # x2_guess, v2_guess, a2_guess = GameSol.MPC_guess_robot_calc(x2_state, v2_state, x2_des, x_partner=x1_guess)
                x1_guess, v1_guess, a1_guess, x2_guess, v2_guess, a2_guess, _ = GameSol.Centralized_MPC_calc(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha)

                z0 = np.zeros_like(GameSol.z0)
                z0[GameSol.indx_x1:GameSol.indx_x1 + N + 1] = x1_guess[:, 0]
                z0[GameSol.indx_y1:GameSol.indx_y1 + N + 1] = x1_guess[:, 1]
                z0[GameSol.indx_vx1:GameSol.indx_vx1 + N + 1] = v1_guess[:, 0]
                z0[GameSol.indx_vy1:GameSol.indx_vy1 + N + 1] = v1_guess[:, 1]
                z0[GameSol.indx_ax1:GameSol.indx_ax1 + N] = a1_guess[:, 0]
                z0[GameSol.indx_ay1:GameSol.indx_ay1 + N] = a1_guess[:, 1]
                z0[GameSol.indx_x2:GameSol.indx_x2 + N + 1] = x2_guess[:, 0]
                z0[GameSol.indx_y2:GameSol.indx_y2 + N + 1] = x2_guess[:, 1]
                z0[GameSol.indx_vx2:GameSol.indx_vx2 + N + 1] = v2_guess[:, 0]
                z0[GameSol.indx_vy2:GameSol.indx_vy2 + N + 1] = v2_guess[:, 1]
                z0[GameSol.indx_ax2:GameSol.indx_ax2 + N] = a2_guess[:, 0]
                z0[GameSol.indx_ay2:GameSol.indx_ay2 + N] = a2_guess[:, 1]
                
                # Calculate Solution with alpha = alpha+dalpha for feasibility check
                Ntries = 3
                dalpha = (0.9-alpha)/Ntries
                dbeta = (0.9-beta)/Ntries
                for i_dalpha in range(0,Ntries+1):
                    GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha+dalpha*i_dalpha, beta+dbeta*i_dalpha, z0=z0, log=True)
                    if GameSol.success: break
                if i_dalpha>0 and GameSol.success:
                    success = False
                    for i_dalpha_back in range(i_dalpha,-1,-1):
                        GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha + dalpha*i_dalpha_back, beta + dbeta*i_dalpha_back, log=True)
                        if GameSol.success: 
                            success = True
                    if success: 
                        GameSol.success = True
                    else:
                        GeneralGameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, log=True)
                
            if SolverType == 'Centralized':
                GameSol.Centralized_MPC_calc(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha)
                GameSol.sol = GameSol.Centralized_MPC_sol
            if GameSol.success:
                i_acc = 0
            else:
                i_acc += 1
                i_acc = min(i_acc, N - 1)
            
                
        # Pure persuit acceleration command for human
        if Human_PreDefined_Traj:
            a1_cmd = np.zeros((1,2))
            # Pursuit point
            pp_dist = np.linalg.norm(x1_des - x1_state)
            pp_point = x1_des + pp_dist * pp_factor * np.array([np.cos(pp_theta), np.sin(pp_theta)])
            dpos_3d = np.array( [pp_point[0,0] - x1_state[0,0], pp_point[0,1] - x1_state[0,1], 0])
            v_3d = np.array( [v1_state[0,0], v1_state[0,1], 0])
            lam_dot = np.cross(dpos_3d, v_3d) / (np.linalg.norm(dpos_3d)**2 + 1e-6)
            a1_cmd[0,:] = 2.0 * np.linalg.norm(v1_state) * np.cross(v_3d, lam_dot)[0:2]
            # Velocity matching term
            tau = 0.5
            Vel_near_Obs = 1.0
            if Scenario.obstacles:
                for Obstcle in Scenario.obstacles:
                    dist_to_obs = np.linalg.norm(x1_state - Obstcle['Pos']) - Obstcle['diam']/2
                    Vel_near_Obs = Vel_near_Obs*min(1.0, dist_to_obs / (Obstcle['diam']/2))
            a1_cmd += (min(max(0.0,pp_dist-0.2)/(3*tau),Vel_near_Obs*GameSol.v1_max) -np.linalg.norm(v1_state)) / (tau) * (pp_point - x1_state) / np.linalg.norm(dpos_3d)
            a1_cmd = LimitedCmd(a1_cmd, a1_acc_limit)
        else:
            a1_cmd = GameSol.sol.a1[i_acc, :]
        if n_mc >= 1:
            a1_cmd += 1.0 * np.random.normal(0.0, 1.0, 2)
                        
        if Scenario.obstacles and Human_PreDefined_Traj:
            for Obstcle in Scenario.obstacles:
                dist_to_obs = np.linalg.norm(x1_state - Obstcle['Pos']) - Obstcle['diam']/2
                if dist_to_obs < 0.75*Obstcle['diam']:
                    r_obs = (x1_state - Obstcle['Pos']) / (np.linalg.norm(x1_state - Obstcle['Pos']) + 1e-6)
                    if np.dot(a1_cmd[0,:], r_obs[0,:]) < 0:
                        a1_cmd = a1_cmd - np.dot(a1_cmd[0,:], r_obs[0,:]) * r_obs
                if dist_to_obs < 0.25*Obstcle['diam']:
                    a1_cmd += a1_acc_limit * r_obs * (0.25*Obstcle['diam'] - dist_to_obs) / (0.25*Obstcle['diam'])
        
        # Random walk acceleration command for human
        if Human_RandomWalk_Traj:
            a1_cmd = 2.0 * np.random.normal(0.0, 1.0, 2)
            if np.linalg.norm(v1_state) >= 0.8 * GameSol.v1_max:
                a_on_v = np.dot(a1_cmd, v1_state.T) / (np.linalg.norm(v1_state) + 1e-6)
                if a_on_v > 0:
                    a1_cmd = a1_cmd - a_on_v * (v1_state / (np.linalg.norm(v1_state) + 1e-6))
        
        
        # Apply acceleration commands and propagate states
        # a1_cmd = LimitedCmd(a1_cmd, GameSol.a1_max)
        a1_cmd = LimitedCmd(a1_cmd, a1_acc_limit)
        a2_cmd = np.zeros((1,2))
        a2_cmd[0,0] = interp1d(GameSol.sol.time+np.linspace(0,(N-1)*GameSol.dt, N), GameSol.sol.a2[:,0], kind='previous', fill_value='extrapolate')(t)
        a2_cmd[0,1] = interp1d(GameSol.sol.time+np.linspace(0,(N-1)*GameSol.dt, N), GameSol.sol.a2[:,1], kind='previous', fill_value='extrapolate')(t)
        # a2_cmd[0,1] = np.interp(t, GameSol.sol.time+np.linspace(0,(N-1)*GameSol.dt, N), GameSol.sol.a2[:,1])
        # a2_cmd[0,0] = GameSol.sol.a2[i_acc,0]
        # a2_cmd[0,1] = GameSol.sol.a2[i_acc,1]
        # a2_cmd = LimitedCmd(np.array(a2_cmd), GameSol.a2_max)
        x1_state = x1_state + dt_Sim * v1_state + 0.5 * dt_Sim**2 * a1_cmd
        v1_state = v1_state + dt_Sim * a1_cmd
        x2_state = x2_state + dt_Sim * v2_state + 0.5 * dt_Sim**2 * a2_cmd
        v2_state = v2_state + dt_Sim * a2_cmd
        t += dt_Sim
        # Log data for analysis and plotting
        x1_hist = np.vstack((x1_hist, x1_state))
        v1_hist = np.vstack((v1_hist, v1_state))
        a1_hist = np.vstack((a1_hist, a1_cmd))
        x2_hist = np.vstack((x2_hist, x2_state))
        v2_hist = np.vstack((v2_hist, v2_state))
        a2_hist = np.vstack((a2_hist, a2_cmd))
        alpha_hist = np.vstack((alpha_hist, np.array([[GameSol.sol.alpha]])))
        beta_hist = np.vstack((beta_hist, np.array([[GameSol.sol.beta]])))
        confidence_hist = np.vstack((confidence_hist, np.array([GameSol.sol.confidence], dtype=float)))
        t_hist = np.vstack((t_hist, t))

        if np.linalg.norm(x2_state - x1_state) > GameSol.d + 10*GameSol.delta_d:
            infeasible_run = True
            EndSimulation = True
            print('Terminating MC run due to excessive separation.')

        if rt_plot:
            p1_plot.set_data([x1_state[0, 0]], [x1_state[0, 1]])
            p2_plot.set_data([x2_state[0, 0]], [x2_state[0, 1]])
            p12_line.set_data([x1_state[0, 0], x2_state[0, 0]], [x1_state[0, 1], x2_state[0, 1]])
            indx = np.linspace(0, N, 1 + len(p12_line_pred))
            for i in range(len(p12_line_pred)):
                i_indx = int(indx[i + 1])
                p12_line_pred[i].set_data([GameSol.sol.x1[i_indx, 0], GameSol.sol.x2[i_indx, 0]], [GameSol.sol.x1[i_indx, 1], GameSol.sol.x2[i_indx, 1]])
            p1_pred.set_data(GameSol.sol.x1[:, 0], GameSol.sol.x1[:, 1])
            p2_pred.set_data(GameSol.sol.x2[:, 0], GameSol.sol.x2[:, 1])
            
            # p2_pred_positive.set_data(x2_positive[:, 0], x2_positive[:, 1])
            # p2_pred_negative.set_data(x2_negative[:, 0], x2_negative[:, 1])
            
            p1_hist.set_data(x1_hist[:, 0], x1_hist[:, 1])
            p2_hist.set_data(x2_hist[:, 0], x2_hist[:, 1])
            tgt1_plot.set_data([x1_des[0, 0]], [x1_des[0, 1]])
            tgt2_plot.set_data([x2_des[0, 0]], [x2_des[0, 1]])
            ax_xy.set_title(f'Time: {t:3.3}[Sec], Alpha is {GameSol.sol.alpha}, Beta is {GameSol.sol.beta}, Confidence is {GameSol.sol.confidence}')
            ax_xy.legend(ncol=4, loc='upper center')

            t_pred = np.linspace(GameSol.sol.time - dt_Solver, t - dt_Solver + N * dt_Solver, N + 1)
            p1_vel.set_data([t], [np.linalg.norm(v1_state)])
            p2_vel.set_data([t], [np.linalg.norm(v2_state)])
            p1_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v1, axis=1))
            p2_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v2, axis=1))
            p1_vel_hist.set_data(t_hist, np.linalg.norm(v1_hist, axis=1))
            p2_vel_hist.set_data(t_hist, np.linalg.norm(v2_hist, axis=1))

            p1_acc.set_data([t], [np.linalg.norm(a1_cmd)])
            p2_acc.set_data([t], [np.linalg.norm(a2_cmd)])
            p1_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a1, axis=1))
            p2_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a2, axis=1))
            p1_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a1_hist, axis=1))
            p2_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a2_hist, axis=1))

            p12_dist.set_data([t], [np.linalg.norm(x1_state - x2_state)])
            p12_dist_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.x1 - GameSol.sol.x2, axis=1))
            p12_dist_hist.set_data(t_hist[:, 0], np.linalg.norm(x1_hist - x2_hist, axis=1))
            if SolverType == 'DG':
                for i in range(len(p_sigL)):
                    p_sigL[i].set_data([np.linspace(0,GameSol.sol.sigL.shape[0]-1, GameSol.sol.sigL.shape[0])], [GameSol.sol.sigL[:, i]])
                ax_sigL.set_ylim([-0.1, np.max( [1.0, 1.1*np.max( GameSol.sol.sigL) ] ) ])
                if Scenario.obstacles:
                    for i in range(len(p_sigOA)):
                        p_sigOA[i].set_data([np.linspace(0,GameSol.sol.sigOA.shape[0]-1, GameSol.sol.sigOA.shape[0])], [GameSol.sol.sigOA[:, i]])
                    ax_sigOA.set_ylim([-0.1, np.max( [1.0, 1.1*np.max( GameSol.sol.sigOA) ] ) ])

            if t_hist.shape[0] % (10 * int(dt_Solver/dt_Sim)) == 0:
                k = t_hist.shape[0]//(10 * int(dt_Solver/dt_Sim))-1
                p12_line_hist[k].set_data([x1_hist[-1, 0], x2_hist[-1, 0]], [x1_hist[-1, 1], x2_hist[-1, 1]])

            plt.pause(0.1)

            fig.canvas.draw()
            if rt_plot and n_mc == 0:
                frame = capture_frame_agg(fig, canvas, w_target, h_target)
                Frames.append(frame)

        if EndSimulation or (t >= Tf) or (np.linalg.norm(x1_state - x1_des) + np.linalg.norm(x2_state - x2_des) < 1.0):
            EndSimulation = True
            if rt_plot:
                for i in range(len(p12_line_pred)):
                    p12_line_pred[i].set_data([], [])
                for i in range(len(p12_line_hist)):
                    p12_line_hist[i].set_data([], [])
            
            if rt_plot and fig is not None:
                fig.savefig(f"{Scenario.name}_traj_final.png")
            if rt_plot and n_mc == 0:
                ax_xy.plot(x1_hist[:, 0], x1_hist[:, 1], '-', color=hist_col[ialpha], linewidth=2, label=f'alpha={alpha}')
                ax_xy.plot(x2_hist[:, 0], x2_hist[:, 1], '--', color=hist_col[ialpha], linewidth=2)
                ax_xy.legend(ncol=4, loc='upper center')
                ax_xy.set_title('Trajectories')
                if ialpha == 0:
                    ax_vel.plot(t_hist, np.linalg.norm(v1_hist, axis=1), '-', color=hist_col[ialpha], linewidth=2, label='Human')
                    ax_vel.plot(t_hist, np.linalg.norm(v2_hist, axis=1), '--', color=hist_col[ialpha], linewidth=2, label='Robot')
                else:
                    ax_vel.plot(t_hist, np.linalg.norm(v1_hist, axis=1), '-', color=hist_col[ialpha], linewidth=2)
                    ax_vel.plot(t_hist, np.linalg.norm(v2_hist, axis=1), '--', color=hist_col[ialpha], linewidth=2)
                
                p1_vel.set_data([], [])
                p2_vel.set_data([], [])
                p1_vel_pred.set_data([], [])
                p2_vel_pred.set_data([], [])
                ax_vel.legend()    
                ax_acc.plot(t_hist[:-1], np.linalg.norm(a1_hist, axis=1), '-', color=hist_col[ialpha], linewidth=2)
                ax_acc.plot(t_hist[:-1], np.linalg.norm(a2_hist, axis=1), '--', color=hist_col[ialpha], linewidth=2)
                ax_dist.plot(t_hist, np.linalg.norm(x1_hist - x2_hist, axis=1), '-', color=hist_col[ialpha], linewidth=2)
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
                
                for k in range(9, np.shape(x1_hist)[0]-1, 10):
                    ax_xy.plot([x1_hist[k, 0], x2_hist[k, 0]], [x1_hist[k, 1], x2_hist[k, 1]], ':', color=hist_col[ialpha], linewidth=1)
                if fig is not None:
                    fig.savefig(f"{Scenario.name}_final.png")

    dist_series = np.linalg.norm(x1_hist - x2_hist, axis=1)
    a1_mag = np.linalg.norm(a1_hist, axis=1)  # magnitude gives total acceleration per step
    a2_mag = np.linalg.norm(a2_hist, axis=1)
    scenario_time = t_hist[-1, 0]
    confidence_series = np.asarray(confidence_hist)
    if confidence_series.ndim > 1:
        confidence_series = confidence_series.mean(axis=1)
    if infeasible_run:
        dist_mean = -1.0
        dist_std = -1.0
        a1_mean = -1.0
        a1_std = -1.0
        a2_mean = -1.0
        a2_std = -1.0
        time_std = -1.0
        confidence_mean = -1.0
        confidence_std = -1.0
    else:
        dist_mean = dist_series.mean()
        dist_std = dist_series.std()
        a1_mean = a1_mag.mean()
        a1_std = a1_mag.std()
        a2_mean = a2_mag.mean()
        a2_std = a2_mag.std()
        time_std = np.std([scenario_time])
        confidence_mean = confidence_series.mean()
        confidence_std = confidence_series.std()
    return np.array(
        [
            alpha,
            n_mc + 1,
            dist_mean,
            dist_std,
            a1_mean,
            a1_std,
            a2_mean,
            a2_std,
            scenario_time,
            time_std,
            confidence_mean,
            confidence_std,
        ],
        dtype=float,
    )


def run_scenario(Scenario: Scenario):
    plot_context = {'Frames': []} if RT_Plot else None
    mc_run_stats = []

    # for SolverType in ['DG', 'Centralized']:
    for SolverType in ['DG']:
        for ialpha, alpha in enumerate(alpha_vec):
            beta = beta_vec[ialpha]
            if Scenario.Nmc <= 0:
                continue
            if RT_Plot and Scenario.Nmc == 1:
                mc_run_stats.append(run_single_mc(Scenario, alpha, beta, ialpha, 0, SolverType, plot_context))
                mc_start = 1
            else:
                mc_start = 0

            # mc_run_stats.append(run_single_mc(Scenario, alpha, beta, ialpha, 2, SolverType, plot_context))
            
            mc_indices = list(range(mc_start, Scenario.Nmc))
            if not mc_indices:
                continue
            if not RT_Plot and len(mc_indices) == 1:
                mc_run_stats.append(run_single_mc(Scenario, alpha, beta, ialpha, mc_indices[0], SolverType, None))
                continue
            max_workers = max(1, min(len(mc_indices), (os.cpu_count()-12 or 1)))
            # Use spawn so Julia is initialized fresh per worker (fork can cause ReadOnlyMemoryError)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
                futures = [
                    executor.submit(run_single_mc, Scenario, alpha, beta, ialpha, mc_idx, SolverType, None)
                    for mc_idx in mc_indices
                ]
                for fut in futures:
                    mc_run_stats.append(fut.result())

    if plot_context is not None:
        Frames = plot_context['Frames']
        fig = plot_context.get('fig')
        if fig is not None and Frames:
            writer = imageio.get_writer(
                f"{Scenario.name}.mp4",
                fps=3,
                codec='libx264',
                quality=8,
                macro_block_size=16,
            )
            w_target = plot_context['w_target']
            h_target = plot_context['h_target']
            for frame in Frames:
                assert frame.shape == (h_target, w_target, 3), f"Bad frame shape {frame.shape}"
                writer.append_data(frame)
            writer.close()
        if fig is not None:
            plt.close(fig)

    if mc_run_stats and Scenario.Nmc > 1:
        header = ",".join(
            [
                "alpha",
                "mc_run",
                "distance_mean",
                "distance_std",
                "a1_acc_mean",
                "a1_acc_std",
                "a2_acc_mean",
                "a2_acc_std",
                "scenario_time",
                "scenario_time_std",
                "confidence_mean",
                "confidence_std",
            ]
        )
        np.savetxt(
            f"{Scenario.name}_mc_stats.csv",
            np.vstack(mc_run_stats),
            delimiter=",",
            header=header,
            comments="",
        )


def main():
    if not Scenarios:
        return

    for scenario in Scenarios:
        run_scenario(scenario)


if __name__ == '__main__':
    main()
