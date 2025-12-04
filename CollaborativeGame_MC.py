#!/usr/bin/env python3
import os
os.system('clear')

import numpy as np
import math
from dataclasses import dataclass

import CollaborativeGameSolver as GameSolver

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple



@dataclass
class Scenario:
    name: str
    x1_init: np.ndarray
    x2_init: np.ndarray
    x1_des: np.ndarray
    theta_des: float
    obstacles: List[Dict[str, np.ndarray]]
    Nmc: int = 1

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

hist_col = ['c','m','k','y']

Tf = 20.0
alpha_vec = [0.05, 0.1, 0.3, 0.5]
# alpha_vec = [0.05]

N = 10
dt = 0.1

Obstcles = []
Obstcles.append({'Pos': np.array([[6,0]]), "diam": 0.5})

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

Scenario(name="Scenario_1", x1_init=np.array([[-1.0, 6.0]]), x2_init=np.array([[-1-3/np.sqrt(2), 6-3/np.sqrt(2)]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[], Nmc=200),
# Scenario(name="Scenario_2", x1_init=np.array([[-1.5, 6.5]]), x2_init=np.array([[-1.5-3/np.sqrt(2), 6.5-3/np.sqrt(2)]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.0, 4.0]]), "diam": 1.0}]),
# Scenario(name="Scenario_3", x1_init=np.array([[-1.5, 6.5]]), x2_init=np.array([[-1.5+3*np.cos(np.deg2rad(225)), 6.5+3*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.0, 4.0]]), "diam": 1.0}], Nmc=5),
# Scenario(name="Scenario_6_WithObs", x1_init=np.array([[3.0, 0.0]]), x2_init=np.array([[0.0, 0.0]]), x1_des=np.array([[7.0, 0.0]]), theta_des=np.deg2rad(60.0), obstacles=[{"Pos": np.array([[6.0, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_6_Switch_WithObs", x1_init=np.array([[0.0, 0.0]]), x2_init=np.array([[3.0, 0.0]]), x1_des=np.array([[7.0, 4.0]]), theta_des=np.deg2rad(240.0), obstacles=[{"Pos": np.array([[4.5, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_7", x1_init=np.array([[2.0, -1.0]]), x2_init=np.array([[5.0, -1.0]]), x1_des=np.array([[2.0, 8.0]]), theta_des=np.deg2rad(0.0), obstacles=[{"Pos": np.array([[3.0, 4.0]]), "diam": 0.5}]),
]


Scenarios = Scenarios + random_scenarios(num=0)

Human_PreDefined_Traj = True
pp_theta = np.deg2rad(180.0)
# pp_theta = np.atan2( (Scenarios[0].x1_des[0,1]-Scenarios[0].x1_init[0,1]), (Scenarios[0].x1_des[0,0]-Scenarios[0].x1_init[0,0]) )
pp_factor = 0.5

def run_single_mc(Scenario: Scenario, alpha: float, n_mc: int, enable_plot: bool = False, plot_color: str | None = None):

    np.random.seed(100 + n_mc)
    x1_init, x2_init = Scenario.x1_init.copy(), Scenario.x2_init.copy()
    x1_des, theta_des = Scenario.x1_des.copy(), Scenario.theta_des
    Obstcles = Scenario.obstacles

    d_init = np.linalg.norm(x1_init - x2_init)
    x2_des = x1_des + d_init * np.array([np.cos(theta_des), np.sin(theta_des)])

    GameSol = GameSolver.CollaborativeGame(N=N, dt=dt, d=d_init, Obstcles=Scenario.obstacles)

    rt_plot = RT_Plot and enable_plot
    FIG_INCHES = 6.3
    DPI = 160
    Frames = []
    fig = None
    canvas = None
    w_target = h_target = None
    if rt_plot:
        run_color = plot_color or hist_col[min(n_mc, len(hist_col) - 1)]
        fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI, constrained_layout=False)
        ax_xy = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax_xy.plot(x1_init[0, 0], x1_init[0, 1], 'gs', markersize=10, label='Human Start')
        ax_xy.plot(x2_init[0, 0], x2_init[0, 1], 'bs', markersize=10, label='Robot Start')
        ax_xy.plot(x1_des[0, 0], x1_des[0, 1], 'go', markersize=10, label='Human Target')
        ax_xy.plot(x2_des[0, 0], x2_des[0, 1], 'bo', markersize=10, label='Robot Target')
        p1_plot = ax_xy.plot([], [], 'gs', markersize=6, label='Human')[0]
        p2_plot = ax_xy.plot([], [], 'bs', markersize=6, label='Robot')[0]
        p12_line = ax_xy.plot([], [], 'r-', linewidth=3)[0]
        p12_line_pred = []
        for _ in range(4):
            p12_line_pred.append(ax_xy.plot([], [], 'r:', linewidth=3)[0])
        p1_pred = ax_xy.plot([], [], 'g-.', linewidth=3)[0]
        p2_pred = ax_xy.plot([], [], 'b-.', linewidth=3)[0]
        p1_hist = ax_xy.plot([], [], 'g-', linewidth=3)[0]
        p2_hist = ax_xy.plot([], [], 'b-', linewidth=3)[0]
        tgt1_plot = ax_xy.plot([], [], 'go', markersize=6)[0]
        tgt2_plot = ax_xy.plot([], [], 'bo', markersize=6)[0]
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_xlim([-5, 12])
        ax_xy.set_ylim([-1, 10])
        ax_xy.grid(True)
        ax_xy.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05))
        for Obstcle in Obstcles:
            x, y = Obstcle['diam'] / 2 * np.cos(np.linspace(0, 2 * np.pi, 100)), Obstcle['diam'] / 2 * np.sin(np.linspace(0, 2 * np.pi, 100))
            ax_xy.plot(Obstcle['Pos'][0, 0] + x, Obstcle['Pos'][0, 1] + y, 'k-', linewidth=3)

        ax_vel = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        p1_vel = ax_vel.plot([], [], 'gs', markersize=6, label='Human')[0]
        p2_vel = ax_vel.plot([], [], 'bs', markersize=6, label='Robot')[0]
        p1_vel_pred = ax_vel.plot([], [], 'g-.', markersize=3)[0]
        p2_vel_pred = ax_vel.plot([], [], 'b-.', markersize=3)[0]
        p1_vel_hist = ax_vel.plot([], [], 'g-', markersize=3)[0]
        p2_vel_hist = ax_vel.plot([], [], 'b-', markersize=3)[0]
        ax_vel.set_xlim([0, Tf])
        ax_vel.set_ylim([0, max(GameSol.v1_max, GameSol.v2_max) + 1])
        ax_vel.plot([0, Tf], [GameSol.v1_max, GameSol.v1_max], 'g:', markersize=3)
        ax_vel.plot([0, Tf], [GameSol.v2_max, GameSol.v2_max], 'b:', markersize=3)
        ax_vel.grid(True)
        ax_vel.legend()
        ax_vel.set_xlabel('Time [Sec]')
        ax_vel.set_ylabel('Velocity [m/s]')

        ax_acc = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
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
        ax_acc.set_xlabel('Time [Sec]')
        ax_acc.set_ylabel('Acceleration [m/s^2]')

        ax_dist = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        p12_dist = ax_dist.plot([], [], 'rs', markersize=6)[0]
        p12_dist_pred = ax_dist.plot([], [], 'b-', markersize=3)[0]
        p12_dist_hist = ax_dist.plot([], [], 'b-.', markersize=3)[0]
        ax_dist.plot([0, Tf], [GameSol.d_min, GameSol.d_min], 'b:', markersize=3)
        ax_dist.plot([0, Tf], [GameSol.d_max, GameSol.d_max], 'b:', markersize=3)
        ax_dist.set_xlim([0, Tf])
        ax_dist.set_ylim([GameSol.d_min - 2*GameSol.delta_d, GameSol.d_max + 2*GameSol.delta_d])
        ax_dist.grid(True)
        ax_dist.set_xlabel('Time [Sec]')
        ax_dist.set_ylabel('Distance [m]')

        canvas = FigureCanvas(fig)
        w_target = int(FIG_INCHES * DPI)
        h_target = int(FIG_INCHES * DPI)

    # print(f"Scenario: {Scenario.name}, Alpha: {alpha}, MC: {n_mc+1}/{Scenario.Nmc}")
    t = 0.0
    t_hist = np.array([[t]])
    x1_hist, v1_hist, a1_hist = x1_init, np.zeros((1, 2)), np.zeros((0, 2))
    x2_hist, v2_hist, a2_hist = x2_init, np.zeros((1, 2)), np.zeros((0, 2))

    x1_state, x2_state = x1_init, x2_init
    v1_state, v2_state = np.zeros((1, 2)), np.zeros((1, 2))
    EndSimulation = False
    infeasible_run = False
    i_acc = 0
    GameSol.success = False
    avoid_Obs = 0.0
    while not EndSimulation:
        if GameSol.success:
            z0 = GameSol.z0
            z0[GameSol.indx_x1:GameSol.indx_x1 + N] = GameSol.sol.x1_sol[1:, 0]
            z0[GameSol.indx_y1:GameSol.indx_y1 + N] = GameSol.sol.x1_sol[1:, 1]
            z0[GameSol.indx_vx1:GameSol.indx_vx1 + N] = GameSol.sol.v1_sol[1:, 0]
            z0[GameSol.indx_vy1:GameSol.indx_vy1 + N] = GameSol.sol.v1_sol[1:, 1]
            z0[GameSol.indx_ax1:GameSol.indx_ax1 + N - 1] = GameSol.sol.a1_sol[1:, 0]
            z0[GameSol.indx_ay1:GameSol.indx_ay1 + N - 1] = GameSol.sol.a1_sol[1:, 1]
            z0[GameSol.indx_x2:GameSol.indx_x2 + N] = GameSol.sol.x2_sol[1:, 0]
            z0[GameSol.indx_y2:GameSol.indx_y2 + N] = GameSol.sol.x2_sol[1:, 1]
            z0[GameSol.indx_vx2:GameSol.indx_vx2 + N] = GameSol.sol.v2_sol[1:, 0]
            z0[GameSol.indx_vy2:GameSol.indx_vy2 + N] = GameSol.sol.v2_sol[1:, 1]
            z0[GameSol.indx_ax2:GameSol.indx_ax2 + N - 1] = GameSol.sol.a2_sol[1:, 0]
            z0[GameSol.indx_ay2:GameSol.indx_ay2 + N - 1] = GameSol.sol.a2_sol[1:, 1]

            GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, avoid_Obs=avoid_Obs, log=enable_plot)
        if not GameSol.success:
            x1_guess, v1_guess, a1_guess = GameSol.MPC_guess_human_calc(x1_state, v1_state, x1_des)
            x2_guess, v2_guess, a2_guess = GameSol.MPC_guess_robot_calc(x2_state, v2_state, x2_des, x_partner=x1_guess)

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

            GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, avoid_Obs=avoid_Obs, log=enable_plot)

        if GameSol.success:
            i_acc = 0
        else:
            i_acc += 1
        if i_acc >= N - 1:
            i_acc = N - 1
            
        # Puer persuit acceleration command for human
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
            if t <4.0:
                a1_cmd += (min(pp_dist/(2*tau),GameSol.v1_max*0.6) -np.linalg.norm(v1_state)) / (tau) * (pp_point - x1_state) / np.linalg.norm(dpos_3d)
            else:
                a1_cmd += (min(pp_dist/(2*tau),GameSol.v1_max) -np.linalg.norm(v1_state)) / (tau) * (pp_point - x1_state) / np.linalg.norm(dpos_3d)
            a1_cmd = LimitedCmd(a1_cmd, GameSol.a1_max)
        else:
            a1_cmd = GameSol.sol.a1_sol[i_acc, :]
        if n_mc > 0:
            a1_cmd += 1.0 * np.random.normal(0.0, 1.0, 2)
        
        a1_cmd = LimitedCmd(a1_cmd, GameSol.a1_max)
        a2_cmd = LimitedCmd(GameSol.sol.a2_sol[i_acc, :], GameSol.a2_max)
        x1_state = x1_state + dt * v1_state + 0.5 * dt**2 * a1_cmd
        v1_state = v1_state + dt * a1_cmd
        x2_state = x2_state + dt * v2_state + 0.5 * dt**2 * a2_cmd
        v2_state = v2_state + dt * a2_cmd

        t += dt

        x1_hist = np.vstack((x1_hist, x1_state))
        v1_hist = np.vstack((v1_hist, v1_state))
        a1_hist = np.vstack((a1_hist, a1_cmd))
        x2_hist = np.vstack((x2_hist, x2_state))
        v2_hist = np.vstack((v2_hist, v2_state))
        a2_hist = np.vstack((a2_hist, a2_cmd))
        t_hist = np.vstack((t_hist, t))

        if np.linalg.norm(x2_state - x1_state) > GameSol.d + 10 * GameSol.delta_d:
            infeasible_run = True
            EndSimulation = True
            # print("Terminating MC run due to excessive separation.")

        if rt_plot:
            p1_plot.set_data([x1_state[0, 0]], [x1_state[0, 1]])
            p2_plot.set_data([x2_state[0, 0]], [x2_state[0, 1]])
            p12_line.set_data([x1_state[0, 0], x2_state[0, 0]], [x1_state[0, 1], x2_state[0, 1]])
            indx = np.linspace(0, N, 1 + len(p12_line_pred))
            if not Human_PreDefined_Traj:
                for i in range(len(p12_line_pred)):
                    i_indx = int(indx[i + 1])
                    p12_line_pred[i].set_data([GameSol.sol.x1_sol[i_indx, 0], GameSol.sol.x2_sol[i_indx, 0]], [GameSol.sol.x1_sol[i_indx, 1], GameSol.sol.x2_sol[i_indx, 1]])
                p1_pred.set_data(GameSol.sol.x1_sol[:, 0], GameSol.sol.x1_sol[:, 1])
            p2_pred.set_data(GameSol.sol.x2_sol[:, 0], GameSol.sol.x2_sol[:, 1])
            p1_hist.set_data(x1_hist[:, 0], x1_hist[:, 1])
            p2_hist.set_data(x2_hist[:, 0], x2_hist[:, 1])
            tgt1_plot.set_data([x1_des[0, 0]], [x1_des[0, 1]])
            tgt2_plot.set_data([x2_des[0, 0]], [x2_des[0, 1]])
            ax_xy.set_title(f'Alpha is {alpha}, Time: {t:2.2}[Sec]')
            ax_xy.legend(ncol=4, loc='upper center')

            t_pred = np.linspace(t - dt, t - dt + N * dt, N + 1)
            p1_vel.set_data([t], [np.linalg.norm(v1_state)])
            p2_vel.set_data([t], [np.linalg.norm(v2_state)])
            if not Human_PreDefined_Traj:
                p1_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v1_sol, axis=1))
            p2_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v2_sol, axis=1))
            p1_vel_hist.set_data(t_hist, np.linalg.norm(v1_hist, axis=1))
            p2_vel_hist.set_data(t_hist, np.linalg.norm(v2_hist, axis=1))

            p1_acc.set_data([t], [np.linalg.norm(a1_cmd)])
            p2_acc.set_data([t], [np.linalg.norm(a2_cmd)])
            if not Human_PreDefined_Traj:
                p1_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a1_sol, axis=1))
            p2_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a2_sol, axis=1))
            p1_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a1_hist, axis=1))
            p2_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a2_hist, axis=1))

            p12_dist.set_data([t], [np.linalg.norm(x1_state - x2_state)])
            if not Human_PreDefined_Traj:
                p12_dist_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.x1_sol - GameSol.sol.x2_sol, axis=1))
            p12_dist_hist.set_data(t_hist[:, 0], np.linalg.norm(x1_hist - x2_hist, axis=1))
            
            if t_hist.shape[0] % 10 == 0:
                ax_xy.plot([x1_state[0, 0], x2_state[0,0]], [x1_state[0, 1], x2_state[0, 1]], 'r-', linewidth=1)

            plt.pause(0.1)

            fig.canvas.draw()
            if rt_plot and n_mc == 0:
                frame = capture_frame_agg(fig, canvas, w_target, h_target)
                Frames.append(frame)

        if EndSimulation or (t >= Tf) or (max(np.linalg.norm(x1_state - x1_des), np.linalg.norm(x2_state - x2_des)) < 0.1):
            EndSimulation = True
            if rt_plot:
                for i in range(len(p12_line_pred)):
                    p12_line_pred[i].set_data([], [])
                if n_mc == 0:
                    if fig is not None: fig.savefig(f"{Scenario.name}_traj_final.png")
                    ax_xy.plot(x1_hist[:, 0], x1_hist[:, 1], '-', color=run_color, linewidth=2, label=f'alpha={alpha}, mc={n_mc+1}')
                    ax_xy.plot(x2_hist[:, 0], x2_hist[:, 1], '-', color=run_color, linewidth=2)
                    ax_xy.legend(ncol=4, loc='upper center')
                    ax_vel.plot(t_hist, np.linalg.norm(v1_hist, axis=1), '-', color=run_color, linewidth=2)
                    ax_vel.plot(t_hist, np.linalg.norm(v2_hist, axis=1), '-', color=run_color, linewidth=2)
                    ax_acc.plot(t_hist[:-1], np.linalg.norm(a1_hist, axis=1), '-', color=run_color, linewidth=2)
                    ax_acc.plot(t_hist[:-1], np.linalg.norm(a2_hist, axis=1), '-', color=run_color, linewidth=2)
                    ax_dist.plot(t_hist, np.linalg.norm(x1_hist - x2_hist, axis=1), '-', color=run_color, linewidth=2)
                    p1_pred.set_data([], [])
                    p2_pred.set_data([], [])
                    p1_hist.set_data([], [])
                    p2_hist.set_data([], [])
                    
                    for k in range(9, np.shape(x1_hist)[0]-1, 10):
                        ax_xy.plot([x1_hist[k, 0], x2_hist[k, 0]], [x1_hist[k, 1], x2_hist[k, 1]], '-', color=run_color, linewidth=1)
                        if fig is not None: fig.savefig(f"{Scenario.name}_final.png")

    dist_series = np.linalg.norm(x1_hist - x2_hist, axis=1)
    a1_mag = np.linalg.norm(a1_hist, axis=1)  # magnitude gives total acceleration per step
    a2_mag = np.linalg.norm(a2_hist, axis=1)
    scenario_time = t_hist[-1, 0]
    if infeasible_run:
        dist_mean = -1.0
        dist_std = -1.0
        dist_max = -1.0
        a1_mean = -1.0
        a1_std = -1.0
        a2_mean = -1.0
        a2_std = -1.0
        time_std = -1.0
    else:
        dist_mean = np.abs(dist_series-GameSol.d).mean()
        dist_std = np.abs(dist_series-GameSol.d).std()
        dist_max = np.abs(dist_series-GameSol.d).max()
        a1_mean = a1_mag.mean()
        a1_std = a1_mag.std()
        a2_mean = a2_mag.mean()
        a2_std = a2_mag.std()
        time_std = np.std([scenario_time])
    stats_vector = np.array(
        [
            alpha,
            n_mc + 1,
            dist_mean,
            dist_std,
            dist_max,
            a1_mean,
            a1_std,
            a2_mean,
            a2_std,
            scenario_time,
            time_std,
        ],
        dtype=float,
    )

    if rt_plot:
        if Frames:
            writer = imageio.get_writer(
                f"{Scenario.name}.mp4",
                fps=3,
                codec='libx264',
                quality=8,
                macro_block_size=16,
            )
            for frame in Frames:
                assert frame.shape == (h_target, w_target, 3), f"Bad frame shape {frame.shape}"
                writer.append_data(frame)
            writer.close()
        plt.close(fig)

    return stats_vector

###
def run_scenario(Scenario: Scenario):
    mc_run_stats = []
    for alpha in alpha_vec:
        mc_indices = list(range(Scenario.Nmc))
        # Plot the first few MC runs (up to 4); color the first nominal differently
        plot_runs = mc_indices[:4]
        plot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for idx, mc_idx in enumerate(plot_runs):
            color = plot_colors[idx if idx < len(plot_colors) else -1]
            mc_run_stats.append(run_single_mc(Scenario, alpha, mc_idx, enable_plot=True, plot_color=color))
        mc_indices = mc_indices[len(plot_runs):]
        if mc_indices:
            max_workers = max(1, min(len(mc_indices), (int(os.cpu_count()/2) or 1) - 1))
            # Use spawn so Julia is initialized fresh per worker (fork + Julia can segfault/ReadOnlyMemoryError)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
                futures = [executor.submit(run_single_mc, Scenario, alpha, mc_idx, False) for mc_idx in mc_indices]
                for fut in futures:
                    mc_run_stats.append(fut.result())

    if mc_run_stats:
        mc_array = np.vstack(mc_run_stats)
        # Sort by alpha then MC index for readability
        mc_array = mc_array[np.lexsort((mc_array[:, 1], mc_array[:, 0]))]
        header = ",".join(
            [
                "alpha",
                "mc_run",
                "distance_mean",
                "distance_std",
                "distance_max",
                "a1_acc_mean",
                "a1_acc_std",
                "a2_acc_mean",
                "a2_acc_std",
                "scenario_time",
                "scenario_time_std",
            ]
        )
        np.savetxt(
            f"{Scenario.name}_mc_stats.csv",
            mc_array,
            delimiter=",",
            header=header,
            comments="",
        )


def main():
    if not Scenarios:
        return

    if len(Scenarios) == 1:
        run_scenario(Scenarios[0])
        return
    workers = min(len(Scenarios), os.cpu_count()-8 or 1)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        executor.map(run_scenario, Scenarios)


if __name__ == '__main__':
    main()
