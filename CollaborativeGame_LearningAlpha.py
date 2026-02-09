#!/usr/bin/env python3
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

import CollaborativeGameSolver as GameSolver
import CollaborativeGameGeneralSolver as GeneralGameSolver

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

from AuxilityFuncs import (
    Scenario,
    init_plot_context,
    limited_cmd,
    random_scenarios,
    update_plot_context,
    finalize_plot_context,
    write_frames_to_mp4,
)

os.system('clear')


Tf = 15.0
alpha_vec = [0.1, 0.25, 0.5, 0.1, 0.25, 0.5, 0.1, 0.25, 0.5]
beta_vec  = [0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5]
dalpha, dbeta = 0.1, 0.1

N = 10
dt_Solver = 0.1
dt_Sim = 0.1
a1_acc_limit = 10.0
Human_PreDefined_Traj = True
Human_RandomWalk_Traj = False

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
Scenario(name="Scenario_4", x1_init=np.array( [[-1.5, 6.5]]), x2_init=np.array([[-1.5+7*np.cos(np.deg2rad(225)), 6.5+7*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, -.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.5, 0.0]]), "diam": 3.0}], Nmc=75),
# Scenario(name="Scenario_3", x1_init=np.array([[-1.5, 6.5]]), x2_init=np.array([[-1.5+3*np.cos(np.deg2rad(225)), 6.5+3*np.sin(np.deg2rad(225))]]), x1_des=np.array([[8.0, 2.0]]), theta_des=np.deg2rad(90.0), obstacles=[{"Pos": np.array([[0.0, 4.0]]), "diam": 1.0}]),
# Scenario(name="Scenario_6_WithObs", x1_init=np.array([[3.0, 0.0]]), x2_init=np.array([[0.0, 0.0]]), x1_des=np.array([[7.0, 0.0]]), theta_des=np.deg2rad(60.0), obstacles=[{"Pos": np.array([[6.0, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_6_Switch_WithObs", x1_init=np.array([[0.0, 0.0]]), x2_init=np.array([[3.0, 0.0]]), x1_des=np.array([[7.0, 4.0]]), theta_des=np.deg2rad(240.0), obstacles=[{"Pos": np.array([[4.5, 2.0]]), "diam": 0.5}]),
# Scenario(name="Scenario_7", x1_init=np.array([[2.0, -1.0]]), x2_init=np.array([[5.0, -1.0]]), x1_des=np.array([[2.0, 8.0]]), theta_des=np.deg2rad(0.0), obstacles=[{"Pos": np.array([[3.0, 4.0]]), "diam": 0.5}]),
]

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
    if rt_plot and plot_context.get("fig") is None:
        plot_context.update(
            init_plot_context(Scenario, GameSol, x1_init, x2_init, x1_des, x2_des, Tf)
        )

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
                dalpha = (0.9-0.1)/Ntries - alpha
                dbeta = (0.9-0.1)/Ntries - beta
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
                dalpha = (0.9-0.1)/Ntries - alpha
                dbeta = (0.9-0.1)/Ntries - beta
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
            a1_cmd = limited_cmd(a1_cmd, a1_acc_limit)
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
        a1_cmd = limited_cmd(a1_cmd, a1_acc_limit)
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
            update_plot_context(
                plot_context,
                t,
                x1_state,
                x2_state,
                v1_state,
                v2_state,
                a1_cmd,
                a2_cmd,
                x1_hist,
                x2_hist,
                v1_hist,
                v2_hist,
                a1_hist,
                a2_hist,
                t_hist,
                GameSol,
                SolverType,
                Scenario,
                n_mc,
                x1_des,
                x2_des,
                dt_Solver,
                dt_Sim,
            )

        if EndSimulation or (t >= Tf) or (np.linalg.norm(x1_state - x1_des) + np.linalg.norm(x2_state - x2_des) < 1.0):
            EndSimulation = True
            if rt_plot:
                finalize_plot_context(
                    plot_context,
                    Scenario,
                    ialpha,
                    alpha,
                    hist_col,
                    t_hist,
                    v1_hist,
                    v2_hist,
                    a1_hist,
                    a2_hist,
                    x1_hist,
                    x2_hist,
                )

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
            beta,
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
            N,
            dt_Solver,
            dt_Sim,
            a1_acc_limit,
            1.0 if Human_PreDefined_Traj else 0.0,
            1.0 if Human_RandomWalk_Traj else 0.0,
            d_init,
            Tf,
            pp_theta,
            pp_factor,
            float(Scenario.Nmc),
            float(theta_des),
            float(x1_init[0, 0]),
            float(x1_init[0, 1]),
            float(x2_init[0, 0]),
            float(x2_init[0, 1]),
            float(x1_des[0, 0]),
            float(x1_des[0, 1]),
            float(len(Obstcles)),
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
            max_workers = max(1, min(25, len(mc_indices), (os.cpu_count()-12 or 1)))
            # Use spawn so Julia is initialized fresh per worker (fork can cause ReadOnlyMemoryError)
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
                futures = [
                    executor.submit(run_single_mc, Scenario, alpha, beta, ialpha, mc_idx, SolverType, None)
                    for mc_idx in mc_indices
                ]
                for fut in futures:
                    mc_run_stats.append(fut.result())

    if plot_context is not None:
        write_frames_to_mp4(plot_context, Scenario.name)

    if mc_run_stats and Scenario.Nmc > 1:
        header = ",".join(
            [
                "alpha",
                "beta",
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
