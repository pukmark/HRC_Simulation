#!/usr/bin/env python3
import os
os.system('clear')

import numpy as np
import casadi as ca

import CollaborativeGameSolver as GameSolver

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

RT_Plot = True

hist_col = ['c','m','k','y']

Tf = 10.0
alpha_vec = [0.01, 0.5, 0.99]

N = 10
dt = 0.1

Obstcles = []
Obstcles.append({'Pos': np.array([[6,0]]), "diam": 0.5})

# Define the initial conditions
Scenarios =[]
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
# Scenarios.append({'Name': 'Scenario_6_WithObs', 'x1_init': np.array([[3, 0]]), 'x2_init':np.array([[0, 0]]), 'x1_des': np.array([[7, 0]]), 'theta_des':np.deg2rad(60), 'Obs': [{'Pos': np.array([[6,2]]), "diam": 0.5}]})
Scenarios.append({'Name': 'Scenario_6_Switch_WithObs', 'x1_init': np.array([[0, 0]]), 'x2_init':np.array([[3, 0]]), 'x1_des': np.array([[7, 4]]), 'theta_des':np.deg2rad(240), 'Obs': [{'Pos': np.array([[4.5,2]]), "diam": 0.5}]})


for Scenario in Scenarios:
    x1_init, x2_init, x1_des, theta_des = Scenario['x1_init'], Scenario['x2_init'], Scenario['x1_des'], Scenario['theta_des']
    Obstcles = Scenario['Obs']

    d_init = np.linalg.norm(x1_init-x2_init)
    x2_des = x1_des + d_init * np.array([np.cos(theta_des), np.sin(theta_des)])


    GameSol = GameSolver.CollaborativeGame(N = N, dt = dt, d=d_init, Obstcles=Scenario['Obs'])

    if RT_Plot:
        FIG_INCHES = 6.3   # → 1008 px with dpi=160
        DPI = 160
        fig = plt.figure(figsize=(FIG_INCHES, FIG_INCHES), dpi=DPI, constrained_layout=False)
        ax_xy = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        p1_plot = ax_xy.plot([],[], 'gs', markersize=6, label='Human')[0]
        p2_plot = ax_xy.plot([],[], 'bs', markersize=6, label='Robot')[0]
        p12_line = ax_xy.plot([],[], 'r-', linewidth=3)[0]
        p12_line_pred = []
        for _ in range(4):
            p12_line_pred.append(ax_xy.plot([],[], 'r:', linewidth=3)[0])
        p1_pred = ax_xy.plot([],[], 'g-', linewidth=3)[0]
        p2_pred = ax_xy.plot([],[], 'b-', linewidth=3)[0]
        p1_hist = ax_xy.plot([],[], 'g-.', linewidth=3)[0]
        p2_hist = ax_xy.plot([],[], 'b-.', linewidth=3)[0]
        tgt1_plot = ax_xy.plot([],[], 'go', markersize=6)[0]
        tgt2_plot = ax_xy.plot([],[], 'bo', markersize=6)[0]
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel('x [m]')
        ax_xy.set_ylabel('y [m]')
        ax_xy.set_xlim([0,10])
        ax_xy.set_ylim([-5,10])
        ax_xy.grid(True)
        ax_xy.legend()
        for Obstcle in Obstcles:
            x, y = Obstcle['diam']/2*np.cos(np.linspace(0,2*np.pi,100)), Obstcle['diam']/2*np.sin(np.linspace(0,2*np.pi,100))
            ax_xy.plot(Obstcle['Pos'][0,0]+x, Obstcle['Pos'][0,1]+y,'k-', linewidth=3)
        

        ax_vel = plt.subplot2grid((3, 3), (0, 2), colspan=1, rowspan=1)
        p1_vel = ax_vel.plot([],[], 'gs', markersize=6, label='Human')[0]
        p2_vel = ax_vel.plot([],[], 'bs', markersize=6, label='Robot')[0]
        p1_vel_pred = ax_vel.plot([], [], 'g-', markersize=3)[0]
        p2_vel_pred = ax_vel.plot([], [], 'b-', markersize=3)[0]
        p1_vel_hist = ax_vel.plot([], [], 'g-.', markersize=3)[0]
        p2_vel_hist = ax_vel.plot([], [], 'b-.', markersize=3)[0]
        ax_vel.set_xlim([0,Tf])
        ax_vel.set_ylim([0,max(GameSol.v1_max, GameSol.v2_max)+1])
        p1_vel_max = ax_vel.plot([0, Tf], [GameSol.v1_max, GameSol.v1_max], 'g:', markersize=3)[0]
        p2_vel_max = ax_vel.plot([0, Tf], [GameSol.v2_max, GameSol.v2_max], 'b:', markersize=3)[0]
        ax_vel.grid(True)
        ax_vel.legend()
        ax_vel.set_xlabel('Time [Sec]')
        ax_vel.set_ylabel('Velocity [m/s]')

        ax_acc = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=1)
        p1_acc = ax_acc.plot([],[], 'gs', markersize=6, label='Human')[0]
        p2_acc = ax_acc.plot([],[], 'bs', markersize=6, label='Robot')[0]
        p1_acc_pred = ax_acc.plot([], [], 'g-', markersize=3)[0]
        p2_acc_pred = ax_acc.plot([], [], 'b-', markersize=3)[0]
        p1_acc_hist = ax_acc.plot([], [], 'g-.', markersize=3)[0]
        p2_acc_hist = ax_acc.plot([], [], 'b-.', markersize=3)[0]
        ax_acc.set_xlim([0,Tf])
        ax_acc.set_ylim([0,max(GameSol.a1_max, GameSol.a2_max)+1])
        p1_acc_max = ax_acc.plot([0, Tf], [GameSol.a1_max, GameSol.a1_max], 'g:', markersize=3)[0]
        p2_acc_max = ax_acc.plot([0, Tf], [GameSol.a2_max, GameSol.a2_max], 'b:', markersize=3)[0]
        ax_acc.grid(True)
        ax_acc.legend()
        ax_acc.set_xlabel('Time [Sec]')
        ax_acc.set_ylabel('Acceleration [m/s^2]')

        ax_dist = plt.subplot2grid((3, 3), (2, 2), colspan=1, rowspan=1)
        p12_dist = ax_dist.plot([],[], 'rs', markersize=6)[0]
        p12_dist_pred = ax_dist.plot([], [], 'b-', markersize=3)[0]
        p12_dist_hist = ax_dist.plot([], [], 'b-.', markersize=3)[0]
        p12_dist_min = ax_dist.plot([0, Tf], [GameSol.d_min, GameSol.d_min], 'b:', markersize=3)[0]
        p12_dist_max = ax_dist.plot([0, Tf], [GameSol.d_max, GameSol.d_max], 'b:', markersize=3)[0]
        ax_dist.set_xlim([0,Tf])
        ax_dist.set_ylim([GameSol.d_min*0.75, GameSol.d_max*1.05])
        ax_dist.grid(True)
        ax_dist.set_xlabel('Time [Sec]')
        ax_dist.set_ylabel('Distance [m]')

        # After creating fig and subplots:
        canvas = FigureCanvas(fig)

        # Lock target pixel dims
        w_target = int(FIG_INCHES * DPI)  # 1008
        h_target = int(FIG_INCHES * DPI)  # 1008
        Frames = []

    for ialpha, alpha in enumerate(alpha_vec):
        t = 0.0
        t_hist = np.array([[t]])
        x1_hist, v1_hist, a1_hist = x1_init, np.zeros((1,2)), np.zeros((0,2))
        x2_hist, v2_hist, a2_hist = x2_init, np.zeros((1,2)), np.zeros((0,2))

        x1_state, x2_state = x1_init, x2_init
        v1_state, v2_state = np.zeros((1,2)), np.zeros((1,2))
        EndSimulation = False
        i_acc = 0
        GameSol.success = False
        while not EndSimulation:
            # Save The game

            if GameSol.success:
                z0 = GameSol.z0
                z0[GameSol.indx_x1:GameSol.indx_x1+N] = GameSol.sol.x1_sol[1:,0]
                z0[GameSol.indx_y1:GameSol.indx_y1+N] = GameSol.sol.x1_sol[1:,1]
                z0[GameSol.indx_vx1:GameSol.indx_vx1+N] = GameSol.sol.v1_sol[1:,0]
                z0[GameSol.indx_vy1:GameSol.indx_vy1+N] = GameSol.sol.v1_sol[1:,1]
                z0[GameSol.indx_ax1:GameSol.indx_ax1+N-1] = GameSol.sol.a1_sol[1:,0]
                z0[GameSol.indx_ay1:GameSol.indx_ay1+N-1] = GameSol.sol.a1_sol[1:,1]
                z0[GameSol.indx_x2:GameSol.indx_x2+N] = GameSol.sol.x2_sol[1:,0]
                z0[GameSol.indx_y2:GameSol.indx_y2+N] = GameSol.sol.x2_sol[1:,1]
                z0[GameSol.indx_vx2:GameSol.indx_vx2+N] = GameSol.sol.v2_sol[1:,0]
                z0[GameSol.indx_vy2:GameSol.indx_vy2+N] = GameSol.sol.v2_sol[1:,1]
                z0[GameSol.indx_ax2:GameSol.indx_ax2+N-1] = GameSol.sol.a2_sol[1:,0]
                z0[GameSol.indx_ay2:GameSol.indx_ay2+N-1] = GameSol.sol.a2_sol[1:,1]

                GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0)
            if not GameSol.success:
                # calculate mpc initial guess for human:
                x1_guess, v1_guess, a1_guess = GameSol.MPC_guess_human_calc(x1_state, v1_state, x1_des)
                x2_guess, v2_guess, a2_guess = GameSol.MPC_guess_robot_calc(x2_state, v2_state, x2_des, x_partner = x1_guess)

                z0 = np.zeros_like(GameSol.z0)
                z0[GameSol.indx_x1:GameSol.indx_x1+N+1] = x1_guess[:,0]
                z0[GameSol.indx_y1:GameSol.indx_y1+N+1] = x1_guess[:,1]
                z0[GameSol.indx_vx1:GameSol.indx_vx1+N+1] = v1_guess[:,0]
                z0[GameSol.indx_vy1:GameSol.indx_vy1+N+1] = v1_guess[:,1]
                z0[GameSol.indx_ax1:GameSol.indx_ax1+N] = a1_guess[:,0]
                z0[GameSol.indx_ay1:GameSol.indx_ay1+N] = a1_guess[:,1]
                z0[GameSol.indx_x2:GameSol.indx_x2+N+1] = x2_guess[:,0]
                z0[GameSol.indx_y2:GameSol.indx_y2+N+1] = x2_guess[:,1]
                z0[GameSol.indx_vx2:GameSol.indx_vx2+N+1] = v2_guess[:,0]
                z0[GameSol.indx_vy2:GameSol.indx_vy2+N+1] = v2_guess[:,1]
                z0[GameSol.indx_ax2:GameSol.indx_ax2+N] = a2_guess[:,0]
                z0[GameSol.indx_ay2:GameSol.indx_ay2+N] = a2_guess[:,1]
                    
                GameSol.Solve(t, x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0)

            if GameSol.success:
                i_acc = 0
            else:
                i_acc += 1

            # Dynamics
            a1_cmd = LimitedCmd(GameSol.sol.a1_sol[i_acc,:] + 0*np.random.normal(0.0, 1.0, 2), GameSol.a1_max)
            a2_cmd = LimitedCmd(GameSol.sol.a2_sol[i_acc,:], GameSol.a2_max)
            x1_state = x1_state + dt * v1_state + 0.5*dt**2*a1_cmd
            v1_state = v1_state + dt * a1_cmd
            x2_state = x2_state + dt * v2_state + 0.5*dt**2*a2_cmd
            v2_state = v2_state + dt * a2_cmd

            t += dt

            # Save state
            x1_hist = np.vstack((x1_hist, x1_state))
            v1_hist = np.vstack((v1_hist, v1_state))
            a1_hist = np.vstack((a1_hist, a1_cmd))
            x2_hist = np.vstack((x2_hist, x2_state))
            v2_hist = np.vstack((v2_hist, v2_state))
            a2_hist = np.vstack((a2_hist, a2_cmd))
            t_hist = np.vstack((t_hist, t))

            if RT_Plot:
                p1_plot.set_data([x1_state[0,0]], [x1_state[0,1]])
                p2_plot.set_data([x2_state[0,0]], [x2_state[0,1]])
                p12_line.set_data([x1_state[0,0], x2_state[0,0]],[x1_state[0,1], x2_state[0,1]])
                indx = np.linspace(0, N,1+len(p12_line_pred))
                for i in range(len(p12_line_pred)):
                    i_indx = int(indx[i+1])
                    p12_line_pred[i].set_data([GameSol.sol.x1_sol[i_indx,0], GameSol.sol.x2_sol[i_indx,0]], [GameSol.sol.x1_sol[i_indx,1], GameSol.sol.x2_sol[i_indx,1]])
                p1_pred.set_data(GameSol.sol.x1_sol[:,0], GameSol.sol.x1_sol[:,1])
                p2_pred.set_data(GameSol.sol.x2_sol[:,0], GameSol.sol.x2_sol[:,1])
                p1_hist.set_data(x1_hist[:,0], x1_hist[:,1])
                p2_hist.set_data(x2_hist[:,0], x2_hist[:,1])
                tgt1_plot.set_data([x1_des[0,0]], [x1_des[0,1]])
                tgt2_plot.set_data([x2_des[0,0]], [x2_des[0,1]])
                ax_xy.set_title(f'Alpha is {alpha}, Time: {t:2.2}[Sec]')
                ax_xy.legend()
                
                t_pred = np.linspace(t-dt, t-dt+N*dt, N+1)
                p1_vel.set_data([t], [np.linalg.norm(v1_state)])
                p2_vel.set_data([t], [np.linalg.norm(v2_state)])
                p1_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v1_sol, axis=1))
                p2_vel_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.v2_sol, axis=1))
                p1_vel_hist.set_data(t_hist, np.linalg.norm(v1_hist, axis=1))
                p2_vel_hist.set_data(t_hist, np.linalg.norm(v2_hist, axis=1))

                p1_acc.set_data([t], [np.linalg.norm(a1_cmd)])
                p2_acc.set_data([t], [np.linalg.norm(a2_cmd)])
                p1_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a1_sol, axis=1))
                p2_acc_pred.set_data(t_pred[:-1], np.linalg.norm(GameSol.sol.a2_sol, axis=1))
                p1_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a1_hist, axis=1))
                p2_acc_hist.set_data(t_hist[:-1], np.linalg.norm(a2_hist, axis=1))

                p12_dist.set_data([t], [np.linalg.norm(x1_state-x2_state)])
                p12_dist_pred.set_data(t_pred, np.linalg.norm(GameSol.sol.x1_sol-GameSol.sol.x2_sol, axis=1))
                p12_dist_hist.set_data(t_hist[:, 0], np.linalg.norm(x1_hist-x2_hist, axis=1))

                plt.pause(0.1)

                fig.canvas.draw()
                frame = capture_frame_agg(fig, canvas, w_target, h_target)
                Frames.append(frame)
            else:
                print("Current State: x1:", x1_state, "x2_state:", x2_state)

            # Check if to End Simulation
            if t >= Tf or ( np.linalg.norm(x1_state - x1_des)<0.2 and np.linalg.norm(x2_state - x2_des)<0.2):
                EndSimulation = True
                if RT_Plot:
                    ax_xy.plot(x1_hist[:,0], x1_hist[:,1],'--', color=hist_col[ialpha], linewidth=2, label=f'alpha={alpha}')
                    ax_xy.plot(x2_hist[:,0], x2_hist[:,1],'--', color=hist_col[ialpha], linewidth=2)


    writer = imageio.get_writer(
    f'{Scenario["Name"]}.mp4',
    fps=3,
    codec='libx264',
    quality=8,
    macro_block_size=16  # safe: 1008 is divisible by 16; no auto-resize
    )
    for frame in Frames:
        # sanity check (defensive)
        assert frame.shape == (h_target, w_target, 3), f"Bad frame shape {frame.shape}"
        writer.append_data(frame)
    writer.close()
    plt.close(fig)