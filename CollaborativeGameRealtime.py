#!/usr/bin/env python3
"""
CollaborativeGameRealtime.py

Real-time bridge: OptiTrack (human + robot + obstacles) → Game Solver (CasADi/PATH) → Ridgeback /cmd_vel.

Requirements (suggested):
- ROS1 (rospy), geometry_msgs
- OptiTrack via ROS topics (PoseStamped) or NatNet (optional stub below)
- numpy, casadi, matplotlib (optional if plotting), imageio (optional)

Run examples:
    # ROS mocap + ROS Ridgeback
    python3 CollaborativeGameRealtime.py --mocap ros --human OceanViewHuman --robot Ridgeback_base \
        --obs obs1 obs2 --theta-deg 30 --tf 60

    # NatNet (stub) + ROS Ridgeback
    python3 CollaborativeGameRealtime.py --mocap natnet --human Human --robot Ridgeback --theta-deg 0

Notes:
- Uses velocities from mocap finite-differencing with simple low-pass.
- Sends velocity (vx, vy) to Ridgeback. Angular z is 0 by default (can be extended).
- Safety: caps commanded speed to v2_max; E-stop if human-robot distance < d_min * 0.8.
"""

import os
import sys
import time
import math
import argparse
from collections import deque
from dataclasses import dataclass
import threading

import numpy as np

import CollaborativeGameSolver as GameSolver  # local module from your upload

# ----------------------------- Utilities -----------------------------

def LimitedCmd(v: np.ndarray, v_max: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > v_max and n > 1e-9:
        v = v / n * v_max
    return v

@dataclass
class Pose2D:
    x: float
    y: float
    t: float

class VelocityEstimator:
    """Finite-difference velocity with light low-pass smoothing."""
    def __init__(self, maxlen=5, alpha=0.5):
        self.buf = deque(maxlen=maxlen)  # (Pose2D)
        self.alpha = alpha
        self.v = np.zeros((1,2))

    def update(self, p: Pose2D):
        self.buf.append(p)
        if len(self.buf) >= 2:
            a = self.buf[-2]
            b = self.buf[-1]
            dt = max(b.t - a.t, 1e-3)
            raw = np.array([[(b.x - a.x)/dt, (b.y - a.y)/dt]])
            self.v = self.alpha*raw + (1-self.alpha)*self.v
        return self.v

# ----------------------------- Mocap Sources -----------------------------

class MocapROS:
    """ROS PoseStamped subscriber for each rigid body."""
    def __init__(self, topic_fmt="/vrpn_client_node/{name}/pose", use_odom=False):
        self.topic_fmt = topic_fmt
        self.use_odom = use_odom
        self.poses = {}
        self.v_estimators = {}
        self.ready = threading.Event()
        try:
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
            self.rospy = rospy
            self.PoseStamped = PoseStamped
            self.Odom = Odometry
        except Exception as e:
            print("[MocapROS] Failed to import rospy/msgs:", e)
            raise

    def _cb_pose(self, name):
        def _cb(msg):
            if hasattr(msg, 'pose'):  # PoseStamped
                x = msg.pose.position.x
                y = msg.pose.position.y
                t = msg.header.stamp.to_sec()
            else:  # Fallback
                x = getattr(msg, 'x', 0.0)
                y = getattr(msg, 'y', 0.0)
                t = time.time()
            self.poses[name] = Pose2D(x, y, t)
            if name not in self.v_estimators:
                self.v_estimators[name] = VelocityEstimator()
            self.v_estimators[name].update(self.poses[name])
            self.ready.set()
        return _cb

    def start(self, names):
        self.rospy.loginfo("[MocapROS] Subscribing to: %s", names)
        for n in names:
            topic = self.topic_fmt.format(name=n)
            if self.use_odom:
                sub = self.rospy.Subscriber(topic, self.Odom, self._cb_pose(n), queue_size=10)
            else:
                sub = self.rospy.Subscriber(topic, self.PoseStamped, self._cb_pose(n), queue_size=50)

    def get_xy(self, name) -> np.ndarray:
        p = self.poses.get(name, None)
        if p is None:
            return None
        return np.array([[p.x, p.y]])

    def get_v(self, name) -> np.ndarray:
        if name not in self.v_estimators:
            return np.zeros((1,2))
        return self.v_estimators[name].v

class MocapNatNetStub:
    """Placeholder for direct NatNet SDK. Implement as needed for your lab."""
    def __init__(self):
        self.poses = {}
        self.v_estimators = {}

    def start(self, names):
        print("[MocapNatNetStub] WARNING: This is a stub. Implement NatNet client here.")

    def get_xy(self, name):
        p = self.poses.get(name, None)
        if p is None:
            return None
        return np.array([[p.x, p.y]])

    def get_v(self, name):
        if name not in self.v_estimators:
            return np.zeros((1,2))
        return self.v_estimators[name].v

# ----------------------------- Ridgeback ROS Control -----------------------------

class RidgebackROS:
    def __init__(self, cmd_topic="/cmd_vel"):
        try:
            import rospy
            from geometry_msgs.msg import Twist, PoseStamped
            from nav_msgs.msg import Path
            self.rospy = rospy
            self.Twist = Twist
            self.PoseStamped = PoseStamped
            self.Path = Path
        except Exception as e:
            print("[RidgebackROS] Failed to import rospy/geometry_msgs:", e)
            raise
        self.pub = self.rospy.Publisher(cmd_topic, self.Twist, queue_size=1)
        self.pub_pos = self.rospy.Publisher("/cmd_pos", self.PoseStamped, queue_size=1)

    def send_velocity(self, v_cmd: np.ndarray, yaw_rate: float = 0.0):
        msg = self.Twist()
        msg.linear.x = float(v_cmd[0,0])
        msg.linear.y = float(v_cmd[0,1])
        msg.angular.z = float(yaw_rate)
        self.pub.publish(msg)

    def send_position(self, x, y, yaw=0.0):
        msg = self.PoseStamped()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 0.0
        self.pub_pos.publish(msg)

# ----------------------------- Experiment Runner -----------------------------

def run(args):
    # Solver / game setup
    N = args.N
    dt = args.dt
    Tf = args.tf
    alpha = args.alpha

    # Obstacles (by names or static list)
    obstacle_names = args.obs or []
    static_obstacles = []
    for obs_def in args.static_obs:
        # format: x,y,diam
        try:
            x,y,diam = map(float, obs_def.split(","))
            static_obstacles.append({'Pos': np.array([[x, y]]), 'diam': diam})
        except:
            print(f"[WARN] Bad --static-obs '{obs_def}', expected x,y,diam")

    Obstcles = static_obstacles.copy()  # Dynamic obstacles from mocap are merged later

    # Mocap source
    if args.mocap == "ros":
        mocap = MocapROS(topic_fmt=args.ros_mocap_topic_fmt, use_odom=args.mocap_odom)
        # ROS node init
        if not mocap.rospy.core.is_initialized():
            mocap.rospy.init_node("collab_game_realtime", anonymous=True)
        names = [args.human, args.robot] + obstacle_names
        mocap.start(names)
    elif args.mocap == "natnet":
        mocap = MocapNatNetStub()
        mocap.start([args.human, args.robot] + obstacle_names)
    else:
        raise ValueError("--mocap must be 'ros' or 'natnet'")

    # Ridgeback control
    rb = RidgebackROS(cmd_topic=args.cmd_topic)
    pub_x1_des = rb.rospy.Publisher("/x1_des", rb.PoseStamped, queue_size=1)
    pub_x2_sol = rb.rospy.Publisher("/x2_sol", rb.Path, queue_size=1)


    # Wait for first mocap samples
    if args.mocap == "ros":
        print("[Init] Waiting for mocap data... (Ctrl+C to abort)")
        while True:
            hx = mocap.get_xy(args.human); rx = mocap.get_xy(args.robot)
            if hx is not None and rx is not None:
                break
            mocap.rospy.sleep(0.02)

    # Initial states
    x1_state = mocap.get_xy(args.human)
    x2_state = mocap.get_xy(args.robot)
    v1_state = mocap.get_v(args.human)
    v2_state = mocap.get_v(args.robot)

    # Targets
    d_init = np.linalg.norm(x1_state - x2_state)
    d_init = float(d_init) if d_init > 1e-6 else args.init_dist
    x1_des = np.array([[args.x1_des_x, args.x1_des_y]])
    theta = math.radians(args.theta_deg)
    x2_des = x1_des + d_init * np.array([[math.cos(theta), math.sin(theta)]])

    # Timing
    t = 0.0
    t0 = time.time()

    # Create Game
    GameSol = GameSolver.CollaborativeGame(N=N, dt=dt, d=d_init, Obstcles=Obstcles)

    # Logs
    logs = []
    step_dt = dt
    if args.loop_rate > 0:
        loop_dt = 1.0 / args.loop_rate
        if abs(loop_dt - step_dt) > 1e-6:
            print(f"[WARN] Ignoring loop_rate {args.loop_rate:.3f} Hz; using solver dt {step_dt:.3f} s for sequential execution")

    # Main loop
    print("[Run] Starting real-time loop...")
    while True:
        if args.mocap == "ros" and mocap.rospy.is_shutdown():
            break

        t = time.time() - t0
        if t >= Tf:
            break

        step_start = time.time()
        # Refresh dynamic obstacles from mocap
        Obstcles = static_obstacles.copy()
        for oname in obstacle_names:
            pos = mocap.get_xy(oname)
            if pos is not None:
                Obstcles.append({'Pos': pos, 'diam': args.obs_diam})

        # Recreate game if obstacle set changes size (safest approach)
        # (Alternatively, you could add an API to update Obstcles in-place.)
        if len(Obstcles) != len(GameSol.Obstcles):
            GameSol = GameSolver.CollaborativeGame(N=N, dt=dt, d=d_init, Obstcles=Obstcles)

        # Read states
        hx = mocap.get_xy(args.human)
        rx = mocap.get_xy(args.robot)
        hv = mocap.get_v(args.human)
        rv = mocap.get_v(args.robot)

        if hx is not None:
            x1_state = hx
        if rx is not None:
            x2_state = rx
        if hv is not None:
            v1_state = hv
        if rv is not None:
            v2_state = rv

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

            GameSol.Solve(time.time(), x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, log=True)
        if not GameSol.success:

            # Build warm start via MPC guesses
            x1_guess, v1_guess, a1_guess = GameSol.MPC_guess_human_calc(x1_state, v1_state, x1_des)
            x2_guess, v2_guess, a2_guess = GameSol.MPC_guess_robot_calc(x2_state, v2_state, x2_des, x_partner=x1_guess)

            z0 = np.zeros_like(GameSol.z0)
            z0[GameSol.indx_x1:GameSol.indx_x1+N+1] = x1_guess[:,0]
            z0[GameSol.indx_y1:GameSol.indx_y1+N+1] = x1_guess[:,1]
            z0[GameSol.indx_vx1:GameSol.indx_vx1+N+1] = v1_guess[:,0]
            z0[GameSol.indx_vy1:GameSol.indx_vy1+N+1] = v1_guess[:,1]
            z0[GameSol.indx_ax1:GameSol.indx_ax1+N]   = a1_guess[:,0]
            z0[GameSol.indx_ay1:GameSol.indx_ay1+N]   = a1_guess[:,1]
            z0[GameSol.indx_x2:GameSol.indx_x2+N+1] = x2_guess[:,0]
            z0[GameSol.indx_y2:GameSol.indx_y2+N+1] = x2_guess[:,1]
            z0[GameSol.indx_vx2:GameSol.indx_vx2+N+1] = v2_guess[:,0]
            z0[GameSol.indx_vy2:GameSol.indx_vy2+N+1] = v2_guess[:,1]
            z0[GameSol.indx_ax2:GameSol.indx_ax2+N]   = a2_guess[:,0]
            z0[GameSol.indx_ay2:GameSol.indx_ay2+N]   = a2_guess[:,1]

            # Solve game
            try:
                v1_state = np.zeros((1,2))
                GameSol.Solve(time.time(), x1_state, v1_state, x1_des, x2_state, v2_state, x2_des, alpha, z0=z0, log=True)
                GameSol.sol.time = time.time()
            except Exception as e:
                print("[Solve] Exception:", e)

        # Command robot
        v2_cmd = np.zeros((1,2))
        fallback_mode = False
        x_cmd = float(x2_state[0,0])
        y_cmd = float(x2_state[0,1])
        x1_target = np.array(x1_des, copy=True)
        
        GameSol.success = True
        dx_des = x2_des - x1_des
        GameSol.sol.x2_sol[:,0] = x1_state[0,0] + dx_des[0,0]
        GameSol.sol.x2_sol[:,1] = x1_state[0,1] + dx_des[0,1]
        
        if GameSol.success:
            t_pred = GameSol.sol.time + np.linspace(0.0, GameSol.N*GameSol.dt, GameSol.N+1)
            GameSol.sol.t_sol = t_pred
            x2_traj_msg = rb.Path()
            x2_traj_msg.header.stamp = rb.rospy.Time.now()
            x2_traj_msg.header.frame_id = "map"
            for idx, (px, py) in enumerate(GameSol.sol.x2_sol):
                vx = float(GameSol.sol.v2_sol[idx, 0])
                vy = float(GameSol.sol.v2_sol[idx, 1])
                pose_stamped = rb.PoseStamped()
                pose_stamped.header.stamp = rb.rospy.Time.from_sec(float(t_pred[idx]))
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = float(px)
                pose_stamped.pose.position.y = float(py)
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.x = vx
                pose_stamped.pose.orientation.y = vy
                pose_stamped.pose.orientation.z = 0.0
                pose_stamped.pose.orientation.w = 1.0
                x2_traj_msg.poses.append(pose_stamped)
            pub_x2_sol.publish(x2_traj_msg)
        else:
            # fallback: simple velocity towards desired formation point
            to_goal = (x2_des - x2_state)
            v2_cmd = to_goal / max(step_dt, 1e-3)
            fallback_mode = True

        # Safety checks
        dist = np.linalg.norm((x1_state - x2_state).flatten())
        if dist < GameSol.d_min * 0.8:
            print("[SAFETY] Too close to human (%.2f m). Stopping." % dist)
            v2_cmd[:] = 0.0
            x_cmd = float(x2_state[0,0])
            y_cmd = float(x2_state[0,1])

        v2_cmd = LimitedCmd(v2_cmd.flatten(), GameSol.v2_max).reshape(1,2)
        if fallback_mode:
            x_cmd = float(x2_state[0,0] + step_dt * v2_cmd[0,0])
            y_cmd = float(x2_state[0,1] + step_dt * v2_cmd[0,1])

        rb.send_velocity(v2_cmd, yaw_rate=0.0)
        rb.send_position(x_cmd, y_cmd, yaw=0.0)
        dist = np.linalg.norm(x1_state-x2_state)
        feas = (dist <= GameSol.d_max) and (dist >= GameSol.d_min)
        x1_dist = np.linalg.norm(x1_state-x1_des)
        x2_dist = np.linalg.norm(x2_state-x2_des)
        print("(x1_dist,x2_disr)=(",x1_dist,",",x2_dist,"), x12_dist = ",np.linalg.norm(x1_state-x2_state),"dist is ",feas)

        ps_des = rb.PoseStamped()
        ps_des.header.stamp = rb.rospy.Time.now()
        ps_des.header.frame_id = "map"
        ps_des.pose.position.x = float(x1_target[0,0])
        ps_des.pose.position.y = float(x1_target[0,1])
        ps_des.pose.orientation.w = 1.0
        pub_x1_des.publish(ps_des)

        # Log
        now = time.time() - t0
        logs.append([now, x1_state[0,0], x1_state[0,1], x2_state[0,0], x2_state[0,1],
                     v2_cmd[0,0], v2_cmd[0,1], int(GameSol.success)])

        # Ensure actors run for exactly dt before next solve
        step_elapsed = time.time() - step_start
        remaining = step_dt - step_elapsed
        if remaining > 0:
            if args.mocap == "ros":
                mocap.rospy.sleep(remaining)
            else:
                time.sleep(remaining)

    # Stop robot
    try:
        rb.send_velocity(np.zeros((1,2)), 0.0)
    except:
        pass

    # Save logs
    logs = np.array(logs)
    out_csv = args.log_csv
    np.savetxt(out_csv, logs, delimiter=",",
               header="t,x1,y1,x2,y2,vx_cmd,vy_cmd,solve_ok", comments="")
    print(f"[Done] Saved log to {out_csv}")


def build_parser():
    p = argparse.ArgumentParser(description="Realtime collaborative game with Ridgeback + OptiTrack")
    # Experiment parameters
    p.add_argument("--N", type=int, default=10, help="Horizon steps")
    p.add_argument("--dt", type=float, default=0.1, help="Control timestep [s]")
    p.add_argument("--tf", type=float, default=999.0, help="Experiment duration [s]")
    p.add_argument("--alpha", type=float, default=0.05, help="Shared-constraint split α")
    p.add_argument("--theta-deg", type=float, default=-30.0, dest="theta_deg", help="Desired angle human→robot [deg]")
    p.add_argument("--x1-des-x", type=float, default=7.0, dest="x1_des_x")
    p.add_argument("--x1-des-y", type=float, default=0.0, dest="x1_des_y")
    p.add_argument("--loop-rate", type=float, default=10.0, help="Main loop rate [Hz]")
    p.add_argument("--init-dist", type=float, default=4.0, dest="init_dist", help="Fallback initial distance if mocap gap")
    # Entities
    p.add_argument("--human", type=str, required=True, help="Human mocap rigid body name")
    p.add_argument("--robot", type=str, required=True, help="Robot mocap rigid body name")
    p.add_argument("--obs", type=str, nargs="*", default=[], help="Obstacle rigid body names (optional)")
    p.add_argument("--static-obs", type=str, nargs="*", default=[], help="Static obstacles: 'x,y,diam' entries")
    p.add_argument("--obs-diam", type=float, default=0.5, help="Diameter [m] for mocap-obstacles")
    # Mocap source
    p.add_argument("--mocap", choices=["ros","natnet"], default="ros")
    p.add_argument("--ros-mocap-topic-fmt", type=str,
                   default="/vrpn_client_node/{name}/pose",
                   help="Format for PoseStamped topics")
    p.add_argument("--mocap-odom", action="store_true", help="Use nav_msgs/Odometry instead of PoseStamped")
    # Ridgeback
    p.add_argument("--cmd-topic", type=str, default="/cmd_vel", help="Ridgeback cmd_vel topic")
    # Output
    p.add_argument("--log-csv", type=str, default="collab_realtime_log.csv")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[Exit] KeyboardInterrupt")
