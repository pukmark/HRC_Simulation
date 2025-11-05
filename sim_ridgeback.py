#!/usr/bin/env python3
"""
sim_ridgeback.py
ROS1 (rospy) 2D holonomic Ridgeback simulator.
"""
import math
import argparse
import rospy
import tf
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

def q_from_yaw(yaw):
    q = tf.transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(*q)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-name", default="Ridgeback_base")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--amax", type=float, default=10.0)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--child-frame-id", default="base_link")
    args, _ = parser.parse_known_args()

    rospy.init_node("sim_ridgeback")
    robot_name = rospy.get_param("~robot_name", args.robot_name)
    rate_hz = rospy.get_param("~rate", args.rate)
    vmax = rospy.get_param("~vmax", args.vmax)
    amax = rospy.get_param("~amax", args.amax)
    frame_id = rospy.get_param("~frame_id", args.frame_id)
    child_frame_id = rospy.get_param("~child_frame_id", args.child_frame_id)

    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
    pose_pub = rospy.Publisher(f"/vrpn_client_node/{robot_name}/pose", PoseStamped, queue_size=10)
    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)

    cmd = Twist()
    started = False
    def cb_cmd(msg):
        nonlocal cmd
        cmd = msg
        started = True

    cmd_pos = PoseStamped()
    cmd_pos.header.frame_id = frame_id
    cmd_pos.pose.orientation.w = 1.0
    cmd_pos_x = 0.0
    cmd_pos_y = 0.0
    x2_sol_path = []

    def cb_x2_sol(msg):
        nonlocal x2_sol_path
        x2_sol_path = [(pose.header.stamp.to_sec(),
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.orientation.x,
                        pose.pose.orientation.y) for pose in msg.poses]
    rospy.Subscriber("/cmd_vel", Twist, cb_cmd, queue_size=10)
    rospy.Subscriber("/x2_sol", Path, cb_x2_sol, queue_size=10)

    r = rospy.Rate(rate_hz)
    dt = 1.0/max(1.0, rate_hz)

    # State
    x = -1.0; y = 1.0; yaw = 0.0
    vx = 0.0; vy = 0.0; wz = 0.0

    while not rospy.is_shutdown():
        # bound accelerations crudely
        now = rospy.Time.now()
        now_sec = now.to_sec()
        if x2_sol_path:
            times = np.array([p[0] for p in x2_sol_path])
            xs = np.array([p[1] for p in x2_sol_path])
            ys = np.array([p[2] for p in x2_sol_path])
            vxs = np.array([p[3] for p in x2_sol_path])
            vys = np.array([p[4] for p in x2_sol_path])
            cmd_pos_x = float(np.interp(now_sec, times, xs))
            cmd_pos_y = float(np.interp(now_sec, times, ys))
            cmd_vel_x = float(np.interp(now_sec, times, vxs))
            cmd_vel_y = float(np.interp(now_sec, times, vys))
            cmd_pos.header.stamp = now
            cmd_pos.pose.position.x = cmd_pos_x
            cmd_pos.pose.position.y = cmd_pos_y

        if not started:
            if cmd.linear.x**2+cmd.linear.y**2 > 0.001 or cmd_pos_x**2+cmd_pos_y**2 > 0.001:
                started = True
                rospy.loginfo("1) Ridgeback simulation started receiving /cmd_vel ...")
                print("2) Ridgeback simulation started receiving /cmd_vel ...")

        else:
            ax_des = (cmd_vel_x - vx) / dt + 0.1*(cmd_pos_x - x) / dt**2
            ay_des = (cmd_vel_y - vy) / dt + 0.1*(cmd_pos_y - y) / dt**2
            ax = max(min(ax_des, amax), -amax)
            ay = max(min(ay_des, amax), -amax)
            vx = max(min(vx + ax*dt,  vmax), -vmax)
            vy = max(min(vy + ay*dt,  vmax), -vmax)
            wz = cmd.angular.z

            # integrate
            x += vx*dt
            y += vy*dt
            yaw += wz*dt

            print("(x,y)_cmd = (",cmd_pos_x,",",cmd_pos_y,") --> (ex,ey)=(",cmd_pos_x-x,",",cmd_pos_y-y,")")

        # publish odom
        od = Odometry()
        od.header.stamp = now
        od.header.frame_id = frame_id
        od.child_frame_id = child_frame_id
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = 0.0
        od.pose.pose.orientation = q_from_yaw(yaw)
        od.twist.twist.linear.x = vx
        od.twist.twist.linear.y = vy
        od.twist.twist.angular.z = wz
        odom_pub.publish(od)

        # publish tf
        ts = TransformStamped()
        ts.header.stamp = now
        ts.header.frame_id = frame_id
        ts.child_frame_id = child_frame_id
        ts.transform.translation.x = x
        ts.transform.translation.y = y
        ts.transform.translation.z = 0.0
        ts.transform.rotation = q_from_yaw(yaw)
        tf_pub.publish(TFMessage([ts]))

        # publish pose (mimic OptiTrack topic)
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation = q_from_yaw(yaw)
        pose_pub.publish(ps)

        r.sleep()

if __name__ == "__main__":
    main()
