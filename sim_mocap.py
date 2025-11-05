#!/usr/bin/env python3
"""
sim_mocap.py
Publishes PoseStamped for "human" and optional obstacles to mimic OptiTrack topics.
"""
import math
import argparse
import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
import numpy as np

def make_pub(name):
    return rospy.Publisher(f"/vrpn_client_node/{name}/pose", PoseStamped, queue_size=10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", default="Human", help="human rigid body name")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--human-path", choices=["line","circle","lemniscate","static"], default="target")
    parser.add_argument("--obs", nargs="*", default=[], help="obsName:x,y,diam (diam unused)")
    args = parser.parse_args()

    rospy.init_node("sim_mocap")
    r = rospy.Rate(args.rate)
    human_pub = make_pub(args.human)
    obs_pubs = {}
    obs_vals = {}
    for spec in args.obs:
        try:
            name, rest = spec.split(":")
            x,y,diam = rest.split(",")
            obs_pubs[name] = make_pub(name)
            obs_vals[name] = (float(x), float(y), float(diam))
        except Exception as e:
            rospy.logwarn("Bad obstacle spec '%s' (use name:x,y,diam)", spec)

    started = False
    t0 = rospy.Time.now().to_sec()

    cmd = Twist()
    started = False
    def cb_cmd(msg):
        nonlocal cmd
        cmd = msg

    x1_des = [0.0, 0.0]
    x1_init = [1.0, 1.0]
    t_prev = -1

    def cb_x1_des(msg):
        nonlocal x1_des
        x1_des[0] = msg.pose.position.x
        x1_des[1] = msg.pose.position.y
        
    rospy.Subscriber("/cmd_vel", Twist, cb_cmd, queue_size=1)
    rospy.Subscriber("/x1_des", PoseStamped, cb_x1_des, queue_size=1)

    while not rospy.is_shutdown():
        if not started:
            if x1_des[0]**2+x1_des[1]**2 > 0.001:
                started = True
                t0 = rospy.Time.now().to_sec()  # reset time base when we start
                rospy.loginfo("Human simulation started receiving /cmd_vel ...")
                print("Human simulation started receiving /cmd_vel ...")
            else:
                t0 = rospy.Time.now().to_sec()  # reset time base when we start

        t = rospy.Time.now().to_sec() - t0
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = args.frame_id

        v_max = 0.1
        if args.human_path == "line":
            ps.pose.position.x = 0.5 + 0.0001*t
            ps.pose.position.y = 0.5
        elif args.human_path == "target":
            if t_prev > 0.0 and started:
                dt = rospy.Time.now().to_sec() - t_prev
                ps.pose.position.x = np.clip(x1_des[0], x_prev[0] - dt*v_max, x_prev[0] + dt*v_max)
                ps.pose.position.y = np.clip(x1_des[1], x_prev[1] - dt*v_max, x_prev[1] + dt*v_max)
            else:
                ps.pose.position.x = x1_init[0]
                ps.pose.position.y = x1_init[1]
            t_prev = rospy.Time.now().to_sec()
            x_prev = [ps.pose.position.x, ps.pose.position.y]
        elif args.human_path == "circle":
            R = 0.01; w = 0.01
            ps.pose.position.x = 1.0 + R*math.cos(w*t)
            ps.pose.position.y = 1.0 + R*math.sin(w*t)
        elif args.human_path == "lemniscate":
            a = 2.0; w = 0.25
            ps.pose.position.x = 5.0 + (a*math.cos(w*t)) / (1 + math.sin(w*t)**2)
            ps.pose.position.y = 5.0 + (a*math.sin(w*t)*math.cos(w*t)) / (1 + math.sin(w*t)**2)
        else:
            ps.pose.position.x = 3.0
            ps.pose.position.y = 3.0

        ps.pose.position.z = 0.0
        q = quaternion_from_euler(0,0,0)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        human_pub.publish(ps)
        print("(t,x,y) = (",t_prev,",",ps.pose.position.x,",",ps.pose.position.y,")")

        # obstacles (static)
        for name, pub in obs_pubs.items():
            ox, oy, _ = obs_vals[name]
            ops = PoseStamped()
            ops.header.stamp = rospy.Time.now()
            ops.header.frame_id = args.frame_id
            ops.pose.position.x = ox
            ops.pose.position.y = oy
            ops.pose.position.z = 0.0
            ops.pose.orientation.w = 1.0
            pub.publish(ops)

        r.sleep()

if __name__ == "__main__":
    main()
