#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# ============================================================
# CLASS IDS
# ============================================================

CLASS_BG      = 0
CLASS_ROAD    = 1
CLASS_WHITE   = 2
CLASS_YELLOW  = 3
CLASS_VEHICLE = 4


# ============================================================
# ROBUST LANE ESTIMATOR (EDGE-BASED, NOT CENTROID)
# ============================================================

class LaneEstimator:

    def __init__(self):
        self.prev_width = None

    def extract_lane(self, mask):

        h, w = mask.shape

        # lower-middle ROI = most stable BEV region
        roi = mask[int(h * 0.60):int(h * 0.92), :]

        yellow = (roi == CLASS_YELLOW)
        white  = (roi == CLASS_WHITE)

        ys_y, xs_y = np.where(yellow)
        ys_w, xs_w = np.where(white)

        if len(xs_y) < 15 and len(xs_w) < 15:
            return None, None

        # ----------------------------------------------------
        # CASE 1: both boundaries exist
        # ----------------------------------------------------
        if len(xs_y) > 15 and len(xs_w) > 15:

            left_x  = np.median(xs_y)
            right_x = np.median(xs_w)

        # ----------------------------------------------------
        # CASE 2: only yellow
        # ----------------------------------------------------
        elif len(xs_y) > 15:

            left_x = np.median(xs_y)
            right_x = left_x + 240   # assumed lane width fallback

        # ----------------------------------------------------
        # CASE 3: only white
        # ----------------------------------------------------
        else:

            right_x = np.median(xs_w)
            left_x = right_x - 240

        lane_width = right_x - left_x

        # ----------------------------------------------------
        # lane width stabilization (CRITICAL FIX)
        # ----------------------------------------------------
        if self.prev_width is not None:
            if abs(lane_width - self.prev_width) > 90:
                lane_width = self.prev_width

        self.prev_width = lane_width

        center = left_x + lane_width / 2

        return center, lane_width


# ============================================================
# CONTROLLER (CURVE-STABLE)
# ============================================================

class Controller:

    def __init__(self):
        self.kp = 0.0105
        self.kd = 0.0045
        self.prev_error = 0.0

    def compute(self, error, lane_width):

        derivative = error - self.prev_error
        self.prev_error = error

        omega = self.kp * error + self.kd * derivative

        # curve adaptation (tighten steering in narrow/warped BEV)
        if lane_width is not None and lane_width < 200:
            omega *= 1.15

        return float(np.clip(omega, -0.38, 0.38))


# ============================================================
# MAIN NODE
# ============================================================

class AutonomousDrivingNode(Node):

    def __init__(self):

        super().__init__("autonomous_driving")

        self.bridge = CvBridge()

        self.estimator = LaneEstimator()
        self.controller = Controller()

        self.sub = self.create_subscription(
            Image,
            "/seg/bev_mask",
            self.callback,
            1
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.debug_pub = self.create_publisher(Image, "/debug/bev_vis", 1)

        self.get_logger().info("Stable autonomous driving node started")

    # ========================================================
    # CALLBACK
    # ========================================================

    def callback(self, msg):

        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        h, w = mask.shape

        debug = np.zeros((h, w, 3), dtype=np.uint8)

        debug[mask == CLASS_WHITE]  = (255, 255, 255)
        debug[mask == CLASS_YELLOW] = (0, 255, 255)

        lane_center, lane_width = self.estimator.extract_lane(mask)

        cmd = Twist()

        # ====================================================
        # FAILSAFE MODE
        # ====================================================
        if lane_center is None:

            cmd.linear.x = 0.05
            cmd.angular.z = 0.0

            self.cmd_pub.publish(cmd)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))
            return

        # ====================================================
        # STEERING
        # ====================================================

        robot_center = w / 2
        error = robot_center - lane_center

        omega = self.controller.compute(error, lane_width)

        # speed scaling (curve-safe)
        turn = min(abs(omega), 0.5)
        speed = 0.16 - turn * 0.12
        speed = max(speed, 0.07)

        cmd.linear.x = float(speed)
        cmd.angular.z = float(omega)

        # ====================================================
        # DEBUG VISUALIZATION
        # ====================================================

        cv2.circle(debug,
                   (int(lane_center), int(h * 0.80)),
                   6, (255, 0, 0), -1)

        cv2.circle(debug,
                   (int(robot_center), int(h * 0.80)),
                   6, (0, 255, 0), -1)

        self.cmd_pub.publish(cmd)

        self.debug_pub.publish(
            self.bridge.cv2_to_imgmsg(debug, "bgr8")
        )


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = AutonomousDrivingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
