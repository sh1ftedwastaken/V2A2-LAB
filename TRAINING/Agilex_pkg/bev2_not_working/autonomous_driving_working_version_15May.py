#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


CLASS_WHITE   = 2
CLASS_YELLOW  = 3
CLASS_VEHICLE = 4


# ==================================================
# OBSTACLE AWARENESS (LEFT / RIGHT / CENTER)
# ==================================================
class ObstacleDetector:

    def detect_zone(self, mask):

        h, w = mask.shape

        roi = mask[int(h*0.4):int(h*0.7), :]

        vehicle_map = (roi == CLASS_VEHICLE).astype(np.uint8)

        left  = np.sum(vehicle_map[:, :w//3])
        mid   = np.sum(vehicle_map[:, w//3:2*w//3])
        right = np.sum(vehicle_map[:, 2*w//3:])

        if mid > left and mid > right:
            return "CENTER"
        elif left > right:
            return "LEFT"
        else:
            return "RIGHT"

    def exists(self, mask):
        h, w = mask.shape
        roi = mask[int(h*0.4):int(h*0.7), int(w*0.35):int(w*0.65)]
        return np.sum(roi == CLASS_VEHICLE) > 120


# ==================================================
# LANE TRACKER
# ==================================================
class LaneTracker:

    def compute_cte(self, mask):

        h, w = mask.shape
        rows = slice(int(h * 0.5), h)

        yellow = np.where(mask[rows] == CLASS_YELLOW)[1]
        white  = np.where(mask[rows] == CLASS_WHITE)[1]

        robot_x = w / 2

        if len(yellow) > 30 and len(white) > 30:
            center = (np.mean(yellow) + np.mean(white)) / 2

        elif len(yellow) > 30:
            center = np.mean(yellow) + 35

        elif len(white) > 30:
            center = np.mean(white) - 35

        else:
            return None

        return float(robot_x - center)


# ==================================================
# CONTROLLER (stable)
# ==================================================
class Controller:

    def __init__(self):
        self.kp = 0.02
        self.kd = 0.006
        self.prev = 0.0

    def compute(self, cte):
        d = cte - self.prev
        self.prev = cte

        omega = self.kp * cte + self.kd * d
        return float(np.clip(omega, -0.6, 0.6))


# ==================================================
# STATE MACHINE (FIXES OSCILLATION)
# ==================================================
class Behavior:

    def __init__(self):
        self.mode = "FOLLOW"
        self.avoid_dir = None
        self.lock_timer = 0

    def update(self, obstacle, zone):

        # --------------------------
        # ENTER AVOID MODE
        # --------------------------
        if obstacle and self.mode == "FOLLOW":
            self.mode = "AVOID"
            self.lock_timer = 20

            # choose stable direction once
            if zone == "LEFT":
                self.avoid_dir = "RIGHT"
            elif zone == "RIGHT":
                self.avoid_dir = "LEFT"
            else:
                self.avoid_dir = "RIGHT"

        # --------------------------
        # HOLD AVOID MODE
        # --------------------------
        if self.mode == "AVOID":
            self.lock_timer -= 1
            if self.lock_timer <= 0:
                self.mode = "FOLLOW"


# ==================================================
# NODE
# ==================================================
class DriveNode(Node):

    def __init__(self):

        super().__init__("stable_avoid_drive")

        self.bridge = CvBridge()

        self.lane = LaneTracker()
        self.obs = ObstacleDetector()
        self.ctrl = Controller()
        self.beh = Behavior()

        self.sub = self.create_subscription(
            Image,
            "/seg/bev_mask",
            self.cb,
            1
        )

        self.pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.get_logger().info("Stable avoidance driver started")

    def cb(self, msg):

        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")

        obstacle = self.obs.exists(mask)
        zone = self.obs.detect_zone(mask)

        self.beh.update(obstacle, zone)

        cmd = Twist()

        # ==================================================
        # AVOID MODE (STABLE TURN, NO OSCILLATION)
        # ==================================================
        if self.beh.mode == "AVOID":

            speed = 0.08
            omega = 0.35 if self.beh.avoid_dir == "LEFT" else -0.35

            cmd.linear.x = speed
            cmd.angular.z = omega

            self.pub.publish(cmd)
            return

        # ==================================================
        # NORMAL FOLLOW
        # ==================================================
        cte = self.lane.compute_cte(mask)

        if cte is None:
            cmd.linear.x = 0.05
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        omega = self.ctrl.compute(cte)

        cmd.linear.x = 0.15
        cmd.angular.z = omega

        self.pub.publish(cmd)


# ==================================================
# MAIN
# ==================================================
def main():

    rclpy.init()
    node = DriveNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
