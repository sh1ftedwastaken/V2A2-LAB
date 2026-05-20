#!/usr/bin/env python3
# coding=utf-8

"""
autonomous_driving.py
=====================

ROS2 node for autonomous driving using BEV segmentation.

Pipeline:
    /seg/bev_mask  →  lane detection + ACC  →  /cmd_vel

This version is aligned with:
    limo_segmentation_node.py
    seg_bev_node.py

Author: Stuti (updated)
"""

import json
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge


# -------------------------------
# CLASS DEFINITIONS
# -------------------------------
CLASS_BG = 0
CLASS_ROAD = 1
CLASS_WHITE = 2
CLASS_YELLOW = 3
CLASS_VEHICLE = 4


# -------------------------------
# VISUALIZATION
# -------------------------------
CLASS_COLORS = {
    CLASS_BG:      (40, 40, 40),
    CLASS_ROAD:    (80, 120, 80),
    CLASS_WHITE:   (220, 220, 220),
    CLASS_YELLOW:  (0, 200, 220),
    CLASS_VEHICLE: (50, 50, 220),
}


def colorise_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    return out


# -------------------------------
# LANE DETECTOR
# -------------------------------
class LaneDetector:
    LANE_OFFSET_PX = 35

    def __init__(self, bev_w=160, bev_h=160):
        self.bev_w = bev_w
        self.bev_h = bev_h

    def _centroid_x(self, mask, cls, rows):
        xs = np.where(mask[rows, :] == cls)[1]
        return float(xs.mean()) if xs.size > 5 else None

    def detect(self, bev_mask, lane_pos=0.5):
        rows = slice(int(self.bev_h * 0.4), self.bev_h)
        robot_x = self.bev_w / 2

        yellow = self._centroid_x(bev_mask, CLASS_YELLOW, rows)
        white = self._centroid_x(bev_mask, CLASS_WHITE, rows)

        if yellow is not None and white is not None:
            target = yellow + lane_pos * (white - yellow)
            confidence = 1.0
        elif yellow is not None:
            target = yellow + 2 * self.LANE_OFFSET_PX
            confidence = 0.5
        elif white is not None:
            target = white - 2 * self.LANE_OFFSET_PX
            confidence = 0.5
        else:
            return dict(cte=0.0, confidence=0.0)

        cte = robot_x - target
        return dict(cte=cte, confidence=confidence)


# -------------------------------
# STEERING (PD)
# -------------------------------
class SteeringController:
    def __init__(self, kp=0.012, kd=0.004, max_ang=0.55):
        self.kp = kp
        self.kd = kd
        self.max_ang = max_ang
        self.prev_cte = 0.0

    def compute(self, cte, confidence):
        if confidence == 0:
            return 0.0

        d = cte - self.prev_cte
        omega = (self.kp * cte + self.kd * d) * confidence
        self.prev_cte = cte

        return float(np.clip(omega, -self.max_ang, self.max_ang))


# -------------------------------
# ADAPTIVE CRUISE
# -------------------------------
class AdaptiveCruise:
    def __init__(self):
        self.buf = deque(maxlen=5)

    def update(self, bev_mask):
        vehicle_pixels = np.sum(bev_mask == CLASS_VEHICLE)
        self.buf.append(vehicle_pixels)
        avg = np.mean(self.buf)

        if avg > 200:
            return 0.0, "STOP"
        elif avg > 80:
            return 0.08, "SLOW"
        else:
            return 0.15, "CRUISE"


# -------------------------------
# ROS NODE
# -------------------------------
class AutonomousDrivingNode(Node):
    def __init__(self):
        super().__init__("autonomous_driving")

        self.bridge = CvBridge()

        # Params
        self.declare_parameter("kp", 0.012)
        self.declare_parameter("kd", 0.004)
        self.declare_parameter("lane_position", 0.5)

        kp = self.get_parameter("kp").value
        kd = self.get_parameter("kd").value
        self.lane_pos = self.get_parameter("lane_position").value

        # Modules
        self.lane = LaneDetector()
        self.steer = SteeringController(kp, kd)
        self.acc = AdaptiveCruise()

        # Sub / Pub
        self.sub = self.create_subscription(
            Image,
            "/seg/bev_mask",
            self.callback,
            1
        )

        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 1)
        self.pub_dbg = self.create_publisher(Image, "/debug/bev", 1)
        self.pub_status = self.create_publisher(String, "/debug/status", 1)

        self.get_logger().info("Autonomous driving node started.")

    def callback(self, msg):
        bev = self.bridge.imgmsg_to_cv2(msg, "mono8")

        # ---- LANE ----
        lane_info = self.lane.detect(bev, self.lane_pos)

        # ---- STEERING ----
        omega = self.steer.compute(
            lane_info["cte"],
            lane_info["confidence"]
        )

        # ---- SPEED ----
        speed, state = self.acc.update(bev)

        # ---- COMMAND ----
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(omega)
        self.pub_cmd.publish(cmd)

        # ---- DEBUG ----
        dbg = colorise_mask(bev)
        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        status = {
            "cte": round(lane_info["cte"], 2),
            "confidence": lane_info["confidence"],
            "omega": round(omega, 3),
            "speed": speed,
            "state": state,
        }
        self.pub_status.publish(String(data=json.dumps(status)))


# -------------------------------
# MAIN
# -------------------------------
def main():
    rclpy.init()
    node = AutonomousDrivingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
