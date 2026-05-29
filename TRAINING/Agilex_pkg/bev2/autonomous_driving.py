#!/usr/bin/env python3
# coding=utf-8

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


CLASS_WHITE = 2
CLASS_YELLOW = 3
CLASS_ROAD = 1

STATE_BOTH = "BOTH_LANES"
STATE_LEFT_ONLY = "LEFT_LANE_ONLY"
STATE_RIGHT_ONLY = "RIGHT_LANE_ONLY"
STATE_DRIVABLE = "DRIVABLE_AREA_MODE"
STATE_LOST = "LOST"

DEFAULT_LANE_WIDTH = 100.0
DEFAULT_MAX_SPEED = 0.20
DEFAULT_CENTER_OFFSET_PX = 17.0
DEFAULT_ROI_START_RATIO = 0.60
DEFAULT_ROI_END_RATIO = 0.84

# =========================================================
# LANE TRACKER
# =========================================================
class EdgeLaneTracker:

    def __init__(
        self,
        default_lane_width=DEFAULT_LANE_WIDTH,
        center_offset_px=DEFAULT_CENTER_OFFSET_PX,
        roi_start_ratio=DEFAULT_ROI_START_RATIO,
        roi_end_ratio=DEFAULT_ROI_END_RATIO,
    ):
        self.running_lane_width = float(default_lane_width)
        self.center_offset_px = float(center_offset_px)
        self.roi_start_ratio = float(roi_start_ratio)
        self.roi_end_ratio = float(roi_end_ratio)
        self.alpha_width = 0.05

    def roi_bounds(self, mask):
        h, _ = mask.shape
        return int(h * self.roi_start_ratio), int(h * self.roi_end_ratio)

    def lane_measurement(self, roi, cls):
        _, xs = np.where(roi == cls)

        if len(xs) < 12:
            return None, 0

        return float(np.median(xs)), len(xs)

    def update_lane_width(self, yellow_x, white_x):
        measured_width = abs(white_x - yellow_x)
        self.running_lane_width = (
            self.alpha_width * measured_width
            + (1.0 - self.alpha_width) * self.running_lane_width
        )

    def calculate_target_center(self, mask):
        _, w = mask.shape
        car_center = (w / 2.0)
        half_lane = self.running_lane_width / 2.0

        y0, y1 = self.roi_bounds(mask)
        roi = mask[y0:y1, :]

        yellow_x, yellow_count = self.lane_measurement(roi, CLASS_YELLOW)
        white_x, white_count = self.lane_measurement(roi, CLASS_WHITE)

        if yellow_x is None and white_x is None:
            return None, STATE_LOST

        if yellow_x is not None and white_x is not None:
            if abs(white_x - yellow_x) >= 35:
                self.update_lane_width(yellow_x, white_x)
                return (yellow_x + white_x) / 2.0, STATE_BOTH

            if yellow_count >= white_count:
                white_x = None
            else:
                yellow_x = None

        if yellow_x is not None:
            if yellow_x < car_center:
                return yellow_x + half_lane, STATE_LEFT_ONLY
            return yellow_x - half_lane, STATE_RIGHT_ONLY

        if white_x is not None:
            if white_x > car_center:
                return white_x - half_lane, STATE_RIGHT_ONLY
            return white_x + half_lane, STATE_LEFT_ONLY

        return None, STATE_LOST

    def fallback_drivable_area(self, mask):
        y0, y1 = self.roi_bounds(mask)
        roi = mask[y0:y1, :]

        _, xs = np.where(roi == CLASS_ROAD)

        if len(xs) > 50:
            return float(np.median(xs))

        return None

    def update(self, mask):
        center, state = self.calculate_target_center(mask)
        multiplier = 1.0

        if state == STATE_LOST:
            center = self.fallback_drivable_area(mask)
            if center is not None:
                state = STATE_DRIVABLE
        elif state == STATE_RIGHT_ONLY:
            multiplier = -1.0
            
        return center, state, multiplier


# =========================================================
# LOW-PASS FILTER
# =========================================================
class LowPassFilter:

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.initialized = False
        self.current_value = 0.0

    def filter(self, value):

        if not self.initialized:
            self.current_value = value
            self.initialized = True
            return value

        self.current_value = (
            self.alpha * value
            + (1.0 - self.alpha) * self.current_value
        )
        return self.current_value


# =========================================================
# CONTROLLER
# =========================================================
class Controller:

    def __init__(self):
        self.kp = 0.012
        self.kd = 0.004
        self.prev = 0.0

    def compute(self, error):

        d = error - self.prev
        self.prev = error

        omega = self.kp * error + self.kd * d
        return float(np.clip(omega, -0.4, 0.4))


# =========================================================
# NODE
# =========================================================
class Driver(Node):

    def __init__(self):

        super().__init__("no_circle_driver")

        self.declare_parameter("default_lane_width", DEFAULT_LANE_WIDTH)
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("center_offset_px", DEFAULT_CENTER_OFFSET_PX)
        default_lane_width = self.get_parameter("default_lane_width").value
        self.max_speed = self.get_parameter("max_speed").value
        self.center_offset_px = self.get_parameter("center_offset_px").value

        self.bridge = CvBridge()
        self.lane = EdgeLaneTracker(
            default_lane_width=default_lane_width,
            center_offset_px=self.center_offset_px,
        )
        self.ctrl = Controller()
        self.error_filter = LowPassFilter(alpha=0.25)

        self.create_subscription(Image, "/seg/bev_mask", self.cb, 1)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.get_logger().info("NO-CIRCLE LANE DRIVER ACTIVE")

    def cb(self, msg):

        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        _, w = mask.shape

        center, state, multiplier = self.lane.update(mask)

        cmd = Twist()

        if center is None:
            self.get_logger().warn(
                "CRITICAL BLINDNESS: NO TRACKING ANCHORS FOUND"
            )
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        desired_center = (
            (w / 2.0) + (self.center_offset_px * multiplier)
            if state != STATE_BOTH
            else (w / 2.0)
        )
        raw_error = desired_center - center
        smoothed_error = self.error_filter.filter(raw_error)
        omega = self.ctrl.compute(smoothed_error)

        speed = self.max_speed - abs(omega) * 0.08
        if state in (STATE_LEFT_ONLY, STATE_RIGHT_ONLY, STATE_DRIVABLE):
            speed = min(speed, self.max_speed * 0.8)
        speed = float(np.clip(speed, 0.05, self.max_speed))

        cmd.linear.x = speed
        cmd.angular.z = omega

        self.pub.publish(cmd)


def main():

    rclpy.init()
    node = Driver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
