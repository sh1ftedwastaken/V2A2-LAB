#!/usr/bin/env python3
# coding=utf-8

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Import shared lane analyzer
from lane_analyzer import (
    LaneAnalyzer,
    STATE_BOTH,
    STATE_LEFT_ONLY,
    STATE_RIGHT_ONLY,
    STATE_DRIVABLE,
    STATE_LOST,
    DEFAULT_LANE_WIDTH_PX,
    DEFAULT_ROI_START_RATIO,
    DEFAULT_ROI_END_RATIO,
    CLASS_ROAD,
    CLASS_YELLOW,
    CLASS_WHITE,
)

# =========================================================
# UNIVERSAL STATES (behavior states for overtaking logic)
# =========================================================
STATE_LANE_KEEP = "LANE_KEEP"
STATE_CHANGING_LANE = "CHANGING_LANE"
STATE_PASSING = "PASSING"
STATE_RETURNING = "RETURNING"

DEFAULT_MAX_SPEED        = 0.20
DEFAULT_CENTER_OFFSET_PX = 0.0


# =========================================================
# LOW-PASS FILTER
# =========================================================
class LowPassFilter:

    def __init__(self, alpha=0.25):
        self.alpha         = alpha
        self.initialized   = False
        self.current_value = 0.0

    def filter(self, value):

        if not self.initialized:
            self.current_value = value
            self.initialized   = True
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
        self.kp   = 0.015
        self.kd   = 0.005
        self.prev = 0.0

    def compute(self, error):

        d = error - self.prev
        self.prev = error

        omega = self.kp * error + self.kd * d
        return float(np.clip(omega, -0.45, 0.45))


# =========================================================
# NODE
# =========================================================
class Driver(Node):
    
    def __init__(self):
        super().__init__("no_circle_driver")

        self.declare_parameter("default_lane_width", DEFAULT_LANE_WIDTH_PX)
        self.declare_parameter("max_speed", DEFAULT_MAX_SPEED)
        self.declare_parameter("center_offset_px", DEFAULT_CENTER_OFFSET_PX)
        self.declare_parameter("roi_start_ratio", DEFAULT_ROI_START_RATIO)
        self.declare_parameter("roi_end_ratio", DEFAULT_ROI_END_RATIO)
        
        default_lane_width    = self.get_parameter("default_lane_width").value
        self.max_speed        = self.get_parameter("max_speed").value
        self.center_offset_px = self.get_parameter("center_offset_px").value
        roi_start_ratio       = self.get_parameter("roi_start_ratio").value
        roi_end_ratio         = self.get_parameter("roi_end_ratio").value

        self.bridge = CvBridge()
        
        # Use shared LaneAnalyzer for polynomial lane fitting
        self.lane_analyzer = LaneAnalyzer(
            lane_width_px=default_lane_width,
            camera_offset_x_px=0.0,  # BEV is robot-centered, no camera offset
            roi_start_ratio=roi_start_ratio,
            roi_end_ratio=roi_end_ratio,
            alpha_lane_width=0.07,
        )
        self.ctrl = Controller()
        self.error_filter = LowPassFilter(alpha=0.25)
        
        self.obstacle_dist = float("inf")
        self.left_clear    = True
        self.right_clear   = True
        
        self.behavior_state = STATE_LANE_KEEP
        self.overtake_side  = None
        self.direction_sign = 0           # +1 left, -1 right, 0 none    
        
        self.create_subscription(LaserScan, "/scan", self.lidar_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/seg/bev_mask", self.cb, 1)
        
        self.pub = self.create_publisher(Twist, "/cmd_vel", 1)

        self.get_logger().info("NO-CIRCLE LANE DRIVER ACTIVE (with polynomial lane tracking)")


    def lidar_cb(self, msg):
        front_ranges = []
        left_ranges  = []
        right_ranges = []
        
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + (i * msg.angle_increment)
            
            if msg.range_min < dist < msg.range_max:
                if -0.26 <= angle <= 0.26:
                    front_ranges.append(dist)
                elif 0.52 <= angle <= 1.31:
                    left_ranges.append(dist)
                elif -1.31 <= angle <= -0.52:
                    right_ranges.append(dist)
        
        self.obstacle_dist = min(front_ranges) if front_ranges else float("inf")
        self.left_clear = min(left_ranges) > 0.35 if left_ranges else True
        self.right_clear = min(right_ranges) > 0.35 if right_ranges else True
            
            
    def cb(self, msg):

        mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
        _, w = mask.shape

        cmd = Twist()
        
        SAFETY_STOP_DISTANCE = 0.3 # cm
        if self.obstacle_dist < SAFETY_STOP_DISTANCE:
            self.get_logger().warn(
                f"Obstacle ahead! Dist: {self.obstacle_dist:.2f} m."
            )
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return
        
        # Get polynomial lane fits from shared analyzer
        _, _, center_path, center_y_min, center_y_max, state, multiplier = self.lane_analyzer.analyze(mask)

        if center_path is None:
            self.get_logger().warn(
                "CRITICAL BLINDNESS: NO TRACKING ANCHORS FOUND"
            )
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.pub.publish(cmd)
            return

        # Target lookahead distance in Y
        roi_y = (center_y_min + center_y_max) / 2.0

        # Evaluate center path by finding the waypoint closest to our target Y
        distances = np.abs(center_path[:, 1] - roi_y)
        closest_idx = int(np.argmin(distances))
        center = float(center_path[closest_idx, 0])

        # Desired center is the robot center in BEV coordinates (BEV is robot-centered)
        desired_center = w / 2.0 + (self.center_offset_px * multiplier) if state != STATE_BOTH else w / 2.0
        raw_error = desired_center - center
        smoothed_error = self.error_filter.filter(raw_error)
        omega = self.ctrl.compute(smoothed_error)

        speed = self.max_speed - abs(omega) * 0.08
        if state in (STATE_LEFT_ONLY, STATE_RIGHT_ONLY, STATE_DRIVABLE):
            speed = min(speed, self.max_speed * 0.70)
        speed = float(np.clip(speed, 0.1, self.max_speed))

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