"""
seg_bev_node.py
---------------
ROS2 node: subscribes to a segmentation mask,
warps it to BEV, publishes cleaned BEV mask.

Topics subscribed:
    /seg/mask_raw               (sensor_msgs/Image)  - raw seg mask (mono8)

Topics published:
    /seg/bev_mask               (sensor_msgs/Image)  - BEV cleaned mask (mono8)
    /seg/bev                    (sensor_msgs/Image)  - colourised BEV for debugging
    /seg/bev_overlay            (sensor_msgs/Image)  - driving ROI and center debug overlay

Run:
    ros2 run limo_seg seg_bev_node
    or
    ros2 launch limo_seg seg_bev_launch.py
"""
from __future__ import annotations #keep this since the limo car has a python version older
import os
import sys

import cv2
import numpy as np
import yaml
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bev.bev_transform import undistort_mask, to_bev, cleanup_bev, scale_intrinsics


CLASS_BG = 0
CLASS_ROAD = 1
CLASS_WHITE = 2
CLASS_YELLOW = 3
CLASS_VEHICLE = 4

CLASS_COLORS = {
    CLASS_BG: (0, 255, 0),
    CLASS_ROAD: (100, 100, 100),
    CLASS_WHITE: (255, 255, 255),
    CLASS_YELLOW: (0, 255, 255),
    CLASS_VEHICLE: (0, 0, 255),
}

DEBUG_USED_YELLOW = (255, 0, 255)
DEBUG_USED_WHITE = (255, 160, 0)
DEBUG_CENTER = (255, 0, 0)
DEBUG_ROBOT_CENTER = (0, 255, 0)
DEBUG_ROI = (180, 180, 180)

STATE_BOTH = "BOTH_LANES"
STATE_LEFT_ONLY = "LEFT_LANE_ONLY"
STATE_RIGHT_ONLY = "RIGHT_LANE_ONLY"
STATE_DRIVABLE = "DRIVABLE_AREA_MODE"
STATE_LOST = "LOST"


class DrivingLaneDebug:

    def __init__(self, default_lane_width=100.0):
        self.running_lane_width = float(default_lane_width)
        self.alpha_width = 0.05

    def lane_measurement(self, roi: np.ndarray, cls: int):
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

    def calculate_target_center(self, mask: np.ndarray):
        h, w = mask.shape
        car_center = w / 2.0
        half_lane = self.running_lane_width / 2.0

        roi = mask[int(h * 0.70):int(h * 0.94), :]

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

    def fallback_drivable_area(self, mask: np.ndarray):
        h, _ = mask.shape
        roi = mask[int(h * 0.70):int(h * 0.94), :]

        _, xs = np.where(roi == CLASS_ROAD)

        if len(xs) > 50:
            return float(np.median(xs))

        return None

    def update(self, mask: np.ndarray):
        center, state = self.calculate_target_center(mask)

        if state == STATE_LOST:
            center = self.fallback_drivable_area(mask)
            if center is not None:
                state = STATE_DRIVABLE

        return center, state


class SegBEVNode(Node):

    def __init__(self):
        super().__init__("seg_bev_node")

        self.declare_parameter("mask_topic", "/seg/mask_raw")
        self.declare_parameter("camera_params", "bev2/camera_params.txt")
        self.declare_parameter("bev_size", 160)
        self.declare_parameter("mask_width", 160)
        self.declare_parameter("mask_height", 120)
        self.declare_parameter("default_lane_width", 100.0)

        mask_topic = self.get_parameter("mask_topic").value
        params_path = self.get_parameter("camera_params").value
        self.bev_size = self.get_parameter("bev_size").value
        self.mask_w = self.get_parameter("mask_width").value
        self.mask_h = self.get_parameter("mask_height").value
        default_lane_width = self.get_parameter("default_lane_width").value

        with open(params_path, "r", encoding="utf-8") as handle:
            cam = yaml.safe_load(handle)

        k_native = np.array(cam["k"]).reshape(3, 3)
        self.d = np.array(cam["d"])
        self.k = scale_intrinsics(
            k_native,
            native_w=cam["width"],
            native_h=cam["height"],
            target_w=self.mask_w,
            target_h=self.mask_h,
        )

        src_points = np.float32([
            [8, 118],
            [152, 118],
            [90, 80],
            [45, 80],
        ])
        dst_points = np.float32([
            [20, 155],
            [140, 155],
            [140, 5],
            [20, 5],
        ])
        self.h_matrix, _ = cv2.findHomography(src_points, dst_points)

        self.bridge = CvBridge()
        self.lane_debug = DrivingLaneDebug(default_lane_width=default_lane_width)
        self._frames = 0
        self.sub = self.create_subscription(Image, mask_topic, self.image_callback, 10)
        self.pub_bev_mask = self.create_publisher(Image, "/seg/bev_mask", 10)
        self.pub_bev = self.create_publisher(Image, "/seg/bev", 10)
        self.pub_overlay = self.create_publisher(Image, "/seg/bev_overlay", 10)

        self.get_logger().info(f"Subscribed to mask topic: {mask_topic}")
        self.get_logger().info(f"Expected mask size     : {self.mask_w}x{self.mask_h}")
        self.get_logger().info(f"BEV size               : {self.bev_size}x{self.bev_size}")
        self.get_logger().info("seg_bev_node ready - waiting for masks")

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls, bgr in CLASS_COLORS.items():
            color[mask == cls] = bgr
        return color

    def driving_overlay(self, mask: np.ndarray) -> np.ndarray:
        overlay = self.colorize(mask)
        h, w = mask.shape
        y0 = int(h * 0.70)
        y1 = int(h * 0.94)

        roi_mask = mask[y0:y1, :]
        roi_overlay = overlay[y0:y1, :]
        roi_overlay[roi_mask == CLASS_YELLOW] = DEBUG_USED_YELLOW
        roi_overlay[roi_mask == CLASS_WHITE] = DEBUG_USED_WHITE

        cv2.rectangle(overlay, (0, y0), (w - 1, y1 - 1), DEBUG_ROI, 1)

        center, state = self.lane_debug.update(mask)
        robot_center = w / 2

        cv2.line(
            overlay,
            (int(robot_center), y0),
            (int(robot_center), y1 - 1),
            DEBUG_ROBOT_CENTER,
            1,
        )

        if center is not None:
            center_x = int(np.clip(center, 0, w - 1))
            center_y = int((y0 + y1) / 2)
            cv2.line(
                overlay,
                (center_x, y0),
                (center_x, y1 - 1),
                DEBUG_CENTER,
                2,
            )
            cv2.circle(overlay, (center_x, center_y), 4, DEBUG_CENTER, -1)

        cv2.putText(
            overlay,
            state,
            (5, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            DEBUG_CENTER if state != STATE_LOST else (0, 0, 255),
            1,
        )

        return overlay

    def image_callback(self, msg: Image):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        if mask.shape[:2] != (self.mask_h, self.mask_w):
            mask = cv2.resize(mask, (self.mask_w, self.mask_h), interpolation=cv2.INTER_NEAREST)

        mask_undist = undistort_mask(mask, self.k, self.d)
        bev_mask = to_bev(
            mask_undist,
            self.h_matrix,
            bev_size=(self.bev_size, self.bev_size),
        )
        bev_clean = cleanup_bev(bev_mask)

        bev_msg = self.bridge.cv2_to_imgmsg(bev_clean, encoding="mono8")
        bev_msg.header = msg.header
        self.pub_bev_mask.publish(bev_msg)

        bev_color = self.colorize(bev_clean)
        bev_vis_msg = self.bridge.cv2_to_imgmsg(bev_color, encoding="bgr8")
        bev_vis_msg.header = msg.header
        self.pub_bev.publish(bev_vis_msg)

        overlay = self.driving_overlay(bev_clean)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        self._frames += 1
        if self._frames == 1:
            self.get_logger().info(
                f"First mask received - classes in frame: {np.unique(mask).tolist()}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = SegBEVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
