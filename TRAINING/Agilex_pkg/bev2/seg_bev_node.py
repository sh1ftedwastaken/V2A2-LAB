#!/usr/bin/env python3
"""
seg_bev_node.py
---------------
ROS2 node that warps a segmentation mask into bird's-eye view.

This version intentionally uses a direct cv2.warpPerspective path. On the real
robot this is easier to calibrate than the intrinsics/undistortion path, and it
matches Stuti's working BEV workflow while keeping the tuning params from bev2.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


CLASS_COLORS_BGR = np.array([
    [0, 255, 0],
    [80, 80, 80],
    [255, 255, 255],
    [0, 255, 255],
    [0, 0, 255],
], dtype=np.uint8)

DEFAULT_SRC = np.float32([
    [10, 119],   # near-left
    [150, 119],  # near-right
    [110, 65],   # far-right
    [50, 65],    # far-left
])

SAVE_DIR = os.path.expanduser("~/debug_bev")
SAVE_EVERY = 5


def load_src_from_file(path: str) -> Optional[np.ndarray]:
    """Read four x,y source points from a simple camera_params.txt file."""
    if not path or not os.path.isfile(path):
        return None

    try:
        values = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                values.extend(float(v) for v in line.split())
        if len(values) < 8:
            return None
        return np.asarray(values[:8], dtype=np.float32).reshape(4, 2)
    except Exception:
        return None


def make_dst(bev_width: int, bev_height: int, margin_ratio: float) -> np.ndarray:
    """Destination rectangle centered in the BEV canvas."""
    margin_x = int(bev_width * margin_ratio)
    margin_y = 0
    return np.float32([
        [margin_x, bev_height - 1 - margin_y],
        [bev_width - margin_x, bev_height - 1 - margin_y],
        [bev_width - margin_x, margin_y],
        [margin_x, margin_y],
    ])


class SegBEVNode(Node):

    def __init__(self):
        super().__init__("seg_bev_node")

        self.declare_parameter("mask_topic", "/seg/mask_raw")
        self.declare_parameter("camera_params", "")
        self.declare_parameter("bev_size", 160)
        self.declare_parameter("bev_width", 0)
        self.declare_parameter("bev_height", 0)
        self.declare_parameter("mask_width", 160)
        self.declare_parameter("mask_height", 120)
        self.declare_parameter("src_points", [])
        self.declare_parameter("src_x_offset_px", 0.0)
        self.declare_parameter("src_y_offset_px", 0.0)
        self.declare_parameter("dst_x_offset_px", 0.0)
        self.declare_parameter("dst_y_offset_px", 0.0)
        self.declare_parameter("dst_margin_ratio", 0.22)
        self.declare_parameter("calibrate_mode", False)
        self.declare_parameter("save_debug", False)
        self.declare_parameter("heartbeat_sec", 2.0)

        mask_topic = str(self.get_parameter("mask_topic").value)
        params_path = str(self.get_parameter("camera_params").value)
        bev_size = int(self.get_parameter("bev_size").value)
        bev_width = int(self.get_parameter("bev_width").value) or bev_size
        bev_height = int(self.get_parameter("bev_height").value) or bev_size
        self.mask_w = int(self.get_parameter("mask_width").value)
        self.mask_h = int(self.get_parameter("mask_height").value)
        src_points_param = self.get_parameter("src_points").value
        src_x_offset = float(self.get_parameter("src_x_offset_px").value)
        src_y_offset = float(self.get_parameter("src_y_offset_px").value)
        dst_x_offset = float(self.get_parameter("dst_x_offset_px").value)
        dst_y_offset = float(self.get_parameter("dst_y_offset_px").value)
        margin_ratio = float(self.get_parameter("dst_margin_ratio").value)
        self.calibrate_mode = bool(self.get_parameter("calibrate_mode").value)
        self.save_debug = bool(self.get_parameter("save_debug").value)
        heartbeat_sec = float(self.get_parameter("heartbeat_sec").value)

        self.bev_w = bev_width
        self.bev_h = bev_height
        if self.save_debug:
            os.makedirs(SAVE_DIR, exist_ok=True)

        src = self._resolve_src_points(src_points_param, params_path)
        dst = make_dst(self.bev_w, self.bev_h, margin_ratio)

        src[:, 0] += src_x_offset
        src[:, 1] += src_y_offset
        dst[:, 0] += dst_x_offset
        dst[:, 1] += dst_y_offset

        self.src = src.astype(np.float32)
        self.dst = dst.astype(np.float32)
        self.h_matrix = cv2.getPerspectiveTransform(self.src, self.dst)
        self.kernel = np.ones((3, 3), np.uint8)

        self.bridge = CvBridge()
        self.frames = 0
        self.sub = self.create_subscription(Image, mask_topic, self.image_callback, 1)
        self.pub_bev_mask = self.create_publisher(Image, "/seg/bev_mask", 1)
        self.pub_bev = self.create_publisher(Image, "/seg/bev", 1)
        self.pub_bev_overlay = self.create_publisher(Image, "/seg/bev_overlay", 1)
        self.pub_bev_debug = self.create_publisher(Image, "/seg/bev_debug", 1)
        self.pub_bev_grid = self.create_publisher(Image, "/seg/bev_grid", 1)
        self.create_timer(max(0.5, heartbeat_sec), self._heartbeat)

        self.get_logger().info(f"Subscribed to mask topic: {mask_topic}")
        self.get_logger().info(f"Expected mask size     : {self.mask_w}x{self.mask_h}")
        self.get_logger().info(f"BEV output size        : {self.bev_w}x{self.bev_h}")
        self.get_logger().info(f"src_points             : {self.src.tolist()}")
        self.get_logger().info(f"dst_points             : {self.dst.tolist()}")
        self.get_logger().info(f"calibrate_mode         : {self.calibrate_mode}")
        self.get_logger().info("seg_bev_node ready - waiting for masks")

    def _resolve_src_points(self, src_points_param, params_path: str) -> np.ndarray:
        if src_points_param:
            values = np.asarray(src_points_param, dtype=np.float32)
            if values.size >= 8:
                return values[:8].reshape(4, 2)
            self.get_logger().warn("src_points parameter had fewer than 8 values; ignoring it.")

        src = load_src_from_file(params_path)
        if src is not None:
            self.get_logger().info(f"Loaded BEV source points from {params_path}")
            return src

        self.get_logger().warn(
            f"Could not read simple BEV points from '{params_path}'. "
            "Using DEFAULT_SRC; run with calibrate_mode:=True to tune."
        )
        return DEFAULT_SRC.copy()

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        return CLASS_COLORS_BGR[np.clip(mask, 0, len(CLASS_COLORS_BGR) - 1)]

    def _draw_calib(self, mask: np.ndarray) -> np.ndarray:
        vis = self.colorize(mask)
        pts = self.src.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        labels = ["0:NL", "1:NR", "2:FR", "3:FL"]
        for (x, y), label in zip(self.src.astype(int), labels):
            cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(vis, label, (int(x) + 4, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def image_callback(self, msg: Image):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        if mask.shape[:2] != (self.mask_h, self.mask_w):
            mask = cv2.resize(mask, (self.mask_w, self.mask_h), interpolation=cv2.INTER_NEAREST)

        if self.calibrate_mode:
            grid_msg = self.bridge.cv2_to_imgmsg(self._draw_calib(mask), encoding="bgr8")
            grid_msg.header = msg.header
            self.pub_bev_grid.publish(grid_msg)

        bev = cv2.warpPerspective(
            mask,
            self.h_matrix,
            (self.bev_w, self.bev_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bev = cv2.morphologyEx(bev, cv2.MORPH_OPEN, self.kernel)
        bev = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, self.kernel)

        bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding="mono8")
        bev_msg.header = msg.header
        self.pub_bev_mask.publish(bev_msg)

        bev_color = self.colorize(bev)
        bev_color_msg = self.bridge.cv2_to_imgmsg(bev_color, encoding="bgr8")
        bev_color_msg.header = msg.header
        self.pub_bev.publish(bev_color_msg)
        self.pub_bev_overlay.publish(bev_color_msg)
        self.pub_bev_debug.publish(bev_color_msg)

        self.frames += 1
        if self.frames == 1:
            self.get_logger().info(f"First mask received - classes in frame: {np.unique(mask).tolist()}")

        if self.save_debug and self.frames % SAVE_EVERY == 0:
            ts = time.time()
            cv2.imwrite(os.path.join(SAVE_DIR, f"{ts}_bev_mask_scaled.png"), bev * 40)
            cv2.imwrite(os.path.join(SAVE_DIR, f"{ts}_bev_debug.png"), bev_color)
            if self.calibrate_mode:
                cv2.imwrite(os.path.join(SAVE_DIR, f"{ts}_calib.png"), self._draw_calib(mask))

    def _heartbeat(self):
        if self.frames == 0:
            self.get_logger().warn("Still waiting for masks. Check /seg/mask_raw is publishing.")


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
