#!/usr/bin/env python3

import os
import cv2
import json
import time
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


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


class DebugVisualizer(Node):

    def __init__(self):
        super().__init__("debug_visualizer")

        self.declare_parameter("display_width", 960)
        self.declare_parameter("display_height", 180)

        self.display_width = int(self.get_parameter("display_width").value)
        self.display_height = int(self.get_parameter("display_height").value)

        self.bridge = CvBridge()

        self.create_subscription(Image, "/camera/color/image_raw", self.raw_cb, 1)
        self.create_subscription(Image, "/seg/cam_overlay", self.cam_overlay_cb, 1)
        self.create_subscription(Image, "/seg/mask_raw", self.seg_cb, 1)
        self.create_subscription(Image, "/seg/bev_mask", self.bev_cb, 1)
        self.create_subscription(Image, "/seg/bev_overlay", self.center_cb, 1)
        self.create_subscription(String, "/debug/status", self.status_cb, 1)

        self.raw = None
        self.cam_overlay = None
        self.seg = None
        self.bev = None
        self.center = None
        self.status = {}

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"/home/agilex/debug_runs/run_{ts}"

        for d in ["raw", "seg", "bev", "bev_center", "canvas"]:
            os.makedirs(os.path.join(self.save_dir, d), exist_ok=True)

        self.frame_id = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_canvas = cv2.VideoWriter(
            self.save_dir + "/full.mp4", fourcc, 30, (2560, 480)
        )

        cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEBUG", self.display_width, self.display_height)

        self.timer = self.create_timer(0.03, self.render)

        self.get_logger().info("Debug visualizer started")
        self.get_logger().info(
            f"Debug window size: {self.display_width}x{self.display_height}"
        )

    def raw_cb(self, msg):
        self.raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def cam_overlay_cb(self, msg):
        self.cam_overlay = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def seg_cb(self, msg):
        self.seg = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def bev_cb(self, msg):
        self.bev = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def center_cb(self, msg):
        self.center = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def status_cb(self, msg):
        try:
            self.status = json.loads(msg.data)
        except:
            self.status = {}

    def colorize(self, mask):
        if mask is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = mask.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)

        for k, v in CLASS_COLORS.items():
            out[mask == k] = v

        return out

    def camera_overlay_panel(self, raw):
        overlay = self.cam_overlay if self.cam_overlay is not None else raw
        raw_panel = cv2.resize(raw, (640, 480))
        overlay_panel = cv2.resize(overlay, (640, 480))

        panel = overlay_panel.copy()
        inset_w, inset_h = 170, 128
        margin = 10
        inset = cv2.resize(raw_panel, (inset_w, inset_h))
        x0 = margin
        y0 = panel.shape[0] - inset_h - margin
        panel[y0:y0 + inset_h, x0:x0 + inset_w] = inset
        cv2.rectangle(
            panel,
            (x0, y0),
            (x0 + inset_w, y0 + inset_h),
            (235, 235, 235),
            2,
        )
        return panel

    def render(self):

        if self.raw is None:
            return

        raw = self.raw.copy()
        camera_overlay = self.camera_overlay_panel(raw)
        seg = self.colorize(self.seg)
        bev = self.colorize(self.bev)

        canvas = np.hstack([
            camera_overlay,
            cv2.resize(seg, (640, 480)),
            cv2.resize(bev, (640, 480)),
            cv2.resize(self.center if self.center is not None else bev, (640, 480)),
        ])

        cv2.imshow("DEBUG", canvas)
        cv2.waitKey(1)

        self.video_canvas.write(canvas)

        self.frame_id += 1


def main():
    rclpy.init()
    node = DebugVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
