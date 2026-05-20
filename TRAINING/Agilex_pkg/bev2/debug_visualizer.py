#!/usr/bin/env python3
# debug_visualizer.py

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


# =========================================================
# CLASS COLORS
# =========================================================
CLASS_BG = 0
CLASS_ROAD = 1
CLASS_WHITE = 2
CLASS_YELLOW = 3
CLASS_VEHICLE = 4

CLASS_COLORS = {
    CLASS_BG:      (40, 40, 40),
    CLASS_ROAD:    (60, 120, 60),
    CLASS_WHITE:   (255, 255, 255),
    CLASS_YELLOW:  (0, 255, 255),
    CLASS_VEHICLE: (0, 0, 255),
}


# =========================================================
# DEBUG VISUALIZER
# =========================================================
class DebugVisualizer(Node):

    def __init__(self):

        super().__init__("debug_visualizer")

        self.bridge = CvBridge()

        # =================================================
        # SUBSCRIBERS
        # =================================================
        self.sub_raw = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.raw_cb,
            1
        )

        self.sub_seg = self.create_subscription(
            Image,
            "/seg/mask_raw",
            self.seg_cb,
            1
        )

        self.sub_bev = self.create_subscription(
            Image,
            "/seg/bev_mask",
            self.bev_cb,
            1
        )

        # OPTIONAL DEBUG BEV
        self.sub_debug_bev = self.create_subscription(
            Image,
            "/debug/bev",
            self.debug_bev_cb,
            1
        )

        self.sub_status = self.create_subscription(
            String,
            "/debug/status",
            self.status_cb,
            1
        )

        # =================================================
        # STORAGE
        # =================================================
        self.raw = None
        self.seg = None
        self.bev = None
        self.debug_bev = None

        self.status = {}

        # =================================================
        # SAVE DIRECTORY
        # =================================================
        ts = time.strftime("%Y%m%d_%H%M%S")

        self.save_dir = f"/home/agilex/debug_runs/run_{ts}"

        os.makedirs(self.save_dir, exist_ok=True)

        # FRAME DIRECTORIES
        self.raw_dir = os.path.join(self.save_dir, "raw")
        self.seg_dir = os.path.join(self.save_dir, "seg")
        self.overlay_dir = os.path.join(self.save_dir, "overlay")
        self.bev_dir = os.path.join(self.save_dir, "bev")
        self.canvas_dir = os.path.join(self.save_dir, "canvas")

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)
        os.makedirs(self.bev_dir, exist_ok=True)
        os.makedirs(self.canvas_dir, exist_ok=True)

        self.frame_id = 0

        # =================================================
        # VIDEO WRITERS
        # =================================================
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.video_raw = cv2.VideoWriter(
            os.path.join(self.save_dir, "raw.mp4"),
            fourcc,
            30,
            (640, 480)
        )

        self.video_seg = cv2.VideoWriter(
            os.path.join(self.save_dir, "seg.mp4"),
            fourcc,
            30,
            (640, 480)
        )

        self.video_overlay = cv2.VideoWriter(
            os.path.join(self.save_dir, "overlay.mp4"),
            fourcc,
            30,
            (640, 480)
        )

        self.video_bev = cv2.VideoWriter(
            os.path.join(self.save_dir, "bev.mp4"),
            fourcc,
            30,
            (640, 480)
        )

        self.video_canvas = cv2.VideoWriter(
            os.path.join(self.save_dir, "full_debug.mp4"),
            fourcc,
            30,
            (1920, 480)
        )

        # =================================================
        # TIMER
        # =================================================
        self.timer = self.create_timer(0.03, self.render)

        self.get_logger().info("Debug visualizer started.")

    # =====================================================
    # CALLBACKS
    # =====================================================
    def raw_cb(self, msg):
        self.raw = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def seg_cb(self, msg):
        self.seg = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def bev_cb(self, msg):
        self.bev = self.bridge.imgmsg_to_cv2(msg, "mono8")

    def debug_bev_cb(self, msg):
        self.debug_bev = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def status_cb(self, msg):

        try:
            self.status = json.loads(msg.data)
        except:
            self.status = {}

    # =====================================================
    # COLORIZE
    # =====================================================
    def colorize_mask(self, mask):

        h, w = mask.shape

        out = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, color in CLASS_COLORS.items():
            out[mask == cls_id] = color

        return out

    # =====================================================
    # DRAW STATUS
    # =====================================================
    def draw_status(self, img):

        y = 35

        for key, value in self.status.items():

            text = f"{key}: {value}"

            cv2.putText(
                img,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            y += 28

    # =====================================================
    # SAVE FRAME
    # =====================================================
    def save_frame(self, folder, idx, img):

        cv2.imwrite(
            os.path.join(folder, idx),
            img
        )

    # =====================================================
    # MAIN RENDER
    # =====================================================
    def render(self):

        if self.raw is None:
            return

        raw = self.raw.copy()

        h, w = raw.shape[:2]

        # =================================================
        # SEGMENTATION
        # =================================================
        if self.seg is not None:

            seg_color = self.colorize_mask(self.seg)

            seg_color = cv2.resize(
                seg_color,
                (w, h)
            )

            overlay = cv2.addWeighted(
                raw,
                0.7,
                seg_color,
                0.3,
                0
            )

        else:

            seg_color = np.zeros_like(raw)

            overlay = raw.copy()

        # =================================================
        # STATUS TEXT
        # =================================================
        self.draw_status(overlay)

        # =================================================
        # BEV
        # =================================================
        if self.debug_bev is not None:

            bev_vis = cv2.resize(
                self.debug_bev,
                (w, h)
            )

        elif self.bev is not None:

            bev_vis = self.colorize_mask(self.bev)

            bev_vis = cv2.resize(
                bev_vis,
                (w, h)
            )

        else:

            bev_vis = np.zeros_like(raw)

        # =================================================
        # PANEL LABELS
        # =================================================
        cv2.putText(
            raw,
            "RAW",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            "SEGMENTATION",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        cv2.putText(
            bev_vis,
            "BEV",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # =================================================
        # FINAL CANVAS
        # =================================================
        canvas = np.hstack([
            raw,
            overlay,
            bev_vis
        ])

        canvas = cv2.resize(
            canvas,
            (1920, 480)
        )

        # =================================================
        # SHOW
        # =================================================
        cv2.imshow("Segmentation Viewer", canvas)

        key = cv2.waitKey(1)

        if key == ord('q'):
            rclpy.shutdown()

        # =================================================
        # SAVE IMAGES
        # =================================================
        idx = f"{self.frame_id:06d}.png"

        self.save_frame(self.raw_dir, idx, raw)

        self.save_frame(self.seg_dir, idx, seg_color)

        self.save_frame(self.overlay_dir, idx, overlay)

        self.save_frame(self.bev_dir, idx, bev_vis)

        self.save_frame(self.canvas_dir, idx, canvas)

        # =================================================
        # SAVE VIDEOS
        # =================================================
        raw_v = cv2.resize(raw, (640, 480))

        seg_v = cv2.resize(seg_color, (640, 480))

        overlay_v = cv2.resize(overlay, (640, 480))

        bev_v = cv2.resize(bev_vis, (640, 480))

        self.video_raw.write(raw_v)

        self.video_seg.write(seg_v)

        self.video_overlay.write(overlay_v)

        self.video_bev.write(bev_v)

        self.video_canvas.write(canvas)

        self.frame_id += 1

    # =====================================================
    # CLEANUP
    # =====================================================
    def destroy_node(self):

        self.video_raw.release()

        self.video_seg.release()

        self.video_overlay.release()

        self.video_bev.release()

        self.video_canvas.release()

        cv2.destroyAllWindows()

        super().destroy_node()


# =========================================================
# MAIN
# =========================================================
def main():

    rclpy.init()

    node = DebugVisualizer()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
