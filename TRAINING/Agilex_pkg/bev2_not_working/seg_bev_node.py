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
    /seg/bev_overlay            (sensor_msgs/Image)  - colourised BEV alias for debugging

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


CLASS_COLORS = {
    0: (0, 255, 0),
    1: (100, 100, 100),
    2: (255, 255, 255),
    3: (0, 255, 255),
    4: (0, 0, 255),
}


class SegBEVNode(Node):

    def __init__(self):
        super().__init__("seg_bev_node")

        self.declare_parameter("mask_topic", "/seg/mask_raw")
        self.declare_parameter("camera_params", "bev/camera_params.txt")
        self.declare_parameter("bev_size", 160)
        self.declare_parameter("mask_width", 160)
        self.declare_parameter("mask_height", 120)

        mask_topic = self.get_parameter("mask_topic").value
        params_path = self.get_parameter("camera_params").value
        self.bev_size = self.get_parameter("bev_size").value
        self.mask_w = self.get_parameter("mask_width").value
        self.mask_h = self.get_parameter("mask_height").value

        with open(params_path, "r", encoding="utf-8") as handle:
            cam = yaml.safe_load(handle)
            
        if not isinstance(cam, dict):
            self.get_logger().error(
                f"Failed to parse '{params_path}' as a dictionary. "
                f"Parsed type: {type(cam).__name__}. Ensure the file is valid YAML/JSON "
                "and that colons are followed by a space (e.g., 'k: [...]' not 'k:[...]').")
            raise TypeError(f"Camera parameters at {params_path} must be a YAML dictionary.")

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
        self.pub_overlay.publish(bev_vis_msg)

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
