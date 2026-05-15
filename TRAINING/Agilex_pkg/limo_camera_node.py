#!/usr/bin/env python3
"""
limo_camera_node.py  —  ROS 2 Humble
=====================================
Camera node for the LIMO Pro robot car.

Key behaviour (ROS 2 Humble version)
--------------------------------------
  • Subscribes to the LIMO camera topic.
  • Upscales each frame (default 320×240 → 640×480) via bicubic interpolation.
  • Saves EVERY incoming frame as a timestamped PNG  (save_every = 1 by default).
  • Publishes the upscaled frame on /limo/camera/image_upscaled.
  • Publishes a /limo/camera/frame_saved (std_msgs/Bool) on every save.
  • Publishes status on /limo/camera/status (std_msgs/String).
  • Optionally runs U-Net segmentation on each saved frame.

ROS 2 Parameters (declare in launch or pass with --ros-args -p name:=value)
---------------------------------------------------------------------------
  camera_topic      string   /camera/rgb/image_raw
  output_topic      string   /limo/camera/image_upscaled
  save_dir          string   ~/limo_frames
  save_every        int      1          ← saves every single frame
  target_width      int      640
  target_height     int      480
  run_segmentation  bool     false
  checkpoint        string   ''
  num_classes       int      2

Launch
------
    ros2 launch limo_camera_pkg limo_camera.launch.py
"""

import os
import sys
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError

# Optional U-Net
_UNET_AVAILABLE = False
try:
    _pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_pkg_root / 'unet'))
    from predict import UNetPredictor
    _UNET_AVAILABLE = True
except ImportError:
    pass


class LimoCameraNode(Node):

    def __init__(self):
        super().__init__('limo_camera_node')

        self.declare_parameter('camera_topic',     '/camera/rgb/image_raw')
        self.declare_parameter('output_topic',     '/limo/camera/image_upscaled')
        self.declare_parameter('save_dir',         os.path.expanduser('~/limo_frames'))
        self.declare_parameter('save_every',       1)
        self.declare_parameter('target_width',     640)
        self.declare_parameter('target_height',    480)
        self.declare_parameter('run_segmentation', False)
        self.declare_parameter('checkpoint',       '')
        self.declare_parameter('num_classes',      2)

        cam_topic = self.get_parameter('camera_topic').value
        out_topic = self.get_parameter('output_topic').value
        self.save_dir = os.path.expanduser(self.get_parameter('save_dir').value)
        self.save_every = self.get_parameter('save_every').value
        self.tgt_w = self.get_parameter('target_width').value
        self.tgt_h = self.get_parameter('target_height').value
        run_seg = self.get_parameter('run_segmentation').value
        checkpoint = self.get_parameter('checkpoint').value
        num_classes = self.get_parameter('num_classes').value

        self.bridge = CvBridge()
        self.frame_count = 0
        self.saved_count = 0
        self._lock = threading.Lock()

        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Save directory : {self.save_dir}')
        self.get_logger().info(f'save_every     : {self.save_every}  (1 = every frame)')
        self.get_logger().info(f'Target size    : {self.tgt_w}×{self.tgt_h}')

        self.predictor = None
        if run_seg:
            if not _UNET_AVAILABLE:
                self.get_logger().warn('U-Net not importable — segmentation disabled.')
            elif not checkpoint:
                self.get_logger().warn('checkpoint param not set — segmentation disabled.')
            else:
                self.get_logger().info(f'Loading U-Net from {checkpoint}')
                self.predictor = UNetPredictor(
                    checkpoint=checkpoint,
                    img_size=(self.tgt_w, self.tgt_h),
                    num_classes=num_classes,
                )
                self.get_logger().info('U-Net loaded ✓')

        self.pub_image = self.create_publisher(Image, out_topic, 2)
        self.pub_saved = self.create_publisher(Bool, '/limo/camera/frame_saved', 2)
        self.pub_status = self.create_publisher(String, '/limo/camera/status', 2)

        if self.predictor:
            self.pub_seg = self.create_publisher(Image, '/limo/camera/segmentation', 2)

        self.sub = self.create_subscription(Image, cam_topic, self._image_callback, 1)
        self.create_timer(1.0, self._heartbeat)

        self.get_logger().info(f'Subscribed to  : {cam_topic}')
        self.get_logger().info('Node ready — waiting for images...')

    def _image_callback(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'cv_bridge error: {exc}')
            return

        self.frame_count += 1
        upscaled = cv2.resize(img_bgr, (self.tgt_w, self.tgt_h), interpolation=cv2.INTER_CUBIC)

        try:
            out_msg = self.bridge.cv2_to_imgmsg(upscaled, encoding='bgr8')
            out_msg.header = msg.header
            self.pub_image.publish(out_msg)
        except CvBridgeError as exc:
            self.get_logger().error(f'Publish error: {exc}')

        if self.frame_count % self.save_every == 0:
            self._save_frame(upscaled)

    def _save_frame(self, img_bgr: np.ndarray):
        with self._lock:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            fname = f'frame_{self.saved_count:06d}_{ts}.png'
            fpath = os.path.join(self.save_dir, fname)

            cv2.imwrite(fpath, img_bgr)
            self.saved_count += 1

        self.get_logger().info(f'Saved [{self.saved_count:>6}]: {fname}')
        self.pub_saved.publish(Bool(data=True))
        self.pub_status.publish(String(data=f'saved:{fpath}|total:{self.saved_count}'))

        if self.predictor is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask = self.predictor.predict(img_rgb)
                overlay = self.predictor.overlay(img_rgb, mask)
                ov_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

                seg_path = fpath.replace('.png', '_seg.png')
                cv2.imwrite(seg_path, ov_bgr)

                seg_msg = self.bridge.cv2_to_imgmsg(ov_bgr, encoding='bgr8')
                self.pub_seg.publish(seg_msg)
            except Exception as exc:
                self.get_logger().warn(f'Segmentation failed: {exc}')

    def _heartbeat(self):
        self.pub_status.publish(
            String(data=f'running|frames:{self.frame_count}|saved:{self.saved_count}')
        )


def main(args=None):
    rclpy.init(args=args)
    node = LimoCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
