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
import warnings
from typing import NamedTuple, Optional, Tuple

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

DEFAULT_LANE_WIDTH_PX = 100.0
MIN_POLYFIT_POINTS = 30
MIN_OBSTACLE_AREA_PX = 8
LANE_FIT_BIN_HEIGHT_PX = 4
MIN_LANE_FIT_BINS = 8
CURVE_RMSE_IMPROVEMENT = 0.18
CURVE_MIN_X_SPAN_PX = 12.0
DEFAULT_ROI_START_RATIO = 0.60
DEFAULT_ROI_END_RATIO = 0.84
MIN_BOTH_LANE_GAP_PX = 35.0
MAX_COMPONENT_ROI_DISTANCE_PX = 28.0

_RANK_WARNING_TYPES = tuple(
    warning_type
    for warning_type in (
        getattr(np, "RankWarning", None),
        getattr(getattr(np, "exceptions", None), "RankWarning", None),
    )
    if warning_type is not None
)
_POLYFIT_ERRORS = (
    np.linalg.LinAlgError,
    ValueError,
    TypeError,
    FloatingPointError,
) + _RANK_WARNING_TYPES

CLASS_COLORS = {
    CLASS_BG: (0, 255, 0),
    CLASS_ROAD: (100, 100, 100),
    CLASS_WHITE: (255, 255, 255),
    CLASS_YELLOW: (0, 255, 255),
    CLASS_VEHICLE: (0, 0, 255),
}

CURVE_YELLOW = (255, 0, 255)
CURVE_WHITE = (255, 128, 0)
CURVE_CENTER = (255, 0, 0)
OBSTACLE_BOX = (0, 30, 255)
EGO_AXIS = (255, 210, 40)

STATE_BOTH = "BOTH_LANES"
STATE_LEFT_ONLY = "LEFT_LANE_ONLY"
STATE_RIGHT_ONLY = "RIGHT_LANE_ONLY"
STATE_LOST = "LOST"


class LaneFit(NamedTuple):
    coeffs: np.ndarray
    y_min: int
    y_max: int
    roi_x: float
    count: int
    rmse: float


class SegBEVNode(Node):

    def __init__(self):
        super().__init__("seg_bev_node")

        self.declare_parameter("mask_topic", "/seg/mask_raw")
        self.declare_parameter("camera_params", "bev2/camera_params.txt")
        self.declare_parameter("bev_size", 160)
        self.declare_parameter("mask_width", 160)
        self.declare_parameter("mask_height", 120)
        self.declare_parameter("default_lane_width", DEFAULT_LANE_WIDTH_PX)
        self.declare_parameter("camera_offset_x_px", 0.0)
        self.declare_parameter("roi_start_ratio", DEFAULT_ROI_START_RATIO)
        self.declare_parameter("roi_end_ratio", DEFAULT_ROI_END_RATIO)
        
        self.declare_parameter("bev_src_bottom_left_x", 8.0)
        self.declare_parameter("bev_src_bottom_left_y", 118.0)
        self.declare_parameter("bev_src_bottom_right_x", 152.0)
        self.declare_parameter("bev_src_bottom_right_y", 118.0)
        self.declare_parameter("bev_src_top_right_x", 120.0)
        self.declare_parameter("bev_src_top_right_y", 80.0)
        self.declare_parameter("bev_src_top_left_x", 30.0)
        self.declare_parameter("bev_src_top_left_y", 80.0)
        self.declare_parameter("bev_dst_margin_x", 20.0)
        self.declare_parameter("bev_dst_top_y", 5.0)
        self.declare_parameter("bev_dst_bottom_y", 155.0)

        self.bev_size      = self.get_parameter("bev_size").value
        self.mask_w        = self.get_parameter("mask_width").value
        self.mask_h        = self.get_parameter("mask_height").value
        mask_topic         = self.get_parameter("mask_topic").value
        params_path        = self.get_parameter("camera_params").value
        self.camera_offset_x_px = self.get_parameter("camera_offset_x_px").value
        self.lane_width_px      = self.get_parameter("default_lane_width").value
        self.roi_start_ratio    = self.get_parameter("roi_start_ratio").value
        self.roi_end_ratio      = self.get_parameter("roi_end_ratio").value
        self.alpha_lane_width   = 0.05

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
            [self.get_parameter("bev_src_bottom_left_x").value,
             self.get_parameter("bev_src_bottom_left_y").value],
            [self.get_parameter("bev_src_bottom_right_x").value,
             self.get_parameter("bev_src_bottom_right_y").value],
            [self.get_parameter("bev_src_top_right_x").value,
             self.get_parameter("bev_src_top_right_y").value],
            [self.get_parameter("bev_src_top_left_x").value,
             self.get_parameter("bev_src_top_left_y").value],
        ])
        dst_margin_x = self.get_parameter("bev_dst_margin_x").value
        dst_top_y    = self.get_parameter("bev_dst_top_y").value
        dst_bottom_y = self.get_parameter("bev_dst_bottom_y").value
        dst_points   = np.float32([
            [dst_margin_x, dst_bottom_y],
            [self.bev_size - dst_margin_x, dst_bottom_y],
            [self.bev_size - dst_margin_x, dst_top_y],
            [dst_margin_x, dst_top_y],
        ])
        self.h_matrix, _ = cv2.findHomography(src_points, dst_points)

        self.bridge = CvBridge()
        self._frames = 0
        
        self.sub          = self.create_subscription(Image, mask_topic, self.image_callback, 10)
        self.pub_bev_mask = self.create_publisher(Image, "/seg/bev_mask", 10)
        self.pub_bev      = self.create_publisher(Image, "/seg/bev", 10)
        self.pub_overlay  = self.create_publisher(Image, "/seg/bev_overlay", 10)

        self.get_logger().info(f"Subscribed to mask topic: {mask_topic}")
        self.get_logger().info(f"Expected mask size     : {self.mask_w}x{self.mask_h}")
        self.get_logger().info(f"BEV size               : {self.bev_size}x{self.bev_size}")
        self.get_logger().info(f"BEV source points      : {src_points.tolist()}")
        self.get_logger().info(f"BEV destination points : {dst_points.tolist()}")
        self.get_logger().info("seg_bev_node ready - waiting for masks")

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls, bgr in CLASS_COLORS.items():
            color[mask == cls] = bgr
        return color

    def _polyfit_safe(self, y_points: np.ndarray, x_points: np.ndarray, degree: int) -> Optional[np.ndarray]:
        if y_points.size <= degree or np.unique(y_points).size <= degree:
            return None

        try:
            with warnings.catch_warnings():
                for warning_type in _RANK_WARNING_TYPES:
                    warnings.simplefilter("error", warning_type)
                coeffs = np.polyfit(
                    y_points.astype(np.float64),
                    x_points.astype(np.float64),
                    degree,
                )
        except _POLYFIT_ERRORS:
            return None

        if not np.all(np.isfinite(coeffs)):
            return None
        return coeffs

    def _binned_lane_points(
        self,
        y_points: np.ndarray,
        x_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(y_points)
        y_sorted = y_points[order]
        x_sorted = x_points[order].astype(np.float64) - float(self.camera_offset_x_px)
        bins = y_sorted // LANE_FIT_BIN_HEIGHT_PX
        unique_bins, starts = np.unique(bins, return_index=True)

        med_y = np.empty(unique_bins.size, dtype=np.float64)
        med_x = np.empty(unique_bins.size, dtype=np.float64)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < starts.size else y_sorted.size
            med_y[i] = np.median(y_sorted[start:end])
            med_x[i] = np.median(x_sorted[start:end])
        return med_y, med_x

    def _roi_bounds(self, mask: np.ndarray) -> Tuple[int, int]:
        h, _ = mask.shape
        y0 = int(h * float(self.roi_start_ratio))
        y1 = int(h * float(self.roi_end_ratio))
        return max(0, y0), min(h, max(y0 + 1, y1))

    def _lane_measurement(self, roi: np.ndarray, cls: int) -> Tuple[Optional[float], int]:
        _, xs = np.where(roi == cls)
        if xs.size < 12:
            return None, 0
        return float(np.median(xs)), int(xs.size)

    def _selected_component_points(
        self,
        mask: np.ndarray,
        cls: int,
        roi_x: float,
        y0: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        class_mask = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        best_label = None
        best_distance = float("inf")
        best_count = 0
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] <= MIN_POLYFIT_POINTS:
                continue

            roi_labels = labels[y0:y1, :]
            roi_ys, roi_xs = np.where(roi_labels == label)
            if roi_xs.size < 12:
                continue

            distance = abs(float(np.median(roi_xs)) - roi_x)
            if distance < best_distance:
                best_distance = distance
                best_count = int(roi_xs.size)
                best_label = label

        if best_label is None or best_distance > MAX_COMPONENT_ROI_DISTANCE_PX:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        ys, xs = np.where(labels == best_label)
        if ys.size < max(MIN_POLYFIT_POINTS, best_count):
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        return ys, xs

    def _fit_selected_lane(
        self,
        mask: np.ndarray,
        cls: int,
        roi_x: Optional[float],
        count: int,
        y0: int,
        y1: int,
    ) -> Optional[LaneFit]:
        if roi_x is None:
            return None

        lane_y, lane_x = self._selected_component_points(mask, cls, roi_x, y0, y1)
        if lane_y.size == 0:
            return None
        return self._fit_lane_poly(lane_y, lane_x, roi_x, count)

    def _fit_lane_poly(
        self,
        y_points: np.ndarray,
        x_points: np.ndarray,
        roi_x: float,
        count: int,
    ) -> Optional[LaneFit]:
        if y_points.size <= MIN_POLYFIT_POINTS or np.unique(y_points).size < 3:
            return None

        fit_y, fit_x = self._binned_lane_points(y_points, x_points)
        if fit_y.size < MIN_LANE_FIT_BINS:
            return None

        linear = self._polyfit_safe(fit_y, fit_x, 1)
        if linear is None:
            return None

        linear_x = np.polyval(linear, fit_y)
        linear_rmse = float(np.sqrt(np.mean((linear_x - fit_x) ** 2)))
        best_coeffs = linear
        best_rmse = linear_rmse

        x_span = float(np.ptp(fit_x))
        if x_span < CURVE_MIN_X_SPAN_PX:
            best_coeffs = best_coeffs.astype(np.float64, copy=True)
            best_coeffs[-1] += float(self.camera_offset_x_px)
            return LaneFit(
                best_coeffs,
                int(np.min(fit_y)),
                int(np.max(fit_y)),
                float(roi_x),
                int(count),
                best_rmse,
            )

        curved = self._polyfit_safe(fit_y, fit_x, 2)
        if curved is not None:
            curved_x = np.polyval(curved, fit_y)
            curved_rmse = float(np.sqrt(np.mean((curved_x - fit_x) ** 2)))
            improvement = 0.0 if linear_rmse <= 1e-6 else (linear_rmse - curved_rmse) / linear_rmse
            if improvement >= CURVE_RMSE_IMPROVEMENT and curved_rmse < best_rmse:
                best_coeffs = curved

        best_coeffs = best_coeffs.astype(np.float64, copy=True)
        best_coeffs[-1] += float(self.camera_offset_x_px)
        return LaneFit(
            best_coeffs,
            int(np.min(fit_y)),
            int(np.max(fit_y)),
            float(roi_x),
            int(count),
            best_rmse,
        )

    def _draw_polynomial(
        self,
        overlay: np.ndarray,
        coeffs: Optional[np.ndarray],
        color: Tuple[int, int, int],
        thickness: int,
        y_min: int = 0,
        y_max: Optional[int] = None,
    ) -> None:
        if coeffs is None:
            return

        h, w = overlay.shape[:2]
        if y_max is None:
            y_max = h - 1
        y_min = int(np.clip(y_min, 0, h - 1))
        y_max = int(np.clip(y_max, y_min, h - 1))
        ys = np.arange(y_min, y_max + 1, dtype=np.float64)
        xs = np.polyval(coeffs, ys)
        xs = np.clip(np.rint(xs), 0, w - 1).astype(np.int32)
        pts = np.column_stack((
            xs,
            ys.astype(np.int32),
        )).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)

    def _average_lane_coeffs(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        degree = max(left.size, right.size)
        left_pad = np.pad(left, (degree - left.size, 0), mode="constant")
        right_pad = np.pad(right, (degree - right.size, 0), mode="constant")
        return (left_pad + right_pad) / 2.0

    def _shift_lane_coeffs(self, fit: LaneFit, offset_px: float) -> np.ndarray:
        coeffs = fit.coeffs.copy()
        coeffs[-1] += offset_px
        return coeffs

    def _update_lane_width(self, measured_width: float) -> None:
        self.lane_width_px = (
            self.alpha_lane_width * measured_width
            + (1.0 - self.alpha_lane_width) * float(self.lane_width_px)
        )

    def driving_overlay(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape

        # 1. Near-car ROI measurement, mirroring EdgeLaneTracker.
        y0, y1 = self._roi_bounds(mask)
        roi = mask[y0:y1, :]
        ego_center = float(w / 2.0 + self.camera_offset_x_px)
        half_lane = float(self.lane_width_px) / 2.0
        obstacle_mask = (mask == CLASS_VEHICLE).astype(np.uint8)

        yellow_x, yellow_count = self._lane_measurement(roi, CLASS_YELLOW)
        white_x, white_count = self._lane_measurement(roi, CLASS_WHITE)

        use_yellow = yellow_x is not None
        use_white = white_x is not None
        if use_yellow and use_white:
            measured_width = abs(float(white_x) - float(yellow_x))
            if measured_width >= MIN_BOTH_LANE_GAP_PX:
                self._update_lane_width(measured_width)
                half_lane = float(self.lane_width_px) / 2.0
            elif yellow_count >= white_count:
                use_white = False
            else:
                use_yellow = False

        # 2. Fit only the locally selected lane component(s).
        yellow_fit = self._fit_selected_lane(
            mask, CLASS_YELLOW, yellow_x if use_yellow else None, yellow_count, y0, y1
        )
        white_fit = self._fit_selected_lane(
            mask, CLASS_WHITE, white_x if use_white else None, white_count, y0, y1
        )

        # 3. Unified centerline solver with EdgeLaneTracker single-lane direction logic.
        center_coeffs = None
        center_y_min = y0
        center_y_max = y1 - 1
        if yellow_fit is not None and white_fit is not None:
            center_coeffs = self._average_lane_coeffs(yellow_fit.coeffs, white_fit.coeffs)
            center_y_min = max(yellow_fit.y_min, white_fit.y_min)
            center_y_max = min(yellow_fit.y_max, white_fit.y_max)
            if center_y_min > center_y_max:
                center_y_min = min(yellow_fit.y_min, white_fit.y_min)
                center_y_max = max(yellow_fit.y_max, white_fit.y_max)
        elif yellow_fit is not None:
            offset = half_lane if yellow_fit.roi_x < ego_center else -half_lane
            center_coeffs = self._shift_lane_coeffs(yellow_fit, offset)
            center_y_min = yellow_fit.y_min
            center_y_max = yellow_fit.y_max
        elif white_fit is not None:
            offset = -half_lane if white_fit.roi_x > ego_center else half_lane
            center_coeffs = self._shift_lane_coeffs(white_fit, offset)
            center_y_min = white_fit.y_min
            center_y_max = white_fit.y_max

        # 4. Obstacle Clustering & Bounding Boxes
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Canvas Composition & Graphical Overlay Rendering
        overlay = self.colorize(mask)

        # Draw Polynomial Curves across full image height
        if yellow_fit is not None:
            self._draw_polynomial(
                overlay, yellow_fit.coeffs, CURVE_YELLOW, 2, yellow_fit.y_min, yellow_fit.y_max
            )
        if white_fit is not None:
            self._draw_polynomial(
                overlay, white_fit.coeffs, CURVE_WHITE, 2, white_fit.y_min, white_fit.y_max
            )
        self._draw_polynomial(overlay, center_coeffs, CURVE_CENTER, 3, center_y_min, center_y_max)

        # Draw Obstacle Bounding Boxes
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_OBSTACLE_AREA_PX:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), OBSTACLE_BOX, 2)

        # Draw Ego Indicators (dotted vertical axis with camera offset calibration)
        ego_center = int(np.clip(round(ego_center), 0, w - 1))
        for y_dot in range(0, h, 6):
            cv2.circle(overlay, (ego_center, y_dot), 1, EGO_AXIS, -1)

        return overlay

    def image_callback(self, msg: Image):
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        if mask.shape[:2] != (self.mask_h, self.mask_w):
            mask = cv2.resize(
                mask, 
                (self.mask_w, self.mask_h), 
                interpolation=cv2.INTER_NEAREST
            )

        mask_undist = undistort_mask(mask, self.k, self.d)
        bev_mask = to_bev(
            mask_undist,
            self.h_matrix,
            bev_size=(self.bev_size, self.bev_size)
        )
        bev_clean = cleanup_bev(bev_mask)

        bev_msg        = self.bridge.cv2_to_imgmsg(bev_clean, encoding="mono8")
        bev_msg.header = msg.header
        self.pub_bev_mask.publish(bev_msg)

        bev_color   = self.colorize(bev_clean)
        bev_vis_msg = self.bridge.cv2_to_imgmsg(bev_color, encoding="bgr8")
        bev_vis_msg.header = msg.header
        self.pub_bev.publish(bev_vis_msg)

        overlay     = self.driving_overlay(bev_clean)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.pub_overlay.publish(overlay_msg)

        self._frames += 1
        if self._frames == 1:
            self.get_logger().info(
                f"First mask received - classes in frame: {np.unique(mask).tolist()}"
            )
        elif self._frames % 100 == 0:
            self.get_logger().info(
                f"Frames processed: {self._frames} - classes in latest frame: {np.unique(mask).tolist()}"
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
