"""
lane_analyzer.py
================
Shared lane analysis and overlay rendering module.

Extracted from seg_bev_node.py to eliminate duplication between:
- seg_bev_node.py (BEV overlay)
- autonomous_driving.py (lane tracking for control)
- limo_segmentation_node.py (camera overlay)

This module has ZERO ROS dependencies — pure numpy/OpenCV.
"""

from __future__ import annotations
import warnings
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# CONSTANTS (shared across all nodes)
# =========================================================
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
DEFAULT_ROI_START_RATIO = 0.75
DEFAULT_ROI_END_RATIO = 0.95
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
STATE_DRIVABLE = "DRIVABLE_AREA_MODE"


# =========================================================
# DATA STRUCTURES
# =========================================================
class LaneFit(NamedTuple):
    """Parametric lane path with metadata."""
    path_points: np.ndarray     # N x 2 array of [x, y] coordinates
    y_min: int                  # minimum y where fit is valid
    y_max: int                  # maximum y where fit is valid
    roi_x: float                # x position in ROI used for selection
    count: int                  # number of pixels in fit
    rmse: float                 # root mean square error of fit


# =========================================================
# LANE ANALYZER (stateless fitting logic)
# =========================================================
class LaneAnalyzer:
    """
    Stateless lane polynomial fitting.
    
    All configuration passed via __init__ — no hidden state between calls.
    """

    def __init__(
        self,
        lane_width_px: float = DEFAULT_LANE_WIDTH_PX,
        camera_offset_x_px: float = 0.0,
        roi_start_ratio: float = DEFAULT_ROI_START_RATIO,
        roi_end_ratio: float = DEFAULT_ROI_END_RATIO,
        alpha_lane_width: float = 0.05,
    ):
        self.lane_width_px = float(lane_width_px)
        self.camera_offset_x_px = float(camera_offset_x_px)
        self.roi_start_ratio = float(roi_start_ratio)
        self.roi_end_ratio = float(roi_end_ratio)
        self.alpha_lane_width = float(alpha_lane_width)

    # ---- Core fitting pipeline ----

    def roi_bounds(self, mask: np.ndarray) -> Tuple[int, int]:
        """Return (y0, y1) pixel bounds of the ROI."""
        h, _ = mask.shape
        y0 = int(h * self.roi_start_ratio)
        y1 = int(h * self.roi_end_ratio)
        return max(0, y0), min(h, max(y0 + 1, y1))

    def lane_measurement(self, roi: np.ndarray, cls: int) -> Tuple[Optional[float], int]:
        """Measure median x position of class in ROI. Returns (x, count) or (None, 0)."""
        _, xs = np.where(roi == cls)
        if xs.size < 12:
            return None, 0
        return float(np.median(xs)), int(xs.size)

    def _polyfit_safe(self, y_points: np.ndarray, x_points: np.ndarray, degree: int) -> Optional[np.ndarray]:
        """Safe polynomial fit with warning/error handling."""
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
        """Bin points by y and compute median x per bin."""
        order = np.argsort(y_points)
        y_sorted = y_points[order]
        x_sorted = x_points[order].astype(np.float64)
        bins = y_sorted // LANE_FIT_BIN_HEIGHT_PX
        unique_bins, starts = np.unique(bins, return_index=True)

        med_y = np.empty(unique_bins.size, dtype=np.float64)
        med_x = np.empty(unique_bins.size, dtype=np.float64)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < starts.size else y_sorted.size
            med_y[i] = np.median(y_sorted[start:end])
            med_x[i] = np.median(x_sorted[start:end])
        return med_y, med_x

    def _selected_component_points(
        self,
        mask: np.ndarray,
        cls: int,
        roi_x: float,
        y0: int,
        y1: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract points from the connected component closest to roi_x in the ROI."""
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
            
        # Parametric variable t (cumulative distance)
        # Sort bottom to top (highest Y to lowest Y from robot's perspective)
        order = np.argsort(fit_y)[::-1]
        sy = fit_y[order]
        sx = fit_x[order]
        
        dt = np.sqrt(np.diff(sx)**2 + np.diff(sy)**2)
        t = np.zeros(len(sx))
        t[1:] = np.cumsum(dt)
        if t[-1] == 0:
            return None
        t_norm = t / t[-1]  # Normalize distance from 0.0 to 1.0
        
        # Fit X and Y independently based on distance t
        degree = 2 if len(t_norm) > 3 else 1
        coeffs_x = np.polyfit(t_norm, sx, degree)
        coeffs_y = np.polyfit(t_norm, sy, degree)
        
        # Evaluate 50 discrete waypoints along the curve
        t_eval = np.linspace(0.0, 1.0, 50)
        path_x = np.polyval(coeffs_x, t_eval)
        path_y = np.polyval(coeffs_y, t_eval)
        
        path_points = np.column_stack((path_x, path_y))
        path_points[:, 0] += float(self.camera_offset_x_px)
        
        # Simple Euclidean RMSE against bins
        rmse = 0.0 
        
        return LaneFit(
            path_points,
            int(np.min(fit_y)),
            int(np.max(fit_y)),
            float(roi_x),
            int(count),
            rmse,
        )

    # ---- Public API ----

    def analyze(self, mask: np.ndarray) -> Tuple[Optional[LaneFit], Optional[LaneFit], Optional[np.ndarray], int, int, str]:
        """
        Full lane analysis on a mask.
        
        Returns:
            yellow_fit, white_fit, center_coeffs, center_y_min, center_y_max, state
        """
        h, w = mask.shape
        y0, y1 = self.roi_bounds(mask)
        roi = mask[y0:y1, :]
        ego_center = float(w / 2.0)
        half_lane = float(self.lane_width_px) / 2.0
        obstacle_mask = (mask == CLASS_VEHICLE).astype(np.uint8)

        # 1. ROI measurement (mirrors EdgeLaneTracker)
        yellow_x, yellow_count = self.lane_measurement(roi, CLASS_YELLOW)
        white_x, white_count = self.lane_measurement(roi, CLASS_WHITE)

        use_yellow = yellow_x is not None
        use_white = white_x is not None

        if use_yellow and use_white:
            measured_width = abs(float(white_x) - float(yellow_x))
            if measured_width >= MIN_BOTH_LANE_GAP_PX:
                self.update_lane_width(measured_width)
                half_lane = float(self.lane_width_px) / 2.0
            elif yellow_count >= white_count:
                use_white = False
            else:
                use_yellow = False

        # 2. Fit selected lane components
        yellow_fit = self._fit_selected_lane(
            mask, CLASS_YELLOW, yellow_x if use_yellow else None, yellow_count, y0, y1
        )
        white_fit = self._fit_selected_lane(
            mask, CLASS_WHITE, white_x if use_white else None, white_count, y0, y1
        )

        # 3. Unified centerline solver
        center_path = None
        center_y_min = y0
        center_y_max = y1 - 1
        state = STATE_LOST
        multiplier = 1.0

        if yellow_fit is not None and white_fit is not None:
            center_path = self._average_lane_paths(yellow_fit, white_fit)
            center_y_min = max(yellow_fit.y_min, white_fit.y_min)
            center_y_max = min(yellow_fit.y_max, white_fit.y_max)
            if center_y_min > center_y_max:
                center_y_min = min(yellow_fit.y_min, white_fit.y_min)
                center_y_max = max(yellow_fit.y_max, white_fit.y_max)
            state = STATE_BOTH
        elif yellow_fit is not None:
            offset = half_lane if yellow_fit.roi_x < ego_center else -half_lane
            center_path = self._shift_lane_path(yellow_fit, offset)
            center_y_min = yellow_fit.y_min
            center_y_max = yellow_fit.y_max
            state = STATE_LEFT_ONLY if yellow_fit.roi_x < ego_center else STATE_RIGHT_ONLY
        elif white_fit is not None:
            offset = -half_lane if white_fit.roi_x > ego_center else half_lane
            center_path = self._shift_lane_path(white_fit, offset)
            center_y_min = white_fit.y_min
            center_y_max = white_fit.y_max
            state = STATE_RIGHT_ONLY if white_fit.roi_x > ego_center else STATE_LEFT_ONLY

        if state == STATE_RIGHT_ONLY:
            multiplier = -1.0

        # 4. Fallback to drivable area if completely lost
        if center_path is None:
            _, xs = np.where(roi == CLASS_ROAD)
            if len(xs) > 50:
                med_x = float(np.median(xs))
                # Create a vertical straight path
                center_path = np.column_stack((
                    np.full(50, med_x),
                    np.linspace(center_y_max, center_y_min, 50)
                ))
                state = STATE_DRIVABLE

        return yellow_fit, white_fit, center_path, center_y_min, center_y_max, state, multiplier

    def update_lane_width(self, measured_width: float) -> None:
        """EMA update of running lane width."""
        self.lane_width_px = (
            self.alpha_lane_width * measured_width
            + (1.0 - self.alpha_lane_width) * float(self.lane_width_px)
        )

    # ---- Helpers ----

    def _average_lane_paths(self, left_fit: LaneFit, right_fit: LaneFit) -> np.ndarray:
        # Averages the 50 aligned waypoints (t_eval points match 1:1)
        return (left_fit.path_points + right_fit.path_points) / 2.0

    def _shift_lane_path(self, fit: LaneFit, offset_px: float) -> np.ndarray:
        shifted = fit.path_points.copy()
        shifted[:, 0] += offset_px
        return shifted


# =========================================================
# LANE OVERLAY RENDERER (stateless drawing)
# =========================================================
class LaneOverlayRenderer:
    """
    Stateless overlay drawing.
    
    All colors and style constants passed via __init__ or use module-level defaults.
    """

    def __init__(
        self,
        curve_yellow: Tuple[int, int, int] = CURVE_YELLOW,
        curve_white: Tuple[int, int, int] = CURVE_WHITE,
        curve_center: Tuple[int, int, int] = CURVE_CENTER,
        obstacle_box: Tuple[int, int, int] = OBSTACLE_BOX,
        ego_axis: Tuple[int, int, int] = EGO_AXIS,
    ):
        self.curve_yellow = curve_yellow
        self.curve_white = curve_white
        self.curve_center = curve_center
        self.obstacle_box = obstacle_box
        self.ego_axis = ego_axis

    def colorize(self, mask: np.ndarray) -> np.ndarray:
        """Convert single-channel class mask to BGR color image."""
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls, bgr in CLASS_COLORS.items():
            color[mask == cls] = bgr
        return color

    def draw_path(
        self,
        overlay: np.ndarray,
        path_points: Optional[np.ndarray],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        """Draw parametric path on overlay."""
        if path_points is None or len(path_points) == 0:
            return
        
        h, w = overlay.shape[:2]
        pts = path_points.copy()
        pts[:, 0] = np.clip(np.rint(pts[:, 0]), 0, w - 1)
        pts[:, 1] = np.clip(np.rint(pts[:, 1]), 0, h - 1)
        
        pts = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=thickness)

    def draw_obstacle_boxes(self, overlay: np.ndarray, mask: np.ndarray, min_area: int = MIN_OBSTACLE_AREA_PX) -> None:
        """Draw bounding boxes around vehicle-class obstacles."""
        obstacle_mask = (mask == CLASS_VEHICLE).astype(np.uint8)
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), self.obstacle_box, 2)

    def draw_ego_axis(self, overlay: np.ndarray, ego_center_x: float) -> None:
        """Draw dotted vertical ego center line."""
        h, w = overlay.shape[:2]
        ego_center = int(np.clip(round(ego_center_x), 0, w - 1))
        for y_dot in range(0, h, 6):
            cv2.circle(overlay, (ego_center, y_dot), 1, self.ego_axis, -1)

    def driving_overlay(
        self,
        mask: np.ndarray,
        analyzer: LaneAnalyzer,
        lane_width_px: float,
        camera_offset_x_px: float,
        roi_ratios: Tuple[float, float],
    ) -> np.ndarray:
        h, w = mask.shape

        # Run analysis (reuses analyzer's internal logic)
        yellow_fit, white_fit, center_path, center_y_min, center_y_max, _, _ = analyzer.analyze(mask)

        # Compose overlay
        overlay = self.colorize(mask)

        # Draw path curves
        if yellow_fit is not None:
            self.draw_path(overlay, yellow_fit.path_points, self.curve_yellow, 2)
        if white_fit is not None:
            self.draw_path(overlay, white_fit.path_points, self.curve_white, 2)
        self.draw_path(overlay, center_path, self.curve_center, 3)

        # Draw obstacle boxes & ego axis
        self.draw_obstacle_boxes(overlay, mask)
        ego_center = float(w / 2.0 + camera_offset_x_px)
        self.draw_ego_axis(overlay, ego_center)

        return overlay