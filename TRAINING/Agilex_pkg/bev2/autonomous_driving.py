#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    from cv_bridge import CvBridge
except ImportError as exc:  # pragma: no cover - robot-only path
    raise RuntimeError("ROS2 and cv_bridge are required to run this node.") from exc


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


def colorise_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        out[mask == cls_id] = color
    return out


class LaneDetector:
    def __init__(
        self,
        lane_offset_ratio: float = 0.22,
        preferred_lane: str = "auto",
        lane_switch_margin_ratio: float = 0.08,
        min_line_pixels: int = 12,
        lane_width_alpha: float = 0.2,
        lost_frame_limit: int = 4,
        robot_center_bias_ratio: float = 0.0,
        auto_switch_confirm_frames: int = 3,
        lock_initial_lane: bool = True,
    ):
        self.lane_offset_ratio = lane_offset_ratio
        self.preferred_lane = preferred_lane
        self.lane_switch_margin_ratio = lane_switch_margin_ratio
        self.min_line_pixels = min_line_pixels
        self.lane_width_alpha = lane_width_alpha
        self.lost_frame_limit = lost_frame_limit
        self.robot_center_bias_ratio = robot_center_bias_ratio
        self.auto_switch_confirm_frames = auto_switch_confirm_frames
        self.active_side: Optional[str] = None
        self.estimated_lane_width_px: Optional[float] = None
        self.last_yellow_x: Optional[float] = None
        self.last_target_x: Optional[float] = None
        self.lost_frames = 0
        self._pending_side: Optional[str] = None
        self._pending_side_frames = 0
        self.last_curve_direction: str = "straight"

        # ── Role tracking (inner/outer) ────────────────────────────────────────
        # ISOLATED: The role system is kept here for future overtaking use but
        # is NOT used for normal lane following. The inner/outer detection was
        # causing the car to switch from inner to outer lane because curve_direction
        # flips on a circular track, inverting the role→side mapping each time.
        # Re-enable by setting lock_initial_lane=False and using role in detect().
        self.active_role: Optional[str] = None
        self._pending_role: Optional[str] = None
        self._pending_role_frames = 0

        # ── Lane lock ──────────────────────────────────────────────────────────
        # When True: once the starting side is confirmed, it is NEVER changed
        # automatically. The car stays in whichever lane it started in.
        # Set to False only when you want to re-enable auto-switching (e.g. for
        # overtaking experiments).
        self.lock_initial_lane: bool = lock_initial_lane
        self._side_locked: bool = False   # becomes True after first confirmation
        self._lane_lock_override: bool = False

    def _centroid_x(self, bev_mask: np.ndarray, cls_id: int, row_slice: slice) -> Optional[float]:
        xs = np.where(bev_mask[row_slice, :] == cls_id)[1]
        return float(xs.mean()) if xs.size >= self.min_line_pixels else None

    def _centroid_from_xs(self, xs: np.ndarray) -> Optional[float]:
        return float(xs.mean()) if xs.size >= self.min_line_pixels else None

    def _split_white_lanes(
        self,
        bev_mask: np.ndarray,
        row_slice: slice,
        yellow_x: Optional[float],
        robot_x: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        # White pixels can belong to both borders. Split them into left and right
        # candidates around the yellow divider when possible. If yellow is missing,
        # fall back to the last known divider before using the robot center.
        xs = np.where(bev_mask[row_slice, :] == CLASS_WHITE)[1]
        if xs.size < self.min_line_pixels:
            return None, None

        split_x = yellow_x if yellow_x is not None else self.last_yellow_x
        if split_x is None:
            split_x = robot_x
        margin = max(3.0, bev_mask.shape[1] * 0.03)

        left_xs = xs[xs < split_x - margin]
        right_xs = xs[xs > split_x + margin]
        return self._centroid_from_xs(left_xs), self._centroid_from_xs(right_xs)

    def _lane_center(
        self,
        side: str,
        yellow_x: Optional[float],
        white_left_x: Optional[float],
        white_right_x: Optional[float],
        lane_pos: float,
        lane_offset_px: float,
    ) -> Optional[float]:
        # When both boundaries of a corridor are present, use the geometric
        # center inside that corridor. This is the most stable mode.
        if side == "left":
            if yellow_x is not None and white_left_x is not None:
                return white_left_x + lane_pos * (yellow_x - white_left_x)
            return None

        if yellow_x is not None and white_right_x is not None:
            return yellow_x + lane_pos * (white_right_x - yellow_x)
        return None

    def _side_from_yellow(self, robot_x: float, yellow_x: Optional[float]) -> Optional[str]:
        # Yellow is the road divider. If it is right of the robot, the robot is
        # in the left corridor; if it is left of the robot, the robot is in the
        # right corridor.
        if yellow_x is None:
            return None
        return "left" if robot_x < yellow_x else "right"

    def _choose_side(
        self,
        robot_x: float,
        left_center_x: Optional[float],
        right_center_x: Optional[float],
        width: int,
        forced_side: Optional[str] = None,
    ) -> Optional[str]:
        # Explicit configuration always wins.
        if forced_side in {"left", "right"}:
            self.active_side = forced_side
            self._side_locked = True
            return self.active_side

        # Use yellow line as ground truth for side selection if available
        yellow_side = self._side_from_yellow(robot_x, self.last_yellow_x)
        if yellow_side is not None and not self._side_locked:
            self.active_side = yellow_side
            if self.lock_initial_lane:
                self._side_locked = True
            return self.active_side

        candidates = {}
        if left_center_x is not None:
            candidates["left"] = abs(robot_x - left_center_x)
        if right_center_x is not None:
            candidates["right"] = abs(robot_x - right_center_x)
        if not candidates:
            return self.active_side

        # ── Lane lock: once confirmed, never auto-switch ───────────────────────
        # Prevents the inner→outer drift caused by curve_direction flipping the
        # role→side map on a circular track.
        if self._side_locked and self.active_side in {"left", "right"} and not self._lane_lock_override:
            return self.active_side

        # Not yet locked — pick best side from evidence.
        if self.active_side in candidates and len(candidates) == 2:
            other_side = "right" if self.active_side == "left" else "left"
            margin = width * self.lane_switch_margin_ratio
            if candidates[other_side] + margin < candidates[self.active_side]:
                self.active_side = other_side
            if self.lock_initial_lane:
                self._side_locked = True
            return self.active_side

        self.active_side = min(candidates, key=candidates.get)
        if self.lock_initial_lane:
            self._side_locked = True
        return self.active_side

    def _update_auto_side(self, evidence_side: Optional[str]) -> Optional[str]:
        # In auto mode, keep the current side stable, but allow a switch if the
        # opposite side is supported for several consecutive frames. This lets
        # the car recover when you physically move it from one corridor to the
        # other during runtime.
        if self.preferred_lane in {"left", "right"}:
            self.active_side = self.preferred_lane
            self._pending_side = None
            self._pending_side_frames = 0
            return self.active_side

        if evidence_side is None:
            return self.active_side

        if self.active_side is None:
            self.active_side = evidence_side
            self._pending_side = None
            self._pending_side_frames = 0
            return self.active_side

        if evidence_side == self.active_side:
            self._pending_side = None
            self._pending_side_frames = 0
            return self.active_side

        if self._pending_side == evidence_side:
            self._pending_side_frames += 1
        else:
            self._pending_side = evidence_side
            self._pending_side_frames = 1

        if self._pending_side_frames >= self.auto_switch_confirm_frames:
            self.active_side = evidence_side
            self._pending_side = None
            self._pending_side_frames = 0

        return self.active_side

    def _update_auto_role(self, evidence_role: Optional[str]) -> Optional[str]:
        if self.preferred_lane in {"inner", "outer"}:
            self.active_role = self.preferred_lane
            self._pending_role = None
            self._pending_role_frames = 0
            return self.active_role

        if evidence_role is None:
            return self.active_role

        if self.active_role is None:
            self.active_role = evidence_role
            self._pending_role = None
            self._pending_role_frames = 0
            return self.active_role

        if evidence_role == self.active_role:
            self._pending_role = None
            self._pending_role_frames = 0
            return self.active_role

        if self._pending_role == evidence_role:
            self._pending_role_frames += 1
        else:
            self._pending_role = evidence_role
            self._pending_role_frames = 1

        if self._pending_role_frames >= self.auto_switch_confirm_frames:
            self.active_role = evidence_role
            self._pending_role = None
            self._pending_role_frames = 0

        return self.active_role

    def _update_lane_width(self, lane_width_px: Optional[float]):
        # Smooth the learned lane width so single-lane mode can project a stable
        # center when one boundary disappears.
        if lane_width_px is None or lane_width_px <= 1.0:
            return
        if self.estimated_lane_width_px is None:
            self.estimated_lane_width_px = lane_width_px
        else:
            alpha = self.lane_width_alpha
            self.estimated_lane_width_px = (1.0 - alpha) * self.estimated_lane_width_px + alpha * lane_width_px

    def _project_single_lane_target(
        self,
        lane_side: Optional[str],
        yellow_x: Optional[float],
        white_left_x: Optional[float],
        white_right_x: Optional[float],
        lane_pos: float,
        default_lane_width_px: float,
    ) -> Optional[float]:
        # Offset tracking mode. Once we know the approximate lane width from
        # previous dual-line frames, we can keep following the corridor using a
        # single visible boundary.
        lane_width_px = self.estimated_lane_width_px or default_lane_width_px

        if lane_side == "left":
            if white_left_x is not None:
                return white_left_x + lane_pos * lane_width_px
            if yellow_x is not None:
                return yellow_x - (1.0 - lane_pos) * lane_width_px
            return None

        if lane_side == "right":
            if yellow_x is not None:
                return yellow_x + lane_pos * lane_width_px
            if white_right_x is not None:
                return white_right_x - (1.0 - lane_pos) * lane_width_px
            return None

        return None

    def _infer_side_from_visible_lines(
        self,
        robot_x: float,
        yellow_x: Optional[float],
        white_left_x: Optional[float],
        white_right_x: Optional[float],
    ) -> Optional[str]:
        # Startup fallback. When the car begins in a curve, we may not have a
        # dual-lane observation yet. In that case infer the lane side from the
        # single visible boundary instead of waiting for the yellow line.
        if self.preferred_lane in {"left", "right"}:
            return self.preferred_lane

        if self.active_side in {"left", "right"}:
            return self.active_side

        if white_left_x is not None and white_right_x is None:
            return "left"
        if white_right_x is not None and white_left_x is None:
            return "right"

        if yellow_x is not None:
            # If the divider is left of the robot center, we are probably in
            # the right corridor; if it is right of center, we are in the left.
            return "right" if yellow_x < robot_x else "left"

        if white_left_x is not None and white_right_x is not None:
            return "left" if abs(robot_x - white_left_x) < abs(white_right_x - robot_x) else "right"

        return None

    def _infer_role_from_yellow(
        self,
        robot_x: float,
        yellow_x: Optional[float],
        curve_direction: str,
    ) -> Optional[str]:
        # Yellow is the divider between the inner and outer corridors. If the
        # robot center is on one side of yellow, that tells us which corridor
        # the car currently occupies.
        if yellow_x is None or curve_direction not in {"left", "right"}:
            return None

        if curve_direction == "left":
            return "inner" if robot_x < yellow_x else "outer"
        return "inner" if robot_x > yellow_x else "outer"

    def _curve_anchor_dx(
        self,
        bev_mask: np.ndarray,
        cls_id: int,
        near_rows: slice,
        far_rows: slice,
    ) -> Optional[float]:
        near_x = self._centroid_x(bev_mask, cls_id, near_rows)
        far_x = self._centroid_x(bev_mask, cls_id, far_rows)
        if near_x is None or far_x is None:
            return None
        return far_x - near_x

    def _estimate_curve_direction(
        self,
        bev_mask: np.ndarray,
        yellow_x: Optional[float],
        white_left_x: Optional[float],
        white_right_x: Optional[float],
    ) -> str:
        # Estimate whether the road ahead bends left or right by comparing line
        # centroids between near and far BEV rows. If the evidence is weak,
        # reuse the last non-straight direction so inner/outer mapping stays
        # stable through short straight-looking intervals.
        height = bev_mask.shape[0]
        near_rows = slice(int(height * 0.68), height)
        far_rows = slice(int(height * 0.30), int(height * 0.58))

        deltas = []
        for cls_id in (CLASS_YELLOW, CLASS_WHITE):
            dx = self._curve_anchor_dx(bev_mask, cls_id, near_rows, far_rows)
            if dx is not None:
                deltas.append(dx)

        if not deltas:
            return self.last_curve_direction

        mean_dx = float(np.mean(deltas))
        threshold_px = max(3.0, bev_mask.shape[1] * 0.03)
        if mean_dx > threshold_px:
            self.last_curve_direction = "right"
        elif mean_dx < -threshold_px:
            self.last_curve_direction = "left"
        elif self.last_curve_direction not in {"left", "right"}:
            self.last_curve_direction = "straight"

        return self.last_curve_direction

    def _role_side_map(self, curve_direction: str) -> dict:
        if curve_direction == "left":
            return {"inner": "left", "outer": "right"}
        if curve_direction == "right":
            return {"inner": "right", "outer": "left"}
        if self.last_curve_direction == "left":
            return {"inner": "left", "outer": "right"}
        if self.last_curve_direction == "right":
            return {"inner": "right", "outer": "left"}
        return {}

    def _project_target_by_role(
        self,
        lane_role: Optional[str],
        role_side_map: dict,
        yellow_x: Optional[float],
        white_left_x: Optional[float],
        white_right_x: Optional[float],
        lane_pos: float,
        lane_width_px: float,
    ) -> Optional[float]:
        if lane_role not in {"inner", "outer"}:
            return None

        side = role_side_map.get(lane_role)
        if side == "left":
            if lane_role == "inner" and yellow_x is not None:
                return yellow_x - (1.0 - lane_pos) * lane_width_px
            if white_left_x is not None:
                return white_left_x + lane_pos * lane_width_px
            if yellow_x is not None:
                return yellow_x - (1.0 - lane_pos) * lane_width_px
            return None

        if side == "right":
            if lane_role == "inner" and yellow_x is not None:
                return yellow_x + lane_pos * lane_width_px
            if white_right_x is not None:
                return white_right_x - (1.0 - lane_pos) * lane_width_px
            if yellow_x is not None:
                return yellow_x + lane_pos * lane_width_px
            return None

        return None

    def _estimate_curve_direction_from_road(self, mask: np.ndarray) -> str:
        # Use the full camera-perspective segmentation to estimate whether the
        # road bends left or right. This is more stable than using only the
        # BEV crop because the raw view contains more of the road ahead.
        height, width = mask.shape[:2]
        near_rows = slice(int(height * 0.70), int(height * 0.88))
        far_rows = slice(int(height * 0.35), int(height * 0.55))

        near_x = self._centroid_x(mask, CLASS_ROAD, near_rows)
        far_x = self._centroid_x(mask, CLASS_ROAD, far_rows)
        if near_x is None or far_x is None:
            return self.last_curve_direction

        dx = far_x - near_x
        threshold_px = max(6.0, width * 0.04)
        if dx > threshold_px:
            self.last_curve_direction = "right"
        elif dx < -threshold_px:
            self.last_curve_direction = "left"
        elif self.last_curve_direction not in {"left", "right"}:
            self.last_curve_direction = "straight"

        return self.last_curve_direction

    def _infer_role_from_camera_mask(
        self,
        camera_mask: np.ndarray,
        curve_direction: str,
    ) -> Tuple[Optional[str], Optional[float]]:
        # Infer whether the robot currently occupies the inner or outer
        # corridor using the full camera mask. We only need the yellow divider
        # near the robot and the current bend direction.
        height, width = camera_mask.shape[:2]
        rows = slice(int(height * 0.72), int(height * 0.92))
        robot_x = width * (0.5 + self.robot_center_bias_ratio)
        yellow_x = self._centroid_x(camera_mask, CLASS_YELLOW, rows)
        if yellow_x is None or curve_direction not in {"left", "right"}:
            return None, yellow_x

        if curve_direction == "left":
            return ("inner" if robot_x < yellow_x else "outer"), yellow_x
        return ("inner" if robot_x > yellow_x else "outer"), yellow_x

    def detect(self, bev_mask: np.ndarray, lane_pos: float = 0.50, camera_mask: Optional[np.ndarray] = None) -> dict:
        height, width = bev_mask.shape[:2]
        rows = slice(int(height * 0.45), height)
        robot_x = width * (0.5 + self.robot_center_bias_ratio)
        lane_offset_px = width * self.lane_offset_ratio

        yellow_x = self._centroid_x(bev_mask, CLASS_YELLOW, rows)
        if yellow_x is not None:
            self.last_yellow_x = yellow_x
        white_xs = np.where(bev_mask[rows, :] == CLASS_WHITE)[1]
        white_x = self._centroid_from_xs(white_xs)
        white_left_x, white_right_x = self._split_white_lanes(bev_mask, rows, yellow_x, robot_x)
        if camera_mask is not None:
            curve_direction = self._estimate_curve_direction_from_road(camera_mask)
            evidence_role, camera_yellow_x = self._infer_role_from_camera_mask(camera_mask, curve_direction)
        else:
            curve_direction = self._estimate_curve_direction(bev_mask, yellow_x, white_left_x, white_right_x)
            evidence_role = self._infer_role_from_yellow(robot_x, yellow_x, curve_direction)
            camera_yellow_x = None

        # ── Role system (ISOLATED — not used for lane following) ───────────────
        # Kept here for future overtaking use. Do not feed role into lane_side
        # selection below — it was causing inner→outer switching because
        # curve_direction flips on a circular track and inverts the role→side map.
        role_side_map = self._role_side_map(curve_direction)
        if evidence_role is None and yellow_x is not None:
            evidence_role = next((r for r, s in role_side_map.items() if s == self.active_side), None)
        lane_role = self._update_auto_role(evidence_role)
        # ──────────────────────────────────────────────────────────────────────

        # ── Side selection (lock-based, role-free) ────────────────────────────
        forced_side: Optional[str] = None
        if self.preferred_lane in {"left", "right"}:
            forced_side = self.preferred_lane

        left_center_x = self._lane_center("left", yellow_x, white_left_x, white_right_x, lane_pos, lane_offset_px)
        right_center_x = self._lane_center("right", yellow_x, white_left_x, white_right_x, lane_pos, lane_offset_px)
        dual_side = self._choose_side(robot_x, left_center_x, right_center_x, width, forced_side=forced_side)
        evidence_side = dual_side if dual_side is not None else self._infer_side_from_visible_lines(
            robot_x, yellow_x, white_left_x, white_right_x
        )
        # Only force a role-derived side when the user explicitly asks for
        # inner/outer. In auto mode, stay in the physically detected corridor.
        desired_side_from_role = (
            role_side_map.get(lane_role)
            if self.preferred_lane in {"inner", "outer"} and lane_role in {"inner", "outer"}
            else None
        )
        if desired_side_from_role in {"left", "right"}:
            lane_side = desired_side_from_role
            self.active_side = lane_side
        else:
            # Fall back to side auto-updates only when role evidence is not
            # strong enough to define the corridor.
            if not self._side_locked:
                lane_side = self._update_auto_side(evidence_side)
            else:
                lane_side = self.active_side
        # ──────────────────────────────────────────────────────────────────────

        actual_side = self._side_from_yellow(robot_x, yellow_x)
        left_lane_width  = (yellow_x - white_left_x)  if yellow_x is not None and white_left_x  is not None else None
        right_lane_width = (white_right_x - yellow_x) if yellow_x is not None and white_right_x is not None else None

        if lane_side == "left":
            self._update_lane_width(left_lane_width)
        elif lane_side == "right":
            self._update_lane_width(right_lane_width)

        # Primary CTE path: follow the currently selected corridor. Averaging
        # yellow with every white pixel can target the middle of the full road
        # when both white boundaries are visible, so use only the white boundary
        # that belongs to the locked side.
        lane_width_px = self.estimated_lane_width_px or (lane_offset_px * 2.0)
        selected_white_x = white_left_x if lane_side == "left" else white_right_x if lane_side == "right" else None
        if yellow_x is not None and selected_white_x is not None:
            measured_width = abs(selected_white_x - yellow_x)
            self._update_lane_width(measured_width)
            if lane_side == "left":
                target_x = selected_white_x + lane_pos * (yellow_x - selected_white_x)
            else:
                target_x = yellow_x + lane_pos * (selected_white_x - yellow_x)
            mode = "dual_lane"
        elif yellow_x is not None:
            side = lane_side or actual_side or "right"
            direction = -1.0 if side == "left" else 1.0
            target_x = yellow_x + direction * lane_pos * lane_width_px
            mode = "single_lane"
        elif selected_white_x is not None:
            side = lane_side or self.active_side or ("left" if white_x < robot_x else "right")
            direction = 1.0 if side == "left" else -1.0
            target_x = selected_white_x + direction * (1.0 - lane_pos) * lane_width_px
            mode = "single_lane"
        elif white_x is not None:
            # Last fallback: if split-white failed but white exists, still
            # project using the locked side. A hardcoded offset to one side can
            # pull the car into the opposite corridor at curve exits.
            side = lane_side or actual_side or self.active_side or ("left" if white_x < robot_x else "right")
            direction = 1.0 if side == "left" else -1.0
            target_x = white_x + direction * (1.0 - lane_pos) * lane_width_px
            mode = "single_lane"
        else:
            target_x = None
            mode = "lost"

        if target_x is None and self.last_target_x is not None and self.lost_frames < self.lost_frame_limit:
            # Short grace window to absorb flicker without immediately dropping
            # to a zero target.
            target_x = self.last_target_x
            mode = "grace_hold"

        if target_x is None:
            self.lost_frames += 1
            confidence = 0.0
        else:
            self.last_target_x = target_x
            self.lost_frames = 0
            if mode == "dual_lane":
                confidence = 1.0
            elif mode == "single_lane":
                confidence = 0.70
            else:
                confidence = 0.35

        cte = (robot_x - target_x) if target_x is not None else 0.0
        return {
            "center_x": target_x,
            "cte": cte,
            "yellow_x": yellow_x,
            "white_x": white_x,
            "white_left_x": white_left_x,
            "white_right_x": white_right_x,
            "left_center_x": left_center_x,
            "right_center_x": right_center_x,
            "lane_side": lane_side,
            "lane_role": lane_role,
            "evidence_role": evidence_role,
            "curve_direction": curve_direction,
            "lane_width_px": self.estimated_lane_width_px,
            "mode": mode,
            "lost_frames": self.lost_frames,
            "evidence_side": evidence_side,
            "actual_side": actual_side,
            "wrong_side": False,
            "camera_yellow_x": camera_yellow_x,
            "confidence": confidence,
        }


class SteeringController:
    def __init__(self, kp: float = 0.012, kd: float = 0.004, max_angular: float = 0.55, smooth_alpha: float = 0.6):
        self.kp = kp
        self.kd = kd
        self.max_angular = max_angular
        self.alpha = smooth_alpha
        self._prev_cte = 0.0
        self._prev_omega = 0.0

    def compute(self, cte: float, confidence: float) -> float:
        if confidence == 0.0:
            self._prev_cte = 0.0
            return self._prev_omega * 0.7

        d_cte = cte - self._prev_cte
        raw = (self.kp * cte + self.kd * d_cte) * confidence
        clipped = float(np.clip(raw, -self.max_angular, self.max_angular))
        omega = self.alpha * clipped + (1.0 - self.alpha) * self._prev_omega
        self._prev_cte = cte
        self._prev_omega = omega
        return omega


class AdaptiveCruiseController:
    FAR_ROWS = (0.00, 0.33)
    FAR_COLS = (0.25, 0.75)
    NEAR_ROWS = (0.33, 0.67)
    NEAR_COLS = (0.31, 0.69)

    FAR_SLOW_THRESH = 12
    NEAR_OBS_THRESH = 15

    SPEED_CRUISE = 0.15
    SPEED_SLOW = 0.09
    SPEED_LANE_CHANGE = 0.08
    SPEED_STOP = 0.00

    HISTORY = 5
    LC_STEER_DURATION = 2.0
    LC_HOLD_DURATION = 1.5
    LC_STEER_MAX_DURATION = 3.0
    LC_STEER_ANGULAR = 0.35
    LC_COOLDOWN = 1.5

    def __init__(self, target_speed: float = 0.15):
        # Keep the old workflow behavior: very small requested target speeds
        # still get promoted to a practical cruise value on the robot.
        self.target_speed = target_speed
        self._far_buf = deque(maxlen=self.HISTORY)
        self._near_buf = deque(maxlen=self.HISTORY)
        self.state = "CRUISE"
        self.commanded_v = self.target_speed
        self._lc_active = False
        self._lc_phase = None
        self._lc_phase_end = 0.0
        self._lc_direction = 1
        self._lc_cooldown_until = 0.0
        self.obstacle_bbox: Optional[Tuple[int, int, int, int]] = None

    @staticmethod
    def _zone_pixels(bev_mask: np.ndarray, row_bounds: Tuple[float, float], col_bounds: Tuple[float, float]) -> np.ndarray:
        height, width = bev_mask.shape[:2]
        r0 = int(height * row_bounds[0])
        r1 = int(height * row_bounds[1])
        c0 = int(width * col_bounds[0])
        c1 = int(width * col_bounds[1])
        return bev_mask[r0:r1, c0:c1]

    def _count(self, bev_mask: np.ndarray, row_bounds: Tuple[float, float], col_bounds: Tuple[float, float]) -> int:
        zone = self._zone_pixels(bev_mask, row_bounds, col_bounds)
        return int((zone == CLASS_VEHICLE).sum())

    @staticmethod
    def _vehicle_bbox(bev_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(bev_mask == CLASS_VEHICLE)
        if xs.size < 10:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _decide_lc_direction(bbox: Optional[Tuple[int, int, int, int]], width: int) -> int:
        if bbox is None:
            return 1
        x1, _, x2, _ = bbox
        return 1 if (x1 + x2) / 2.0 > width / 2.0 else -1

    @staticmethod
    def _detect_zone(bev_mask: np.ndarray) -> str:
        height, width = bev_mask.shape[:2]
        roi = bev_mask[int(height * 0.40):int(height * 0.70), :]
        vehicle = roi == CLASS_VEHICLE
        left = int(vehicle[:, :width // 3].sum())
        center = int(vehicle[:, width // 3:2 * width // 3].sum())
        right = int(vehicle[:, 2 * width // 3:].sum())
        if center > left and center > right:
            return "CENTER"
        if left > right:
            return "LEFT"
        return "RIGHT"

    @staticmethod
    def _zone_to_lc_direction(zone: str) -> int:
        # Positive angular override turns left in this controller.
        if zone == "LEFT":
            return -1
        return 1

    @staticmethod
    def _avoid_dir_to_lc_direction(avoid_dir: Optional[str]) -> Optional[int]:
        if avoid_dir == "LEFT":
            return 1
        if avoid_dir == "RIGHT":
            return -1
        return None

    def update(self, bev_mask: np.ndarray, preferred_avoid_dir=None):
        now = time.time()
        height, width = bev_mask.shape[:2]

        far_px  = self._count(bev_mask, self.FAR_ROWS,  self.FAR_COLS)
        near_px = self._count(bev_mask, self.NEAR_ROWS, self.NEAR_COLS)
        self._far_buf.append(far_px)
        self._near_buf.append(near_px)
        avg_far  = float(np.mean(self._far_buf))
        avg_near = float(np.mean(self._near_buf))

        self.obstacle_bbox = self._vehicle_bbox(bev_mask)
        obstacle_zone      = self._detect_zone(bev_mask)
        angular_override   = 0.0

        if self._lc_active:
            if self._lc_phase == "STEER":
                angular_override   = self._lc_direction * self.LC_STEER_ANGULAR
                self.commanded_v   = self.SPEED_LANE_CHANGE
                self.state         = "LANE_CHANGE"

                # ── Exit condition: obstacle cleared OR safety timeout ──
                near_clear   = avg_near < self.NEAR_OBS_THRESH * 0.4
                far_clear    = avg_far  < self.FAR_SLOW_THRESH * 0.5
                timed_out    = now >= self._lc_phase_end           # hard cap

                if (near_clear and far_clear) or timed_out:
                    self._lc_phase     = "HOLD"
                    self._lc_phase_end = now + self.LC_HOLD_DURATION

            elif self._lc_phase == "HOLD":
                angular_override = 0.0
                self.commanded_v = self.SPEED_LANE_CHANGE
                self.state = "LANE_CHANGE"

                # Exit HOLD only when lane lines are visible again OR timeout
                lane_visible = self._count(bev_mask, (0.45, 1.0), (0.0, 1.0)) > 0  # any white/yellow
                if (now >= self._lc_phase_end) or lane_visible:
                    self._lc_active = False
                    self._lc_phase = None
                    self._lc_cooldown_until = now + self.LC_COOLDOWN
                    self.state = "CRUISE"
                    self.commanded_v = self.target_speed
        else:
            obstacle_near = avg_near >= self.NEAR_OBS_THRESH
            cooldown_ok   = now >= self._lc_cooldown_until

            if obstacle_near and cooldown_ok:
                self._lc_active    = True
                self._lc_phase     = "STEER"
                self._lc_phase_end = now + self.LC_STEER_MAX_DURATION  # safety cap
                self._lc_direction = (
                    self._avoid_dir_to_lc_direction(preferred_avoid_dir)
                    or self._zone_to_lc_direction(obstacle_zone)
                )
                self.state       = "LANE_CHANGE"
                self.commanded_v = self.SPEED_LANE_CHANGE

            elif obstacle_near and not cooldown_ok:
                # Still in cooldown — hold a gentle steer rather than stopping
                angular_override = self._lc_direction * self.LC_STEER_ANGULAR * 0.4
                self.state, self.commanded_v = "AVOID_COOLDOWN", self.SPEED_LANE_CHANGE

            elif avg_far >= self.FAR_SLOW_THRESH:
                self.state, self.commanded_v = "SLOW", self.SPEED_SLOW
            else:
                self.state, self.commanded_v = "CRUISE", self.target_speed

        debug = {
            "far_px_avg":        round(avg_far, 1),
            "near_px_avg":       round(avg_near, 1),
            "state":             self.state,
            "commanded_v":       self.commanded_v,
            "lc_direction":      self._lc_direction,
            "avoid_direction":   "LEFT" if self._lc_direction > 0 else "RIGHT",
            "preferred_avoid_dir": preferred_avoid_dir,
            "lc_phase":          self._lc_phase,
            "bbox":              self.obstacle_bbox,
            "obstacle_zone":     obstacle_zone,
            "mask_size":         [width, height],
        }
        return self.commanded_v, angular_override, self.state, debug


class AutonomousDrivingNode(Node):
    def __init__(self):
        super().__init__("autonomous_driving")

        self.declare_parameter("bev_mask_topic", "/seg/bev_mask")
        self.declare_parameter("camera_mask_topic", "/seg/cam_mask_raw")
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("status_topic", "/cruise/status")
        self.declare_parameter("debug_topic", "/cruise/debug_bev")
        self.declare_parameter("target_speed", 0.05)
        self.declare_parameter("kp", 0.005)
        self.declare_parameter("kd", 0.002)
        self.declare_parameter("max_angular", 0.4)
        self.declare_parameter("lane_position", 0.50)
        self.declare_parameter("preferred_lane", "auto")
        self.declare_parameter("robot_center_bias_ratio", -0.04)
        self.declare_parameter("auto_switch_confirm_frames", 3)
        self.declare_parameter("lock_initial_lane", True)
        self.declare_parameter("debug_viz", True)
        self.declare_parameter("warmup_frames", 10)
        self.declare_parameter("single_lane_speed_scale", 0.9)
        self.declare_parameter("grace_speed_scale", 0.8)
        self.declare_parameter("controller_mode", "advanced")
        self.declare_parameter("simple_speed", 0.15)
        self.declare_parameter("simple_lost_speed", 0.05)
        self.declare_parameter("simple_avoid_speed", 0.08)
        self.declare_parameter("simple_avoid_omega", 0.35)
        self.declare_parameter("simple_avoid_frames", 20)
        self.declare_parameter("avoid_strategy", "lane")

        bev_mask_topic = self.get_parameter("bev_mask_topic").value
        camera_mask_topic = self.get_parameter("camera_mask_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        status_topic = self.get_parameter("status_topic").value
        debug_topic = self.get_parameter("debug_topic").value
        target_speed = float(self.get_parameter("target_speed").value)
        kp = float(self.get_parameter("kp").value)
        kd = float(self.get_parameter("kd").value)
        max_angular = float(self.get_parameter("max_angular").value)
        self.lane_pos = float(self.get_parameter("lane_position").value)
        preferred_lane = str(self.get_parameter("preferred_lane").value).lower()
        robot_center_bias_ratio = float(self.get_parameter("robot_center_bias_ratio").value)
        auto_switch_confirm_frames = int(self.get_parameter("auto_switch_confirm_frames").value)
        lock_initial_lane = bool(self.get_parameter("lock_initial_lane").value)
        self.debug = bool(self.get_parameter("debug_viz").value)
        self.warmup_frames = int(self.get_parameter("warmup_frames").value)
        self.single_lane_speed_scale = float(self.get_parameter("single_lane_speed_scale").value)
        self.grace_speed_scale = float(self.get_parameter("grace_speed_scale").value)
        self.controller_mode = str(self.get_parameter("controller_mode").value).lower()
        self.simple_speed = float(self.get_parameter("simple_speed").value)
        self.simple_lost_speed = float(self.get_parameter("simple_lost_speed").value)
        self.simple_avoid_speed = float(self.get_parameter("simple_avoid_speed").value)
        self.simple_avoid_omega = float(self.get_parameter("simple_avoid_omega").value)
        self.simple_avoid_frames = int(self.get_parameter("simple_avoid_frames").value)
        self.avoid_strategy = str(self.get_parameter("avoid_strategy").value).lower()

        self.bridge = CvBridge()
        self.lanes = LaneDetector(
            preferred_lane=preferred_lane,
            robot_center_bias_ratio=robot_center_bias_ratio,
            auto_switch_confirm_frames=auto_switch_confirm_frames,
            lock_initial_lane=lock_initial_lane,
        )
        self.steer = SteeringController(kp=kp, kd=kd, max_angular=max_angular)
        self.acc = AdaptiveCruiseController(target_speed=target_speed)

        self.sub = self.create_subscription(Image, bev_mask_topic, self._cb, 1)
        self.sub_camera = self.create_subscription(Image, camera_mask_topic, self._camera_cb, 1)
        self.pub_cmd = self.create_publisher(Twist, cmd_topic, 1)
        self.pub_status = self.create_publisher(String, status_topic, 1)
        self.pub_debug = self.create_publisher(Image, debug_topic, 1) if self.debug else None

        self._frame = 0
        self._t0 = time.time()
        self._bev_frames_received = 0
        self._camera_frames_received = 0
        self._bev_mask_topic = bev_mask_topic
        self._camera_mask_topic = camera_mask_topic
        self._latest_camera_mask: Optional[np.ndarray] = None
        self._simple_prev_cte = 0.0
        self._simple_avoid_timer = 0
        self._simple_avoid_dir = "RIGHT"
        self.create_timer(2.0, self._subscription_heartbeat)

        self.get_logger().info(f"Subscribed to BEV mask : {bev_mask_topic}")
        self.get_logger().info(f"Subscribed to cam mask : {camera_mask_topic}")
        self.get_logger().info(f"Velocity topic         : {cmd_topic}")
        self.get_logger().info(f"Status topic           : {status_topic}")
        self.get_logger().info(f"Preferred lane         : {preferred_lane}")
        self.get_logger().info(f"Robot center bias      : {robot_center_bias_ratio}")
        self.get_logger().info(f"Auto switch frames     : {auto_switch_confirm_frames}")
        self.get_logger().info(f"Lock initial lane      : {lock_initial_lane}")
        self.get_logger().info(f"Controller mode        : {self.controller_mode}")
        self.get_logger().info(f"Avoid strategy         : {self.avoid_strategy}")
        if self.debug:
            self.get_logger().info(f"Debug topic            : {debug_topic}")
        self.get_logger().info("autonomous_driving ready - waiting for BEV masks")

    def _cb(self, msg: Image):
        try:
            bev_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge error: {exc}")
            return

        self._bev_frames_received += 1
        if self._bev_frames_received == 1:
            self.get_logger().info(
                f"First BEV mask received from {self._bev_mask_topic} with shape={bev_mask.shape}"
            )

        self._run(bev_mask, msg.header)

    def _camera_cb(self, msg: Image):
        try:
            camera_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge camera mask error: {exc}")
            return

        self._camera_frames_received += 1
        self._latest_camera_mask = camera_mask

    def _subscription_heartbeat(self):
        if self._bev_frames_received == 0:
            self.get_logger().warn(
                f"Still waiting for BEV masks on {self._bev_mask_topic}. "
                "Check that seg_bev_node is running and publishing /seg/bev_mask."
            )
        if self._camera_frames_received == 0:
            self.get_logger().warn(
                f"Still waiting for camera masks on {self._camera_mask_topic}. "
                "Check that limo_segmentation_node is publishing /seg/cam_mask_raw."
            )

    def _preferred_avoid_direction(self, lane_info: dict) -> Optional[str]:
        if self.avoid_strategy == "zone":
            return None
        if self.avoid_strategy in {"left", "right"}:
            return self.avoid_strategy.upper()

        # Lane-based avoidance: if we are tracking the right corridor, escape
        # left; if we are tracking the left corridor, escape right. This avoids
        # relying on BEV obstacle-zone orientation when the homography is noisy.
        lane_side = lane_info.get("lane_side") or lane_info.get("actual_side") or lane_info.get("evidence_side")
        if lane_side == "right":
            return "LEFT"
        if lane_side == "left":
            return "RIGHT"
        return None

    def _run(self, bev_mask: np.ndarray, header=None):
        if self.controller_mode == "stuti":
            self._run_stuti(bev_mask, header)
            return

        lane_info = self.lanes.detect(bev_mask, lane_pos=self.lane_pos, camera_mask=self._latest_camera_mask)
        steer_omega = self.steer.compute(lane_info["cte"], lane_info["confidence"])
        preferred_avoid_dir = self._preferred_avoid_direction(lane_info)
        linear_v, lc_omega, _, acc_dbg = self.acc.update(bev_mask, preferred_avoid_dir=preferred_avoid_dir)
        avoidance_active = acc_dbg.get("lc_phase") is not None or acc_dbg.get("state") == "AVOID_COOLDOWN"
        self.lanes._lane_lock_override = avoidance_active
        if not avoidance_active and self.lanes._lane_lock_override is False:
            actual = lane_info.get("actual_side")
            if actual in {"left", "right"}:
                self.lanes.active_side = actual
                self.lanes._side_locked = True
        final_omega = lc_omega if acc_dbg["lc_phase"] is not None else steer_omega + lc_omega
        omega_limit = max(self.steer.max_angular, abs(lc_omega), self.acc.LC_STEER_ANGULAR)
        final_omega = float(np.clip(final_omega, -omega_limit, omega_limit))

        cmd = Twist()
        if self._frame < self.warmup_frames:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Safety rule:
            # - dual lane: full speed
            # - single lane / grace hold: slow down
            # - lane fully lost for several frames: stop instead of guessing
            if lane_info["mode"] == "lost" and lane_info["lost_frames"] >= self.lanes.lost_frame_limit:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                if lane_info["mode"] == "dual_lane":
                    speed_scale = 1.0
                elif lane_info["mode"] == "single_lane":
                    speed_scale = self.single_lane_speed_scale
                else:
                    speed_scale = self.grace_speed_scale
                cmd.linear.x = float(linear_v) * speed_scale
                cmd.angular.z = final_omega
        self.pub_cmd.publish(cmd)

        now = time.time()
        fps = 1.0 / max(now - self._t0, 1e-6)
        status = {
            "frame": self._frame,
            "fps": round(fps, 1),
            "cte": round(float(lane_info["cte"]), 2),
            "confidence": lane_info["confidence"],
            "lane_side": lane_info["lane_side"],
            "lane_role": lane_info["lane_role"],
            "evidence_side": lane_info["evidence_side"],
            "evidence_role": lane_info["evidence_role"],
            "curve_direction": lane_info["curve_direction"],
            "mode": lane_info["mode"],
            "lost_frames": lane_info["lost_frames"],
            "omega": round(final_omega, 4),
            "linear_v": round(float(cmd.linear.x), 3),
            "acc": acc_dbg,
        }
        self.pub_status.publish(String(data=json.dumps(status)))

        if self.pub_debug is not None:
            debug_image = self._build_debug_view(bev_mask, lane_info, acc_dbg, fps, cmd)
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
                if header is not None:
                    debug_msg.header = header
                self.pub_debug.publish(debug_msg)
            except Exception as exc:
                self.get_logger().warn(f"debug publish error: {exc}")

        self._frame += 1
        self._t0 = now

    def _simple_lane_cte(self, bev_mask: np.ndarray) -> Optional[float]:
        height, width = bev_mask.shape[:2]
        rows = slice(int(height * 0.50), height)
        yellow = np.where(bev_mask[rows, :] == CLASS_YELLOW)[1]
        white = np.where(bev_mask[rows, :] == CLASS_WHITE)[1]
        robot_x = width * (0.5 + self.lanes.robot_center_bias_ratio)

        if yellow.size > 30 and white.size > 30:
            center = (float(yellow.mean()) + float(white.mean())) / 2.0
        elif yellow.size > 30:
            center = float(yellow.mean()) + 35.0
        elif white.size > 30:
            center = float(white.mean()) - 35.0
        else:
            return None

        return float(robot_x - center)

    def _simple_obstacle_exists(self, bev_mask: np.ndarray) -> bool:
        height, width = bev_mask.shape[:2]
        roi = bev_mask[int(height * 0.40):int(height * 0.70), int(width * 0.35):int(width * 0.65)]
        return int((roi == CLASS_VEHICLE).sum()) > 120

    def _simple_obstacle_zone(self, bev_mask: np.ndarray) -> str:
        height, width = bev_mask.shape[:2]
        roi = bev_mask[int(height * 0.40):int(height * 0.70), :]
        vehicle = roi == CLASS_VEHICLE
        left = int(vehicle[:, :width // 3].sum())
        center = int(vehicle[:, width // 3:2 * width // 3].sum())
        right = int(vehicle[:, 2 * width // 3:].sum())
        if center > left and center > right:
            return "CENTER"
        if left > right:
            return "LEFT"
        return "RIGHT"

    def _simple_omega(self, cte: float) -> float:
        d_cte = cte - self._simple_prev_cte
        self._simple_prev_cte = cte
        omega = 0.02 * cte + 0.006 * d_cte
        return float(np.clip(omega, -0.6, 0.6))

    def _run_stuti(self, bev_mask: np.ndarray, header=None):
        cmd = Twist()
        obstacle = self._simple_obstacle_exists(bev_mask)
        zone = self._simple_obstacle_zone(bev_mask)

        if obstacle and self._simple_avoid_timer <= 0:
            self._simple_avoid_timer = max(1, self.simple_avoid_frames)
            if zone == "LEFT":
                self._simple_avoid_dir = "RIGHT"
            elif zone == "RIGHT":
                self._simple_avoid_dir = "LEFT"
            else:
                self._simple_avoid_dir = "RIGHT"

        mode = "FOLLOW"
        cte = self._simple_lane_cte(bev_mask)

        if self._frame < self.warmup_frames:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            mode = "WARMUP"
        elif self._simple_avoid_timer > 0:
            # Frame-locked avoid mode mirrors Stuti's working script: one
            # decision is held long enough for the robot to physically rotate.
            cmd.linear.x = self.simple_avoid_speed
            cmd.angular.z = self.simple_avoid_omega if self._simple_avoid_dir == "LEFT" else -self.simple_avoid_omega
            self._simple_avoid_timer -= 1
            mode = "AVOID"
        elif cte is None:
            cmd.linear.x = self.simple_lost_speed
            cmd.angular.z = 0.0
            mode = "LOST"
        else:
            cmd.linear.x = self.simple_speed
            cmd.angular.z = self._simple_omega(cte)

        self.pub_cmd.publish(cmd)

        now = time.time()
        fps = 1.0 / max(now - self._t0, 1e-6)
        status = {
            "frame": self._frame,
            "fps": round(fps, 1),
            "controller_mode": "stuti",
            "mode": mode,
            "cte": None if cte is None else round(float(cte), 2),
            "obstacle": obstacle,
            "obstacle_zone": zone,
            "avoid_dir": self._simple_avoid_dir,
            "avoid_timer": self._simple_avoid_timer,
            "omega": round(float(cmd.angular.z), 4),
            "linear_v": round(float(cmd.linear.x), 3),
        }
        self.pub_status.publish(String(data=json.dumps(status)))

        if self.pub_debug is not None:
            debug = colorise_mask(bev_mask)
            cv2.putText(debug, f"Mode: {mode}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, f"CTE: {0.0 if cte is None else cte:.2f}", (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, f"Obs: {obstacle} {zone} dir={self._simple_avoid_dir}", (5, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, f"v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}", (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
                if header is not None:
                    debug_msg.header = header
                self.pub_debug.publish(debug_msg)
            except Exception as exc:
                self.get_logger().warn(f"debug publish error: {exc}")

        self._frame += 1
        self._t0 = now

    def _build_debug_view(self, bev_mask: np.ndarray, lane_info: dict, acc_dbg: dict, fps: float, cmd: Twist) -> np.ndarray:
        debug = colorise_mask(bev_mask)
        height, width = bev_mask.shape[:2]
        robot_x = int(width * (0.5 + self.lanes.robot_center_bias_ratio))

        cv2.line(debug, (robot_x, 0), (robot_x, height - 1), (255, 0, 255), 1)

        # Visualize the lane references used by the controller so you can see
        # whether it is targeting the intended corridor.
        if lane_info["yellow_x"] is not None:
            cv2.line(debug, (int(lane_info["yellow_x"]), 0), (int(lane_info["yellow_x"]), height - 1), (0, 255, 255), 1)
        if lane_info["white_left_x"] is not None:
            cv2.line(debug, (int(lane_info["white_left_x"]), 0), (int(lane_info["white_left_x"]), height - 1), (255, 255, 255), 1)
        if lane_info["white_right_x"] is not None:
            cv2.line(debug, (int(lane_info["white_right_x"]), 0), (int(lane_info["white_right_x"]), height - 1), (255, 255, 255), 1)
        if lane_info["left_center_x"] is not None:
            cv2.line(debug, (int(lane_info["left_center_x"]), 0), (int(lane_info["left_center_x"]), height - 1), (255, 100, 100), 1)
        if lane_info["right_center_x"] is not None:
            cv2.line(debug, (int(lane_info["right_center_x"]), 0), (int(lane_info["right_center_x"]), height - 1), (100, 100, 255), 1)
        if lane_info["center_x"] is not None:
            cx = int(lane_info["center_x"])
            cv2.line(debug, (cx, 0), (cx, height - 1), (0, 0, 255), 2)

        bbox = acc_dbg.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)

        info_lines = [
            f"FPS: {fps:.1f}",
            f"State: {acc_dbg['state']}",
            f"v={cmd.linear.x:.3f} w={cmd.angular.z:.3f}",
            f"Curve: {lane_info['curve_direction']}",
            f"Lane: {lane_info['lane_side']}/{lane_info['lane_role']}",
            f"Ev  : {lane_info['evidence_side']}/{lane_info['evidence_role']} mode={lane_info['mode']}",
            f"CTE={lane_info['cte']:.2f} conf={lane_info['confidence']:.2f}",
            f"Lost={lane_info['lost_frames']} width={0.0 if lane_info['lane_width_px'] is None else lane_info['lane_width_px']:.1f}",
        ]
        y = 16
        for line in info_lines:
            cv2.putText(debug, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 16

        return debug


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousDrivingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
