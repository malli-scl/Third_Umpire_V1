#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire – Heatmap Analytics (Final Version with Movement Trails + Reports)
-------------------------------------------------------------------------------
FEATURES:
• YOLOv8n continuous detection (frame-skipped, device-aware)
• Movement Trails (walking people → smooth trails)
• Dwell-Time Boost (standing people → brighter heat)
• Permanent daily heat accumulation with gentle decay
• Initial snapshots after startup (per camera)
• Hourly snapshots (per camera + combined 2×N grid)
• Final-day heatmap snapshots with ranking numbers (1–5) per camera
• Combined daily JSON report with per-camera heat-zone counts
• Camera selection + preview dashboard
"""

import os
import cv2
import time
import json
import torch
import platform
import warnings
import sys
import numpy as np
from datetime import datetime, timedelta

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QScrollArea,
    QGridLayout,
    QCheckBox,
    QHBoxLayout,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap, QImage

from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


# ==========================================================
#   URL Normalization Helper
# ==========================================================
def normalize_rtsp_url(url: str) -> str:
    """
    Normalize RTSP URL to fix common formatting issues.
    NOTE: For CP Plus and Hikvision, leading zeros in channel numbers (0101, 0201) are REQUIRED.
    This function only normalizes Dahua format URLs, not Hikvision/CP Plus format.
    """
    if not url or not isinstance(url, str):
        return url

    if "/Streaming/Channels/" in url:
        return url

    return url


# ==========================================================
#   CUSTOM HEATMAP COLOR MAP (Yellow → Orange → Red)
#   Yellow = Low activity, Orange = Medium, Red = High
# ==========================================================
def apply_custom_heatmap(heat: np.ndarray) -> np.ndarray:
    """
    Custom heatmap with:
      - Blue-ish background
      - Yellow = low activity
      - Red = high activity
    """
    heat = heat.astype(np.float32)
    heat_nonzero = heat[heat > 0]

    # Base background color (BGR)
    blue_bg = np.array([255, 180, 80], dtype=np.float32)

    if len(heat_nonzero) == 0:
        blue_bg_uint8 = blue_bg.astype(np.uint8)
        return np.full((*heat.shape, 3), blue_bg_uint8, dtype=np.uint8)

    hmin = float(heat_nonzero.min())
    hmax = float(heat_nonzero.max())

    if hmax > hmin:
        norm = np.where(heat > 0, (heat - hmin) / max(hmax - hmin, 1e-6), 0.0)
    else:
        norm = np.where(heat > 0, 0.5, 0.0)

    norm = np.clip(norm, 0.0, 1.0)

    yellow = np.array([0, 255, 255], dtype=np.float32)
    medium_red = np.array([0, 50, 255], dtype=np.float32)

    heat_bgr = np.full((heat.shape[0], heat.shape[1], 3), blue_bg, dtype=np.float32)

    # 0 → 0.5 : blue → yellow
    mask1 = (norm >= 0.0) & (norm < 0.5) & (heat > 0)
    if mask1.any():
        t1 = (norm[mask1] / 0.5).clip(0.0, 1.0)
        t1_3d = t1[:, None]
        heat_bgr[mask1] = blue_bg * (1 - t1_3d) + yellow * t1_3d

    # 0.5 → 1.0 : yellow → red
    mask2 = (norm >= 0.5) & (norm <= 1.0) & (heat > 0)
    if mask2.any():
        t2 = ((norm[mask2] - 0.5) / 0.5).clip(0.0, 1.0)
        t2_3d = t2[:, None]
        heat_bgr[mask2] = yellow * (1 - t2_3d) + medium_red * t2_3d

    # Exact zero / one safety
    mask_zero = (norm == 0.0) & (heat > 0)
    if mask_zero.any():
        heat_bgr[mask_zero] = yellow

    mask_one = (norm == 1.0) & (heat > 0)
    if mask_one.any():
        heat_bgr[mask_one] = medium_red

    heat_bgr = np.clip(heat_bgr, 0.0, 255.0)
    return heat_bgr.astype(np.uint8)


# ==========================================================
#   CENTRALIZED CAMERA CONFIGURATION
# ==========================================================
def configure_camera(cap: cv2.VideoCapture) -> cv2.VideoCapture:
    """
    Centralized function to configure camera properties.
    Sets FPS, focus, compression, and exposure for optimal performance.
    """
    if cap is None or not cap.isOpened():
        return cap

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 50)
    except Exception:
        pass

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_QUALITY, 100)
    except Exception:
        pass

    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    except Exception:
        pass

    return cap


# ==========================================================
#   CAMERA PREVIEW WORKER
# ==========================================================
class CameraPreviewWorker(QThread):
    frame_ready = pyqtSignal(QPixmap, int)

    def __init__(self, idx: int, url: str):
        super().__init__()
        self.idx = idx
        self.url = normalize_rtsp_url(url)
        self.running = True

    def run(self):
        # Ensure FFmpeg RTSP logs stay fully quiet for previews
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;5000000|"
            "max_delay;500000|"
            "buffersize;10000000|"
            "loglevel;0"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap = configure_camera(cap)

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.2)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                300, 200, Qt.AspectRatioMode.KeepAspectRatio
            )
            self.frame_ready.emit(pix, self.idx)
            time.sleep(0.2)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(300)


# ==========================================================
#   HEATMAP PROCESSOR — TRAILS + DWELL + ACCUMULATION
# ==========================================================
class HeatmapProcessor(QThread):
    """
    Continuous heatmap accumulation with movement trails and daily reports.

    Emits:
      • heatmap_ready(overlay_frame: np.ndarray, cam_index: int)
    """

    heatmap_ready = pyqtSignal(object, int)

    def __init__(self, urls, cam_names):
        super().__init__()
        self.urls = [normalize_rtsp_url(url) for url in urls]
        self.cam_names = cam_names
        self.running = True

        # YOLO setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt")
        try:
            self.model.fuse()
        except Exception:
            pass
        self.model.to(self.device)

        self.SCALE = 1.0  # full-res heatmap
        self.BLUR_INTERVAL = 10
        self.POS_CLEAR_INTERVAL = 200
        self.loop_counter = 0

        # Adaptive config for low‑power vs high‑power devices
        if self._detect_low_power_device():
            self.img_size = 320
            self.frame_skip = 6
            self.heat_blur_size = 11
            self.loop_sleep = 0.10
            self.DECAY = 0.98
            torch.set_num_threads(2)
            print("[HEATMAP] Low-power device detected → optimized settings applied.")
        else:
            self.img_size = 640
            self.frame_skip = 3
            self.heat_blur_size = 15
            self.loop_sleep = 0.05
            self.DECAY = 0.99
            print("[HEATMAP] High-performance device → standard settings applied.")

        self.frame_counter = {i: 0 for i in range(len(urls))}
        print(f"[HEATMAP] YOLOv8 loaded on {self.device}")

        now = datetime.now()
        self.active_day = now.strftime("%Y-%m-%d")
        current_hour_start = now.replace(minute=0, second=0, microsecond=0)
        self.next_snapshot_time = current_hour_start + timedelta(hours=1)

        today_final = now.replace(hour=22, minute=59, second=50, microsecond=0)
        if now < today_final:
            self.daily_final_time = today_final
        else:
            self.daily_final_time = today_final + timedelta(days=1)

        self.initial_snapshots_done = False
        self.daily_final_saved = False

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.record_dir = os.path.join(base_dir, "..", "..", "recordings", "heatmap")
        os.makedirs(self.record_dir, exist_ok=True)

        # Heat storage
        self.daily_heat = {i: None for i in range(len(urls))}
        self.highest_heatmap = {i: None for i in range(len(urls))}
        self.highest_accumulated_heat = {i: 0.0 for i in range(len(urls))}

        # Tracking / diagnostics
        self.dwell_map = {i: {} for i in range(len(urls))}
        self.prev_positions = {i: {} for i in range(len(urls))}
        self.person_id_counter = {i: 0 for i in range(len(urls))}
        self.frames_processed = {i: 0 for i in range(len(urls))}
        self.consecutive_failures = {i: 0 for i in range(len(urls))}
        self.frame_drop_count = {i: 0 for i in range(len(urls))}
        self.reconnection_logged = {i: False for i in range(len(urls))}
        self.frame_drop_warned = {i: False for i in range(len(urls))}

        # Final-day JSON aggregation buffer
        self.final_day_json_data = []

    # ------------------------------------------------------
    #  Helper: device type
    # ------------------------------------------------------
    def _detect_low_power_device(self) -> bool:
        """Detect if running on low-power hardware (ARM, small CPU, etc.)."""
        if self.device == "cuda" or torch.cuda.is_available():
            return False

        machine = platform.machine().lower()
        if machine.startswith("arm") or machine.startswith("aarch64"):
            return True

        platform_name = platform.platform().lower()
        low_power_indicators = [
            "raspberry",
            "orange pi",
            "banana pi",
            "odroid",
            "jetson",
            "rockchip",
            "allwinner",
            "broadcom",
        ]
        if any(ind in platform_name for ind in low_power_indicators):
            return True

        try:
            cpu_count = os.cpu_count() or 0
            if cpu_count <= 4:
                return True
        except Exception:
            pass

        return False

    # ------------------------------------------------------
    #  Helper: YOLO predict
    # ------------------------------------------------------
    def _safe_predict(self, frame_resized: np.ndarray):
        """Simple YOLO prediction without blocking."""
        try:
            results = self.model.predict(
                frame_resized,
                conf=0.20,
                classes=[0],
                verbose=False,
                device=self.device,
                imgsz=self.img_size,
            )
            boxes = (
                results[0].boxes.xyxy.cpu().numpy()
                if len(results[0].boxes)
                else np.empty((0, 4), dtype=np.float32)
            )
            return boxes
        except Exception:
            return np.empty((0, 4), dtype=np.float32)

    # ------------------------------------------------------
    #  Helper: safe camera read
    # ------------------------------------------------------
    def _safe_camera_read(self, cap: cv2.VideoCapture, timeout: float = 2.0):
        """Safe camera read with basic timeout."""
        try:
            start_time = time.time()
            ok, frame = cap.read()
            read_time = time.time() - start_time
            # If a read takes too long, silently treat it as a failure (no noisy logs)
            if read_time > timeout:
                return False, None
            if not ok or frame is None:
                return False, None
            return True, frame
        except Exception as e:
            # Swallow low-level read errors to avoid spamming logs; caller will handle reconnection.
            return False, None

    # ------------------------------------------------------
    #  Overlay builder for live view
    # ------------------------------------------------------
    def create_heatmap_overlay(
        self, cam_idx: int, frame: np.ndarray, heat: np.ndarray
    ) -> np.ndarray | None:
        """Create heatmap overlay on frame."""
        if frame is None or heat is None:
            return None

        # Resize heat to frame size
        if heat.shape[:2] != frame.shape[:2]:
            heat = cv2.resize(
                heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # Blur for smooth zones
        blur_size = getattr(self, "heat_blur_size", 21)
        if heat.max() > 0:
            heat = cv2.GaussianBlur(heat, (blur_size, blur_size), 0)

        colored = apply_custom_heatmap(heat)

        # Apply blue tint to original frame
        blue_bg_color = np.array([255, 180, 80], dtype=np.float32)
        if frame.dtype != np.uint8:
            base = np.clip(frame, 0, 255).astype(np.float32)
        else:
            base = frame.astype(np.float32)

        blue_tint = np.full_like(base, blue_bg_color, dtype=np.float32)
        base = base * 0.7 + blue_tint * 0.3

        # Blend heatmap overlay
        colored_f = colored.astype(np.float32)
        blended = base * 0.35 + colored_f * 0.65
        overlay = np.clip(blended, 0, 255).astype(np.uint8)
        return overlay

    # ------------------------------------------------------
    #  MAIN LOOP — continuous motion heat accumulation
    # ------------------------------------------------------
    def run(self):
        # 1) Connect all cameras
        caps: list[cv2.VideoCapture | None] = []
        for i, url in enumerate(self.urls):
            cap = None
            connected = False

            for retry in range(3):
                try:
                    # Silence ffmpeg noise during open
                    old_stderr = sys.stderr
                    try:
                        sys.stderr = open(os.devnull, "w")
                        # Keep FFmpeg completely quiet while opening RTSP streams
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                            "rtsp_transport;tcp|"
                            "stimeout;5000000|"
                            "max_delay;500000|"
                            "buffersize;10000000|"
                            "loglevel;0"
                        )
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    finally:
                        try:
                            sys.stderr.close()
                        except Exception:
                            pass
                        sys.stderr = old_stderr

                    cap = configure_camera(cap)
                    time.sleep(2.0)

                    if cap.isOpened():
                        ok = False
                        for _ in range(5):
                            ok, test_frame = cap.read()
                            if ok and test_frame is not None and test_frame.size > 0:
                                ok = True
                                break
                            time.sleep(0.5)

                        if ok:
                            caps.append(cap)
                            connected = True
                            print(
                                f"[HEATMAP] ✅ Camera {i+1} ({self.cam_names[i]}) connected"
                            )
                            break
                        else:
                            print(
                                f"[HEATMAP] ⚠️ Camera {i+1} opened but failed to read frame"
                            )

                    if cap:
                        cap.release()
                        cap = None
                except Exception:
                    if cap:
                        cap.release()
                        cap = None
                    if retry < 2:
                        time.sleep(1.0)

            if not connected:
                print(
                    f"[HEATMAP] ⚠️ Camera {i+1} ({self.cam_names[i]}) failed to connect after 3 retries"
                )
                caps.append(None)

        active_cam_count = sum(1 for c in caps if c is not None)
        print(
            f"[HEATMAP] Running on {active_cam_count}/{len(self.urls)} active cameras…"
        )

        last_frames: dict[int, np.ndarray] = {}

        # 2) Main loop
        while self.running:
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")

            # --- Daily final snapshot ---
            if not self.daily_final_saved and now >= self.daily_final_time:
                print("[HEATMAP] 🎯 Saving daily final snapshots…")
                fresh_frames_for_daily: dict[int, np.ndarray] = {}
                for i, cap in enumerate(caps):
                    if cap is None or not cap.isOpened():
                        continue
                    try:
                        fresh_frame = None
                        for attempt in range(5):
                            ok, frame_read = cap.read()
                            if (
                                ok
                                and frame_read is not None
                                and len(frame_read.shape) == 3
                                and frame_read.shape[0] > 0
                                and frame_read.shape[1] > 0
                            ):
                                fresh_frame = frame_read
                                if attempt < 3:
                                    # warm up a little
                                    continue
                                break
                            time.sleep(0.2)

                        if fresh_frame is not None:
                            fresh_frames_for_daily[i] = fresh_frame
                        elif i in last_frames:
                            fresh_frames_for_daily[i] = last_frames[i]
                    except Exception as e:
                        print(
                            f"[HEATMAP] ⚠️ Failed to read fresh frame for final snapshot (cam {i+1}): {e}"
                        )
                        if i in last_frames:
                            fresh_frames_for_daily[i] = last_frames[i]

                self.save_final_daily_heatmap(fresh_frames_for_daily)
                self.daily_final_saved = True
                print(
                    f"[HEATMAP] ✅ Daily final snapshots saved at {now.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # --- New day reset ---
            if today_str != self.active_day:
                print(
                    f"[HEATMAP] 🌅 New day detected: {today_str} - Resetting heatmaps…"
                )
                self.active_day = today_str
                self.initial_snapshots_done = False
                self.daily_final_saved = False

                today_final = now.replace(
                    hour=22, minute=59, second=50, microsecond=0
                )
                self.daily_final_time = today_final

                self.daily_heat = {i: None for i in range(len(self.urls))}
                self.highest_heatmap = {i: None for i in range(len(self.urls))}
                self.highest_accumulated_heat = {i: 0.0 for i in range(len(self.urls))}
                self.dwell_map = {i: {} for i in range(len(self.urls))}
                self.prev_positions = {i: {} for i in range(len(self.urls))}
                self.person_id_counter = {i: 0 for i in range(len(self.urls))}
                self.frames_processed = {i: 0 for i in range(len(self.urls))}

            # --- Per-camera processing ---
            for i, cap in enumerate(caps):
                if cap is None or not cap.isOpened():
                    continue

                # Read frame safely
                ok, frame = self._safe_camera_read(cap, timeout=2.0)
                if not ok or frame is None:
                    self.consecutive_failures[i] = self.consecutive_failures.get(i, 0) + 1

                    # Attempt reconnection after many failures
                    if self.consecutive_failures[i] >= 10 and not self.reconnection_logged.get(
                        i, False
                    ):
                        self.reconnection_logged[i] = True
                        try:
                            if caps[i] is not None:
                                caps[i].release()
                            time.sleep(2.0)
                            new_cap = cv2.VideoCapture(self.urls[i], cv2.CAP_FFMPEG)
                            new_cap = configure_camera(new_cap)
                            time.sleep(1.0)
                            if new_cap.isOpened():
                                caps[i] = new_cap
                                self.consecutive_failures[i] = 0
                                self.reconnection_logged[i] = False
                                print(f"[HEATMAP] ✅ Camera {i+1} reconnected")
                        except Exception:
                            pass
                    time.sleep(0.1)
                    continue

                if self.consecutive_failures.get(i, 0) > 0:
                    self.consecutive_failures[i] = 0
                    self.reconnection_logged[i] = False

                last_frames[i] = frame.copy()
                self.frame_counter[i] += 1

                # Frame skipping for performance
                if self.frame_counter[i] % self.frame_skip != 0:
                    self.frame_drop_count[i] = self.frame_drop_count.get(i, 0) + 1
                    continue

                if self.frame_counter[i] % 500 == 0:
                    dropped = self.frame_drop_count.get(i, 0)
                    drop_rate = (
                        dropped / self.frame_counter[i] * 100.0
                        if self.frame_counter[i] > 0
                        else 0.0
                    )
                    if drop_rate > 50 and not self.frame_drop_warned.get(i, False):
                        print(
                            f"[HEATMAP] ⚠️ Camera {i+1} - High frame drop rate: {drop_rate:.1f}%"
                        )
                        self.frame_drop_warned[i] = True

                # Resize for YOLO
                if self.img_size != 640:
                    frame_resized = cv2.resize(frame, (self.img_size, self.img_size))
                else:
                    frame_resized = frame

                boxes = self._safe_predict(frame_resized)

                # Map boxes back to original resolution
                if boxes.size and self.img_size != 640:
                    h_orig, w_orig = frame.shape[:2]
                    scale_x = w_orig / self.img_size
                    scale_y = h_orig / self.img_size
                    boxes = boxes * np.array(
                        [scale_x, scale_y, scale_x, scale_y], dtype=np.float32
                    )

                h, w = frame.shape[:2]

                if self.daily_heat[i] is None:
                    self.daily_heat[i] = np.zeros((h, w), dtype=np.float32)

                heat = self.daily_heat[i]
                # Gentle decay keeps long-term accumulation but slowly fades
                heat *= self.DECAY

                current_heat = np.zeros((h, w), dtype=np.float32)
                current_positions: dict[int, tuple[int, int, int]] = {}

                # Reset tracking dict occasionally to prevent unbounded growth
                if self.frame_counter[i] % self.POS_CLEAR_INTERVAL == 0:
                    self.prev_positions[i] = {}

                for (x1, y1, x2, y2) in boxes:
                    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    box_width = int(x2 - x1)
                    box_height = int(y2 - y1)

                    if box_width <= 0 or box_height <= 0:
                        continue

                    if cx < 0 or cy < 0 or cx >= w or cy >= h:
                        continue

                    # Clamp bounding box to valid range
                    x1_clamped = max(0, int(x1))
                    y1_clamped = max(0, int(y1))
                    x2_clamped = min(w, int(x2))
                    y2_clamped = min(h, int(y2))
                    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
                        continue

                    ground_y = min(int(y2_clamped), h - 1)

                    # Track identity by nearest previous position
                    matched_pid = None
                    min_dist = float("inf")
                    for pid, (px, py, _) in self.prev_positions[i].items():
                        dist = np.hypot(cx - px, cy - py)
                        if dist < min_dist and dist < 100:
                            min_dist = dist
                            matched_pid = pid

                    if matched_pid is not None:
                        person_id = matched_pid
                    else:
                        person_id = self.person_id_counter[i]
                        self.person_id_counter[i] += 1

                    move_dist = 0.0
                    if matched_pid is not None:
                        prev_x, prev_y, _ = self.prev_positions[i][matched_pid]
                        move_dist = np.hypot(cx - prev_x, cy - prev_y)

                    # Base heat radius constrained by box size
                    base_heat = 200.0
                    max_radius_x = int(box_width * 0.5)
                    max_radius_y = int(box_height * 0.5)
                    heat_radius = max(15, int(box_width * 0.4))
                    heat_radius = min(heat_radius, max_radius_x, max_radius_y)

                    # Ellipse for body
                    cv2.ellipse(
                        current_heat,
                        (cx, cy),
                        (heat_radius, int(heat_radius * 1.2)),
                        0,
                        0,
                        360,
                        base_heat,
                        -1,
                    )
                    # Circle at feet
                    cv2.circle(
                        current_heat,
                        (cx, ground_y),
                        int(heat_radius * 0.7),
                        base_heat,
                        -1,
                    )

                    # Movement trails
                    min_move = max(5, int(min(box_width, box_height) * 0.20))
                    if matched_pid is not None and move_dist > min_move:
                        prev_x, prev_y, _ = self.prev_positions[i][matched_pid]
                        prev_ground_y = min(
                            int(prev_y + box_height * 0.5), h - 1
                        )

                        movement_boost = min(60, int(move_dist / 2))
                        trail_intensity = min(240, base_heat + movement_boost)
                        trail_radius = min(
                            max(10, int(box_width * 0.3)), heat_radius
                        )

                        if (
                            0 <= prev_x < w
                            and 0 <= prev_ground_y < h
                            and 0 <= cx < w
                            and 0 <= ground_y < h
                        ):
                            cv2.line(
                                current_heat,
                                (int(prev_x), prev_ground_y),
                                (cx, ground_y),
                                trail_intensity,
                                trail_radius,
                            )

                    current_positions[person_id] = (cx, cy, box_width * box_height)

                # Update tracking
                self.prev_positions[i] = current_positions

                # Accumulate current heat
                heat = np.clip(heat + current_heat, 0, 300.0)

                # Occasional blur to soften zones
                if self.loop_counter % self.BLUR_INTERVAL == 0 and heat.max() > 0:
                    blur_kernel = min(self.heat_blur_size, 7)
                    if blur_kernel % 2 == 0:
                        blur_kernel += 1
                    heat = cv2.GaussianBlur(heat, (blur_kernel, blur_kernel), 0)

                self.daily_heat[i] = heat

                # Track highest accumulated heat per camera
                accumulated = float(heat.sum())
                if accumulated > self.highest_accumulated_heat.get(i, 0.0):
                    self.highest_accumulated_heat[i] = accumulated
                    self.highest_heatmap[i] = heat.copy()

                # Emit live overlay every few frames
                if self.frame_counter[i] % 3 == 0:
                    try:
                        overlay = self.create_heatmap_overlay(i, frame, heat)
                        if overlay is not None:
                            self.heatmap_ready.emit(overlay, i)
                    except Exception:
                        pass

                self.frames_processed[i] = self.frames_processed.get(i, 0) + 1

            self.loop_counter += 1

            # --- Initial snapshots (once after ~30 frames per cam) ---
            if not self.initial_snapshots_done and self.frames_processed:
                min_frames = min(self.frames_processed.values())
                if min_frames >= 30:
                    for i, cap in enumerate(caps):
                        if cap is None or not cap.isOpened():
                            continue
                        if i not in last_frames:
                            continue
                        try:
                            self.save_snapshot(i, last_frames[i])
                        except Exception as e:
                            print(
                                f"[HEATMAP] ⚠️ Initial snapshot failed for cam {i+1}: {e}"
                            )
                    self.initial_snapshots_done = True
                    print(
                        f"[HEATMAP] ✅ Initial snapshots saved after {min_frames} frames per camera"
                    )

            # --- Hourly snapshot + combined ---
            if now >= self.next_snapshot_time:
                current_minute = now.minute
                if current_minute < 5:
                    print(
                        f"[HEATMAP] ⏰ Hourly snapshot triggered at {now.strftime('%H:%M:%S')}"
                    )
                    for i, cap in enumerate(caps):
                        if cap is None or not cap.isOpened():
                            continue
                        if i not in last_frames:
                            continue
                        try:
                            self.save_snapshot(i, last_frames[i])
                            print(
                                f"[HEATMAP] ✅ Hourly snapshot saved for cam {i+1} ({self.cam_names[i]})"
                            )
                        except Exception as e:
                            print(
                                f"[HEATMAP] ⚠️ Hourly snapshot failed for cam {i+1}: {e}"
                            )

                    # Reset dwell map for next hour
                    for i in range(len(self.urls)):
                        self.dwell_map[i] = {}

                    # Combined multi-cam grid for this hour
                    try:
                        self.create_combined_snapshot(now, last_frames)
                    except Exception as e:
                        print(
                            f"[HEATMAP] ⚠️ Failed to create combined snapshot: {e}"
                        )

                    print(
                        f"[HEATMAP] 🔄 Hourly snapshot batch completed at {now.strftime('%H:%M:%S')}"
                    )

                current_hour_start = now.replace(minute=0, second=0, microsecond=0)
                self.next_snapshot_time = current_hour_start + timedelta(hours=1)
                print(
                    f"[HEATMAP] ⏰ Next hourly snapshot scheduled for: {self.next_snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            time.sleep(self.loop_sleep)

        # --- Cleanup ---
        for i, cap in enumerate(caps):
            if cap is not None:
                try:
                    cap.release()
                    print(f"[HEATMAP] Camera {i+1} released")
                except Exception:
                    pass

        print("[HEATMAP] Processor stopped cleanly.")

    # ------------------------------------------------------
    #  Combined multi‑cam snapshot (2×N grid)
    # ------------------------------------------------------
    def create_combined_snapshot(self, timestamp: datetime, last_frames: dict[int, np.ndarray]):
        num_cams = len(self.urls)
        if num_cams == 0:
            return

        cols = 2 if num_cams <= 4 else 3
        rows = (num_cams + cols - 1) // cols

        img_width = 640
        img_height = 480
        combined_width = cols * img_width
        combined_height = rows * img_height
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        for i in range(num_cams):
            cam_name = self.cam_names[i] if i < len(self.cam_names) else f"Cam_{i+1}"
            frame = last_frames.get(i, None)
            heat = self.daily_heat.get(i)

            if frame is not None and heat is not None:
                tile = self.create_heatmap_overlay(i, frame, heat)
            elif frame is not None:
                tile = frame.copy()
            else:
                tile = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                cv2.putText(
                    tile,
                    f"{cam_name} - No Feed",
                    (10, img_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

            if tile is None:
                tile = np.zeros((img_height, img_width, 3), dtype=np.uint8)

            tile = cv2.resize(tile, (img_width, img_height))

            row = i // cols
            col = i % cols
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width

            combined_img[y_start:y_end, x_start:x_end] = tile

        date_str = timestamp.strftime("%Y-%m-%d")
        combined_dir = os.path.join(self.record_dir, date_str, "Combined")
        os.makedirs(combined_dir, exist_ok=True)

        combined_path = os.path.join(combined_dir, "Combined.jpg")
        cv2.imwrite(combined_path, combined_img)
        print(f"[HEATMAP] ✅ Saved latest combined snapshot → {combined_path}")

    # ======================================================
    #   SNAPSHOT (per camera, used for initial + hourly)
    # ======================================================
    def save_snapshot(self, cam_idx: int, frame: np.ndarray):
        if frame is None:
            print(f"[HEATMAP] ❌ Snapshot skipped: frame None for cam {cam_idx}")
            return

        heat = self.daily_heat.get(cam_idx)
        if heat is None:
            return

        try:
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(
                    f"[HEATMAP] ⚠️ Invalid frame for cam {cam_idx+1}, skipping snapshot"
                )
                return

            if frame.shape[0] < 100 or frame.shape[1] < 100:
                print(
                    f"[HEATMAP] ⚠️ Frame too small for cam {cam_idx+1}: {frame.shape}, skipping"
                )
                return

            frame = frame.copy()
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            h, w = frame.shape[:2]

            # Ensure heatmap matches frame size
            if heat.shape[:2] != (h, w):
                heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

            # Build overlay for snapshot
            colored = apply_custom_heatmap(heat)
            overlay_f = frame.astype(np.float32)
            colored_f = colored.astype(np.float32)
            blended = overlay_f * 0.70 + colored_f * 0.30
            overlay = np.clip(blended, 0, 255).astype(np.uint8)

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            cam_name = self.cam_names[cam_idx]
            folder = os.path.join(self.record_dir, date_str, cam_name)
            os.makedirs(folder, exist_ok=True)

            hour_str = now.strftime("%H")
            timestamp_str = f"{date_str}_{hour_str}-00-00"
            snapshot_filename = f"snapshot_{timestamp_str}.png"
            snapshot_path = os.path.join(folder, snapshot_filename)

            # Delete existing snapshot for this hour
            if os.path.exists(snapshot_path):
                try:
                    os.remove(snapshot_path)
                    print(
                        f"[HEATMAP] 🗑️ Deleted previous hourly snapshot: {snapshot_filename}"
                    )
                except Exception as e:
                    print(
                        f"[HEATMAP] ⚠️ Failed to delete previous snapshot {snapshot_filename}: {e}"
                    )

            # Delete older snapshots for this camera
            for f in os.listdir(folder):
                if f.startswith("snapshot_") and f.endswith(".png") and f != snapshot_filename:
                    try:
                        os.remove(os.path.join(folder, f))
                        print(f"[HEATMAP] 🗑️ Deleted old snapshot: {f}")
                    except Exception as e:
                        print(
                            f"[HEATMAP] ⚠️ Failed to delete old snapshot {f}: {e}"
                        )

            cv2.imwrite(snapshot_path, overlay, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            accumulated_heat_value = float(heat.sum())
            print(
                f"[HEATMAP] 📸 Snapshot saved → {snapshot_path} (Accumulated heat: {accumulated_heat_value:.1f})"
            )

        except Exception as e:
            print(f"[HEATMAP] ❌ Snapshot overlay failed for cam {cam_idx+1}: {e}")
            import traceback

            traceback.print_exc()

    # ======================================================
    #   FINAL DAILY HEATMAP + RANKING + JSON
    # ======================================================
    def _add_heat_ranking(self, overlay: np.ndarray, heat: np.ndarray, max_ranks: int = 5):
        """Add ranking numbers to overlay based on accumulated heat zones."""
        try:
            if heat is None or heat.max() <= 0:
                return

            heat_np = heat.astype(np.float32)
            h, w = heat_np.shape[:2]

            if overlay.shape[:2] != (h, w):
                heat_np = cv2.resize(
                    heat_np,
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                h, w = heat_np.shape[:2]

            heat_max = float(heat_np.max())
            if heat_max <= 0:
                return

            heat_normalized = (heat_np / heat_max * 255).astype(np.uint8)
            threshold_value = max(10, int(heat_max * 0.3))
            _, binary = cv2.threshold(
                heat_normalized, threshold_value, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return

            regions = []
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw < 20 or bh < 20:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                region_heat = heat_np[mask > 0]
                if region_heat.size == 0:
                    continue

                accumulated_heat = float(region_heat.sum())
                max_heat = float(region_heat.max())
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = x + bw // 2
                    cy = y + bh // 2

                regions.append(
                    {
                        "accumulated_heat": accumulated_heat,
                        "max_heat": max_heat,
                        "center": (cx, cy),
                        "area": cv2.contourArea(contour),
                    }
                )

            if not regions:
                return

            regions.sort(key=lambda r: r["accumulated_heat"], reverse=True)
            top = regions[:max_ranks]

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(
                0.8,
                min(overlay.shape[1] / 800.0, overlay.shape[0] / 600.0),
            )
            thickness = max(2, int(font_scale * 2))

            for rank, reg in enumerate(top, start=1):
                cx, cy = reg["center"]
                text = str(rank)
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                tx = cx - tw // 2
                ty = cy + th // 2

                cv2.putText(
                    overlay,
                    text,
                    (tx - 1, ty - 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    text,
                    (tx + 1, ty + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    text,
                    (tx, ty),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
        except Exception:
            return

    def _count_heat_zones(self, heat: np.ndarray) -> int:
        """Count number of distinct heat zones."""
        if heat is None:
            return 0
        try:
            if heat.max() <= 0:
                return 0

            h, w = heat.shape[:2]
            heat_normalized = (heat / heat.max() * 255).astype(np.uint8)
            threshold_value = max(10, int(heat.max() * 0.3))
            _, binary = cv2.threshold(
                heat_normalized, threshold_value, 255, cv2.THRESH_BINARY
            )
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return 0
            zone_count = 0
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw >= 20 and bh >= 20:
                    zone_count += 1
            return zone_count
        except Exception as e:
            print(f"[HEATMAP] ⚠️ Failed to count heat zones: {e}")
            return 0

    def _save_combined_daily_json(self):
        """Save combined daily JSON from self.final_day_json_data."""
        try:
            if not self.final_day_json_data:
                print(
                    "[HEATMAP] ⚠️ No heat density data to save in combined JSON"
                )
                return

            date_str = self.active_day
            combined = {
                date_str: {
                    "PeakTime": datetime.now().strftime("%H:%M:%S"),
                    "Total": 0,
                    "Cameras": {},
                }
            }

            for json_data in self.final_day_json_data:
                cam_name = json_data.get("camera", "Unknown")
                zones = json_data.get("zones", [])
                cam_total = len(zones)
                combined[date_str]["Cameras"][cam_name] = cam_total
                combined[date_str]["Total"] += cam_total

            sorted_cameras = sorted(
                combined[date_str]["Cameras"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
            combined[date_str]["Cameras"] = dict(sorted_cameras)

            logs_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs"
            )
            os.makedirs(logs_dir, exist_ok=True)
            json_path = os.path.join(logs_dir, "heatmap_density.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=4, ensure_ascii=False)

            print(f"[HEATMAP] ✅ Combined daily JSON saved → {json_path}")
            print(
                f"[HEATMAP] 📊 Total heat zones: {combined[date_str]['Total']}"
            )
            print("[HEATMAP] 📊 Cameras sorted by heat density:")
            for cam_name, zone_count in sorted_cameras:
                print(f"  • {cam_name}: {zone_count} zones")
        except Exception as e:
            print(f"[HEATMAP] ❌ Failed to save combined daily JSON: {e}")
            import traceback

            traceback.print_exc()

    def save_final_daily_heatmap(self, fresh_frames: dict[int, np.ndarray] | None = None):
        """Save final daily heatmap per camera with ranking & JSON."""
        print("[HEATMAP] Saving final daily heatmaps with highest accumulated heat…")
        self.final_day_json_data = []

        for i in range(len(self.urls)):
            heat = self.highest_heatmap.get(i)
            if heat is None:
                heat = self.daily_heat.get(i)

            if heat is None:
                print(
                    f"[HEATMAP] ⚠️ No heatmap data for final snapshot (cam {i+1}), skipping"
                )
                continue

            frame = None
            if fresh_frames and i in fresh_frames:
                frame = fresh_frames[i]

            if frame is None:
                print(
                    f"[HEATMAP] ⚠️ No frame available for daily final snapshot (cam {i+1}), skipping"
                )
                continue

            date_str = self.active_day
            cam_name = self.cam_names[i]
            folder = os.path.join(self.record_dir, date_str, cam_name)
            os.makedirs(folder, exist_ok=True)

            final_filename = f"final_day_{date_str.replace('-', '_')}.png"
            final_path = os.path.join(folder, final_filename)

            try:
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    print(
                        f"[HEATMAP] ⚠️ Invalid frame for cam {i+1}, skipping final snapshot"
                    )
                    continue

                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    print(
                        f"[HEATMAP] ⚠️ Frame too small for cam {i+1}: {frame.shape}, skipping"
                    )
                    continue

                frame = frame.copy()
                if not frame.flags["C_CONTIGUOUS"]:
                    frame = np.ascontiguousarray(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                h, w = frame.shape[:2]
                if heat.shape[:2] != (h, w):
                    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

                final_heat = heat.copy()

                # Blend accumulated heatmap with frame
                blur_size = 11 if self.heat_blur_size <= 15 else 15
                if final_heat.max() > 0:
                    final_heat = cv2.GaussianBlur(
                        final_heat, (blur_size, blur_size), 0
                    )

                colored = apply_custom_heatmap(final_heat)
                overlay_f = frame.astype(np.float32)
                colored_f = colored.astype(np.float32)
                blended = overlay_f * 0.70 + colored_f * 0.30
                overlay = np.clip(blended, 0, 255).astype(np.uint8)

                # Ranking numbers on overlay
                try:
                    self._add_heat_ranking(overlay, heat, max_ranks=5)
                except Exception as e:
                    print(f"[HEATMAP] ⚠️ Ranking failed for cam {i+1}: {e}")

                cv2.imwrite(final_path, overlay, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                highest_heat_value = self.highest_accumulated_heat.get(i, 0.0)
                print(
                    f"[HEATMAP] 📸 Final daily heatmap saved → {final_path} (Highest accumulated heat: {highest_heat_value:.1f})"
                )

                # Count heat zones & extract zone info
                zone_count = self._count_heat_zones(heat)
                zones = []
                try:
                    if heat.max() > 0:
                        h_heat, w_heat = heat.shape[:2]
                        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
                        threshold_value = max(10, int(heat.max() * 0.3))
                        _, binary = cv2.threshold(
                            heat_norm, threshold_value, 255, cv2.THRESH_BINARY
                        )
                        contours, _ = cv2.findContours(
                            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        for contour in contours:
                            x, y, bw, bh = cv2.boundingRect(contour)
                            if bw >= 20 and bh >= 20:
                                mask = np.zeros((h_heat, w_heat), dtype=np.uint8)
                                cv2.drawContours(mask, [contour], -1, 255, -1)
                                region_heat = heat[mask > 0]
                                if region_heat.size == 0:
                                    continue
                                accumulated_heat = float(region_heat.sum())
                                max_heat = float(region_heat.max())
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                else:
                                    cx = x + bw // 2
                                    cy = y + bh // 2
                                zones.append(
                                    {
                                        "center": (cx, cy),
                                        "accumulated_heat": accumulated_heat,
                                        "max_heat": max_heat,
                                        "area": cv2.contourArea(contour),
                                    }
                                )
                except Exception as e:
                    print(
                        f"[HEATMAP] ⚠️ Failed to extract zone info for cam {i+1}: {e}"
                    )

                self.final_day_json_data.append(
                    {
                        "camera": cam_name,
                        "zones": zones,
                        "zone_count": zone_count,
                    }
                )
                print(
                    f"[HEATMAP] 📊 Camera {cam_name}: {zone_count} heat zones detected"
                )

            except Exception as e:
                print(
                    f"[HEATMAP] ❌ Failed to save daily final snapshot for cam {i+1}: {e}"
                )
                import traceback

                traceback.print_exc()

        # After all cameras processed, write combined JSON
        self._save_combined_daily_json()

    def stop(self):
        self.running = False
        self.wait(1000)


# ==========================================================
#   HEATMAP DASHBOARD UI  (Camera Selection + Preview)
# ==========================================================
class HeatmapDashboardScreen(QWidget):
    """Unified layout for Heatmap Live Analytics."""

    def __init__(self, stacked):
        super().__init__()
        self.stacked = stacked
        self.preview_labels = []
        self.cam_checkboxes = []
        self.preview_workers: list[CameraPreviewWorker] = []

        self.selection_file = "selected_heatmap.json"

        # Connect heatmap signal if processor already exists
        if hasattr(self.stacked, "heatmap_worker") and self.stacked.heatmap_worker:
            self.stacked.heatmap_worker.heatmap_ready.connect(
                self.update_heatmap_preview
            )

        self.setStyleSheet(
            """
            QWidget { background:white; color:#0f2027; }
            QLabel { font-size:18px; }
            QPushButton {
                font-size:20px; font-weight:bold;
                color:white;
                border-radius:10px;
                padding:12px 24px;
            }
        """
        )

        layout = QVBoxLayout(self)

        title = QLabel("🔥 Heatmap – Live Analytics")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:26px; font-weight:bold; color:#D32F2F;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(40)

        scroll.setWidget(container)
        layout.addWidget(scroll)

        cams = getattr(self.stacked, "active_cams", [])
        saved = []
        if os.path.exists(self.selection_file):
            try:
                saved = json.load(open(self.selection_file))["names"]
            except Exception:
                saved = []

        if not cams:
            msg = QLabel("No active cameras detected.")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
        else:
            for i, cam in enumerate(cams):
                cam_name = cam["cam_name"]

                lbl = QLabel()
                lbl.setFixedSize(300, 200)
                lbl.setStyleSheet("background:#e8e8e8; border:1px solid #ccc;")

                cb = QCheckBox(cam_name)
                cb.setChecked(cam_name in saved or not saved)

                self.preview_labels.append(lbl)
                self.cam_checkboxes.append(cb)

                vbox = QVBoxLayout()
                vbox.addWidget(lbl)
                vbox.addWidget(cb)

                widget = QWidget()
                widget.setLayout(vbox)

                self.grid.addWidget(widget, i // 3, i % 3)

                worker = CameraPreviewWorker(i, cam["url"])
                worker.frame_ready.connect(self.update_preview)
                worker.start()
                self.preview_workers.append(worker)

        btn_row = QHBoxLayout()

        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("background:#2e7d32;")
        back_btn.setFixedHeight(55)
        back_btn.clicked.connect(
            lambda: self.stacked.stack.setCurrentWidget(self.stacked.ml_screen)
        )
        btn_row.addWidget(back_btn)

        run_btn = QPushButton("▶ Run Analytics")
        run_btn.setStyleSheet("background:#D32F2F;")
        run_btn.setFixedHeight(55)
        run_btn.clicked.connect(self.handle_run)
        btn_row.addWidget(run_btn)

        layout.addLayout(btn_row)

    def update_preview(self, pix: QPixmap, idx: int):
        if idx < len(self.preview_labels):
            self.preview_labels[idx].setPixmap(pix)

    def update_heatmap_preview(self, frame: np.ndarray, idx: int):
        """Update preview label with heatmap overlay frame."""
        if idx >= len(self.preview_labels):
            return
        try:
            h, w, ch = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(
                frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888
            )
            pix = QPixmap.fromImage(qimg).scaled(
                300, 200, Qt.AspectRatioMode.KeepAspectRatio
            )
            self.preview_labels[idx].setPixmap(pix)
        except Exception:
            pass

    def handle_run(self):
        for w in self.preview_workers:
            w.stop()

        cams = getattr(self.stacked, "active_cams", [])
        selected = [i for i, cb in enumerate(self.cam_checkboxes) if cb.isChecked()]

        if not selected:
            self.show_popup("No cameras selected.\nSelect at least one camera.")
            return

        sel_names = [cams[i]["cam_name"] for i in selected]
        sel_urls = [normalize_rtsp_url(cams[i]["url"]) for i in selected]

        json.dump({"names": sel_names}, open(self.selection_file, "w"), indent=4)

        if (
            getattr(self.stacked, "heatmap_worker", None)
            and self.stacked.heatmap_worker.isRunning()
        ):
            print("[HEATMAP] Processor already running.")
            try:
                self.stacked.heatmap_worker.heatmap_ready.disconnect()
            except Exception:
                pass
            self.stacked.heatmap_worker.heatmap_ready.connect(
                self.update_heatmap_preview
            )
        else:
            print(f"[RUN] Starting Heatmap Analytics → {sel_names}")
            self.stacked.start_heatmap(sel_urls, sel_names)
            if hasattr(self.stacked, "heatmap_worker") and self.stacked.heatmap_worker:
                self.stacked.heatmap_worker.heatmap_ready.connect(
                    self.update_heatmap_preview
                )

        self.stacked.stack.setCurrentWidget(self.stacked.ml_screen)

    def show_popup(self, message: str):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Selection Required")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setStyleSheet(
            """
            QMessageBox { background:white; font-size:18px; }
            QPushButton {
                padding:10px 20px;
                background:#D32F2F;
                color:white;
                border-radius:10px;
                font-size:16px;
            }
        """
        )
        msg.exec()


