#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire – Dashboard Screen (Auto-Detected Cameras)
-------------------------------------------------------
• Auto-detects DVR IP
• Builds RTSP list dynamically
• Displays only reachable cameras
• Full-frame 3×N grid
"""

import cv2, time, math
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QGridLayout, QPushButton,
    QSizePolicy, QHBoxLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage

# --- Import DVR helper logic ---
from modules.utils.dvr_logic import auto_detect_dvr_ip, build_rtsp_urls


# ---------------- Stream Thread ---------------- #
class StreamWorker(QThread):
    frame_ready = pyqtSignal(QImage, int)

    def __init__(self, cam_index, url, fps_limit=10):
        super().__init__()
        self.cam_index = cam_index
        self.url = url
        self.running = True
        self.fps_limit = fps_limit

    def run(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

        if not cap.isOpened():
            print(f"[WARN] Cam {self.cam_index+1}: failed to open {self.url}")
            return

        interval = 1.0 / self.fps_limit
        while self.running:
            start = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(img, self.cam_index)
            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)
        cap.release()

    def stop(self):
        """Gracefully stop the stream thread."""
        self.running = False
        try:
            self.wait(500)  # wait up to 0.5 s for Pi / Luckfox
        except:
            pass



# ---------------- Dashboard UI ---------------- #
class DashboardScreen(QWidget):
    back_to_login = pyqtSignal()
    go_next = pyqtSignal()

    def __init__(self, config=None, active_cams=None):
        super().__init__()
        self.config = config or {}
        self.stream_threads, self.labels = [], []
        self.cols = 3
        self.aspect_ratio = 16 / 9
        self.default_tile_width = 420

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("📡 Active Camera Streams")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("font-size:22px; font-weight:bold; color:#1a5fd0;")
        self.main_layout.addWidget(title)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(15)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.main_layout.addLayout(self.grid_layout)

        # Bottom bar
        bottom_bar = QHBoxLayout()
        back_btn = QPushButton("⬅ Back to Login")
        back_btn.clicked.connect(self.handle_back)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color:#1a5fd0; color:white;
                border:none; border-radius:6px;
                padding:8px 20px; font-weight:bold;
            }
            QPushButton:hover { background-color:#1749a8; }
        """)

        next_btn = QPushButton("Next ➡")
        next_btn.clicked.connect(self.handle_next)
        next_btn.setStyleSheet("""
            QPushButton {
                background-color:#2e7d32; color:white;
                border:none; border-radius:6px;
                padding:8px 20px; font-weight:bold;
            }
            QPushButton:hover { background-color:#1b5e20; }
        """)

        bottom_bar.addWidget(back_btn)
        bottom_bar.addStretch()
        bottom_bar.addWidget(next_btn)
        self.main_layout.addStretch()
        self.main_layout.addLayout(bottom_bar)

        # --- Auto detect DVR every 5 minutes ---
        self.dvr_timer = QTimer(self)
        self.dvr_timer.timeout.connect(self.detect_and_load)
        self.dvr_timer.start(300000)  # 5 minutes = 300,000 ms

        # Run once immediately after startup
        QTimer.singleShot(200, self.detect_and_load)

        self.showMaximized()


    def detect_and_load(self):
        try:
            # --- Ensure config is always a dict ---
            if isinstance(self.config, str):
                import json
                try:
                    self.config = json.loads(self.config)
                except Exception:
                    raise RuntimeError("Invalid config format; expected dict, got str")

            now = time.strftime("%H:%M:%S")
            print(f"[DVR] Auto-detection triggered at {now}")
            print("[DVR] Auto-detecting DVR IP...")
            print("[DVR] Auto-detecting DVR IP...")

            # --- Step 1: Detect DVR IP ---
            dvr_ip = auto_detect_dvr_ip()
            print("[DVR] Scanning 1016 hosts for DVR devices...")
            print(f"[DVR] DVR IP detected: {dvr_ip}")

            # --- Step 2: Build RTSP URLs ---
            result = build_rtsp_urls(
                self.config.get("brand", "Hikvision"),
                dvr_ip,
                self.config.get("user", "admin"),
                self.config.get("pass", "")
            )

            # --- Step 3: Parse result safely ---
            urls, cam_names = [], []
            if isinstance(result, list):
                for i, item in enumerate(result):
                    if isinstance(item, dict):
                        urls.append(item.get("url"))
                        cam_names.append(item.get("name", f"Cam {i+1}"))
            elif isinstance(result, tuple) and len(result) >= 2:
                urls, cam_names = result[:2]
            else:
                raise RuntimeError(f"Unexpected return format from build_rtsp_urls(): {type(result)}")

            if not urls:
                raise RuntimeError("No camera URLs generated")

            # --- Step 4: Filter reachable cameras ---
            active_cams = []
            for url, name in zip(urls, cam_names):
                cap = cv2.VideoCapture(url)
                ok = cap.isOpened()
                cap.release()
                if ok:
                    active_cams.append({"url": url, "cam_name": name, "active": True})
                    print(f"[OK] {name} reachable")
                else:
                    print(f"[SKIP] {name} unreachable")

            if not active_cams:
                QMessageBox.warning(self, "No Cameras", "No active camera streams detected.")
                return

            self.active_cams = active_cams
            self.populate_grid()

            # --- Step 5: Print success in your desired format ---
            print(f"[INFO] Login success: {{'brand': '{self.config.get('brand')}', "
                f"'ip': '{dvr_ip}', 'user': '{self.config.get('user')}', "
                f"'pass': '{self.config.get('pass')}'}}")

        except Exception as e:
            QMessageBox.critical(self, "DVR Error", str(e))
            print(f"[ERROR] DVR detection failed: {e}")


    # ---------- Populate Grid ---------- #
    def populate_grid(self):
        # --- Clean up any old preview threads before starting new ones ---
        for w in getattr(self, "stream_threads", []):
            try:
                if w.isRunning():
                    w.stop()
            except:
                pass
        self.stream_threads.clear()
        self.labels.clear()

        total = len(self.active_cams)
        for idx, cam in enumerate(self.active_cams):
            row, col = divmod(idx, self.cols)
            label = QLabel(f"{cam['cam_name']}\nConnecting…")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("""
                QLabel {
                    background:#f0f8ff; border:2px solid #1a5fd0;
                    border-radius:8px; color:#333; font-weight:bold;
                }
            """)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.grid_layout.addWidget(label, row, col)
            self.labels.append(label)

            worker = StreamWorker(idx, cam["url"], fps_limit=10)
            worker.frame_ready.connect(self.update_frame)
            self.stream_threads.append(worker)
            worker.start()

            # Fail message if no frame arrives
            QTimer.singleShot(3000, lambda ci=idx: self.mark_fail(ci))

    def mark_fail(self, ci):
        lbl = self.labels[ci]
        if lbl.pixmap() is None:
            lbl.setText(f"Cam {ci+1}\n❌ No Signal")
            lbl.setStyleSheet("""
                QLabel {
                    background:#ffe6e6; border:2px solid #d9534f;
                    border-radius:8px; color:#d9534f; font-weight:bold;
                }
            """)

    @pyqtSlot(QImage, int)
    def update_frame(self, image, cam_index):
        if cam_index >= len(self.labels):
            return
        label = self.labels[cam_index]
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_tile_sizes()

    def adjust_tile_sizes(self):
        if not self.labels: return
        cols = self.cols
        total = len(self.labels)
        rows = math.ceil(total / cols)
        grid_w = self.width() - 100
        tile_w = grid_w // cols - 20
        tile_h = int(tile_w / self.aspect_ratio)
        for label in self.labels:
            label.setFixedSize(QSize(tile_w, tile_h))
        self.update()

    def handle_back(self):
        if hasattr(self, "dvr_timer"):
            self.dvr_timer.stop()
        for w in getattr(self, "stream_threads", []):
            try:
                if w.isRunning():
                    w.stop()
            except:
                pass
        self.back_to_login.emit()

    def handle_next(self):
        if hasattr(self, "dvr_timer"):
            self.dvr_timer.stop()
        for w in getattr(self, "stream_threads", []):
            try:
                if w.isRunning():
                    w.stop()
            except:
                pass
        self.go_next.emit()

