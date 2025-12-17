#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VisitorsSelectionScreen + VisitorsDashboardScreen
-------------------------------------------------
GLOBAL VISITORS LOGIC READY (Buddy Edition)
• Clean UI + Proper Worker Control
• Prevents double worker start
• Safely stops preview threads
• Works with global identity system
"""

import os, json, cv2
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QScrollArea, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage

from modules.visitors_count.daily_processor import DailyCountProcessor


# ==========================================================
#   CAMERA PREVIEW THREAD (UI ONLY)
# ==========================================================
class CameraPreviewWorker(QThread):
    frame_ready = pyqtSignal(QPixmap, int)

    def __init__(self, idx, url):
        super().__init__()
        self.idx = idx
        self.url = url
        self.running = True

    def run(self):
        print(f"[THREAD] Visitors preview {self.idx} → {self.url}")
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(200)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)

            self.frame_ready.emit(pix, self.idx)
            self.msleep(200)

        cap.release()
        print(f"[THREAD] Visitors preview {self.idx} stopped.")

    def stop(self):
        self.running = False
        self.wait(300)


# ==========================================================
#   CAMERA SELECTION SCREEN
# ==========================================================
class VisitorsSelectionScreen(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.build_ui()

    # ------------------------------------------------------
    def build_ui(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                                            stop:0 #edf3ff,
                                            stop:1 #d6e0ff);
                font-family:'Segoe UI';
            }
            QLabel#title {
                font-size:26px;
                font-weight:700;
                color:#123e91;
            }
            QCheckBox { font-size:16px; padding:6px; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #1a5fd0,
                                            stop:1 #1749a8);
                color:white;
                border:none;
                border-radius:10px;
                padding:10px 20px;
                font-weight:600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #2c6be0,
                                            stop:1 #1749a8);
            }
        """)

        main = QVBoxLayout(self)
        main.setContentsMargins(40, 30, 40, 30)

        title = QLabel("📷 Select Cameras for Daily Visitors")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main.addWidget(title)

        # camera list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        cl = QVBoxLayout(container)

        cams = getattr(self.controller, "active_cams", [])
        self.checkboxes = []

        for cam in cams:
            cb = QCheckBox(f"{cam['cam_name']}  →  {cam['url']}")
            self.checkboxes.append((cb, cam))
            cl.addWidget(cb)

        scroll.setWidget(container)
        main.addWidget(scroll)

        # buttons
        btns = QHBoxLayout()
        back_btn = QPushButton("⬅ Back")
        back_btn.clicked.connect(self.go_back)
        btns.addWidget(back_btn)

        run_btn = QPushButton("Run Analytics ▶")
        run_btn.clicked.connect(self.run_analytics)
        btns.addWidget(run_btn)

        main.addLayout(btns)

    def go_back(self):
        self.controller.stack.setCurrentWidget(self.controller.ml_screen)
        self.controller.fade_to_widget(self.controller.ml_screen)

    # -------- START ANALYTICS (SAFE GLOBAL WORKER START) --------
    def run_analytics(self):
        urls, names = [], []

        for cb, cam in self.checkboxes:
            if cb.isChecked():
                urls.append(cam["url"])
                names.append(cam["cam_name"])

        if not urls:
            print("[VISITOR] No cameras selected.")
            return

        print("[VISITOR] Selected:", names)

        worker = getattr(self.controller, "visitors_worker", None)

        # Safe worker start
        if worker is None or not worker.isRunning():
            print("[VISITOR] Starting GLOBAL visitors…")
            self.controller.start_daily_visitors(urls, names)
        else:
            print("[VISITOR] Global worker already running, cannot start twice.")

        # Go to dashboard
        dash = VisitorsDashboardScreen(self.controller)
        self.controller.stack.addWidget(dash)
        self.controller.stack.setCurrentWidget(dash)
        self.controller.fade_to_widget(dash)


# ==========================================================
#   DASHBOARD (LIVE PREVIEWS + RUN)
# ==========================================================
class VisitorsDashboardScreen(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.preview_workers = []
        self.preview_labels = []
        self.cam_checkboxes = []
        self.selection_file = "selected_visitors.json"

        self.build_ui()

    # ------------------------------------------------------
    def build_ui(self):
        self.setStyleSheet("""
            QWidget { background:white; }
            QCheckBox { font-size:16px; }
            QPushButton {
                font-size:20px; font-weight:bold;
                color:white;
                border-radius:10px; padding:12px 24px;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("👣 Daily Visitors – Live Analytics")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:24px; font-weight:bold; color:#2E7D32;")
        layout.addWidget(title)

        # ------------- Camera Previews -------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        cont = QWidget()
        self.grid = QGridLayout(cont)
        self.grid.setSpacing(25)
        scroll.setWidget(cont)
        layout.addWidget(scroll)

        cams = getattr(self.controller, "active_cams", [])

        # load saved selections
        saved = []
        if os.path.exists(self.selection_file):
            try:
                saved = json.load(open(self.selection_file))["names"]
            except:
                saved = []

        for i, cam in enumerate(cams):
            lbl = QLabel()
            lbl.setFixedSize(300, 200)
            lbl.setStyleSheet(
                "background:#E8F5E9; border:3px solid #81C784; border-radius:8px;"
            )

            cb = QCheckBox(cam["cam_name"])
            cb.setChecked(True if not saved or cam["cam_name"] in saved else False)

            self.preview_labels.append(lbl)
            self.cam_checkboxes.append(cb)

            v = QVBoxLayout()
            v.addWidget(lbl)
            v.addWidget(cb)

            w = QWidget()
            w.setLayout(v)
            self.grid.addWidget(w, i // 3, i % 3)

            worker = CameraPreviewWorker(i, cam["url"])
            worker.frame_ready.connect(self.update_preview)
            worker.start()
            self.preview_workers.append(worker)

        # =============================
        #       BUTTON ROW
        # =============================
        btn_row = QHBoxLayout()

        # Back Button (patched version)
        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("background:#2E7D32;")
        back_btn.setFixedHeight(55)
        back_btn.clicked.connect(self.go_back)
        btn_row.addWidget(back_btn)

        # Run Button (patched)
        run_btn = QPushButton("▶ Run Analytics")
        run_btn.setStyleSheet("background:#388E3C;")
        run_btn.setFixedHeight(55)
        run_btn.clicked.connect(self.handle_run)
        btn_row.addWidget(run_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------
    def go_back(self):
        """Stop preview threads and return safely."""
        for w in self.preview_workers:
            w.stop()

        self.controller.stack.setCurrentWidget(self.controller.ml_screen)
        self.controller.fade_to_widget(self.controller.ml_screen)

    # ------------------------------------------------------
    def update_preview(self, pix, idx):
        if idx < len(self.preview_labels):
            self.preview_labels[idx].setPixmap(pix)

    # ------------------------------------------------------
    def handle_run(self):
        """Start GLOBAL Daily Visitors analytics."""

        # Stop preview threads
        for w in self.preview_workers:
            w.stop()

        cams = getattr(self.controller, "active_cams", [])
        selected = [i for i, cb in enumerate(self.cam_checkboxes) if cb.isChecked()]

        if not selected:
            self.show_popup("No cameras selected.\nPlease select at least one camera.")
            return

        sel_names = [cams[i]["cam_name"] for i in selected]
        sel_urls  = [cams[i]["url"] for i in selected]

        # Save selection
        with open(self.selection_file, "w") as f:
            json.dump({"names": sel_names}, f, indent=4)

        # Safe worker start
        worker = getattr(self.controller, "visitors_worker", None)
        if worker is None or not worker.isRunning():
            print(f"[RUN] Starting GLOBAL Daily Visitors → {sel_names}")
            self.controller.start_daily_visitors(sel_urls, sel_names)
        else:
            print("[VISITORS] Global worker already running.")

        # Return to ML screen
        self.controller.stack.setCurrentWidget(self.controller.ml_screen)
        self.controller.fade_to_widget(self.controller.ml_screen)

    # ------------------------------------------------------
    def show_popup(self, message):
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Selection Required")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setStyleSheet("""
            QMessageBox {
                background:white;
                font-size:18px;
            }
            QPushButton {
                padding:10px 20px;
                background:#388E3C;
                color:white;
                border-radius:10px;
                font-size:16px;
            }
        """)
        msg.exec()
