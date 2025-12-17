#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SettingsScreen – with Device Info, ML Model Time Table, and Logout
------------------------------------------------------------------
• Shows editable time table for ML models inside same page
• Saves schedule to core/ml_model_schedule.json
• Jetson-safe and QSS-compatible
"""

import os, json, socket, psutil
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QMessageBox,
    QTableWidget, QTableWidgetItem, QTimeEdit, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QTime
from modules.utils.gpu_monitor import get_gpu_temp, get_cpu_temp


class SettingsScreen(QWidget):
    """Settings page with Device Info, Model Time Table, and Logout."""
    def __init__(self, stacked, parent_screen):
        super().__init__()
        self.stacked = stacked
        self.parent_screen = parent_screen
        self.schedule_path = os.path.join("core", "ml_model_schedule.json")

        self.models = [
            "Crowd Density",
            "HeatMap",
            "Daily Visitors",
            "Emp–Cus Interaction",
            "Motion Detection"
        ]

        # ---------- QSS ----------
        self.setStyleSheet("""
            QWidget { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 #f7f9ff, stop:1 #e3ebff); color:#0f2027; }
            QLabel#title { font-size:28px; font-weight:700; color:#1749a8; }
            QPushButton {
                font-size:17px; font-weight:600; color:white;
                background:#1565C0; border:none; border-radius:10px;
                padding:10px 25px;
            }
            QPushButton:hover { background:#1e73d2; }
            QPushButton#backBtn {
                background:#2E7D32; color:white; font-weight:600;
                border-radius:8px; padding:10px 25px;
            }
            QPushButton#backBtn:hover { background:#1b5e20; }
            QPushButton#logoutBtn {
                background:#c62828; color:white; font-weight:600;
                border-radius:8px; padding:10px 25px;
            }
            QPushButton#logoutBtn:hover { background:#e53935; }
            QFrame#card {
                background:white; border:2px solid #c7d2ff;
                border-radius:16px; padding:30px;
            }
            QHeaderView::section {
                background:#1749a8; color:white; font-weight:600;
                border:none; padding:6px;
            }
        """)

        # ---------- Layout ----------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 40, 50, 40)
        layout.setSpacing(25)

        # Title
        title = QLabel("⚙ Settings")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addStretch(1)

        # --- Buttons Row ---
        self.btn_device = QPushButton("📟 Device Details")
        self.btn_device.clicked.connect(self.show_device_info)

        self.btn_model_time = QPushButton("⏱ ML Model Time Set")
        self.btn_model_time.clicked.connect(self.toggle_time_table)

        self.btn_logout = QPushButton("🚪 Logout")
        self.btn_logout.setObjectName("logoutBtn")
        self.btn_logout.clicked.connect(self.confirm_logout)

        for b in [self.btn_device, self.btn_model_time, self.btn_logout]:
            b.setFixedWidth(280)
            layout.addWidget(b, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch(1)

        # --- Time Table (hidden initially) ---
        self.card = QFrame()
        self.card.setObjectName("card")
        self.card_layout = QVBoxLayout(self.card)
        self.card_layout.setSpacing(15)

        self.table = QTableWidget(len(self.models), 3)
        self.table.setHorizontalHeaderLabels(["ML Model", "Start Time", "Stop Time"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, name in enumerate(self.models):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, item)

            start = QTimeEdit(); start.setDisplayFormat("HH:mm")
            stop = QTimeEdit(); stop.setDisplayFormat("HH:mm")
            self.table.setCellWidget(row, 1, start)
            self.table.setCellWidget(row, 2, stop)

        self.card_layout.addWidget(self.table)

        # Save Button
        self.save_btn = QPushButton("💾 Save Schedule")
        self.save_btn.setFixedWidth(220)
        self.save_btn.clicked.connect(self.save_schedule)
        self.card_layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.card.setVisible(False)
        layout.addWidget(self.card)

        # --- Back Button ---
        back_btn = QPushButton("⬅ Back to ML Model Selection")
        back_btn.setObjectName("backBtn")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.load_schedule()  # load saved data if present

    # ======================================================
    #  Device Info
    # ======================================================
    def show_device_info(self):
        try:
            hostname = socket.gethostname()
            ip_addr = socket.gethostbyname(hostname)
            disk = psutil.disk_usage('/')
            total_gb = disk.total // (1024**3)
            used_gb = disk.used // (1024**3)
            cpu_temp = get_cpu_temp() or "--"
            gpu_temp = get_gpu_temp() or "--"
        except Exception as e:
            print("[ERROR] Device info:", e)
            hostname = "Unknown"; ip_addr = "N/A"
            total_gb = used_gb = cpu_temp = gpu_temp = "--"

        msg = QMessageBox(self)
        msg.setWindowTitle("Device Details")
        msg.setText(
            f"Device: {hostname}\n"
            f"IP Address: {ip_addr}\n"
            f"CPU Temp: {cpu_temp}°C\n"
            f"GPU Temp: {gpu_temp}°C\n"
            f"Storage: {used_gb} GB / {total_gb} GB\n"
            f"Status: ✅ Active"
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    # ======================================================
    #  ML Model Time Table
    # ======================================================
    def toggle_time_table(self):
        """Show/hide model time table."""
        visible = not self.card.isVisible()
        self.card.setVisible(visible)
        if visible:
            self.load_schedule()

    def load_schedule(self):
        """Load saved times from JSON."""
        if not os.path.exists(self.schedule_path):
            return
        try:
            with open(self.schedule_path, "r") as f:
                data = json.load(f)
            for row, name in enumerate(self.models):
                start_val = data.get(name, {}).get("start", "09:00")
                stop_val = data.get(name, {}).get("stop", "21:00")
                self.table.cellWidget(row, 1).setTime(QTime.fromString(start_val, "HH:mm"))
                self.table.cellWidget(row, 2).setTime(QTime.fromString(stop_val, "HH:mm"))
        except Exception as e:
            print("[WARN] Failed to load schedule:", e)

    def save_schedule(self):
        """Save times to JSON."""
        data = {}
        for row, name in enumerate(self.models):
            start = self.table.cellWidget(row, 1).time().toString("HH:mm")
            stop = self.table.cellWidget(row, 2).time().toString("HH:mm")
            data[name] = {"start": start, "stop": stop}
        try:
            os.makedirs(os.path.dirname(self.schedule_path), exist_ok=True)
            with open(self.schedule_path, "w") as f:
                json.dump(data, f, indent=4)
            QMessageBox.information(self, "Saved", "ML Model schedule saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save schedule:\n{e}")

    # ======================================================
    #  Logout & Back
    # ======================================================
    def confirm_logout(self):
        reply = QMessageBox.question(
            self, "Confirm Logout",
            "Are you sure you want to log out?\nAll ML models will be stopped.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            main_window = self.window()
            if hasattr(main_window, "closeEvent"):
                print("[SETTINGS] Logging out — stopping all models…")
                for worker in [
                    getattr(main_window, "crowd_worker", None),
                    getattr(main_window, "heatmap_processor", None),
                    getattr(main_window, "visitors_worker", None),
                    getattr(main_window, "emp_cus_worker", None),
                    getattr(main_window, "motion_worker", None),
                ]:
                    try:
                        if worker and worker.isRunning():
                            worker.stop()
                    except Exception as e:
                        print(f"[WARN] Worker stop failed: {e}")
                if hasattr(main_window, "return_to_login"):
                    main_window.return_to_login()

    def go_back(self):
        """Return to ML Model Selection screen."""
        self.stacked.setCurrentWidget(self.parent_screen)
        if hasattr(self.stacked.parent(), "fade_to_widget"):
            self.stacked.parent().fade_to_widget(self.parent_screen)
