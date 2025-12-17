#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLSelectionScreen – Clean Highlighted Edition (with Settings Page)
-------------------------------------------------------------------
• Adds ⚙ Settings button (bottom-right)
• Opens new Settings page with Back button
• Maintains full QSS compatibility, Jetson-safe
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from modules.analytics.settings_screen import SettingsScreen



# ==========================================================
#   Reusable Model Card
# ==========================================================
class ModelCard(QFrame):
    """Static card with supported QSS highlight on hover."""
    def __init__(self, icon, title, desc, signal_emit):
        super().__init__()
        self.signal_emit = signal_emit
        self.title_text = title

        self.setFixedSize(380, 380)
        self.setStyleSheet(self.normal_style())

        vbox = QVBoxLayout(self)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.setSpacing(12)

        # Icon
        self.icon_lbl = QLabel(icon)
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_lbl.setStyleSheet("font-size:78px; color:#1749a8;")

        # Title
        self.title_lbl = QLabel(title)
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_lbl.setStyleSheet("""
            font-size:22px;
            font-weight:700;
            color:#1749a8;
        """)

        # Description
        self.desc_lbl = QLabel(desc)
        self.desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_lbl.setWordWrap(True)
        self.desc_lbl.setStyleSheet("""
            font-size:15px;
            color:#444;
            background: transparent;
            padding: 4px 14px;
        """)

        # Launch button
        self.launch_btn = QPushButton("Launch ▶")
        self.launch_btn.setFixedWidth(150)
        self.launch_btn.clicked.connect(lambda: self.signal_emit.emit(self.title_text))

        
        # --- Buttons container ---
        buttons_container = QHBoxLayout()
        buttons_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        buttons_container.setSpacing(10)

        # Employees button (only for Emp–Cus Interaction)
        self.employees_btn = None
        if title == "Emp–Cus Interaction":
            self.employees_btn = QPushButton("Employees")
            self.employees_btn.setFixedWidth(120)
            self.employees_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                                stop:0 #4CAF50,
                                                stop:1 #388E3C);
                    color:white;
                    border:none;
                    border-radius:8px;
                    padding:10px 16px;
                    font-weight:600;
                    letter-spacing:0.5px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                                stop:0 #66BB6A,
                                                stop:1 #4CAF50);
                }
            """)

            self.employees_btn.clicked.connect(lambda: self.signal_emit.emit("EmpCus_EMPLOYEE"))

            buttons_container.addWidget(self.employees_btn)

        buttons_container.addWidget(self.launch_btn)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_container)


        vbox.addWidget(self.icon_lbl)
        vbox.addWidget(self.title_lbl)
        vbox.addWidget(self.desc_lbl)
        vbox.addStretch()
        vbox.addWidget(buttons_widget)


    def enterEvent(self, e): self.setStyleSheet(self.hover_style())
    def leaveEvent(self, e): self.setStyleSheet(self.normal_style())

    def normal_style(self):
        return """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #ffffff,
                                            stop:0.5 #f4f7ff,
                                            stop:1 #e3e9ff);
                border: 2px solid #c7d2ff;
                border-radius: 18px;
                padding: 10px;
            }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #1a5fd0,
                                            stop:1 #1749a8);
                color:white;
                border:none;
                border-radius:8px;
                padding:10px 22px;
                font-weight:600;
                letter-spacing:0.5px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #2c6be0,
                                            stop:1 #1749a8);
            }
        """
    def hover_style(self):
        return """
            QFrame {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #e9f1ff,
                                            stop:0.5 #c5d5ff,
                                            stop:1 #a9beff);
                border: 2px solid #1a5fd0;
                border-radius: 18px;
                padding: 10px;
            }
            QLabel {
                color:#102f70;
            }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #ffffff,
                                            stop:1 #dce6ff);
                color:#1749a8;
                border:none;
                border-radius:8px;
                padding:10px 22px;
                font-weight:700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                                            stop:0 #f5f8ff,
                                            stop:1 #e4ecff);
            }
        """


# ==========================================================
#   ML Selection Screen
# ==========================================================
class MLSelectionScreen(QWidget):
    model_selected = pyqtSignal(str)
    back_to_dashboard = pyqtSignal()

    def __init__(self, stacked=None):
        super().__init__()
        self.stacked = stacked
        self.build_ui()

    def build_ui(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                                            stop:0 #edf3ff,
                                            stop:1 #d6e0ff);
                font-family:'Segoe UI';
            }
            QLabel#title {
                font-size:28px;
                font-weight:700;
                color:#123e91;
            }
            QLabel#subtitle {
                font-size:15px;
                color:#444;
            }
            QPushButton#backBtn {
                background-color:#2e7d32;
                color:white;
                border:none;
                border-radius:10px;
                padding:8px 20px;
                font-weight:600;
            }
            QPushButton#backBtn:hover { background-color:#1b5e20; }
            QPushButton#settingsBtn {
                background-color:#1565C0;
                color:white;
                border:none;
                border-radius:10px;
                padding:8px 20px;
                font-weight:600;
            }
            QPushButton#settingsBtn:hover { background-color:#1e73d2; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(25)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("🧠 Select ML Model for Analytics")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        subtitle = QLabel("Choose the analytics module to launch.")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # --- Row 1 ---
        row1 = QHBoxLayout()
        row1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row1.setSpacing(30)

        models_row1 = [
            ("👥", "Crowd Density", "Counts people in real-time to analyze occupancy."),
            ("🔥", "HeatMap", "Visualizes customer movement and dwell patterns."),
            ("📈", "Daily Visitors", "Tracks unique visitors entering per day."),
        ]
        for icon, name, desc in models_row1:
            card = ModelCard(icon, name, desc, self.model_selected)
            card.setFixedSize(320, 320)
            row1.addWidget(card)

        # --- Row 2 ---
        row2 = QHBoxLayout()
        row2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row2.setSpacing(30)

        models_row2 = [
            ("💬", "Emp–Cus Interaction", "Detects employee–customer engagement instances."),
            ("🎥", "Motion Detection", "Alerts on any motion in the monitored area."),
        ]
        for icon, name, desc in models_row2:
            card = ModelCard(icon, name, desc, self.model_selected)
            card.setFixedSize(320, 320)
            row2.addWidget(card)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addStretch()

        # --- Bottom Buttons Row ---
        foot = QHBoxLayout()
        foot.setSpacing(10)

        back_btn = QPushButton("⬅ Back to Dashboard")
        back_btn.setObjectName("backBtn")
        back_btn.setFixedWidth(220)
        back_btn.clicked.connect(self.back_to_dashboard.emit)
        foot.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        foot.addStretch(1)

        # ✅ New Settings Button (bottom-right)
        settings_btn = QPushButton("⚙ Settings")
        settings_btn.setObjectName("settingsBtn")
        settings_btn.setFixedWidth(150)
        settings_btn.clicked.connect(self.open_settings)
        foot.addWidget(settings_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addLayout(foot)

    def open_settings(self):
        """Open the new Settings page."""
        if self.stacked is None:
            print("[WARN] MLSelectionScreen.stacked not set!")
            return
        settings = SettingsScreen(self.stacked, self)
        self.stacked.addWidget(settings)
        self.stacked.setCurrentWidget(settings)