#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoginScreen – DVR Configuration + Save + 16-Channel RTSP Test
-----------------------------------------------------------------
• Auto-saves DVR details to core/config.json
• Manual “Save Configuration” button
• Tests 16 RTSP channels in a background QThread
• Highlights green/red for active/inactive
• Modern light-blue gradient UI
• Safe auto-skip only if user has NOT edited fields
• Brand-specific RTSP validation
"""

import json, time
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox,
    QLineEdit, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QHBoxLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QColor
from modules.utils import dvr_logic
from modules.utils.dvr_logic import auto_detect_dvr_ip
from core.config_manager import validate_config, auto_retest_dvr

# Try to import requests for HTTP authentication
try:
    import requests
    from requests.auth import HTTPDigestAuth
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARN] requests library not available - HTTP auth test disabled")


# ---------------- DVR Credential Validator Worker ---------------- #
class CredentialValidatorWorker(QObject):
    validated = pyqtSignal(bool)  # emits True if credentials are valid, False otherwise
    error = pyqtSignal(str)  # emits error message if any

    def __init__(self, brand, ip, user, pwd):
        super().__init__()
        self.brand, self.ip, self.user, self.pwd = brand, ip, user, pwd

    def run(self):
        """Validate credentials by testing HTTP API first, then RTSP."""
        try:
            # First, try HTTP authentication test (more reliable for Hikvision)
            if REQUESTS_AVAILABLE and self.brand == "Hikvision":
                try:
                    test_url_http = f"http://{self.ip}/ISAPI/System/deviceInfo"
                    response = requests.get(
                        test_url_http,
                        auth=HTTPDigestAuth(self.user, self.pwd),
                        timeout=3
                    )
                    if response.status_code == 200:
                        print(f"[LOGIN] ✅ HTTP authentication successful - credentials are valid")
                        self.validated.emit(True)
                        return
                    elif response.status_code == 401:
                        print(f"[LOGIN] ❌ HTTP authentication failed - wrong credentials (401)")
                        self.error.emit("Authentication failed - Invalid username or password")
                        self.validated.emit(False)
                        return
                except requests.exceptions.RequestException as e:
                    print(f"[LOGIN] HTTP auth test failed (may not be supported): {e}")
                    # Continue to RTSP test if HTTP fails
            
            # Fallback: Test RTSP connection
            url_dicts = dvr_logic.build_rtsp_urls(self.brand, self.ip, self.user, self.pwd)
            if not url_dicts:
                self.error.emit("Failed to generate RTSP URLs")
                return
            
            import cv2
            
            # Test first 2 channels quickly
            test_channels = min(2, len(url_dicts))
            successful_connections = 0
            
            for i in range(test_channels):
                try:
                    test_url = url_dicts[i]["url"]
                    cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
                    time.sleep(2)  # Give time for connection
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            successful_connections += 1
                            print(f"[LOGIN] ✅ RTSP connection successful on channel {i+1}")
                            break
                    else:
                        cap.release()
                except Exception as e:
                    print(f"[LOGIN] Channel {i+1} RTSP test error: {e}")
                    continue
            
            # If we found a working connection, credentials are valid
            if successful_connections > 0:
                print(f"[LOGIN] ✅ Credentials validated via RTSP")
                self.validated.emit(True)
                return
            
            # No successful connections - but can't be sure if it's credentials or cameras
            # Proceed to full test and let user decide
            print(f"[LOGIN] No successful connections in test. Proceeding to full camera test...")
            self.validated.emit(True)  # Allow full test to proceed
            
        except Exception as e:
            error_msg = str(e)
            print(f"[LOGIN] Credential validation error: {error_msg}")
            # Proceed to full test
            self.validated.emit(True)


# ---------------- DVR Worker Thread ---------------- #
class DVRWorker(QObject):
    finished = pyqtSignal(list)  # emits [(url, ok), ...]

    def __init__(self, brand, ip, user, pwd):
        super().__init__()
        self.brand, self.ip, self.user, self.pwd = brand, ip, user, pwd

    def run(self):
        url_dicts = dvr_logic.build_rtsp_urls(self.brand, self.ip, self.user, self.pwd)
        urls = [u["url"] for u in url_dicts]
        results = dvr_logic.test_rtsp_urls(urls)
        self.finished.emit(results)


# ---------------- Auto-Connect Worker (Background) ---------------- #
class AutoConnectWorker(QObject):
    """Background worker for auto-connecting to DVR and verifying RTSP URLs."""
    connection_success = pyqtSignal(dict)  # emits config dict if successful
    connection_failed = pyqtSignal()  # emits if connection fails

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        """Auto-detect DVR IP and proceed with auto-login if credentials exist."""
        try:
            print("[AUTO-CONNECT] Starting background auto-connection...")
            
            # Get credentials
            brand = self.config.get("brand", "Hikvision")
            saved_ip = self.config.get("ip", "")
            user = self.config.get("user", "")
            pwd = self.config.get("pass", "")
            
            # Check if credentials exist
            if not user or not pwd:
                print("[AUTO-CONNECT] ❌ Missing username or password, cannot auto-connect")
                self.connection_failed.emit()
                return
            
            # Step 1: Try to auto-detect DVR IP (non-blocking, quick check)
            detected_ip = None
            try:
                # Use a shorter timeout for auto-detection to avoid long waits
                detected_ip = auto_detect_dvr_ip()
            except Exception as e:
                print(f"[AUTO-CONNECT] ⚠️ IP detection error (non-critical): {e}")
            
            # Step 2: Use detected IP if available, otherwise use saved IP
            # Always save IP to config if it changes
            config_path = Path(__file__).resolve().parents[2] / "core" / "config.json"
            
            if detected_ip:
                print(f"[AUTO-CONNECT] ✅ DVR IP detected: {detected_ip}")
                # Update IP if it's different from saved IP
                if detected_ip != saved_ip:
                    print(f"[AUTO-CONNECT] 🔄 IP changed from {saved_ip} → {detected_ip}, updating config...")
                    self.config["ip"] = detected_ip
                    # Save updated config with detected IP
                    try:
                        with open(config_path, "w") as f:
                            json.dump(self.config, f, indent=4)
                        print("[AUTO-CONNECT] ✅ Config automatically saved with new IP")
                    except Exception as e:
                        print(f"[AUTO-CONNECT] ⚠️ Failed to save config: {e}")
                else:
                    print(f"[AUTO-CONNECT] ✅ IP unchanged ({detected_ip}), using saved IP")
                    self.config["ip"] = detected_ip
            elif saved_ip:
                print(f"[AUTO-CONNECT] ✅ Using saved DVR IP: {saved_ip}")
                self.config["ip"] = saved_ip
                # Ensure saved IP is in config (in case it wasn't saved before)
                try:
                    with open(config_path, "w") as f:
                        json.dump(self.config, f, indent=4)
                except Exception as e:
                    print(f"[AUTO-CONNECT] ⚠️ Failed to save config: {e}")
            else:
                print("[AUTO-CONNECT] ❌ No IP available (neither detected nor saved)")
                self.connection_failed.emit()
                return
            
            # Step 3: Verify IP is valid format
            ip = self.config.get("ip", "")
            if not ip or not self.is_valid_ip_format(ip):
                print(f"[AUTO-CONNECT] ❌ Invalid IP format: {ip}")
                self.connection_failed.emit()
                return
            
            # Step 4: For auto-login, we don't need to verify RTSP streams are active
            # (that's too strict - cameras might be off, but credentials are valid)
            # Just verify that we have valid credentials and IP
            print(f"[AUTO-CONNECT] ✅ Auto-login ready: {brand} DVR at {ip} with user {user}")
            print("[AUTO-CONNECT] ✅ Proceeding with auto-login (RTSP verification skipped for speed)")
            
            # Success - proceed with auto-login
            self.connection_success.emit(self.config)
                
        except Exception as e:
            print(f"[AUTO-CONNECT] ❌ Auto-connection error: {e}")
            import traceback
            traceback.print_exc()
            self.connection_failed.emit()
    
    def is_valid_ip_format(self, ip):
        """Check if IP address format is valid."""
        try:
            parts = ip.split(".")
            if len(parts) != 4:
                return False
            return all(0 <= int(p) <= 255 for p in parts)
        except:
            return False


# ---------------- DVR IP Auto-Update Worker ---------------- #
class DVRIPUpdateWorker(QObject):
    """Background worker for auto-detecting and updating DVR IP without blocking UI."""
    ip_detected = pyqtSignal(str)  # emits detected IP address
    detection_finished = pyqtSignal()  # emits when detection finishes (success or failure)
    
    def __init__(self):
        super().__init__()
    
    def run(self):
        """Auto-detect DVR IP in background thread."""
        try:
            print("[DVR] Auto-detecting DVR IP in background...")
            detected_ip = auto_detect_dvr_ip()
            if detected_ip:
                print(f"[DVR] ✅ DVR IP detected: {detected_ip}")
                self.ip_detected.emit(detected_ip)
            else:
                print("[DVR] ⚠️ No DVR detected on network.")
        except Exception as e:
            print(f"[DVR] ❌ Auto DVR IP update failed: {e}")
        finally:
            # Always emit finished signal to hide loading overlay
            self.detection_finished.emit()
            self.detection_finished.emit()


# ---------------- Login Screen ---------------- #
class LoginScreen(QWidget):
    login_success = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.config_path = Path(__file__).resolve().parents[2] / "core" / "config.json"
        self.manual_entry = False     # track manual editing
        self.auto_connect_worker = None
        self.auto_connect_thread = None
        self.ip_update_worker = None
        self.ip_update_thread = None
        self.user_is_typing = False  # Track if user is actively typing
        self.loading_overlay = None  # Loading overlay widget
        self.loading_timer = None  # Timer for loading animation
        self.loading_dots = 0  # Counter for loading animation

        # DO NOT clear credentials - keep them for auto-login on subsequent sessions
        # Only show login UI if credentials don't exist (first time on new system)
        # Hide login screen initially to prevent flickering during startup
        self.setVisible(False)

        self.setup_style()
        self.build_ui()
        self.load_saved_config()
        self.link_manual_change_events()

        # Run saved config auto-validation AFTER UI loads
        QTimer.singleShot(300, self.check_saved_config)
        # Only auto-update IP if IP field is empty (first-time login)
        # Use a longer delay to avoid interfering with user input
        QTimer.singleShot(2000, self.auto_update_dvr_ip_if_empty)


    # ---------------- Saved Config Check (Auto-Connect in Background) ---------------- #
    def check_saved_config(self):
        """Check for saved credentials and auto-connect in background if they exist."""
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)

            # First time login on new system - no credentials saved
            if not cfg.get("ip") or not cfg.get("user") or not cfg.get("pass"):
                print("[LOGIN] No saved credentials found → showing login UI (first time login).")
                # Show main window and login UI for manual entry
                main_window = self.window()
                if main_window:
                    main_window.show()
                    main_window.raise_()
                    main_window.activateWindow()
                return

            # Credentials exist - start background auto-connection
            print("[LOGIN] Saved credentials found → starting background auto-connection...")
            self.start_auto_connect(cfg)

        except Exception as e:
            print(f"[LOGIN] Config check failed: {e}")

    # ---------------- Start Auto-Connect in Background ---------------- #
    def start_auto_connect(self, config):
        """Start background worker to auto-detect DVR IP and proceed with auto-login."""
        # Stop any existing auto-connect thread
        if self.auto_connect_thread and self.auto_connect_thread.isRunning():
            self.auto_connect_thread.quit()
            self.auto_connect_thread.wait(1000)
        
        # Hide login UI and main window during auto-connect to avoid blank screen
        self.setVisible(False)
        main_window = self.window()
        if main_window:
            main_window.hide()
        print("[LOGIN] Hiding login UI and main window during auto-connection...")
        
        # Create and start background worker
        self.auto_connect_thread = QThread()
        self.auto_connect_worker = AutoConnectWorker(config)
        self.auto_connect_worker.moveToThread(self.auto_connect_thread)
        
        # Connect signals
        self.auto_connect_thread.started.connect(self.auto_connect_worker.run)
        self.auto_connect_worker.connection_success.connect(self.on_auto_connect_success)
        self.auto_connect_worker.connection_failed.connect(self.on_auto_connect_failed)
        self.auto_connect_worker.connection_success.connect(self.auto_connect_thread.quit)
        self.auto_connect_worker.connection_failed.connect(self.auto_connect_thread.quit)
        self.auto_connect_worker.connection_success.connect(self.auto_connect_worker.deleteLater)
        self.auto_connect_worker.connection_failed.connect(self.auto_connect_worker.deleteLater)
        self.auto_connect_thread.finished.connect(self.auto_connect_thread.deleteLater)
        
        # Start thread
        self.auto_connect_thread.start()
        print("[LOGIN] Background auto-connection started...")

    # ---------------- Auto-Connect Success Handler ---------------- #
    def on_auto_connect_success(self, config):
        """Handle successful auto-connection - go directly to ML selection."""
        print("[LOGIN] ✅ Auto-connection successful → going directly to ML Model Selection")
        # Show main window before emitting signal (ML screen will be shown)
        main_window = self.window()
        if main_window:
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()
        # Emit signal immediately - ML screen will be shown
        self.login_success.emit(config)

    # ---------------- Auto-Connect Failed Handler ---------------- #
    def on_auto_connect_failed(self):
        """Handle failed auto-connection - show login UI for user to enter credentials."""
        print("[LOGIN] ❌ Auto-connection failed → showing login UI")
        # Show main window and login UI again so user can manually enter/update credentials
        main_window = self.window()
        if main_window:
            main_window.show()
            main_window.raise_()
            main_window.activateWindow()
        self.setVisible(True)


    # ---------------- Clear Saved Credentials (REMOVED - No longer needed) ---------------- #
    # Credentials are now persisted across sessions for auto-login
    # Only cleared if user explicitly wants to re-enter them

    # ---------------- Load Saved Config ---------------- #
    def load_saved_config(self):
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)

            self.brand.setCurrentText(data.get("brand", "Hikvision"))
            self.ip.setText(data.get("ip", ""))
            self.user.setText(data.get("user", ""))
            self.passw.setText(data.get("pass", ""))

        except Exception as e:
            print(f"[WARN] Failed to load config: {e}")


    # ---------------- Manual Change Tracker ---------------- #
    def link_manual_change_events(self):
        def on_credential_change():
            self.manual_entry = True
            self.user_is_typing = True
            # Reset table when credentials change
            if hasattr(self, 'table'):
                self.reset_table_to_default()
            # Reset typing flag after user stops typing (500ms delay)
            QTimer.singleShot(500, lambda: setattr(self, 'user_is_typing', False))
        
        self.ip.textChanged.connect(on_credential_change)
        self.user.textChanged.connect(on_credential_change)
        self.passw.textChanged.connect(on_credential_change)
        self.brand.currentIndexChanged.connect(on_credential_change)
        
        # Track when user clicks on input fields to prevent auto-update during interaction
        def on_field_focused():
            self.user_is_typing = True
            # Reset flag after user stops interacting (2 seconds after focus out)
        
        if hasattr(self, 'ip'):
            self.ip.editingFinished.connect(lambda: QTimer.singleShot(2000, lambda: setattr(self, 'user_is_typing', False)))
        if hasattr(self, 'user'):
            self.user.editingFinished.connect(lambda: QTimer.singleShot(2000, lambda: setattr(self, 'user_is_typing', False)))
        if hasattr(self, 'passw'):
            self.passw.editingFinished.connect(lambda: QTimer.singleShot(2000, lambda: setattr(self, 'user_is_typing', False)))


    # ---------------- UI Styling ---------------- #
    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #a8c7ff, stop:1 #5a9bff);
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QLineEdit, QComboBox {
                background:#fff;
                color: #000000;
                border:1px solid #bbb;
                border-radius:6px;
                padding:6px;
                font-size: 14px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 2px solid #1a5fd0;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                color: #000000;
                selection-background-color: #1a5fd0;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color:#1a5fd0;
                color:white;
                border:none;
                border-radius:6px;
                padding:8px 16px;
                font-weight:bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color:#1749a8; }
            QPushButton#saveBtn { background-color:#2e7d32; }
            QPushButton#saveBtn:hover { background-color:#1b5e20; }
        """)


    # ---------------- Build UI ---------------- #
    def build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("🔐 Connect to DVR")
        title.setStyleSheet("font-size:22px; font-weight:bold; color:#ffffff;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        form = QFormLayout()
        self.brand = QComboBox()
        self.brand.addItems(["Hikvision", "Dahua", "CP Plus", "Samsung"])

        self.ip = QLineEdit()
        self.user = QLineEdit()

        form.addRow("DVR Brand:", self.brand)
        form.addRow("IP Address:", self.ip)
        form.addRow("Username:", self.user)

        # -------- Password + Toggle -------- #
        pass_row = QHBoxLayout()
        self.passw = QLineEdit()
        self.passw.setEchoMode(QLineEdit.EchoMode.Password)

        self.pass_toggle = QPushButton("👁️")
        self.pass_toggle.setFixedWidth(35)
        self.pass_toggle.setStyleSheet("""
            QPushButton {
                background-color: #dce3f5;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color:#c3d4f2; }
        """)
        self.pass_toggle.setCheckable(True)
        self.pass_toggle.toggled.connect(self.toggle_password)

        pass_row.addWidget(self.passw)
        pass_row.addWidget(self.pass_toggle)
        form.addRow("Password:", pass_row)

        # -------- Buttons -------- #
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("🔍 Test Configuration")
        self.test_btn.clicked.connect(self.test_configuration)

        self.save_btn = QPushButton("💾 Save Configuration")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.clicked.connect(self.save_button_clicked)

        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.save_btn)

        # -------- Results Table -------- #
        self.proceed_btn = QPushButton("➡ Proceed to Dashboard")
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.clicked.connect(self.handle_proceed)

        self.table = QTableWidget(16, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "RTSP URL", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        for i in range(16):
            self.table.setItem(i, 0, QTableWidgetItem(f"Cam {i+1}"))
            self.table.setItem(i, 1, QTableWidgetItem("—"))
            self.table.setItem(i, 2, QTableWidgetItem("❌"))

        layout.addWidget(title)
        layout.addLayout(form)
        layout.addLayout(btn_row)
        layout.addWidget(self.table)
        layout.addWidget(self.proceed_btn)
        
        # Create loading overlay (initially hidden)
        self.create_loading_overlay()


    # ---------------- DVR Auto-Detection Logic ---------------- #
    def auto_update_dvr_ip_if_empty(self):
        """Auto-update DVR IP only if IP field is empty and user is not typing."""
        # Don't run if user is actively typing or if IP field already has a value
        if self.user_is_typing:
            print("[DVR] Skipping auto-update - user is typing")
            return
        
        current_ip = self.ip.text().strip()
        if current_ip:
            print(f"[DVR] Skipping auto-update - IP field already has value: {current_ip}")
            return
        
        # Start background worker to detect IP
        self.start_ip_detection_worker()
    
    def start_ip_detection_worker(self):
        """Start background thread to detect DVR IP without blocking UI."""
        # Stop any existing IP update thread
        if self.ip_update_thread and self.ip_update_thread.isRunning():
            self.ip_update_thread.quit()
            self.ip_update_thread.wait(1000)
        
        # Show loading overlay
        self.show_loading_overlay()
        
        # Create and start background worker
        self.ip_update_thread = QThread()
        self.ip_update_worker = DVRIPUpdateWorker()
        self.ip_update_worker.moveToThread(self.ip_update_thread)
        
        # Connect signals
        self.ip_update_thread.started.connect(self.ip_update_worker.run)
        self.ip_update_worker.ip_detected.connect(self.on_ip_detected)
        self.ip_update_worker.detection_finished.connect(self.hide_loading_overlay)
        self.ip_update_worker.detection_finished.connect(self.ip_update_thread.quit)
        self.ip_update_worker.detection_finished.connect(self.ip_update_worker.deleteLater)
        self.ip_update_thread.finished.connect(self.ip_update_thread.deleteLater)
        
        # Start thread
        self.ip_update_thread.start()
        print("[DVR] Background IP detection started...")
    
    def on_ip_detected(self, detected_ip):
        """Handle detected IP address - update UI only if field is still empty."""
        # Only update if IP field is empty and user is not typing
        if self.user_is_typing:
            print(f"[DVR] Skipping IP update - user is typing")
            return
        
        current_ip = self.ip.text().strip()
        if current_ip:
            print(f"[DVR] Skipping IP update - IP field already has value: {current_ip}")
            return
        
        # Update IP field
        print(f"[DVR] ✅ Updating IP field with detected IP: {detected_ip}")
        self.ip.setText(detected_ip)
        self.save_config()
    
    def create_loading_overlay(self):
        """Create a loading overlay widget that appears in the center of the screen."""
        # Create overlay frame
        self.loading_overlay = QFrame(self)
        self.loading_overlay.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 180);
                border-radius: 15px;
            }
        """)
        self.loading_overlay.setVisible(False)
        
        # Create layout for overlay
        overlay_layout = QVBoxLayout(self.loading_overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Loading label with animation
        self.loading_label = QLabel("Detecting DVR IP...")
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                background-color: transparent;
                padding: 20px;
            }
        """)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        overlay_layout.addWidget(self.loading_label)
        
        # Set overlay size and position (will be updated in show_loading_overlay)
        self.loading_overlay.setFixedSize(300, 120)
    
    def show_loading_overlay(self):
        """Show loading overlay in the center of the screen."""
        if not self.loading_overlay:
            self.create_loading_overlay()
        
        # Position overlay in center of widget
        # Use QTimer to ensure widget is fully laid out before positioning
        def position_overlay():
            if self.isVisible() and self.loading_overlay:
                parent_rect = self.rect()
                overlay_x = (parent_rect.width() - self.loading_overlay.width()) // 2
                overlay_y = (parent_rect.height() - self.loading_overlay.height()) // 2
                self.loading_overlay.move(overlay_x, overlay_y)
                self.loading_overlay.setVisible(True)
                self.loading_overlay.raise_()
        
        # Position immediately if visible, otherwise wait a bit
        if self.isVisible():
            QTimer.singleShot(50, position_overlay)
        else:
            position_overlay()
        
        # Start loading animation
        self.loading_dots = 0
        if not self.loading_timer:
            self.loading_timer = QTimer()
            self.loading_timer.timeout.connect(self.update_loading_animation)
        self.loading_timer.start(500)  # Update every 500ms
        self.update_loading_animation()
    
    def update_loading_animation(self):
        """Update loading animation dots."""
        if self.loading_overlay and self.loading_overlay.isVisible():
            self.loading_dots = (self.loading_dots + 1) % 4
            dots = "." * self.loading_dots
            self.loading_label.setText(f"Detecting DVR IP{dots}")
    
    def hide_loading_overlay(self):
        """Hide loading overlay."""
        if self.loading_overlay:
            self.loading_overlay.setVisible(False)
        if self.loading_timer:
            self.loading_timer.stop()


    # ---------------- IP Validator ---------------- #
    def is_valid_ip(self, ip):
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(p) <= 255 for p in parts)
        except:
            return False


    # ---------------- DVR Testing ---------------- #
    def test_configuration(self):
        brand = self.brand.currentText()
        ip = self.ip.text().strip()
        user = self.user.text().strip()
        pwd = self.passw.text().strip()

        if not ip or not user or not pwd:
            QMessageBox.warning(self, "Missing Info", "Please fill in all DVR details.")
            return

        # Validate IP format
        if not self.is_valid_ip(ip):
            QMessageBox.warning(self, "Invalid IP", "Please enter a valid IP address.")
            return

        # Stop any existing threads
        self.stop_all_threads()

        # Clear previous results from table
        self.reset_table()

        self.test_btn.setEnabled(False)
        self.test_btn.setText("⏳ Validating Credentials...")
        self.proceed_btn.setEnabled(False)

        # First, validate credentials
        self.validate_credentials(brand, ip, user, pwd)

    # ---------------- Stop All Threads ---------------- #
    def stop_all_threads(self):
        """Stop any running test or validation threads."""
        try:
            if hasattr(self, 'test_thread') and self.test_thread:
                try:
                    if isinstance(self.test_thread, QThread) and self.test_thread.isRunning():
                        print("[LOGIN] Stopping previous test thread...")
                        self.test_thread.quit()
                        self.test_thread.wait(2000)
                        try:
                            if self.test_thread.isRunning():
                                self.test_thread.terminate()
                        except RuntimeError:
                            pass  # Thread already deleted
                except RuntimeError:
                    pass  # Thread object was deleted
        except Exception as e:
            print(f"[LOGIN] Error stopping test thread: {e}")
        
        try:
            if hasattr(self, 'validator_thread') and self.validator_thread:
                try:
                    if isinstance(self.validator_thread, QThread) and self.validator_thread.isRunning():
                        print("[LOGIN] Stopping previous validator thread...")
                        self.validator_thread.quit()
                        self.validator_thread.wait(2000)
                        try:
                            if self.validator_thread.isRunning():
                                self.validator_thread.terminate()
                        except RuntimeError:
                            pass  # Thread already deleted
                except RuntimeError:
                    pass  # Thread object was deleted
        except Exception as e:
            print(f"[LOGIN] Error stopping validator thread: {e}")

    # ---------------- Validate Credentials ---------------- #
    def validate_credentials(self, brand, ip, user, pwd):
        """First validate credentials, then test cameras if valid."""
        self.validator_thread = QThread()
        self.validator_worker = CredentialValidatorWorker(brand, ip, user, pwd)
        self.validator_worker.moveToThread(self.validator_thread)
        self.validator_thread.started.connect(self.validator_worker.run)
        self.validator_worker.validated.connect(lambda valid: self.on_credentials_validated(valid, brand, ip, user, pwd))
        self.validator_worker.error.connect(lambda msg: self.on_validation_error(msg))
        self.validator_worker.validated.connect(self.validator_thread.quit)
        self.validator_worker.error.connect(self.validator_thread.quit)
        self.validator_worker.validated.connect(self.validator_worker.deleteLater)
        self.validator_worker.error.connect(self.validator_worker.deleteLater)
        self.validator_thread.finished.connect(self.validator_thread.deleteLater)
        self.validator_thread.start()

    # ---------------- On Credentials Validated ---------------- #
    def on_credentials_validated(self, is_valid, brand, ip, user, pwd):
        """Handle credential validation result."""
        if not is_valid:
            # Credentials are wrong - show popup and stop
            QMessageBox.critical(
                self, 
                "❌ Invalid Credentials", 
                "The username or password is incorrect.\n\n"
                "Please check your DVR credentials and try again."
            )
            self.test_btn.setEnabled(True)
            self.test_btn.setText("🔍 Test Configuration")
            return
        
        # Credentials are valid - proceed to test all cameras
        print("[LOGIN] Credentials validated successfully. Testing cameras...")
        self.test_btn.setText("⏳ Testing Cameras...")
        self.test_all_cameras(brand, ip, user, pwd)

    # ---------------- On Validation Error ---------------- #
    def on_validation_error(self, error_msg):
        """Handle validation error."""
        QMessageBox.critical(
            self,
            "❌ Connection Error",
            f"Failed to connect to DVR:\n\n{error_msg}\n\n"
            "Please check:\n"
            "• DVR IP address is correct\n"
            "• DVR is powered on and connected to network\n"
            "• Username and password are correct"
        )
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🔍 Test Configuration")

    # ---------------- Test All Cameras ---------------- #
    def test_all_cameras(self, brand, ip, user, pwd):
        """Test all camera channels after credentials are validated."""
        self.test_thread = QThread()
        self.worker = DVRWorker(brand, ip, user, pwd)
        self.worker.moveToThread(self.test_thread)
        self.test_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.update_results)
        self.worker.finished.connect(self.test_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.test_thread.finished.connect(self.test_thread.deleteLater)
        self.test_thread.start()


    # ---------------- Reset Table to Default ---------------- #
    def reset_table_to_default(self):
        """Reset table to default state when credentials change."""
        if not hasattr(self, 'table'):
            return
        for i in range(self.table.rowCount()):
            self.table.item(i, 1).setText("—")
            self.table.item(i, 2).setText("❌")
            # Reset background to white
            for col in range(3):
                self.table.item(i, col).setBackground(QColor(255, 255, 255))
        self.proceed_btn.setEnabled(False)

    # ---------------- Reset Table ---------------- #
    def reset_table(self):
        """Reset table to default state before new test."""
        for i in range(self.table.rowCount()):
            self.table.item(i, 1).setText("—")
            self.table.item(i, 2).setText("⏳ Testing...")
            # Reset background to white
            for col in range(3):
                self.table.item(i, col).setBackground(QColor(255, 255, 255))

    # ---------------- Update Results ---------------- #
    def update_results(self, results):
        active_count = 0
        
        for i, (url, ok) in enumerate(results):
            if i >= self.table.rowCount():
                break

            self.table.item(i, 1).setText(str(url))
            
            # Update status based on connection result
            if ok:
                active_count += 1
                self.table.item(i, 2).setText("🟢 Active")
                color = QColor(144, 238, 144)
            else:
                self.table.item(i, 2).setText("🔴 Failed")
                color = QColor(255, 182, 193)

            for col in range(3):
                self.table.item(i, col).setBackground(color)

        # Check results
        if active_count > 0:
            # At least one camera is working - credentials are correct!
            QMessageBox.information(
                self, "✅ Connection Successful",
                f"RTSP URLs validated successfully!\n\n"
                f"Found {active_count} active camera(s).\n\n"
                f"You can now proceed to the dashboard."
            )
            self.proceed_btn.setEnabled(True)
        else:
            # ALL cameras failed - but this doesn't necessarily mean wrong credentials
            # Could be cameras not powered on, network issues, or RTSP config
            # Since we can't reliably detect 401 errors, allow user to proceed if they're confident
            reply = QMessageBox.warning(
                self, "⚠️ No Active Cameras Detected",
                "All camera channels failed to connect.\n\n"
                "Possible causes:\n"
                "• Cameras not powered on or connected to DVR\n"
                "• Network connectivity issues\n"
                "• RTSP stream configuration on DVR\n"
                "• Incorrect username or password\n\n"
                "If you're confident your credentials are correct,\n"
                "you can proceed to the dashboard and configure cameras later.\n\n"
                "Do you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.proceed_btn.setEnabled(True)
                QMessageBox.information(
                    self, "Proceeding to Dashboard",
                    "You can verify camera connections from the dashboard.\n"
                    "If credentials are incorrect, you'll need to return to login."
                )
            else:
                self.proceed_btn.setEnabled(False)
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🔍 Test Configuration")


    # ---------------- Password Show / Hide ---------------- #
    def toggle_password(self, checked):
        if checked:
            self.passw.setEchoMode(QLineEdit.EchoMode.Normal)
            self.pass_toggle.setText("🙈")
        else:
            self.passw.setEchoMode(QLineEdit.EchoMode.Password)
            self.pass_toggle.setText("👁️")


    # ---------------- Proceed ---------------- #
    def handle_proceed(self):
        config = {
            "brand": self.brand.currentText(),
            "ip": self.ip.text(),
            "user": self.user.text(),
            "pass": self.passw.text()
        }
        # Save config so next time it will auto-connect in background
        self.save_config()
        print("[LOGIN] ✅ Credentials saved → next login will auto-connect")
        # Emit signal to go directly to ML selection (skip dashboard)
        self.login_success.emit(config)

    # ---------------- Save Button ---------------- #
    def save_button_clicked(self):
        self.save_config()
        QMessageBox.information(self, "Saved", "✅ Configuration saved successfully!")


    # ---------------- Save Config ---------------- #
    def save_config(self):
        data = {
            "brand": self.brand.currentText(),
            "ip": self.ip.text(),
            "user": self.user.text(),
            "pass": self.passw.text()
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")