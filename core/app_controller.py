#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire – App Controller (Fade + Parallel ML Models + Status Bar)
-----------------------------------------------------------------------
Controls navigation between:
  1. Login Screen
  2. Dashboard Screen
  3. ML Model Selection
  4. Per-Model Dashboards (Crowd / Heatmap / Visitors / Emp–Cus / Motion)
  ✅ Auto-skip login if saved DVR config is valid
  ✅ Auto-retest DVR if IP changed
  ✅ All models run in parallel safely
  ✅ Bottom Status Bar: live ML activity + system stats (FPS, GPU/CPU temp)
"""
import json, threading, time
from datetime import datetime, timedelta

import os, sys, psutil
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QGraphicsOpacityEffect, QApplication,
    QStatusBar, QLabel
)
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QTimer
from PyQt6.QtGui import QColor

# --- Core Screens ---
from modules.login.login_screen import LoginScreen
from modules.dashboard.dashboard_screen import DashboardScreen
from modules.analytics.ml_selection_screen import MLSelectionScreen

# --- System Monitor ---
from modules.utils.gpu_monitor import get_gpu_temp, get_cpu_temp, get_gpu_fps

# --- ML Workers ---
from modules.crowd_density.crowd_density import CrowdDensityWorker
# HeatmapProcessor from heatmap_screen (heatmap_module deleted)
from modules.heatmap.heatmap_screen import HeatmapProcessor

# --- Config Manager for Auto-Skip ---
from core.config_manager import load_config, validate_config, auto_retest_dvr
from modules.utils.dvr_logic import auto_detect_dvr_ip
from datetime import datetime

import os, cv2

# For better parallel threading (safe, no flow changes)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

cv2.setUseOptimized(True)
cv2.setNumThreads(1)

class AppController(QMainWindow):
    """Central window switching between screens & managing threads."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Third Umpire Analytics Suite")
        self.resize(1280, 720)

        # Stack manager
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.setCurrentWidget = self.stack.setCurrentWidget

        # Login Screen
        self.login_screen = LoginScreen()
        # After first login, go directly to ML selection (skip dashboard)
        self.login_screen.login_success.connect(self.load_ml_selection_direct)
        self.stack.addWidget(self.login_screen)

        self.fade_duration = 500  # ms

        # --------------------------------------
        # Safe Recording Manager Initialization
        # --------------------------------------
        try:
            from core.unified_recording_manager import UnifiedRecordingManager
            self.recording_manager = UnifiedRecordingManager()

        except Exception as e:
            print("[WARN] RecordingManager not available:", e)
            self.recording_manager = None


        # Thread handles
        self.crowd_worker = None
        self.heatmap_worker = None
        # Alias used by some screens/settings for consistency
        self.heatmap_processor = None
        self.visitors_worker = None
        self.emp_cus_worker = None
        self.motion_worker = None
        
        # ✅ Auto-create empty selection JSON files if missing
        for name in [
            "selected_crowd_density.json",
            "selected_heatmap.json",
            "selected_visitors.json",
            "selected_emp_cus.json",
            "selected_motion.json",
        ]:
            path = os.path.join(os.getcwd(), name)
            if not os.path.isfile(path):
                with open(path, "w") as f:
                    f.write('{"names": []}')
                print(f"[INIT] Created missing file: {name}")
        # ==================================================
        # Bottom Status Bar Setup
        # ==================================================
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_crowd = QLabel("🔴 Crowd Density")
        self.lbl_heatmap = QLabel("🔴 Heatmap")
        self.lbl_visitors = QLabel("🔴 Daily Visitors")
        self.lbl_empcus = QLabel("🔴 Emp–Cus Interaction")
        self.lbl_motion = QLabel("🔴 Motion Detection")
        self.lbl_sysinfo = QLabel("GPU: --°C | CPU: --°C | FPS: --")

        for lbl in [
            self.lbl_crowd,
            self.lbl_heatmap,
            self.lbl_visitors,
            self.lbl_empcus,
            self.lbl_motion,
            self.lbl_sysinfo,
        ]:
            lbl.setStyleSheet("font-size:14px; font-weight:bold; padding:4px 10px;")

        self.status_bar.addPermanentWidget(self.lbl_crowd)
        self.status_bar.addPermanentWidget(self.lbl_heatmap)
        self.status_bar.addPermanentWidget(self.lbl_visitors)
        self.status_bar.addPermanentWidget(self.lbl_empcus)
        self.status_bar.addPermanentWidget(self.lbl_motion)
        self.status_bar.addPermanentWidget(self.lbl_sysinfo)

        # Auto refresh timer for status bar
        # Don't start until after login - keep buttons red before login
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_bar)
        # Status timer will be started after login in load_dashboard()

        # --- Periodic System Metrics Logger (every 5 min) ---
        from modules.utils.system_monitor import log_system_metrics
        self.metric_timer = QTimer(self)
        self.metric_timer.timeout.connect(log_system_metrics)
        self.metric_timer.start(300000)

        # --- Auto Scheduler (DISABLED - models only start when user launches them) ---
        # Scheduler is disabled - models will only start when user explicitly clicks to launch them
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.timeout.connect(self.check_model_schedule)
        # Scheduler will NOT start - models only run when user launches them
        print("[SCHEDULER] Auto model scheduler disabled - models only start when user launches them.")

        # ==================================================
        # === 🧩 AUTO-FLOW: Let Login Screen Handle Auto-Connect ===
        # ==================================================
        # Login screen will check for saved credentials and auto-connect in background
        # If credentials exist and connection succeeds, it will emit login_success
        # which goes directly to ML selection. If no credentials or connection fails,
        # login UI will be shown for user to enter credentials.
        # Hide window initially to prevent flickering during startup/auto-connect
        self.hide()
        print("[APP] Window hidden during startup - will show after auto-connect or if login needed.")
        self.stack.setCurrentWidget(self.login_screen)
        
        # Start DVR Monitor immediately (even during login) to auto-detect IP changes
        # This ensures IP is always up-to-date in the configuration
        self.dvr_monitor = DVRMonitor(controller=self)
        self.dvr_monitor.start()
        print("[APP] ✅ DVR IP monitor started - will auto-update config when IP changes")

    # ==================================================
    # Fade Helper
    # ==================================================
    def fade_to_widget(self, widget):
        """Smooth fade-in transition for widgets."""
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(self.fade_duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._current_animation = anim
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        widget.show()

    # ==================================================
    # Safe Stop All
    # ==================================================
    def safe_stop_all(self):
        """Stop all timers and threads safely."""
        for t in [self.status_timer, self.metric_timer, self.scheduler_timer]:
            try:
                if t.isActive():
                    t.stop()
            except Exception:
                pass

        if self.recording_manager:
            try:
                self.recording_manager.stop_recording("Crowd Density")
                self.recording_manager.stop_recording("HeatMap")
                self.recording_manager.stop_recording("Daily Visitors")
                self.recording_manager.stop_recording("Employee–Customer Interaction")
                self.recording_manager.stop_recording("Motion Detection")
            except Exception as e:
                print("[WARN] Recording stop error:", e)

        # Stop individual workers safely
        for worker in [
            self.crowd_worker,
            self.heatmap_worker,
            self.visitors_worker,
            self.emp_cus_worker,
        ]:
            try:
                if worker and worker.isRunning():
                    worker.stop()
                    worker.wait(2000)
                    worker.deleteLater()
            except Exception as e:
                print(f"[WARN] Safe stop failed for worker: {e}")

        # ✅ Handle Motion Detection (list of threads)
        if isinstance(self.motion_worker, list):
            for w in self.motion_worker:
                try:
                    if w.isRunning():
                        w.stop()
                        w.wait(2000)
                        w.deleteLater()
                except Exception as e:
                    print(f"[WARN] Safe stop failed for Motion worker: {e}")
        else:
            try:
                if self.motion_worker and self.motion_worker.isRunning():
                    self.motion_worker.stop()
                    self.motion_worker.wait(2000)
                    self.motion_worker.deleteLater()
            except Exception as e:
                print(f"[WARN] Safe stop failed for Motion worker: {e}")


    # ==================================================
    # Dashboard Loader
    # ==================================================
    def load_dashboard(self, config):
        print("[INFO] Login success:", config)
        
        # ✅ Start scheduler only after successful login
        if not self.scheduler_timer.isActive():
            self.scheduler_timer.start(60000)
            print("[SCHEDULER] Auto model scheduler started after login.")
        
        # ✅ Start status bar timer after login
        if not self.status_timer.isActive():
            self.status_timer.start(2000)
            print("[STATUS] Status bar timer started after login.")
        
        # ✅ Start 5-min DVR IP monitor
        self.dvr_monitor = DVRMonitor(controller=self)
        self.dvr_monitor.last_ip = config.get("ip")
        self.dvr_monitor.start()

        try:
            # Use dvr_logic to build RTSP URLs based on brand
            from modules.utils import dvr_logic
            brand = config.get('brand', 'Hikvision')
            url_dicts = dvr_logic.build_rtsp_urls(
                brand, 
                config['ip'], 
                config['user'], 
                config['pass'], 
                max_channels=8
            )
            self.active_cams = [
                {
                    "cam_name": item["name"],
                    "url": item["url"]
                }
                for item in url_dicts[:8]  # Take first 8 cameras
            ]

            self.dashboard = DashboardScreen(self, config, self.active_cams)
            self.dashboard.back_to_login.connect(self.return_to_login)
            self.dashboard.go_next.connect(self.load_ml_selection)

            self.stack.addWidget(self.dashboard)
            self.stack.setCurrentWidget(self.dashboard)
            self.fade_to_widget(self.dashboard)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] Failed to load dashboard: {e}")

    # ==================================================
    def return_to_login(self):
        # Stop scheduler when returning to login
        if self.scheduler_timer.isActive():
            self.scheduler_timer.stop()
            print("[SCHEDULER] Auto model scheduler stopped (returned to login).")
        
        # Stop status bar timer and reset buttons to red
        if self.status_timer.isActive():
            self.status_timer.stop()
            print("[STATUS] Status bar timer stopped (returned to login).")
        
        # Stop all recordings
        self.recording_manager.stop_recording("Crowd Density")
        self.recording_manager.stop_recording("HeatMap")
        self.recording_manager.stop_recording("Daily Visitors")
        self.recording_manager.stop_recording("Employee–Customer Interaction")
        self.recording_manager.stop_recording("Motion Detection")
        
        # Reset all status buttons to red
        self.lbl_crowd.setText("🔴 Crowd Density")
        self.lbl_heatmap.setText("🔴 Heatmap")
        self.lbl_visitors.setText("🔴 Daily Visitors")
        self.lbl_empcus.setText("🔴 Emp–Cus Interaction")
        self.lbl_motion.setText("🔴 Motion Detection")
        
        if hasattr(self, "dashboard"):
            for worker in getattr(self.dashboard, "stream_threads", []):
                try:
                    worker.stop()
                except Exception:
                    pass
        self.stack.setCurrentWidget(self.login_screen)
        self.fade_to_widget(self.login_screen)
        print("[INFO] Returned to Login")

    # ==================================================
    def load_ml_selection_direct(self, config):
        """Load ML selection screen directly after login (skipping dashboard)."""
        # Set up active cameras for ML models
        try:
            # Use dvr_logic to build RTSP URLs based on brand
            from modules.utils import dvr_logic
            brand = config.get('brand', 'Hikvision')
            url_dicts = dvr_logic.build_rtsp_urls(
                brand, 
                config['ip'], 
                config['user'], 
                config['pass'], 
                max_channels=8
            )
            self.active_cams = [
                {
                    "cam_name": item["name"],
                    "url": item["url"]
                }
                for item in url_dicts[:8]  # Take first 8 cameras
            ]
            
            # Scheduler is disabled - models only start when user launches them
            # Don't start scheduler timer - models will only run when explicitly launched
            # if not self.scheduler_timer.isActive():
            #     self.scheduler_timer.start(60000)
            #     print("[SCHEDULER] Auto model scheduler started after login.")
            
            if not self.status_timer.isActive():
                self.status_timer.start(2000)
                print("[STATUS] Status bar timer started after login.")
            
            # Start DVR monitor
            self.dvr_monitor = DVRMonitor(controller=self)
            self.dvr_monitor.last_ip = config.get("ip")
            self.dvr_monitor.start()
            
        except Exception as e:
            print(f"[WARN] Failed to setup cameras: {e}")
        
        # Load ML selection screen FIRST (this will show it immediately)
        self.load_ml_selection()
        
        # Then hide login screen after ML screen is visible (prevents blank screen)
        if hasattr(self, 'login_screen'):
            self.login_screen.setVisible(False)
            self.login_screen.hide()
    
    # ==================================================
    def load_ml_selection(self):
        # Only create ML screen if it doesn't exist to avoid duplicates
        if not hasattr(self, 'ml_screen') or self.ml_screen is None:
            self.ml_screen = MLSelectionScreen(self.stack)
            self.ml_screen.model_selected.connect(self.handle_model_selection)
            # If dashboard exists, go back to it, otherwise go to login
            self.ml_screen.back_to_dashboard.connect(self.return_to_dashboard)
            self.stack.addWidget(self.ml_screen)
        
        # Switch to ML selection screen immediately - this makes it visible
        self.stack.setCurrentWidget(self.ml_screen)
        
        # Ensure ML screen is fully visible and active
        self.ml_screen.show()
        self.ml_screen.setVisible(True)
        self.ml_screen.raise_()
        self.ml_screen.activateWindow()
        
        # Ensure the main window is also visible and active (important for auto-boot)
        self.show()
        self.raise_()
        self.activateWindow()
        
        # Apply fade effect
        self.fade_to_widget(self.ml_screen)
        print("[INFO] ML Model Selection screen loaded and displayed immediately")

    def return_to_dashboard_or_login(self):
        """Return to dashboard if it exists, otherwise go to login."""
        if hasattr(self, 'dashboard') and self.dashboard:
            self.stack.setCurrentWidget(self.dashboard)
            self.fade_to_widget(self.dashboard)
            print("[INFO] Returned to Dashboard")
        else:
            # Dashboard doesn't exist (skipped), go to login
            self.return_to_login()
    
    def return_to_dashboard(self):
        if hasattr(self, 'dashboard') and self.dashboard:
            self.stack.setCurrentWidget(self.dashboard)
            self.fade_to_widget(self.dashboard)
            print("[INFO] Returned to Dashboard")

    def handle_model_selection(self, model_name):
        print(f"[INFO] Selected ML Model → {model_name}")

        try:
            # ==================================================
            # 1) CROWD DENSITY
            # ==================================================
            if model_name == "Crowd Density":
                from modules.crowd_density.crowd_density import CrowdDashboardScreen
                self.crowd_dashboard = CrowdDashboardScreen(self)
                self.stack.addWidget(self.crowd_dashboard)
                self.stack.setCurrentWidget(self.crowd_dashboard)
                self.fade_to_widget(self.crowd_dashboard)
                return

            # ==================================================
            # 2) HEATMAP
            # ==================================================
            elif model_name.lower() == "heatmap":
                from modules.heatmap.heatmap_screen import HeatmapDashboardScreen
                self.heatmap_dashboard = HeatmapDashboardScreen(self)
                self.stack.addWidget(self.heatmap_dashboard)
                self.stack.setCurrentWidget(self.heatmap_dashboard)
                self.fade_to_widget(self.heatmap_dashboard)
                return

            # ==================================================
            # 3) DAILY VISITORS
            # ==================================================
            elif model_name == "Daily Visitors":
                from modules.visitors_count.visitors_screen import VisitorsDashboardScreen
                self.visitors_dashboard = VisitorsDashboardScreen(self)
                self.stack.addWidget(self.visitors_dashboard)
                self.stack.setCurrentWidget(self.visitors_dashboard)
                self.fade_to_widget(self.visitors_dashboard)
                return

            # ==================================================
            # 4) EMP–CUS INTERACTION (MAIN ENTRY)
            # ==================================================
            elif model_name == "Emp–Cus Interaction":
                self.load_emp_cus_selection()
                return

            # ==================================================
            # 4A) EMP–CUS → EMPLOYEE BUTTON
            # ==================================================
            elif model_name == "EmpCus_EMPLOYEE":

                # Reuse existing EmployeeManagementScreen if running
                if hasattr(self, 'emp_cus_employee') and self.emp_cus_employee:
                    try:
                        if hasattr(self.emp_cus_employee, 'streams_running') and \
                        self.emp_cus_employee.streams_running:
                            print("[EMP-CUS] Showing existing EmployeeManagementScreen (no reload)")
                            self.stack.setCurrentWidget(self.emp_cus_employee)
                            self.fade_to_widget(self.emp_cus_employee)
                            return
                    except:
                        pass

                # Otherwise create new one
                from modules.emp_cus_interaction.emp_cus_interaction import EmployeeManagementScreen
                if not hasattr(self, 'emp_cus_employee') or not self.emp_cus_employee:
                    self.emp_cus_employee = EmployeeManagementScreen(self)
                    self.stack.addWidget(self.emp_cus_employee)

                self.stack.setCurrentWidget(self.emp_cus_employee)
                self.fade_to_widget(self.emp_cus_employee)
                return

            # ==================================================
            # 5) MOTION DETECTION
            # ==================================================
            elif model_name in ["Motion Detection", "Motion Detect"]:
                from modules.motion_detect.motion_detection import MotionDetectionScreen
                self.motion_screen = MotionDetectionScreen(self)
                self.stack.addWidget(self.motion_screen)
                self.stack.setCurrentWidget(self.motion_screen)
                self.fade_to_widget(self.motion_screen)
                return

            # ==================================================
            # UNKNOWN MODEL
            # ==================================================
            else:
                print(f"[WARN] Unknown ML model selected: {model_name}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] Failed to load ML model screen: {e}")

    # ==================================================
    # DVR Reconnect Logic (Triggered by DVRMonitor)
    # ==================================================
    def reconnect_dvr(self, new_ip):
        """Rebuild DVR URLs and refresh the dashboard when IP changes."""
        print(f"[DVR] 🔄 Reconnecting DVR with new IP: {new_ip}")

        try:
            # 1️⃣ Load saved config (use same path as login screen)
            config_path = get_config_path()
            cfg = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)

            # IP is already updated in config by DVRMonitor, but ensure it's set
            cfg["ip"] = new_ip
            
            # Save config to ensure it's persisted
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=4)
            print(f"[DVR] ✅ Config updated with new IP: {new_ip}")

            # Use dvr_logic to build RTSP URLs based on brand
            from modules.utils import dvr_logic
            brand = cfg.get('brand', 'Hikvision')
            url_dicts = dvr_logic.build_rtsp_urls(
                brand, 
                cfg['ip'], 
                cfg['user'], 
                cfg['pass'], 
                max_channels=8
            )
            self.active_cams = [
                {
                    "cam_name": item["name"],
                    "url": item["url"]
                }
                for item in url_dicts[:8]  # Take first 8 cameras
            ]

            # 3️⃣ Refresh dashboard if already loaded
            if hasattr(self, "dashboard"):
                print("[DVR] Reloading dashboard with new stream URLs…")
                self.dashboard.close()
                self.dashboard = DashboardScreen(self, cfg, self.active_cams)
                self.dashboard.back_to_login.connect(self.return_to_login)
                self.dashboard.go_next.connect(self.load_ml_selection)
                self.stack.addWidget(self.dashboard)
                self.stack.setCurrentWidget(self.dashboard)
                self.fade_to_widget(self.dashboard)
            else:
                print("[DVR] No dashboard yet (probably in login). Skipping reload.")
        except Exception as e:
            print(f"[DVR] ❌ Reconnect failed: {e}")


    # ==================================================
    def stop_if_running(self, worker, name):
        try:
            if worker and worker.isRunning():
                print(f"[INFO] Stopping previous {name} worker before restart…")
                worker.stop()
        except Exception as e:
            print(f"[WARN] Could not stop previous {name} worker: {e}")

    # ==================================================
    # ML Model Starters
    # ==================================================
    def start_crowd_density(self, urls, cam_names):
        try:
            if self.crowd_worker and self.crowd_worker.isRunning():
                print("[CROWD] Existing worker detected — stopping before restart.")
                self.crowd_worker.stop()
                self.crowd_worker.wait()
                # Stop recording for previous instance
                self.recording_manager.stop_recording("Crowd Density")

            print(f"[CROWD] Starting YOLO model for selected cameras → {cam_names}")
            self.crowd_worker = CrowdDensityWorker(urls, cam_names)
            self.crowd_worker.count_updated.connect(self.handle_crowd_update)
            self.crowd_worker.start()

            # ⭐ ONLY CROWD DENSITY RECORDING ENABLED
            if self.recording_manager:
                self.recording_manager.start_recording("Crowd Density", urls, cam_names)

            print("[CROWD] Worker successfully started.")
        except Exception as e:
            print(f"[CROWD] ❌ Failed to start crowd density worker: {e}")


    def start_heatmap(self, urls, cam_names):
        try:
            if self.heatmap_worker and self.heatmap_worker.isRunning():
                print("[HEATMAP] Existing worker detected — stopping before restart.")
                self.heatmap_worker.stop()
                self.heatmap_worker.wait()
                # Stop recording for previous instance
                if self.recording_manager:
                    self.recording_manager.stop_recording("HeatMap")

            print(f"[HEATMAP] Starting processor for selected cameras → {cam_names}")
            self.heatmap_worker = HeatmapProcessor(urls, cam_names)
            # Keep alias for settings/logout flows that expect `heatmap_processor`
            self.heatmap_processor = self.heatmap_worker
            self.heatmap_worker.start()
            
            # Start auto-recording
            #self.recording_manager.start_recording("HeatMap", urls, cam_names)
            
            print("[HEATMAP] Worker successfully started.")
        except Exception as e:
            print(f"[HEATMAP] ❌ Failed to start heatmap worker: {e}")

    def start_daily_visitors(self, urls, cam_names):
        from modules.visitors_count.daily_processor import DailyCountProcessor
        print(f"[INFO] Starting Daily Visitors Worker → {cam_names}")
        self.stop_if_running(self.visitors_worker, "Daily Visitors")
        # Stop recording for previous instance
        self.recording_manager.stop_recording("Daily Visitors")
        
        try:
            self.visitors_worker = DailyCountProcessor(urls, cam_names)
            # Connect signal to update status bar when thread actually starts
            self.visitors_worker.started_signal.connect(lambda: self.update_status_bar())
            self.visitors_worker.start()
            
            # Start auto-recording
           #self.recording_manager.start_recording("Daily Visitors", urls, cam_names)
            
            # Update status bar immediately
            self.update_status_bar()
            # Update again after delays to ensure thread has started
            QTimer.singleShot(200, self.update_status_bar)
            QTimer.singleShot(500, self.update_status_bar)
            QTimer.singleShot(1000, self.update_status_bar)
            # Debug: Check if worker is running
            print(f"[VISITORS] Worker created, isRunning(): {self.visitors_worker.isRunning()}")
            print(f"[VISITORS] Worker running attribute: {getattr(self.visitors_worker, 'running', 'N/A')}")
            print("[INFO] Daily Visitors worker started successfully")
        except Exception as e:
            print(f"[ERROR] Failed to start Daily Visitors: {e}")
            import traceback
            traceback.print_exc()

    def start_emp_cus_interaction(self, urls, cam_names):
        print(f"[INFO] Starting Emp–Cus Interaction → {cam_names}")
        print(f"[EMP-CUS] Received {len(urls) if urls else 0} URLs")
        if urls:
            print(f"[EMP-CUS] First URL: {urls[0]}")

        # ✅ ADDED: Reuse existing EmployeeManagementScreen if it exists and is running
        if hasattr(self, 'emp_cus_employee') and self.emp_cus_employee:
            try:
                # If streams are already running, just show the existing screen
                if hasattr(self.emp_cus_employee, 'streams_running') and self.emp_cus_employee.streams_running:
                    print("[EMP-CUS] Reusing existing EmployeeManagementScreen")
                    self.stack.setCurrentWidget(self.emp_cus_employee)
                    self.fade_to_widget(self.emp_cus_employee)
                    return
            except:
                pass

        # Stop previous screen (if any)
        try:
            # ✅ CHANGED: Added condition to exclude emp_cus_employee from cleanup
            if self.emp_cus_worker and self.emp_cus_worker != self.emp_cus_employee:
                print("[EMP-CUS] Closing previous screen…")
                self.emp_cus_worker.close()
                # Stop recording for previous instance
                self.recording_manager.stop_recording("Employee–Customer Interaction")
        except:
            pass

        try:
            # ✅ CHANGED: Import changed from EmpCusDashboardScreen to EmployeeManagementScreen
            from modules.emp_cus_interaction.emp_cus_interaction import EmployeeManagementScreen

            # ✅ CHANGED: Create or reuse EmployeeManagementScreen (the live ML screen) with URLs and camera names
            print(f"[EMP-CUS] Creating EmployeeManagementScreen with {len(urls) if urls else 0} URLs")
            if hasattr(self, 'emp_cus_employee') and self.emp_cus_employee:
                # Update URLs if screen exists but streams aren't running
                self.emp_cus_employee.provided_urls = urls
                self.emp_cus_employee.provided_cam_names = cam_names
            else:
                # ✅ CHANGED: Creates EmployeeManagementScreen instead of EmpCusDashboardScreen
                self.emp_cus_employee = EmployeeManagementScreen(self, urls=urls, cam_names=cam_names)
                self.stack.addWidget(self.emp_cus_employee)

            self.emp_cus_worker = self.emp_cus_employee

            # Show the screen
            self.stack.setCurrentWidget(self.emp_cus_employee)
            self.fade_to_widget(self.emp_cus_employee)

            print("[INFO] Emp–Cus Interaction screen loaded successfully")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[ERROR] Failed to start Emp–Cus Interaction: {e}")

    def start_motion_detection(self, urls, cam_names):
        from modules.motion_detect.motion_detection import MotionDetectWorker
        print(f"[INFO] Starting Motion Detection → {cam_names}")
        self.stop_if_running(self.motion_worker, "Motion Detection")
        # Stop recording for previous instance
        self.recording_manager.stop_recording("Motion Detection")

        try:
            # ✅ Fix: start one worker per camera, not a list
            self.motion_worker = []
            for i, url in enumerate(urls):
                worker = MotionDetectWorker(i, url)
                worker.start()
                self.motion_worker.append(worker)
            
            # Start auto-recording
            #self.recording_manager.start_recording("Motion Detection", urls, cam_names)
            
            self.update_status_bar()
            print(f"[INFO] Motion Detection workers started ({len(self.motion_worker)} streams)")
        except Exception as e:
            print(f"[ERROR] Failed to start Motion Detection: {e}")

    # ==================================================
    # EMP–CUS CAMERA SELECTION SCREEN LOADER
    # ==================================================
    def load_emp_cus_selection(self):
        print("[EMP-CUS] Opening cam selection...")

        # Destroy previous screen if exists
        if hasattr(self, 'emp_cus_screen'):
            try:
                self.emp_cus_screen.deleteLater()
            except:
                pass

        # Build camera list
        urls  = [c["url"] for c in self.active_cams]
        names = [c["cam_name"] for c in self.active_cams]

        # FIXED import path
        from modules.emp_cus_interaction.emp_cus_interaction import EmpCusDashboardScreen

        # Create heatmap-style preview screen
        self.emp_cus_screen = EmpCusDashboardScreen(self, urls, names)

        # FIXED stack variable
        self.stack.addWidget(self.emp_cus_screen)
        self.stack.setCurrentWidget(self.emp_cus_screen)

        self.fade_to_widget(self.emp_cus_screen)


    
    def check_model_schedule(self):
        """Auto-stop ML models based on saved schedule (but don't auto-start)."""
        # Don't run scheduler if we're on login screen
        if self.stack.currentWidget() == self.login_screen:
            return
        
        schedule_path = os.path.join("core", "ml_model_schedule.json")
        if not os.path.exists(schedule_path):
            return

        try:
            with open(schedule_path, "r") as f:
                schedule = json.load(f)
        except Exception as e:
            print(f"[SCHEDULER] Failed to read schedule: {e}")
            return

        now_dt = datetime.now()
        now_t = now_dt.time()
        now_str = now_dt.strftime("%H:%M")
        print(f"[SCHEDULER] Checking schedule at {now_str}...")

        def get_selected(model_key):
            import json
            sel_file = {
                "Crowd Density": "selected_crowd_density.json",
                "HeatMap": "selected_heatmap.json",
                "Daily Visitors": "selected_visitors.json",
                "Emp": "selected_emp_cus.json",
                "Motion": "selected_motion.json",
            }

            file_name = sel_file.get(model_key, "")
            file_path_root = os.path.join(os.getcwd(), file_name)
            file_path_mod = os.path.join("modules", "analytics", file_name)

            # ✅ Choose whichever exists *and* is a file
            if os.path.isfile(file_path_root):
                file_path = file_path_root
            elif os.path.isfile(file_path_mod):
                file_path = file_path_mod
            else:
                file_path = None

            if file_path:
                try:
                    with open(file_path, "r") as f:
                        sel_names = json.load(f).get("names", [])
                    if sel_names:
                        all_cams = getattr(self, "active_cams", [])
                        selected = [c for c in all_cams if c["cam_name"] in sel_names]
                        return [c["url"] for c in selected], [c["cam_name"] for c in selected]
                except Exception as e:
                    print(f"[WARN] Selection read failed for {model_key}: {e}")

            # Fallback if no valid JSON
            cams = getattr(self, "active_cams", [])
            return [c["url"] for c in cams], [c["cam_name"] for c in cams]


        for model, times in schedule.items():
            start_str = times.get("start")
            stop_str = times.get("stop")
            if not (start_str and stop_str):
                continue

            try:
                start = datetime.strptime(start_str, "%H:%M").time()
                stop = datetime.strptime(stop_str, "%H:%M").time()
            except Exception:
                continue

            active = (start <= now_t < stop) if start < stop else (now_t >= start or now_t < stop)

            worker = None
            if model == "Crowd Density":
                worker = self.crowd_worker
            elif model == "HeatMap":
                worker = self.heatmap_worker
            elif model == "Daily Visitors":
                worker = self.visitors_worker
            elif model.startswith("Emp"):
                worker = self.emp_cus_worker
            elif model.startswith("Motion"):
                worker = self.motion_worker

            urls, names = get_selected(model)

            # =====================================================
            # Start logic (same as before)
            # =====================================================
            if active and (not worker or (isinstance(worker, list) and not any(w.isRunning() for w in worker)) or (hasattr(worker, "isRunning") and not worker.isRunning())):
                print(f"[SCHEDULER] Starting {model} (active window) → {names}")
                try:
                    if model == "Crowd Density":
                        self.start_crowd_density(urls, names)
                    elif model == "HeatMap":
                        self.start_heatmap(urls, names)
                    elif model == "Daily Visitors":
                        self.start_daily_visitors(urls, names)
                    elif model.startswith("Emp"):
                        self.start_emp_cus_interaction(urls, names)
                    elif model.startswith("Motion"):
                        self.start_motion_detection(urls, names)
                except Exception as e:
                    print(f"[SCHEDULER] Error starting {model}: {e}")

            # =====================================================
            # ✅ FIXED STOP LOGIC (handles both list & single thread)
            # =====================================================
            elif not active and worker:
                try:
                    if isinstance(worker, list):
                        running_workers = [w for w in worker if w.isRunning()]
                        if running_workers:
                            print(f"[SCHEDULER] Stopping {model} (outside window) → {len(running_workers)} threads")
                            for w in running_workers:
                                w.stop()
                    elif worker.isRunning():
                        print(f"[SCHEDULER] Stopping {model} (outside window)")
                        worker.stop()
                except Exception as e:
                    print(f"[SCHEDULER] Stop failed for {model}: {e}")

        self.update_status_bar()

    def update_status_bar(self):
        # Don't update if we're on login screen - keep buttons red
        if self.stack.currentWidget() == self.login_screen:
            return

        def is_worker_running(worker):
            """Unified logic for single QThread, list of threads, or UI-based screens."""
            if worker is None:
                return False

            # Case 1: Motion Detection → list of workers
            if isinstance(worker, list):
                return any(w.isRunning() for w in worker if hasattr(w, "isRunning"))

            # Case 2: QThread-based workers
            if hasattr(worker, "isRunning"):
                try:
                    if worker.isRunning():
                        return True
                except:
                    pass

            # Case 3: Workers using custom flags (running / _thread_started)
            if hasattr(worker, "running") and getattr(worker, "running") is True:
                return True

            if hasattr(worker, "_thread_started") and worker._thread_started:
                return True

            # EMP–CUS Interaction → running if screen exists AND streams_running=True
            if hasattr(self, "emp_cus_employee"):
                try:
                    if getattr(self.emp_cus_employee, "streams_running", False):
                        return True
                except:
                    pass

            return False

        def icon(worker):
            return "🟢" if is_worker_running(worker) else "🔴"

        # Update text with correct icons
        self.lbl_crowd.setText(f"{icon(self.crowd_worker)} Crowd Density")
        self.lbl_heatmap.setText(f"{icon(self.heatmap_worker)} Heatmap")
        self.lbl_visitors.setText(f"{icon(self.visitors_worker)} Daily Visitors")
        self.lbl_empcus.setText(f"{icon(self.emp_cus_worker)} Emp–Cus Interaction")
        self.lbl_motion.setText(f"{icon(self.motion_worker)} Motion Detection")

        gpu_temp = get_gpu_temp() or "--"
        cpu_temp = get_cpu_temp() or "--"
        fps = get_gpu_fps() or "--"
        self.lbl_sysinfo.setText(f"GPU: {gpu_temp}°C | CPU: {cpu_temp}°C | FPS: {fps}")
    
    # ==================================================
    # Crowd Density UI Update Handler
    # ==================================================
    def handle_crowd_update(self, total, per_cam):
        """
        Called by CrowdDensityWorker whenever counts update.
        Updates logs, prints live data, and refreshes status labels.
        """
        try:
            print(f"[CROWD] Live update → Total={total} | Per-Cam={per_cam}")

            # Optionally update dashboard if it has a method
            if hasattr(self, "dashboard") and hasattr(self.dashboard, "update_crowd_status"):
                self.dashboard.update_crowd_status(total, per_cam)

            # Update bottom status bar display
            self.lbl_crowd.setText(f"🟢 Crowd Density ({total})")
        except Exception as e:
            print(f"[CROWD] ⚠️ Error handling update: {e}")

    def closeEvent(self, event):
        print("[INFO] Closing app — stopping all analytics workers...")
        
        # Stop all recordings
        self.recording_manager.stop_recording("Crowd Density")
        self.recording_manager.stop_recording("HeatMap")
        self.recording_manager.stop_recording("Daily Visitors")
        self.recording_manager.stop_recording("Employee–Customer Interaction")
        self.recording_manager.stop_recording("Motion Detection")
        
        # ✅ Stop DVR Monitor thread safely
        if hasattr(self, "dvr_monitor"):
            self.dvr_monitor.stop()
        # Stop all single-thread workers
        for worker in [
            self.crowd_worker,
            self.heatmap_worker,
            self.visitors_worker,
            self.emp_cus_worker,
        ]:
            try:
                if worker and worker.isRunning():
                    worker.stop()
            except Exception as e:
                print(f"[WARN] Cleanup failed: {e}")

        # ✅ Stop multi-thread motion workers
        if isinstance(self.motion_worker, list):
            for w in self.motion_worker:
                try:
                    if w.isRunning():
                        w.stop()
                except Exception as e:
                    print(f"[WARN] Cleanup failed for Motion worker: {e}")
        else:
            try:
                if self.motion_worker and self.motion_worker.isRunning():
                    self.motion_worker.stop()
            except Exception as e:
                print(f"[WARN] Cleanup failed for Motion worker: {e}")

        self.update_status_bar()
        event.accept()

# ==================================================
# 🧩 DVR Auto-Detection Background Monitor
# ==================================================
# Use same config path as login screen (core/config.json)
def get_config_path():
    """Get the config file path (same as login screen uses)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "config.json")

AUTO_DETECT_INTERVAL = 300  # seconds = 5 minutes

class DVRMonitor:
    def __init__(self, controller=None):
        self.controller = controller  # optional, to trigger reconnect if needed
        self.last_ip = None
        self.last_detect_time = None
        self.running = False
        self.config_path = get_config_path()

    def load_config(self):
        """Load config from core/config.json (same path as login screen)."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DVR] ⚠️ Failed to load config: {e}")
                return {}
        return {}

    def save_config(self, cfg):
        """Save config to core/config.json (same path as login screen)."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(cfg, f, indent=4)
            print(f"[DVR] ✅ Config saved to {self.config_path}")
        except Exception as e:
            print(f"[DVR] ❌ Failed to save config: {e}")

    def detect_dvr_ip(self):
        from modules.utils.dvr_logic import auto_detect_dvr_ip
        return auto_detect_dvr_ip()

    def monitor_loop(self):
        """Continuously monitor DVR IP and auto-update config when IP changes."""
        while self.running:
            now = datetime.now()
            # Run every 5 minutes
            if not self.last_detect_time or (now - self.last_detect_time).total_seconds() >= AUTO_DETECT_INTERVAL:
                print(f"[DVR] 🔍 Auto-detecting DVR IP... ({now.strftime('%H:%M:%S')})")
                cfg = self.load_config()
                old_ip = cfg.get("ip", "")
                new_ip = self.detect_dvr_ip()

                if new_ip:
                    if new_ip != old_ip:
                        print(f"[DVR] ⚠️ IP changed from {old_ip} → {new_ip}. Auto-updating config...")
                        cfg["ip"] = new_ip
                        # Always save config when IP changes
                        self.save_config(cfg)
                        print(f"[DVR] ✅ Configuration automatically updated with new IP: {new_ip}")
                        
                        # Update active cameras if controller is available
                        if self.controller:
                            print("[DVR] 🔄 Triggering DVR reconnect with new IP...")
                            try:
                                self.controller.reconnect_dvr(new_ip)
                            except Exception as e:
                                print(f"[DVR] ⚠️ Reconnect error: {e}")
                    else:
                        print(f"[DVR] ✅ IP unchanged ({new_ip}). Config is up to date.")
                else:
                    # IP detection failed, but keep using saved IP
                    if old_ip:
                        print(f"[DVR] ⚠️ IP detection failed, using saved IP: {old_ip}")
                    else:
                        print(f"[DVR] ⚠️ IP detection failed and no saved IP available")

                self.last_detect_time = now
            time.sleep(5)

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.monitor_loop, daemon=True).start()

    def stop(self):
        self.running = False

# ==================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = AppController()
    controller.show()
    sys.exit(app.exec())