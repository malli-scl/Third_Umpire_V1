#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire Analytics – Base Launcher
"""

import sys
import os
import subprocess
import getpass

# =====================================================
# GLOBAL LOG SUPPRESSION (WORKS FOR ALL MODULES)
# =====================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
    "rtsp_transport;tcp|stimeout;2000000|bufsize;0|analyzeduration;0|probesize;32|loglevel;0"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"
os.environ["AV_LOG_FORCE_QUIET"] = "1"
os.environ["LIBAV_LOGLEVEL"] = "quiet"

import cv2
try:
    cv2.setLogLevel(0)
except:
    pass

from PyQt6.QtWidgets import QApplication
from core.app_controller import AppController


# ==========================================================
# AUTO-RUN SYSTEMD SERVICE CREATION
# ==========================================================
def ensure_systemd_service():
    """
    Auto-create and enable systemd service for Third Umpire Analytics.
    Runs only once if not already present.
    """
    service_path = "/etc/systemd/system/third_umpire_analytics.service"
    if os.path.exists(service_path):
        print("[SERVICE] third_umpire_analytics.service already exists → skipping creation.")
        return

    username = getpass.getuser()
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    work_dir = os.path.dirname(script_path)
    log_path = "/var/log/third_umpire_analytics.log"
    
    # Find virtual environment Python if available
    venv_python = os.path.join(work_dir, "venv", "bin", "python3")
    if os.path.exists(venv_python):
        python_path = venv_python

    service_content = f"""[Unit]
Description=Third Umpire Analytics Suite – ML Models & Dashboard
After=network-online.target graphical.target
Wants=network-online.target

[Service]
Type=simple
User={username}
WorkingDirectory={work_dir}
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/{username}/.Xauthority"
ExecStart={python_path} {script_path}
Restart=always
RestartSec=10
StandardOutput=append:{log_path}
StandardError=append:{log_path}
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=graphical.target
"""

    try:
        print("[SERVICE] Creating systemd service...")
        tmp_service = "/tmp/third_umpire_analytics.service"
        with open(tmp_service, "w") as f:
            f.write(service_content)
        subprocess.run(["sudo", "mv", tmp_service, service_path], check=True)
        subprocess.run(["sudo", "chmod", "644", service_path], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "third_umpire_analytics.service"], check=True)
        print(f"[SERVICE] third_umpire_analytics.service installed & enabled ✅")
        print(f"[SERVICE] Start with: sudo systemctl start third_umpire_analytics.service")
        print(f"[LOGS] → tail -f {log_path}")
    except Exception as e:
        print(f"[SERVICE] Failed to create service: {e}")
        print(f"[SERVICE] You can manually create the service file at: {service_path}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Third Umpire Analytics")

    window = AppController()
    # Window is hidden initially in AppController.__init__ to prevent flickering
    # It will be shown when needed (login UI or after auto-connect)
    # window.show()  # Not needed - window is hidden initially and shown when needed

    sys.exit(app.exec())


if __name__ == "__main__":
    # Auto-create systemd service on first run (if not running with --no-service flag)
    if "--no-service" not in sys.argv:
        try:
            ensure_systemd_service()
        except Exception as e:
            print(f"[SERVICE] Service creation skipped: {e}")
    
    app = QApplication(sys.argv)
    app.setApplicationName("Third Umpire Analytics")

    window = AppController()
    # Window is hidden initially in AppController.__init__ to prevent flickering
    # It will be shown when needed (login UI or after auto-connect)
    # window.show()  # Not needed - window is hidden initially and shown when needed
    
    # If running as systemd service or auto-boot is enabled, ensure window shows
    # Check if we're running headless (no DISPLAY) - if so, don't try to show window
    display = os.environ.get("DISPLAY")
    if display and display != ":0":
        # Custom display, show window
        pass  # Will be shown by login/auto-connect flow
    elif not display:
        # No display, might be headless - don't show
        print("[MAIN] No DISPLAY environment variable - running headless")
    
    try:
        sys.exit(app.exec())
    finally:
        print("[INFO] App exited cleanly – all resources released.")
