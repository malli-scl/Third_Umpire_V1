#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Recording Manager (SAFE MINIMAL VERSION)
------------------------------------------------
This file acts as a central recording handler for all ML modules.

Why minimal?
-----------
Most Third Umpire modules DO NOT need real video recording, but the
AppController still calls:
    • start_recording()
    • stop_recording()

Without this file, the app crashes.

So this is a lightweight, no-error, no-dependency version that:
    ✔ Accepts all start/stop calls
    ✔ Prints clean logs
    ✔ Never crashes
    ✔ Works for all modules (Crowd, Heatmap, Visitors, Emp-Cus, Motion)
"""

import os
import cv2
import time
from datetime import datetime


class UnifiedRecordingManager:
    def __init__(self):
        self.current_writer = None
        self.current_path = None
        self.active_tag = None

    # -------------------------------------------------------------
    # START RECORDING (dummy / safe version)
    # -------------------------------------------------------------
    def start_recording(self, tag, urls=None, cam_names=None):
        """
        Minimal version:
        - No real video writing
        - Just prints logs
        - Prevents crashes
        """
        self.active_tag = tag
        print(f"[REC] (IGNORED) start_recording → {tag} | Cameras: {cam_names}")

    # -------------------------------------------------------------
    # STOP RECORDING (dummy / safe version)
    # -------------------------------------------------------------
    def stop_recording(self, tag=None):
        """
        Stops writer safely and clears state.
        Works even if no recording was started.
        """
        try:
            if self.current_writer is not None:
                self.current_writer.release()
                self.current_writer = None

            print(f"[REC] Recording stopped → {tag}")
        except Exception as e:
            print(f"[REC] stop_recording error → {e}")

    # -------------------------------------------------------------
    # Optional real writer if needed in future
    # -------------------------------------------------------------
    def _create_writer(self, save_path, frame_width, frame_height):
        """
        Internal method for future expansion.
        Currently not used.
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_path, fourcc, 15,
                                     (frame_width, frame_height))
            return writer
        except Exception as e:
            print(f"[REC] Writer creation failed: {e}")
            return None
