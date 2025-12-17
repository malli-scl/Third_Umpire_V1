#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DailyCountProcessor – GLOBAL VISITOR COUNTER (Buddy Edition)
-------------------------------------------------------------
• Runs UltraLightVisitors for each selected camera
• GLOBAL IDENTITY shared across all cameras
• LOCAL confirmation inside each model
• Merges results into one global total
• Emits UI updates & AppController trigger
"""

import time
from PyQt6.QtCore import QThread, pyqtSignal

# import your model + pyav reader
from modules.visitors_count.ultralight_visitors import (
    UltraLightVisitors,
    pyav_stream_reader,
    GLOBAL_VISITOR_COUNTED   # <--- KEY: global count reference
)


class DailyCountProcessor(QThread):
    """
    Multi-camera wrapper → all cams share ONE global visitor bank.
    """
    count_updated = pyqtSignal(int, str, list)
    json_update_signal = pyqtSignal(dict)
    started_signal = pyqtSignal()

    def __init__(self, urls, cam_names):
        super().__init__()
        self.urls = urls
        self.cam_names = cam_names
        self.running = True

        self.models = []      # UltraLightVisitors instances (one per camera)
        self.streams = []     # PyAV stream generators
        self.fps = 10         # dynamic FPS estimate

    # --------------------------------------------------------------
    def run(self):
        print(f"[VISITORS-ULTRA] Starting GLOBAL daily visitors on {len(self.urls)} camera(s)...")

        # Notify controller
        self.started_signal.emit()

        # Initialize models + streams
        for url in self.urls:
            print(f"[INIT] Creating model + PyAV stream for → {url}")
            self.models.append(UltraLightVisitors())
            self.streams.append(pyav_stream_reader(url))

        prev_time = time.time()

        # ----------------------------------------------------------
        #                        MAIN LOOP
        # ----------------------------------------------------------
        while self.running:

            # IMPORTANT:
            # global total = size of GLOBAL_VISITOR_COUNTED set
            global_total = len(GLOBAL_VISITOR_COUNTED)

            # iterate through cameras
            for cam_idx in range(len(self.urls)):

                model = self.models[cam_idx]
                stream = self.streams[cam_idx]

                # Read frame
                try:
                    frame = next(stream)
                except StopIteration:
                    print(f"[WARN] Stream ended for cam {cam_idx}, reconnecting…")
                    self.streams[cam_idx] = pyav_stream_reader(self.urls[cam_idx])
                    continue
                except Exception as e:
                    print(f"[ERR] PyAV error on cam {cam_idx}: {e}")
                    time.sleep(0.2)
                    continue

                # Process frame (shared global ID logic)
                _, _ = model.process_frame(frame, self.fps)

            # Update FPS estimate
            now = time.time()
            inst_fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            self.fps = (0.9 * self.fps + 0.1 * inst_fps)

            # Recalculate global count
            global_total = len(GLOBAL_VISITOR_COUNTED)
            status = f"Visitors: {global_total}"

            # Emit to UI
            self.count_updated.emit(global_total, status, self.cam_names)

            # Increased sleep to reduce CPU load when running with other ML models
            # This prevents CPU overload on Raspberry Pi CM5 when all 3 models run simultaneously
            time.sleep(0.15)  # Increased from 0.01s to 0.15s for multi-model compatibility

        print("[VISITORS-ULTRA] Stopped cleanly.")

    # --------------------------------------------------------------
    def stop(self):
        self.running = False
        self.wait(500)
