#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Light Daily Visitors – GLOBAL UNIQUE v7 (Buddy Edition)
------------------------------------------------------------
• REAL global identity (1 person = 1 ID across all cameras)
• Count only ONCE per day (resets at midnight)
• LOCAL confirmation (2 detections on same camera)
• Bottom-zone rule preserved for accuracy
• Clean + Optimized (removed old local-ID logic)
"""

import os, time, csv, cv2, random, threading, tempfile, pickle
import numpy as np
import torch, json
from ultralytics import YOLO

# Optional PyAV import (for alternative stream reading - not required)
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False

import os

# Use the same base path as other modules (crowd_density, etc.)
TU_ROOT = os.path.expanduser("~/Downloads/ThirdUmpire_Analytics")
LOG_ROOT = os.path.join(TU_ROOT, "logs")
VISITOR_JSON = os.path.join(LOG_ROOT, "visitor_count.json")
os.makedirs(LOG_ROOT, exist_ok=True)

# ==========================================================
# GLOBAL MEMORY (shared across all cameras)
# ==========================================================
GLOBAL_VISITOR_REGISTRY = {}      # vid -> {"emb": emb, "last": ts}
GLOBAL_VISITOR_LAST_POS = {}      # vid -> (cx, cy)
GLOBAL_VISITOR_COUNTED = set()    # vid
GLOBAL_VISITOR_HOUR = {}          # vid -> hour (e.g., "09:00") - tracks which hour visitor was counted
GLOBAL_NEXT_ID = 1
GLOBAL_DAY = time.strftime("%Y-%m-%d")


# ==========================================================
# FRAME ENHANCEMENT (lightweight)
# ==========================================================
def enhance_frame(frame):
    frame = cv2.bilateralFilter(frame, 5, 35, 35)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    frame = cv2.addWeighted(frame, 1.4, blur, -0.4, 0)

    gamma = 1.1
    table = np.array([(i/255.0)**(1/gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)


# ==========================================================
# CORE MODEL
# ==========================================================
class UltraLightVisitors:
    def __init__(self, conf=0.4):
        self.model = YOLO("yolov8n.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Raspberry Pi optimization
        import platform
        is_raspberry_pi = platform.machine().startswith('arm') or 'raspberry' in platform.platform().lower()
        self.img_size = 320 if (self.device == "cpu" and is_raspberry_pi) else 640
        if self.device == "cpu" and is_raspberry_pi:
            # More aggressive frame skipping on Pi (increased for multi-model compatibility)
            self.frame_skip = 8  # Increased from 4 to reduce CPU load when running with other models
            # Limit CPU threads for better performance
            torch.set_num_threads(2)
            # Disable expensive frame enhancement on Pi when running multiple models
            self.use_enhancement = False  # Bilateral filter is very CPU-intensive
        else:
            self.frame_skip = 4  # Increased from 2 for multi-model compatibility
            self.use_enhancement = True  # Enable on high-power devices

        self.conf = conf
        self.frame_counter = 0
        self.position_thresh = 90
        self.CONFIRM_FRAMES = 2

        # Local (per camera) state only
        self.local_seen = {}        # vid -> count of detections (for confirmation)
        self.last_count_time = {}   # anti-duplicate window (per camera)
        self.colors = {}            # for drawing
        self.total_visitors = 0
        self.lock = threading.Lock()

        # CSV logging init
        self.csv_path = "visitor_count.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "TotalVisitors"])
        # ---------------------------------------------------------
        # LOAD PREVIOUS VISITOR LOG (PERSISTENT COUNT)
        # ---------------------------------------------------------
        try:
            if os.path.exists(VISITOR_JSON):
                with open(VISITOR_JSON, "r") as f:
                    data = json.load(f)

                    today = time.strftime("%Y-%m-%d")

                    # Check if today's data exists in multi-day format
                    if today in data and isinstance(data[today], dict):
                        # Multi-day format: data[day] = {total, last_id, counted_ids, hourly}
                        day_data = data[today]
                        # last_id is the maximum visitor ID counted that day
                        last_id = day_data.get("last_id", 0)
                        # Next sequential ID to assign (1-based counting)
                        globals()["GLOBAL_NEXT_ID"] = last_id + 1
                        
                        # Restore counted IDs (actual IDs, not sequential)
                        counted = day_data.get("counted_ids", [])
                        for vid in counted:
                            GLOBAL_VISITOR_COUNTED.add(int(vid))
                        
                        # Restore hourly tracking
                        hourly_data = day_data.get("hourly", {})
                        for hour_key, hour_info in hourly_data.items():
                            for vid in hour_info.get("counted_ids", []):
                                GLOBAL_VISITOR_HOUR[int(vid)] = hour_key
                        
                        # Log hourly data if available
                        if hourly_data:
                            hourly_summary = ", ".join([f"{h}: {d.get('total', 0)}" for h, d in sorted(hourly_data.items())])
                            print(f"[VISITORS] Restored: total={day_data.get('total', 0)}, last_id={last_id}, next={last_id + 1}, counted_ids={len(counted)}, hourly=[{hourly_summary}]")
                        else:
                            print(f"[VISITORS] Restored: total={day_data.get('total', 0)}, last_id={last_id}, next={last_id + 1}, counted_ids={len(counted)}")
                    elif isinstance(data, dict) and "day" in data:
                        # Legacy format: {day, counted_ids, last_id} at top level
                        prev_day = data.get("day")
                        if prev_day == today:
                            last_id = data.get("last_id", 0)
                            globals()["GLOBAL_NEXT_ID"] = last_id + 1
                            
                            # Restore counted IDs
                            counted = data.get("counted_ids", [])
                            for vid in counted:
                                GLOBAL_VISITOR_COUNTED.add(int(vid))

                            print(f"[VISITORS] Restored (legacy): last_id={last_id}, next={last_id + 1}, counted_ids={len(counted)}")
                        else:
                            print("[VISITORS] New day detected → starting fresh.")
                    else:
                        print("[VISITORS] No valid data found in log → starting fresh.")
        except Exception as e:
            print(f"[WARN] Failed to restore visitor log: {e}")

        # ---------------------------------------------------------
        # NOTE: Legacy visitor_ids.json restoration removed
        # All restoration now handled via VISITOR_JSON above
        # ---------------------------------------------------------

    # ------------------------------------------------------
    # GLOBAL DAILY RESET
    # ------------------------------------------------------
    def reset_daily_globals_if_needed(self):
        global GLOBAL_DAY, GLOBAL_VISITOR_REGISTRY, GLOBAL_VISITOR_LAST_POS
        global GLOBAL_VISITOR_COUNTED, GLOBAL_VISITOR_HOUR, GLOBAL_NEXT_ID

        current_day = time.strftime("%Y-%m-%d")
        if current_day != GLOBAL_DAY:
            GLOBAL_DAY = current_day
            GLOBAL_VISITOR_REGISTRY.clear()
            GLOBAL_VISITOR_LAST_POS.clear()
            GLOBAL_VISITOR_COUNTED.clear()
            GLOBAL_VISITOR_HOUR.clear()
            GLOBAL_NEXT_ID = 1  # Reset to 1 for new day
            self.total_visitors = 0
            print("[RESET] Midnight → Global registry cleared. IDs reset to 1.")

    # ------------------------------------------------------
    # ID EMBEDDING (DCT + COLOR HIST)
    # ------------------------------------------------------
    def extract_emb(self, crop):
        crop = cv2.resize(crop, (16, 16))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        dct = cv2.dct(np.float32(gray))[0:8, 0:8].flatten()
        dct /= np.linalg.norm(dct) + 1e-6

        hist = cv2.calcHist([crop], [0, 1, 2], None,
                            [4, 4, 4], [0, 256, 0, 256, 0, 256]).flatten()
        hist /= np.linalg.norm(hist) + 1e-6

        emb = np.concatenate([dct, hist])
        emb /= np.linalg.norm(emb) + 1e-6
        return emb

    # ------------------------------------------------------
    # JSON LOG UPDATE
    # ------------------------------------------------------
    def update_json_log(self, timestamp, total):
        day = timestamp.split(" ")[0]
        hour = timestamp.split(" ")[1].split(":")[0]  # Extract hour (HH)
        hour_key = f"{hour}:00"  # Format: "09:00", "10:00", etc.

        # Load existing log (multi-day)
        if os.path.exists(VISITOR_JSON):
            try:
                with open(VISITOR_JSON, "r") as f:
                    data = json.load(f)
            except:
                data = {}
        else:
            data = {}

        # If today not present → create new block
        if day not in data:
            data[day] = {
                "total": 0,
                "last_id": 0,
                "counted_ids": [],
                "hourly": {}
            }

        # Update hourly block - track visitors by the hour they were counted
        if "hourly" not in data[day]:
            data[day]["hourly"] = {}
        
        if hour_key not in data[day]["hourly"]:
            data[day]["hourly"][hour_key] = {
                "total": 0,
                "counted_ids": []
            }
        
        # Get visitors counted in THIS specific hour (from GLOBAL_VISITOR_HOUR)
        hourly_counted_ids = [vid for vid, h in GLOBAL_VISITOR_HOUR.items() if h == hour_key]
        hourly_counted_ids = sorted(list(set(hourly_counted_ids)))
        
        # Update hourly data with actual IDs counted in this hour
        data[day]["hourly"][hour_key]["counted_ids"] = hourly_counted_ids
        data[day]["hourly"][hour_key]["total"] = len(hourly_counted_ids)
        
        # Calculate daily totals from union of all hourly counted_ids
        all_hourly_ids = set()
        for hour_data in data[day]["hourly"].values():
            all_hourly_ids.update(hour_data.get("counted_ids", []))
        
        counted_ids_list = sorted(list(all_hourly_ids))
        max_counted_id = max(counted_ids_list) if counted_ids_list else 0

        # Update today's block - derived from hourly data
        data[day]["total"] = len(counted_ids_list)
        data[day]["last_id"] = max_counted_id  # Use actual max ID
        data[day]["counted_ids"] = counted_ids_list  # Union of all hourly counted_ids

        # Save back to JSON
        try:
            with open(VISITOR_JSON, "w") as f:
                json.dump(data, f, indent=4)
            print(f"[JSON] Updated {VISITOR_JSON} (Day: {day}, Hour: {hour_key}, Total: {total})")
        except Exception as e:
            print(f"[WARN] Failed to write visitor JSON: {e}")

    # ------------------------------------------------------
    # MAIN PROCESS
    # ------------------------------------------------------
    def process_frame(self, frame, fps_est=10):
        # Daily reset
        self.reset_daily_globals_if_needed()

        # Only enhance frame if enabled (disabled on Pi for multi-model compatibility)
        if getattr(self, 'use_enhancement', True):
            frame = enhance_frame(frame)
        # else: use frame as-is (saves significant CPU on Pi)
        self.frame_counter += 1
        # Use optimized size for Pi (smaller = faster)
        resize_w = 320 if self.img_size == 320 else 416
        resize_h = int(resize_w * frame.shape[0] / frame.shape[1])  # Maintain aspect ratio
        small = cv2.resize(frame, (resize_w, resize_h))

        # FRAME SKIP
        if self.frame_counter % self.frame_skip != 0:
            global_total = len(GLOBAL_VISITOR_COUNTED)
            return cv2.resize(small, (960, 540)), global_total

        # YOLO detect persons
        results = self.model.predict(
            small, imgsz=self.img_size, conf=self.conf, verbose=False, classes=[0]
        )
        if not results or len(results[0].boxes) == 0:
            global_total = len(GLOBAL_VISITOR_COUNTED)
            return cv2.resize(small, (960, 540)), global_total

        now = time.time()

        # --------------------------------------------------
        # PERSON LOOP
        # --------------------------------------------------
        for b in results[0].boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            crop = small[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            emb = self.extract_emb(crop)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            # -----------------------------
            # GLOBAL MATCHING
            # -----------------------------
            match_id, best_sim = None, 0
            for vid, data in GLOBAL_VISITOR_REGISTRY.items():
                sim = float(np.dot(data["emb"], emb))
                if sim < 0.75:
                    continue

                # position gate
                if vid in GLOBAL_VISITOR_LAST_POS:
                    px, py = GLOBAL_VISITOR_LAST_POS[vid]
                    if np.hypot(cx - px, cy - py) > self.position_thresh:
                        continue

                if sim > best_sim:
                    best_sim = sim
                    match_id = vid

            # -----------------------------
            # NEW GLOBAL ID
            # -----------------------------
            global GLOBAL_NEXT_ID
            if match_id is None:
                match_id = GLOBAL_NEXT_ID
                GLOBAL_NEXT_ID += 1
                GLOBAL_VISITOR_REGISTRY[match_id] = {"emb": emb, "last": now}
            else:
                # update embedding (smooth)
                GLOBAL_VISITOR_REGISTRY[match_id]["emb"] = (
                    0.7 * GLOBAL_VISITOR_REGISTRY[match_id]["emb"] + 0.3 * emb
                )
                GLOBAL_VISITOR_REGISTRY[match_id]["last"] = now

            GLOBAL_VISITOR_LAST_POS[match_id] = (cx, cy)

            # Init color if missing
            if match_id not in self.colors:
                self.colors[match_id] = tuple(random.randint(60, 255) for _ in range(3))

            # -----------------------------
            # LOCAL CONFIRMATION LOGIC
            # -----------------------------
            self.local_seen[match_id] = self.local_seen.get(match_id, 0) + 1
            if self.local_seen[match_id] < self.CONFIRM_FRAMES:
                continue  # not confirmed yet

            # -----------------------------
            # BOTTOM ZONE
            # -----------------------------
            h, w = small.shape[:2]
            if y2 > h * 0.6 and match_id not in GLOBAL_VISITOR_COUNTED:
                with self.lock:
                    GLOBAL_VISITOR_COUNTED.add(match_id)
                    # Track which hour this visitor was counted in
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    hour_key = ts.split(" ")[1].split(":")[0] + ":00"  # Format: "09:00"
                    GLOBAL_VISITOR_HOUR[match_id] = hour_key
                    # global total (correct)
                    global_total = len(GLOBAL_VISITOR_COUNTED)
                    self.update_json_log(ts, global_total)
                    print(f"[COUNT] Visitor #{match_id} → Global Total = {global_total} (Hour: {hour_key})")
            # -----------------------------
            # DRAW
            # -----------------------------
            color = self.colors[match_id]
            cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                small, f"ID {match_id}",
                (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

        global_total = len(GLOBAL_VISITOR_COUNTED)
        return cv2.resize(small, (960, 540)), global_total


# ==========================================================
# PYAV STREAM (AUTO-RECONNECT)
# ==========================================================
def pyav_stream_reader(rtsp_url):
    """Alternative stream reader using PyAV (optional dependency)."""
    if not AV_AVAILABLE:
        raise ImportError("PyAV not available - install with: pip install av")
    
    opts = {"rtsp_transport": "tcp", "stimeout": "5000000", "max_delay": "500000"}
    while True:
        try:
            container = av.open(rtsp_url, options=opts)
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            print(f"[INFO] Connected via PyAV → {rtsp_url}")

            for frame in container.decode(stream):
                yield frame.to_ndarray(format="bgr24")

        except av.AVError as e:
            print(f"[WARN] Stream lost ({e}); reconnecting…")
            time.sleep(2)
            continue