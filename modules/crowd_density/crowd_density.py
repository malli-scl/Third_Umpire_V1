#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire – Crowd Density (Peak Clip Extraction Only)
Buddy Final v3 — Raspberry Pi Optimized
"""

import os, time, json, cv2, torch, subprocess, re, platform, threading, shutil
from datetime import datetime, timedelta
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QCheckBox, QHBoxLayout
)
from PyQt6.QtGui import QPixmap, QImage
from ultralytics import YOLO


# ============================================================
#                GLOBAL STORAGE CONFIG
# ============================================================

TU_ROOT = os.path.expanduser("~/Downloads/ThirdUmpire_Analytics")

RECORD_ROOT = os.path.join(TU_ROOT, "recordings")
CROWD_ROOT = os.path.join(RECORD_ROOT, "crowd_density")
PEAK_ROOT = os.path.join(CROWD_ROOT, "Peak_clip")
LOG_ROOT = os.path.join(TU_ROOT, "logs")

os.makedirs(CROWD_ROOT, exist_ok=True)
os.makedirs(PEAK_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)


# ============================================================
#             URL Normalization Helper
# ============================================================

def normalize_rtsp_url(url):
    """Return sanitized RTSP URL while keeping Hikvision/CP Plus formatting intact."""
    if not url or not isinstance(url, str):
        return url

    # Hikvision / CP Plus → DO NOT modify
    if '/Streaming/Channels/' in url:
        return url

    # Dahua format untouched (no leading zeros needed)
    return url

# ============================================================
#                HOURLY RECORDER (NO .temp)
#     Saves 1-hour file → allows peak extraction → deletes file
# ============================================================

class HourlyRecorder:
    def __init__(self, cam_name, rtsp_url):
        self.cam_name = cam_name
        self.safe_cam = cam_name.replace(" ", "_")
        self.rtsp_url = rtsp_url

        self.proc = None
        self.running = True
        self.last_hour = None

        # daily folder: crowd_density/YYYY-MM-DD/Cam_1/
        self.day_dir = os.path.join(
            CROWD_ROOT,
            datetime.now().strftime("%Y-%m-%d"),
            self.safe_cam
        )
        os.makedirs(self.day_dir, exist_ok=True)

    # ----------------------------------------------------------
    def update_day_dir(self):
        """Roll over new date folder at midnight."""
        today = datetime.now().strftime("%Y-%m-%d")
        new_dir = os.path.join(CROWD_ROOT, today, self.safe_cam)

        if new_dir != self.day_dir:
            self.day_dir = new_dir
            os.makedirs(self.day_dir, exist_ok=True)

    # ----------------------------------------------------------
    def _get_hour_file(self, hour_dt=None):
        """Return video file path for current hour."""
        if hour_dt is None:
            hour_dt = datetime.now().replace(minute=0, second=0, microsecond=0)

        hour_tag = hour_dt.strftime("%H00")
        return os.path.join(self.day_dir, f"{self.safe_cam}_{hour_tag}.ts")
    # ----------------------------------------------------------
    def start_new_hour(self):
        """Start FFmpeg to record the new hour."""
        hour_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
        out_file = self._get_hour_file(hour_dt)

        # Safety: ensure previous file removed if exists
        if os.path.exists(out_file):
            os.remove(out_file)

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-c:v", "copy",
            "-an",
            "-t", "3600",
            "-f", "mpegts",     # ⭐ TS FORMAT
            out_file
        ]
        # Start recorder
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[REC] Started 1-hour record → {out_file}")

    # ----------------------------------------------------------
    def hourly_worker(self):
        """Monitor hour change and start next-hour recorder."""
        while self.running:
            now = datetime.now()
            current_hour = now.replace(minute=0, second=0, microsecond=0)

            self.update_day_dir()

            if self.last_hour is None:
                self.last_hour = current_hour
                self.start_new_hour()

            # next hour started?
            if current_hour > self.last_hour:
                # kill previous ffmpeg
                if self.proc:
                    try:
                        self.proc.terminate()
                        time.sleep(1)
                        if self.proc.poll() is None:
                            self.proc.kill()
                    except:
                        pass

                self.start_new_hour()
                self.last_hour = current_hour

            time.sleep(1)

    # ----------------------------------------------------------
    def start(self):
        self.last_hour = None
        t = threading.Thread(target=self.hourly_worker, daemon=True)
        t.start()

    # ----------------------------------------------------------
    def stop(self):
        self.running = False

        if self.proc:
            try:
                self.proc.terminate()
                time.sleep(1)
                if self.proc.poll() is None:
                    self.proc.kill()
            except:
                pass

        print(f"[REC] Stopped recorder for {self.cam_name}")

    # ----------------------------------------------------------
    def get_hour_file_for_peak(self, peak_dt):
        """Return the recorded hour file of that peak timestamp."""
        hour_dt = peak_dt.replace(minute=0, second=0, microsecond=0)
        file_path = self._get_hour_file(hour_dt)
        return file_path if os.path.exists(file_path) else None

    # ----------------------------------------------------------
    def delete_hour_file(self, peak_dt):
        """Delete the hour file after extraction."""
        f = self.get_hour_file_for_peak(peak_dt)
        if f and os.path.exists(f):
            os.remove(f)
            print(f"[REC] Deleted hour file → {f}")


# ============================================================
#            PEAK CLIP EXTRACTOR (45s before + 45s after)
#        Uses 1-hour file from HourlyRecorder (NO .temp)
#        Deletes 1-hour file after successful extraction
# ============================================================

class PeakClipExtractor:
    def __init__(self, recorder_map):
        """
        recorder_map = { cam_name: HourlyRecorderInstance }
        This lets extractor know which recorder belongs to which cam.
        """
        self.recorders = recorder_map
        self.out_dir = PEAK_ROOT

    def safe(self, cam_name):
        return cam_name.replace(" ", "_")

    # ----------------------------------------------------------
    def extract_peak_clip(self, cam_name, peak_dt):
        """
        Extract 45 sec BEFORE + 45 sec AFTER (total 90 sec).
        Only within the SAME hour file.
        """

        safe = self.safe(cam_name)
        day = peak_dt.strftime("%Y-%m-%d")

        # 1-hour source file from recorder
        rec = self.recorders.get(cam_name)
        if not rec:
            print(f"[PEAK] ❌ No recorder found for {cam_name}")
            return

        src_file = rec.get_hour_file_for_peak(peak_dt)
        if not src_file or not os.path.exists(src_file):
            print(f"[PEAK] ❌ Hour file missing for: {cam_name} at {peak_dt}")
            return

        # Verify source file is valid and not empty
        try:
            src_size = os.path.getsize(src_file)
            if src_size < 1000:  # Less than 1KB is likely invalid
                print(f"[PEAK] ❌ Source file too small: {src_file} ({src_size} bytes)")
                return
            print(f"[PEAK] Source Hour File: {src_file} ({src_size / 1024 / 1024:.2f} MB)")
        except OSError as e:
            print(f"[PEAK] ❌ Cannot access source file: {src_file} - {e}")
            return

        # ------------------------------------------------------
        # Get actual video duration to ensure we don't exceed it
        # ------------------------------------------------------
        try:
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                src_file
            ]
            probe_result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            if probe_result.returncode == 0:
                actual_duration = float(probe_result.stdout.decode().strip())
            else:
                # Fallback: assume 3600 seconds for hour file
                actual_duration = 3600.0
                print(f"[PEAK] ⚠️ Could not probe duration, assuming 3600s")
        except Exception as e:
            actual_duration = 3600.0
            print(f"[PEAK] ⚠️ Duration probe failed: {e}, assuming 3600s")

        # ------------------------------------------------------
        # Calculate start offset (45 sec before peak)
        # Extract exactly 90 seconds: 45s before + 45s after peak timestamp
        # ------------------------------------------------------
        hour_start = peak_dt.replace(minute=0, second=0, microsecond=0)
        peak_rel_sec = int((peak_dt - hour_start).total_seconds())  # Peak position in hour
        
        # Debug: Print peak time calculation
        print(f"[PEAK] Peak datetime: {peak_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[PEAK] Hour start: {hour_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[PEAK] Peak position: {peak_rel_sec}s into hour ({peak_rel_sec//60}m {peak_rel_sec%60}s)")
        
        # Calculate clip start: 45 seconds BEFORE the peak
        # This ensures peak timestamp is at 45 seconds into the 90-second clip
        clip_start_sec = max(0, peak_rel_sec - 45)
        
        # Desired clip duration: exactly 90 seconds (45s before + 45s after peak)
        desired_duration = 90
        
        # Check if we can extract full 90 seconds without exceeding file boundaries
        clip_end_sec = clip_start_sec + desired_duration
        max_available_duration = actual_duration - clip_start_sec
        
        if clip_end_sec > actual_duration:
            # File is shorter than needed, adjust duration
            duration_sec = max_available_duration
            if duration_sec < 45:
                # Can't even get 45 seconds after peak, skip
                print(f"[PEAK] ❌ Cannot extract clip: file too short")
                print(f"[PEAK]   Peak at {peak_rel_sec}s, file duration: {actual_duration:.1f}s, need {clip_end_sec}s")
                return
            print(f"[PEAK] ⚠️ File boundary reached: adjusted from 90s to {duration_sec:.1f}s")
        else:
            # Full 90 seconds available
            duration_sec = desired_duration

        # Convert to ffmpeg timestamp (HH:MM:SS format)
        hh = clip_start_sec // 3600
        mm = (clip_start_sec % 3600) // 60
        ss = clip_start_sec % 60
        clip_start = f"{hh:02d}:{mm:02d}:{ss:02d}"
        
        # Calculate where peak appears in the clip (should be at 45s for full 90s clip)
        peak_in_clip = peak_rel_sec - clip_start_sec
        
        # Debug: Print calculated clip times
        peak_time_str = peak_dt.strftime('%H:%M:%S')
        print(f"[PEAK] Clip calculation:")
        print(f"[PEAK]   Peak timestamp: {peak_time_str} (at {peak_rel_sec}s into hour)")
        print(f"[PEAK]   Clip start: {clip_start} (at {clip_start_sec}s into hour)")
        print(f"[PEAK]   Clip duration: {duration_sec:.1f}s")
        print(f"[PEAK]   Peak appears at: {peak_in_clip:.1f}s into clip (target: 45.0s)")
        
        # Verify peak is correctly positioned
        if abs(peak_in_clip - 45.0) > 1.0 and duration_sec >= 90:
            print(f"[PEAK] ⚠️ Warning: Peak not at expected 45s position in clip (actual: {peak_in_clip:.1f}s)")

        # ------------------------------------------------------
        # Output folder
        # ------------------------------------------------------
        out_day = os.path.join(self.out_dir, day, safe)
        os.makedirs(out_day, exist_ok=True)

        out_file = os.path.join(
            out_day,
            f"{safe}_PEAK_{peak_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        print(f"[PEAK] Extracting {duration_sec:.1f}-sec clip → {out_file}")
        print(f"[PEAK] Start Offset: {clip_start}")

        # ------------------------------------------------------
        # FFmpeg extraction (re-encode for better compatibility)
        # Using re-encoding instead of copy to ensure playability
        # ------------------------------------------------------
        cmd = [
            "ffmpeg",
            "-y",
            "-i", src_file,
            "-ss", clip_start,
            "-t", str(round(duration_sec, 1)),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            out_file
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,  # Increased timeout for re-encoding
        )

        # Check stderr for warnings even on success
        stderr_output = result.stderr.decode(errors='ignore') if result.stderr else ""

        if result.returncode == 0:
            # Wait a moment for file to be fully written
            time.sleep(0.5)
            
            # Retry check with multiple attempts
            file_valid = False
            for attempt in range(3):
                if os.path.exists(out_file):
                    file_size = os.path.getsize(out_file)
                    if file_size > 1000:  # At least 1KB
                        file_valid = True
                        break
                if attempt < 2:
                    time.sleep(0.5)  # Wait and retry
            
            if file_valid:
                file_size_mb = os.path.getsize(out_file) / 1024 / 1024
                print(f"[PEAK] ✅ Saved → {out_file} ({file_size_mb:.2f} MB)")
            else:
                # Log detailed error information
                if os.path.exists(out_file):
                    actual_size = os.path.getsize(out_file)
                    print(f"[PEAK] ❌ Output file too small: {actual_size} bytes (expected >1000 bytes)")
                else:
                    print(f"[PEAK] ❌ Output file missing: {out_file}")
                
                # Show FFmpeg stderr if available (might contain warnings)
                if stderr_output:
                    error_snippet = stderr_output[:500].strip()
                    if error_snippet:
                        print(f"[PEAK] FFmpeg stderr: {error_snippet}")
                
                # Check source file validity
                if os.path.exists(src_file):
                    src_size = os.path.getsize(src_file)
                    print(f"[PEAK] Source file size: {src_size / 1024 / 1024:.2f} MB")
                else:
                    print(f"[PEAK] ⚠️ Source file missing: {src_file}")
                
                return
        else:
            error_msg = stderr_output[:500] if stderr_output else "Unknown FFmpeg error"
            print(f"[PEAK] ❌ FFmpeg failed (code {result.returncode}) → {error_msg}")
            return

        # ------------------------------------------------------
        # DELETE the 1-hour file after extracting
        # ------------------------------------------------------
        rec.delete_hour_file(peak_dt)


# ============================================================
#    CROWD DENSITY WORKER – 20 sec detection → hourly peak
# ============================================================

class CrowdDensityWorker(QThread):
    count_updated = pyqtSignal(int, list)

    def __init__(self, urls, cam_names):
        super().__init__()

        self.urls = [normalize_rtsp_url(u) for u in urls]
        self.cam_names = cam_names
        self.running = True
        # recorder map per camera
        self.recorders = {}
        # Peak Extractor (now requires recorder_map)
        self.peak = PeakClipExtractor(self.recorders)
        self.live_log_buffer = []
        # Track which hours have been processed for extraction
        self.processed_hours = set()

        # ---------------- YOLO LOAD ----------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt")
        self.model.fuse()
        self.model.to(device)

        is_pi = (
            platform.machine().startswith("arm") or
            "raspberry" in platform.platform().lower()
        )

        self.img_size = 320 if (device == "cpu" and is_pi) else 640
        if device == "cpu" and is_pi:
            torch.set_num_threads(2)

        print(f"[CROWD] Model loaded → {device} | img: {self.img_size}px")

        os.makedirs(LOG_ROOT, exist_ok=True)
        self.json_path = os.path.join(LOG_ROOT, "crowd_density.json")
        if not os.path.exists(self.json_path):
            json.dump({}, open(self.json_path, "w"), indent=4)

        self.live_log_buffer = {}
    # ----------------------------------------------------------
    def _load_json(self):
        try:
            return json.load(open(self.json_path))
        except:
            return {}

    # ----------------------------------------------------------
    def _save_json(self, data):
        json.dump(data, open(self.json_path, "w"), indent=4)

    # ----------------------------------------------------------
    def _log_peak(self, total, per_cam):
        """
        Store 20-sec detected values ONLY in RAM.
        No JSON live writes.
        """
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d %H:00")
        timestamp = now.strftime("%H:%M:%S")

        self.live_log_buffer.setdefault(hour_key, [])
        self.live_log_buffer[hour_key].append((timestamp, total, per_cam))

        print(f"[CROWD] Live → Total={total}, PerCam={per_cam}")

    def _finalize_hour(self, force=False):
        """
        Update JSON EVERY 20 sec ONLY IF:
        - New total > previous total for this hour
        - Otherwise do NOT overwrite
        - When hour flips → start new entry
        """

        now = datetime.now()

        # If current minute < 1 → still finalize previous hour
        if now.minute == 0 and now.second <= 59:
            finalize_hour = (now - timedelta(hours=1))
        else:
            finalize_hour = now

        hour_key = finalize_hour.strftime("%Y-%m-%d %H:00")

        if hour_key not in self.live_log_buffer:
            return

        entries = self.live_log_buffer[hour_key]
        if not entries:
            return

        # ------------------------------------------------------
        # Determine peak from RAM buffer
        # ------------------------------------------------------
        try:
            peak_time, peak_total, peak_per_cam = max(entries, key=lambda x: x[1])
        except:
            peak_time, peak_total, peak_per_cam = ("00:00:00", 0, [0] * len(self.cam_names))

        # ------------------------------------------------------
        # Load existing JSON
        # ------------------------------------------------------
        data = self._load_json()

        # If hour not present → create entry
        if hour_key not in data:
            data[hour_key] = {
                "PeakTime": peak_time,
                "Total": peak_total,
                "Cameras": {
                    self.cam_names[i]: peak_per_cam[i]
                    for i in range(len(self.cam_names))
                }
            }
            self._save_json(data)
            print(f"[CROWD] JSON CREATED → {hour_key} | Peak={peak_total}")
            return

        # ------------------------------------------------------
        # Compare with old JSON stored peak
        # ------------------------------------------------------
        old_total = data[hour_key].get("Total", 0)

        if peak_total > old_total:
            # Update JSON (new peak found)
            data[hour_key] = {
                "PeakTime": peak_time,
                "Total": peak_total,
                "Cameras": {
                    self.cam_names[i]: peak_per_cam[i]
                    for i in range(len(self.cam_names))
                }
            }
            self._save_json(data)
            print(f"[CROWD] JSON UPDATED → {hour_key} | Peak={peak_total} @ {peak_time}")

        else:
            print(f"[CROWD] No update → New={peak_total} < Old={old_total}")

        # ------------------------------------------------------
        # HOUR ROLLOVER → EXTRACT PEAK CLIP FOR PREVIOUS HOUR
        # ------------------------------------------------------
        current_hour_dt = now.replace(minute=0, second=0, microsecond=0)
        buffer_hour_dt = datetime.strptime(hour_key, "%Y-%m-%d %H:00")

        # If we've moved to a new hour, process the previous hour's peak
        if current_hour_dt > buffer_hour_dt:
            # Previous hour is complete, extract peak clip now
            if hour_key not in self.processed_hours:
                self.processed_hours.add(hour_key)
                print(f"[PEAK] ⏰ Hour completed: {hour_key} → Extracting peak clip...")
                
                # ------------------------------------------------------
                # ⭐ TRIGGER PEAK CLIP EXTRACTION FOR COMPLETED HOUR
                # ------------------------------------------------------
                try:
                    # Get final peak data from JSON
                    final_peak_time = data.get(hour_key, {}).get("PeakTime", "00:00:00")
                    
                    # Parse hour_key to get date and hour
                    date_str, hour_str = hour_key.split(" ")
                    hour_num = int(hour_str.replace(":00", ""))
                    
                    # Build real peak datetime - ALWAYS use hour from hour_key for consistency
                    # Peak time format is "HH:MM:SS" where HH is the hour, MM is minutes, SS is seconds
                    # IMPORTANT: Always use hour_num from hour_key, extract only MM:SS from peak_time
                    # This ensures the peak time is always within the correct hour
                    try:
                        # Parse the peak time (format: HH:MM:SS)
                        peak_time_parts = final_peak_time.split(":")
                        if len(peak_time_parts) == 3:
                            peak_hour_from_time = int(peak_time_parts[0])  # Hour from peak time (for validation only)
                            peak_min = int(peak_time_parts[1])  # Minutes (0-59) - USE THIS
                            peak_sec = int(peak_time_parts[2])   # Seconds (0-59) - USE THIS
                            
                            # Validate and clamp values
                            if peak_min < 0 or peak_min > 59:
                                print(f"[PEAK] ⚠️ Invalid minute value {peak_min}, clamping to 0-59")
                                peak_min = max(0, min(59, peak_min))
                            if peak_sec < 0 or peak_sec > 59:
                                print(f"[PEAK] ⚠️ Invalid second value {peak_sec}, clamping to 0-59")
                                peak_sec = max(0, min(59, peak_sec))
                            
                            # ALWAYS use hour from hour_key, minutes/seconds from peak_time
                            # This ensures consistency regardless of what hour is in peak_time
                            peak_dt = datetime.strptime(
                                f"{date_str} {hour_num:02d}:{peak_min:02d}:{peak_sec:02d}",
                                "%Y-%m-%d %H:%M:%S"
                            )

                            # Warn if peak hour doesn't match (for debugging)
                            if peak_hour_from_time != hour_num:
                                print(f"[PEAK] ⚠️ Peak time hour ({peak_hour_from_time}) doesn't match hour_key ({hour_num}), using hour_key hour")
                            
                            print(f"[PEAK] Parsed peak time: '{final_peak_time}' → {peak_dt.strftime('%Y-%m-%d %H:%M:%S')} (using hour from hour_key)")
                            
                        else:
                            # Invalid format, use start of hour
                            print(f"[PEAK] ⚠️ Invalid peak time format '{final_peak_time}' (expected HH:MM:SS), using 00:00:00")
                            peak_dt = datetime.strptime(f"{date_str} {hour_num:02d}:00:00", "%Y-%m-%d %H:%M:%S")
                            
                    except (ValueError, IndexError) as ve:
                        print(f"[PEAK] ⚠️ Peak time parse error '{final_peak_time}': {ve}, using 00:00:00")
                        peak_dt = datetime.strptime(f"{date_str} {hour_num:02d}:00:00", "%Y-%m-%d %H:%M:%S")

                    # Wait longer to ensure hourly file is fully written and closed
                    time.sleep(5)

                    for cam_name in self.cam_names:
                        try:
                            self.peak.extract_peak_clip(cam_name, peak_dt)
                        except Exception as e:
                            print(f"[PEAK] Error extracting clip for {cam_name}: {e}")
                            import traceback
                            traceback.print_exc()

                except Exception as e:
                    print(f"[PEAK] Timestamp parse error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear old buffer after processing
            del self.live_log_buffer[hour_key]

    # ----------------------------------------------------------
    def to_substream(self, url):
        """Automatically generate substream URL for DVR brands."""
        if not url or not isinstance(url, str):
            return url

        if "subtype=0" in url:
            return url.replace("subtype=0", "subtype=1")
        if "subtype=1" in url:
            return url

        # Hikvision: change ...01 → ...02
        m = re.search(r"Channels/(\d+)", url)
        if m:
            num = m.group(1)
            if num.endswith("1"):
                return url.replace(num, num[:-1] + "2")
        return url

    # ----------------------------------------------------------
    def run(self):
        # RTSP stability options
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;4000000|"
            "max_delay;500000|"
            "buffer_size;10000000|"
            "loglevel;0"
        )

        # --------------------------------------------
        # Open RTSP streams
        # --------------------------------------------
        caps = []
        for url in self.urls:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.to_substream(url), cv2.CAP_FFMPEG)
            caps.append(cap)

        # --------------------------------------------
        # Start continuous recorders (.temp)
        # --------------------------------------------
        for i, url in enumerate(self.urls):
            rec = HourlyRecorder(self.cam_names[i], self.to_substream(url))
            rec.start()
            self.recorders[self.cam_names[i]] = rec   # ✔ CORRECT
        # --------------------------------------------
        # Main detection loop (every 20 seconds)
        # --------------------------------------------
        while self.running:
            total = 0
            per_cam = []

            for i, cap in enumerate(caps):
                ok, frame = cap.read()
                if not ok:
                    # try reopening stream
                    url = self.to_substream(self.urls[i])
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    caps[i] = cap
                    per_cam.append(0)
                    continue

                try:
                    img = cv2.resize(frame, (self.img_size, self.img_size))
                    res = self.model.predict(
                        img,
                        imgsz=self.img_size,
                        conf=0.25,
                        iou=0.45,
                        classes=[0],
                        verbose=False
                    )
                    count = len(res[0].boxes) if res else 0
                except:
                    count = 0

                per_cam.append(count)
                total = sum(per_cam) if per_cam else 0  # group max as total

            self.count_updated.emit(total, per_cam)

            self._log_peak(total, per_cam)
            self._finalize_hour()

            time.sleep(20)

        # Clean shutdown
        for r in self.recorders.values():
            r.stop()

        for c in caps:
            c.release()

        print("[CROWD] Worker stopped")

    # ----------------------------------------------------------
    def stop(self):
        self.running = False
        self._finalize_hour(force=True)
        self.wait()


# ============================================================
#      DASHBOARD UI (unchanged from Part 1/2)
# ============================================================

class CameraPreviewWorker(QThread):
    frame_ready = pyqtSignal(QPixmap, int)

    def __init__(self, idx, url):
        super().__init__()
        self.idx = idx
        self.url = normalize_rtsp_url(url)
        self.running = True

    def run(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;4000000|"
            "max_delay;500000|"
            "loglevel;0"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg).scaled(
                300,
                200,
                Qt.AspectRatioMode.KeepAspectRatio
            )

            self.frame_ready.emit(pix, self.idx)
            time.sleep(0.2)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(300)


class CrowdDashboardScreen(QWidget):
    """UI screen for selecting cameras + running analytics."""

    def __init__(self, stacked):
        super().__init__()

        self.stacked = stacked
        self.preview_labels = []
        self.cam_checkboxes = []
        self.preview_workers = []
        self.selection_file = "selected_crowd_density.json"

        self.setStyleSheet("""
            QWidget { background:white; color:#0f2027; }
            QLabel { font-size:18px; }
            QPushButton {
                font-size:20px; font-weight:bold;
                color:white;
                border-radius:10px;
                padding:12px 24px;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("👥 Crowd Density – Live Analytics")
        try:
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except:
            title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:26px; font-weight:bold; color:#1565C0;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(40)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        cams = self.stacked.active_cams
        saved = []

        if os.path.exists(self.selection_file):
            try:
                saved = json.load(open(self.selection_file))["names"]
            except:
                saved = []

        if not cams:
            msg = QLabel("No active cameras found.")
            msg.setAlignment(Qt.AlignCenter)
            layout.addWidget(msg)
        else:
            for i, cam in enumerate(cams):
                cam_name = cam["cam_name"]

                lbl = QLabel()
                lbl.setFixedSize(300, 200)
                lbl.setStyleSheet("background:#e8e8e8; border:1px solid #ccc;")

                cb = QCheckBox(cam_name)
                cb.setChecked(cam_name in saved or not saved)

                self.preview_labels.append(lbl)
                self.cam_checkboxes.append(cb)

                vbox = QVBoxLayout()
                vbox.addWidget(lbl)
                vbox.addWidget(cb)
                widget = QWidget()
                widget.setLayout(vbox)

                self.grid.addWidget(widget, i // 3, i % 3)

                worker = CameraPreviewWorker(i, cam["url"])
                worker.frame_ready.connect(self.update_preview)
                worker.start()
                self.preview_workers.append(worker)

        # Buttons
        btn_row = QHBoxLayout()

        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("background:#2e7d32;")
        back_btn.setFixedHeight(55)
        back_btn.clicked.connect(
            lambda: self.stacked.stack.setCurrentWidget(self.stacked.ml_screen)
        )
        btn_row.addWidget(back_btn)

        run_btn = QPushButton("▶ Run Analytics")
        run_btn.setStyleSheet("background:#1976D2;")
        run_btn.setFixedHeight(55)
        run_btn.clicked.connect(self.handle_run)
        btn_row.addWidget(run_btn)

        layout.addLayout(btn_row)

    # ----------------------------------------------------------
    def update_preview(self, pix, idx):
        if idx < len(self.preview_labels):
            self.preview_labels[idx].setPixmap(pix)

    # ----------------------------------------------------------
    def handle_run(self):
        for w in self.preview_workers:
            w.stop()

        cams = self.stacked.active_cams
        selected = [i for i, cb in enumerate(self.cam_checkboxes) if cb.isChecked()]

        if not selected:
            self.show_popup("No cameras selected.")
            return

        sel_names = [cams[i]["cam_name"] for i in selected]
        sel_urls = [normalize_rtsp_url(cams[i]["url"]) for i in selected]

        json.dump({"names": sel_names}, open(self.selection_file, "w"), indent=4)

        print(f"[RUN] Crowd Density started → {sel_names}")
        self.stacked.start_crowd_density(sel_urls, sel_names)
        self.stacked.stack.setCurrentWidget(self.stacked.ml_screen)

    # ----------------------------------------------------------
    def show_popup(self, message):
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Selection Required")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()