#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Employee Face Snapshot Capture Module
Captures cropped face snapshots when employees are recognized.

This is a separate, removable module. 

USAGE:
- Snapshots are automatically captured when employees are recognized
- Snapshots are saved to: ~/employee_customer_interaction/employee_snapshots/
- Organized by date and employee name: YYYY-MM-DD/EmployeeName/CameraName/timestamp.jpg
- Minimum interval: 5 seconds between captures per employee (configurable)

TO DISABLE:
1. Set ENABLED = False at the top of this file, OR
2. Comment out the import in emp_cus_interaction.py (line ~60), OR
3. Comment out the capture block in detloop() (around line ~625), OR
4. Simply delete this file

CONFIGURATION:
- MIN_SNAPSHOT_INTERVAL: Minimum seconds between snapshots (default: 5.0)
- FACE_PADDING: Pixels to add around face crop (default: 10)
- JPEG_QUALITY: Image quality 0-100 (default: 95)
- ENABLED: Enable/disable capture (default: True)
"""

import os
import cv2
import time
from datetime import datetime
from collections import defaultdict

# =========================
# Configuration
# =========================
# Base directory for snapshots - now in employees_photos folder
# Will be set dynamically when module is imported by main module
HOME = os.path.expanduser("~")
ECI_DIR = os.path.join(HOME, "employee_customer_interaction")
SNAPSHOT_BASE_DIR = os.path.join(ECI_DIR, "employees_photos")  # Default, will be updated by main module

# Minimum time between snapshots for the same employee (seconds)
MIN_SNAPSHOT_INTERVAL = 5.0  # Capture at most once every 5 seconds per employee

# Face crop padding (pixels to add around detected face)
FACE_PADDING = 10

# Image quality settings
JPEG_QUALITY = 95  # 0-100, higher = better quality

# Enable/disable snapshot capture (set to False to disable without removing code)
ENABLED = True


# =========================
# Snapshot Capture Class
# =========================
class EmployeeSnapshotCapture:
    """Handles capturing cropped face snapshots of recognized employees."""
    
    def __init__(self):
        self.last_capture_time = defaultdict(float)  # Track last capture time per employee
        self.capture_count = defaultdict(int)  # Track total captures per employee
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create snapshot directories if they don't exist."""
        if not ENABLED:
            return
        
        try:
            os.makedirs(SNAPSHOT_BASE_DIR, exist_ok=True)
        except Exception as e:
            print(f"[SNAPSHOT] Failed to create directories: {e}")
    
    def capture_employee_face(self, employee_name, frame, face_box, cam_name=None):
        """
        Capture a cropped face snapshot of an employee.
        
        Args:
            employee_name: Name of the recognized employee (must not be "Customer")
            frame: Full frame image (numpy array)
            face_box: Bounding box tuple (x1, y1, x2, y2)
            cam_name: Optional camera name for organizing snapshots
        
        Returns:
            str: Path to saved snapshot, or None if not captured
        """
        if not ENABLED:
            return None
        
        # Don't capture customers
        if employee_name == "Customer":
            return None
        
        # Check minimum interval
        now = time.time()
        last_time = self.last_capture_time[employee_name]
        if (now - last_time) < MIN_SNAPSHOT_INTERVAL:
            return None
        
        try:
            # Extract face region with padding
            x1, y1, x2, y2 = face_box
            h, w = frame.shape[:2]
            
            # Add padding
            x1 = max(0, int(x1) - FACE_PADDING)
            y1 = max(0, int(y1) - FACE_PADDING)
            x2 = min(w, int(x2) + FACE_PADDING)
            y2 = min(h, int(y2) + FACE_PADDING)
            
            # Validate bounds
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop face from frame
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            safe_name = "".join(c for c in employee_name if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_name = safe_name.replace(' ', '_')
            
            # Create employee snapshot directory inside employees_photos/EmployeeName/snapshots/
            emp_snapshot_dir = os.path.join(SNAPSHOT_BASE_DIR, safe_name, "snapshots")
            os.makedirs(emp_snapshot_dir, exist_ok=True)
            
            # Add camera name to directory if provided
            if cam_name:
                cam_dir = os.path.join(emp_snapshot_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)
                snapshot_path = os.path.join(cam_dir, f"{timestamp}.jpg")
            else:
                snapshot_path = os.path.join(emp_snapshot_dir, f"{timestamp}.jpg")
            
            # Save snapshot
            cv2.imwrite(snapshot_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            
            # Update tracking
            self.last_capture_time[employee_name] = now
            self.capture_count[employee_name] += 1
            
            print(f"[SNAPSHOT] Captured {employee_name} → {snapshot_path}")
            return snapshot_path
            
        except Exception as e:
            print(f"[SNAPSHOT] Error capturing snapshot for {employee_name}: {e}")
            return None
    
    def get_stats(self):
        """Get snapshot capture statistics."""
        return {
            "total_captures": sum(self.capture_count.values()),
            "per_employee": dict(self.capture_count),
            "snapshot_dir": SNAPSHOT_BASE_DIR
        }


# =========================
# Global Instance
# =========================
# Create global instance (will be initialized when module is imported)
_snapshot_capture = None

def set_snapshot_base_dir(base_dir):
    """Set the base directory for snapshots (called from main module)."""
    global SNAPSHOT_BASE_DIR
    SNAPSHOT_BASE_DIR = base_dir

def get_snapshot_capture():
    """Get or create the global snapshot capture instance."""
    global _snapshot_capture
    if _snapshot_capture is None:
        _snapshot_capture = EmployeeSnapshotCapture()
    return _snapshot_capture


# =========================
# Convenience Function
# =========================
def capture_employee_snapshot(employee_name, frame, face_box, cam_name=None):
    """
    Convenience function to capture employee face snapshot.
    
    Args:
        employee_name: Name of the recognized employee
        frame: Full frame image (numpy array)
        face_box: Bounding box tuple (x1, y1, x2, y2)
        cam_name: Optional camera name
    
    Returns:
        str: Path to saved snapshot, or None
    """
    if not ENABLED:
        return None
    
    capture = get_snapshot_capture()
    return capture.capture_employee_face(employee_name, frame, face_box, cam_name)

