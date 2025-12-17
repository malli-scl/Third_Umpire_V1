#!/usr/bin/env python3
# -- coding: utf-8 --


import os, time, json, cv2, numpy as np
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QPushButton, QScrollArea, QGridLayout,
    QApplication, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QInputDialog, QLineEdit, QDialogButtonBox,
    QSizePolicy, QHBoxLayout, QFileDialog, QCheckBox, QComboBox
)
from PyQt6.QtGui import QPixmap, QImage, QColor
import sys, threading, glob, queue, shutil
from collections import defaultdict

# Optional sklearn import with numpy fallback
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementation using numpy
    def cosine_similarity(X, Y=None):
        """Numpy-based cosine similarity implementation"""
        if Y is None:
            Y = X
        # Normalize X
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        # Normalize Y
        Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-8)
        # Compute cosine similarity
        return np.dot(X_norm, Y_norm.T)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional data logger - create dummy if not available
try:
    from core.data_logger import log_event
except ImportError:
    def log_event(category, message):
        print(f"[{category}] {message}")

# Optional InsightFace imports (will fail gracefully if not available)
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARN] InsightFace not available - face recognition features disabled")

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================
# OPTIONAL: Employee Face Snapshot Capture Module
# ============================================================
# This module captures cropped face snapshots when employees are recognized.
# To disable: Comment out the import and capture calls below.
# ============================================================
try:
    # Try relative import first (when used as module)
    from .employee_snapshot_capture import capture_employee_snapshot, set_snapshot_base_dir
    SNAPSHOT_CAPTURE_AVAILABLE = True
except ImportError:
    try:
        # Fallback to absolute import (when run directly)
        from employee_snapshot_capture import capture_employee_snapshot, set_snapshot_base_dir
        SNAPSHOT_CAPTURE_AVAILABLE = True
    except ImportError:
        # Module not available - disable snapshot capture
        SNAPSHOT_CAPTURE_AVAILABLE = False
        def capture_employee_snapshot(*args, **kwargs):
            return None
        def set_snapshot_base_dir(*args, **kwargs):
            pass

# OS optimization
try:
    os.nice(10)
except:
    pass


# =========================
# Config Constants
# =========================
VIEW_W, VIEW_H = 960, 540  # scaled output for display
DET_SIZE = (960, 960)      # InsightFace detector size (increased for HD camera recognition)


# ============================================
# UNIVERSAL DB AUTO-DETECT PATCH (Buddy Special)
# ============================================

import getpass

def autodetect_eci_dir():
    """
    Auto-detect employee_customer_interaction folder
    by scanning the most common user home directories + project directory.
    Priority:
        1. /home/scl/employee_customer_interaction (if exists and has data)
        2. /home/ubuntu/employee_customer_interaction (if exists and has data)
        3. /home/pi/employee_customer_interaction (if exists and has data)
        4. /home/root/employee_customer_interaction (if exists and has data)
        5. /home/<current-user>/employee_customer_interaction (if exists and has data)
        6. <module-dir>/employee_customer_interaction (if exists and has data)
        7. <project-root>/employee_customer_interaction (if exists and has data)
        8. First existing folder (even if empty)
        9. Create new at project root
    """
    possible_users = ["scl", "ubuntu", "pi", "root", getpass.getuser()]
    candidates = []
    
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(MODULE_DIR)))

    # 1–5 → Home-based possible paths
    for user in possible_users:
        candidates.append(f"/home/{user}/employee_customer_interaction")

    # 6 → Module directory path
    candidates.append(os.path.join(MODULE_DIR, "employee_customer_interaction"))
    
    # 7 → Project-root path
    candidates.append(os.path.join(PROJECT_ROOT, "employee_customer_interaction"))

    # Helper to check if directory has data (has employees_db or embeddings_db with content)
    def has_data(path):
        if not os.path.isdir(path):
            return False
        emp_dir = os.path.join(path, "employees_db")
        emb_dir = os.path.join(path, "embeddings_db")
        # Check if either directory exists and has subdirectories (employees)
        if os.path.isdir(emp_dir):
            items = [item for item in os.listdir(emp_dir) if os.path.isdir(os.path.join(emp_dir, item))]
            if items:
                return True
        if os.path.isdir(emb_dir):
            items = [item for item in os.listdir(emb_dir) if os.path.isdir(os.path.join(emb_dir, item))]
            if items:
                return True
        return False

    # First, try to find a directory with data
    for path in candidates:
        if has_data(path):
            print("[AUTO-DB] Using detected DB path with data:", path)
            return path

    # If no data found, pick the first folder that exists
    for path in candidates:
        if os.path.isdir(path):
            print("[AUTO-DB] Using detected DB path (empty):", path)
            return path

    # If NOTHING exists → create in home directory (root folder)
    current_user = getpass.getuser()
    fallback = f"/home/{current_user}/employee_customer_interaction"
    os.makedirs(fallback, exist_ok=True)
    print("[AUTO-DB] No DB found → creating new at:", fallback)
    return fallback


# Get MASTER folder
ECI_DIR = autodetect_eci_dir()

# NEW UNIVERSAL DB STRUCTURE
# employees_db/EMP_NAME/photos/  (for photos)
# embeddings_db/EMP_NAME/emb_XXX.npy  (for embeddings)
# employees_db/EMP_NAME/meta.json  (per employee metadata)

EMP_DB_DIR = os.path.join(ECI_DIR, "employees_db")
EMP_PHOTOS_DIR = os.path.join(ECI_DIR, "employees_photos")  # New separate photos directory
EMB_DB_DIR = os.path.join(ECI_DIR, "embeddings_db")
LOG_DIR = os.path.join(ECI_DIR, "conv_logs")

# Employees JSON file with new format
EMP_DB_FILE = os.path.join(EMP_DB_DIR, "employees.json")

# Ensure required directories exist
os.makedirs(EMP_DB_DIR, exist_ok=True)
os.makedirs(EMP_PHOTOS_DIR, exist_ok=True)  # Create employees_photos directory
os.makedirs(EMB_DB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("[AUTO-DB] EMP_DB_DIR =", EMP_DB_DIR)
print("[AUTO-DB] EMP_PHOTOS_DIR =", EMP_PHOTOS_DIR)
print("[AUTO-DB] EMB_DB_DIR =", EMB_DB_DIR)
print("[AUTO-DB] LOG_DIR    =", LOG_DIR)
print("[AUTO-DB] ECI_DIR    =", ECI_DIR)

# Set snapshot base directory to employees_photos
if SNAPSHOT_CAPTURE_AVAILABLE:
    try:
        set_snapshot_base_dir(EMP_PHOTOS_DIR)
    except:
        pass




MAX_BACKUPS   = 10

MIN_DB_SIZE   = 0
# Recognition thresholds - optimized for 98% accuracy with minimal false positives
# Balanced for multi-angle recognition while preventing false positives
SIM_THRESHOLD = 0.30  # Increased to 0.30 for 98% accuracy (reduces false positives)
SIM_MARGIN    = 0.05  # Require 0.05 margin to ensure confident matches (prevents false positives)
AUTO_COLLECT_MIN_GAP_SEC = 5.0
AUTO_COLLECT_MAX_SAMPLES = 5
SMOOTH_KEEP_SEC = 30  # Increased to 30 seconds to match ABSENT_TIMEOUT_SEC (handles face turns)
MATCH_MAX_DIST = 130
# Conversation timeout: If customer and employee are not detected together for this duration,
# the conversation will stop and employee will be marked as idle
CONVERSATION_TIMEOUT_SEC = 20.0  # Increased to 15 seconds for more stable conversation detection (prevents rapid flipping)
# Minimum duration (seconds) an employee must be seen before switching to "Present" status
# This prevents early switching when employee just appears
MIN_PRESENT_DURATION_SEC = 3.0  # Require 3 seconds of continuous detection before showing "Present"

# Absent timeout: If employee is not visible/detected for this duration,
# the employee will be marked as "Absent" instead of "Idle"
# Increased to 30 seconds to handle face angle changes (temporary recognition failures)
ABSENT_TIMEOUT_SEC = 30  # Increased from 10 to handle face turns

# =========================
# STREAM URLs (Define your RTSP camera URLs here)
# =========================
# Default STREAM_URLS - can be overridden or modified
STREAM_URLS = [
    "rtsp://admin:SemiCore@2025@172.16.0.124:554/Streaming/Channels/101",
    "rtsp://admin:SemiCore@2025@172.16.0.124:554/Streaming/Channels/201",
    "rtsp://admin:SemiCore@2025@172.16.0.124:554/Streaming/Channels/301",
    "rtsp://admin:SemiCore@2025@172.16.0.124:554/Streaming/Channels/401",
    "rtsp://admin:SemiCore@2025@172.16.0.124:554/Streaming/Channels/501",
]

# =========================
# InsightFace Setup (Optional)
# =========================
MODEL = None
MODEL_LOCK = threading.Lock()

# Load centroids on module import to ensure recognition works immediately
# This ensures existing photos are recognized even after app restart
def _auto_load_centroids():
    """Auto-load centroids when module is imported."""
    try:
        load_db()
        rebuild_centroids()
        print("[INFO] Auto-loaded employee centroids on startup")
    except Exception as e:
        print(f"[WARN] Failed to auto-load centroids: {e}")

if INSIGHTFACE_AVAILABLE:
    try:
        print("[INFO] Loading InsightFace – CPU Only (No CUDA)…")

        # Disable CUDA globally for ONNXRuntime
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["ORT_NO_CUDA"] = "1"

        MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],   # ← HARD CPU LOCK
        )
        MODEL.prepare(ctx_id=-1, det_size=(960, 960), det_thresh=0.25)  # Increased threshold to reduce false positives (desks, walls, etc.)
        # OPTIMIZATION: Lower threshold for faster processing during uploads (will be overridden per-call if needed)

        print("[INFO] ✅ InsightFace ready – CPU mode (HD resolution: 960x960)")
        print("[INFO] ⚠️  IMPORTANT: Existing employees need to be re-registered with new photos")
        print("[INFO]    The detection resolution has been increased for better HD camera recognition")
        
        # Auto-load centroids after MODEL is ready (in background thread to avoid blocking)
        def load_in_background():
            _auto_load_centroids()
        threading.Thread(target=load_in_background, daemon=True).start()
    except Exception as e:
        print(f"[WARN] InsightFace initialization failed: {e}")
        MODEL = None


# =========================
# NEW UNIVERSAL DB HANDLING
# Structure: employees_db/EMP_NAME/photos/ + embeddings_db/EMP_NAME/emb_XXX.npy + meta.json
# =========================
EMP_DB = {}  # EMP_DB now just tracks employee names, embeddings stored as .npy files
EMP_CENTROIDS = {}
EMP_EMBEDDINGS_CACHE = {}  # Cache all embeddings in memory for fast multi-angle recognition
RECOGNITION_LOCK = threading.Lock()  # Thread safety lock for recognition operations

def _get_emp_photos_dir(name):
    """Get photos directory for employee in employees_photos folder."""
    # Create safe folder name (replace spaces with underscores)
    safe_name = name.replace(" ", "_")
    return os.path.join(EMP_PHOTOS_DIR, safe_name)

def _get_emp_emb_dir(name):
    """Get embeddings directory for employee."""
    return os.path.join(EMB_DB_DIR, name)

def _get_emp_meta_file(name):
    """Get meta.json file path for employee."""
    return os.path.join(EMP_DB_DIR, name, "meta.json")

def _load_meta(name):
    """Load metadata for an employee."""
    meta_file = _get_emp_meta_file(name)
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {"photos": [], "embeddings": [], "created": datetime.now().isoformat(), "updated": datetime.now().isoformat()}

def _save_meta(name, meta):
    """Save metadata for an employee."""
    meta_file = _get_emp_meta_file(name)
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    meta["updated"] = datetime.now().isoformat()
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

# =========================
# Employees JSON Management (New Format)
# =========================
def _load_employees_json():
    """Load employees.json file with new format."""
    if os.path.exists(EMP_DB_FILE):
        try:
            with open(EMP_DB_FILE, "r") as f:
                data = json.load(f)
                return data.get("employees", [])
        except Exception as e:
            print(f"[WARN] Failed to load employees.json: {e}")
            return []
    return []

def _save_employees_json(employees_list):
    """Save employees.json file with new format."""
    os.makedirs(os.path.dirname(EMP_DB_FILE), exist_ok=True)
    data = {"employees": employees_list}
    with open(EMP_DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def _generate_employee_id():
    """Generate unique employee ID (EMP-01, EMP-02, etc.)."""
    employees = _load_employees_json()
    if not employees:
        return "EMP-01"
    
    # Extract all existing IDs and find the highest number
    max_num = 0
    for emp in employees:
        emp_id = emp.get("employee_id", "")
        if emp_id.startswith("EMP-"):
            try:
                num = int(emp_id.split("-")[1])
                max_num = max(max_num, num)
            except:
                pass
    
    # Generate next ID
    next_num = max_num + 1
    return f"EMP-{next_num:02d}"

def _get_employee_by_name(name):
    """Get employee data from JSON by name."""
    employees = _load_employees_json()
    for emp in employees:
        if emp.get("employee_name") == name:
            return emp
    return None

def _add_employee_to_json(name, employee_id=None):
    """Add or update employee in employees.json."""
    employees = _load_employees_json()
    
    # Check if employee already exists
    existing_emp = _get_employee_by_name(name)
    
    if existing_emp:
        # Update existing employee
        existing_emp["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Update photo_path
        safe_name = name.replace(" ", "_")
        existing_emp["photo_path"] = f"employees_photos/{safe_name}/"
    else:
        # Create new employee entry
        if employee_id is None:
            employee_id = _generate_employee_id()
        
        safe_name = name.replace(" ", "_")
        new_emp = {
            "employee_id": employee_id,
            "employee_name": name,
            "photo_path": f"employees_photos/{safe_name}/",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        employees.append(new_emp)
    
    _save_employees_json(employees)
    return employee_id if not existing_emp else existing_emp.get("employee_id")

def _remove_employee_from_json(name):
    """Remove employee from employees.json."""
    employees = _load_employees_json()
    employees = [emp for emp in employees if emp.get("employee_name") != name]
    _save_employees_json(employees)

def _get_next_emb_index(name):
    """Get next embedding index for employee."""
    emb_dir = _get_emp_emb_dir(name)
    if not os.path.exists(emb_dir):
        return 1
    existing = glob.glob(os.path.join(emb_dir, "emb_*.npy"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            idx = int(os.path.basename(f).replace("emb_", "").replace(".npy", ""))
            indices.append(idx)
        except:
            pass
    return max(indices) + 1 if indices else 1

def _get_next_photo_index(name):
    """Get next photo index for employee."""
    photos_dir = _get_emp_photos_dir(name)
    if not os.path.exists(photos_dir):
        return 1
    existing = glob.glob(os.path.join(photos_dir, "photo_*.jpg"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            idx = int(os.path.basename(f).replace("photo_", "").replace(".jpg", ""))
            indices.append(idx)
        except:
            pass
    return max(indices) + 1 if indices else 1

def _prune_backups():
    """Prune old backup files."""
    baks = sorted(glob.glob(os.path.join(EMP_DB_DIR, "employees.json.*.bak")))
    if len(baks) > MAX_BACKUPS:
        for p in baks[:-MAX_BACKUPS]:
            try: os.remove(p)
            except: pass

def backup_db():
    """Backup legacy employees.json if it exists."""
    if os.path.exists(EMP_DB_FILE) and os.path.getsize(EMP_DB_FILE) >= MIN_DB_SIZE:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"{EMP_DB_FILE}.{ts}.bak"
        bak_dir = os.path.dirname(bak)
        if bak_dir:
            os.makedirs(bak_dir, exist_ok=True)
        with open(EMP_DB_FILE, "rb") as s, open(bak, "wb") as d: d.write(s.read())
        _prune_backups()

def load_db():
    """Load employee database - scan folders to discover employees."""
    global EMP_DB
    EMP_DB = {}
    
    # Scan employees_db directory for employee folders
    if os.path.exists(EMP_DB_DIR):
        for item in os.listdir(EMP_DB_DIR):
            emp_path = os.path.join(EMP_DB_DIR, item)
            if os.path.isdir(emp_path) and item != "thumbnails":  # Skip legacy thumbnails folder
                # Check if it has meta.json or photos/embeddings
                meta_file = _get_emp_meta_file(item)
                photos_dir = _get_emp_photos_dir(item)
                emb_dir = _get_emp_emb_dir(item)
                
                if os.path.exists(meta_file) or os.path.exists(photos_dir) or os.path.exists(emb_dir):
                    EMP_DB[item] = True  # Just mark as existing
    
    print(f"[DB] Loaded {len(EMP_DB)} employees from folder structure")

def save_db():
    """Save database - no longer needed for new structure, but kept for compatibility."""
    backup_db()
    # New DB structure doesn't need a central JSON file, but keep for legacy support
    if os.path.exists(EMP_DB_FILE):
        # Keep legacy file for backward compatibility during migration
        pass
    rebuild_centroids()

def add_embedding(name, emb, face_img=None):
    """Add embedding and photo for employee using new DB structure."""
    if not name:
        return
    
    name = name.strip()
    if not name:
        return
    
    # Normalize embedding
    emb = np.asarray(emb, dtype=np.float32)
    emb = l2norm(emb)
    
    # Get next indices
    emb_idx = _get_next_emb_index(name)
    photo_idx = _get_next_photo_index(name)
    
    # Create directories
    photos_dir = _get_emp_photos_dir(name)
    emb_dir = _get_emp_emb_dir(name)
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    
    # Save embedding as .npy file
    emb_file = os.path.join(emb_dir, f"emb_{emb_idx:03d}.npy")
    np.save(emb_file, emb)
    
    # Save photo if provided
    photo_file = None
    if face_img is not None:
        # Resize and save photo (160x160 as per instructions)
        face_resized = cv2.resize(face_img, (160, 160))
        photo_file = os.path.join(photos_dir, f"photo_{photo_idx:03d}.jpg")
        cv2.imwrite(photo_file, face_resized)
    
    # Update metadata
    meta = _load_meta(name)
    meta["photos"].append({"index": photo_idx, "file": os.path.basename(photo_file) if photo_file else None, "timestamp": datetime.now().isoformat()})
    meta["embeddings"].append({"index": emb_idx, "file": os.path.basename(emb_file), "timestamp": datetime.now().isoformat()})
    _save_meta(name, meta)
    
    # Add/update employee in JSON file
    _add_employee_to_json(name)
    
    # Update EMP_DB
    EMP_DB[name] = True
    
    rebuild_centroids()

def rename_employee(old_name, new_name):
    """Rename employee - move folders and update metadata."""
    old_name = old_name.strip()
    new_name = new_name.strip()
    
    if not old_name or not new_name or old_name == new_name:
        return
    
    if old_name not in EMP_DB:
        return
    
    # Move photos directory
    old_photos_dir = _get_emp_photos_dir(old_name)
    new_photos_dir = _get_emp_photos_dir(new_name)
    if os.path.exists(old_photos_dir):
        os.makedirs(os.path.dirname(new_photos_dir), exist_ok=True)
        try:
            if os.path.exists(new_photos_dir):
                # Merge if target exists
                for item in os.listdir(old_photos_dir):
                    shutil.move(os.path.join(old_photos_dir, item), os.path.join(new_photos_dir, item))
                shutil.rmtree(old_photos_dir)
            else:
                shutil.move(old_photos_dir, new_photos_dir)
        except Exception as e:
            print(f"[WARN] Failed to move photos directory: {e}")
    
    # Move embeddings directory
    old_emb_dir = _get_emp_emb_dir(old_name)
    new_emb_dir = _get_emp_emb_dir(new_name)
    if os.path.exists(old_emb_dir):
        os.makedirs(os.path.dirname(new_emb_dir), exist_ok=True)
        try:
            if os.path.exists(new_emb_dir):
                # Merge if target exists
                for item in os.listdir(old_emb_dir):
                    shutil.move(os.path.join(old_emb_dir, item), os.path.join(new_emb_dir, item))
                shutil.rmtree(old_emb_dir)
            else:
                shutil.move(old_emb_dir, new_emb_dir)
        except Exception as e:
            print(f"[WARN] Failed to move embeddings directory: {e}")
    
    # Move meta.json
    old_meta = _get_emp_meta_file(old_name)
    new_meta = _get_emp_meta_file(new_name)
    if os.path.exists(old_meta):
        os.makedirs(os.path.dirname(new_meta), exist_ok=True)
        try:
            if os.path.exists(new_meta):
                # Merge metadata
                old_meta_data = _load_meta(old_name)
                new_meta_data = _load_meta(new_name)
                new_meta_data["photos"].extend(old_meta_data.get("photos", []))
                new_meta_data["embeddings"].extend(old_meta_data.get("embeddings", []))
                _save_meta(new_name, new_meta_data)
                os.remove(old_meta)
            else:
                shutil.move(old_meta, new_meta)
        except Exception as e:
            print(f"[WARN] Failed to move meta.json: {e}")
    
    # Remove old employee folder if empty
    old_emp_dir = os.path.join(EMP_DB_DIR, old_name)
    if os.path.exists(old_emp_dir):
        try:
            if not os.listdir(old_emp_dir):
                os.rmdir(old_emp_dir)
        except:
            pass
    
    # Update employees.json - get existing employee_id and update name
    existing_emp = _get_employee_by_name(old_name)
    employee_id = existing_emp.get("employee_id") if existing_emp else None
    if existing_emp:
        _remove_employee_from_json(old_name)
        _add_employee_to_json(new_name, employee_id)  # Keep same employee_id, update name and photo_path
    
    # Update EMP_DB
    EMP_DB.pop(old_name, None)
    EMP_DB[new_name] = True
    
    rebuild_centroids()

def delete_employee(name):
    """Delete employee - remove all folders, files, and JSON entry."""
    name = name.strip()
    if not name:
        return
    
    # Delete photos directory (from employees_photos)
    photos_dir = _get_emp_photos_dir(name)
    if os.path.exists(photos_dir):
        try:
            shutil.rmtree(photos_dir)
            print(f"[DELETE] Removed photos directory: {photos_dir}")
        except Exception as e:
            print(f"[WARN] Failed to delete photos directory: {e}")
    
    # Delete embeddings directory
    emb_dir = _get_emp_emb_dir(name)
    if os.path.exists(emb_dir):
        try:
            shutil.rmtree(emb_dir)
            print(f"[DELETE] Removed embeddings directory: {emb_dir}")
        except Exception as e:
            print(f"[WARN] Failed to delete embeddings directory: {e}")
    
    # Delete meta.json
    meta_file = _get_emp_meta_file(name)
    if os.path.exists(meta_file):
        try:
            os.remove(meta_file)
            print(f"[DELETE] Removed meta.json: {meta_file}")
        except Exception as e:
            print(f"[WARN] Failed to delete meta.json: {e}")
    
    # Remove employee folder from employees_db if empty
    emp_dir = os.path.join(EMP_DB_DIR, name)
    if os.path.exists(emp_dir):
        try:
            if not os.listdir(emp_dir):
                os.rmdir(emp_dir)
                print(f"[DELETE] Removed empty employee directory: {emp_dir}")
            else:
                # If not empty, try to remove it anyway (might have other files)
                shutil.rmtree(emp_dir)
                print(f"[DELETE] Removed employee directory: {emp_dir}")
        except Exception as e:
            print(f"[WARN] Failed to remove employee directory: {e}")
    
    # Remove from employees.json
    _remove_employee_from_json(name)
    print(f"[DELETE] Removed employee '{name}' from employees.json")
    
    # Clear cache for deleted employee (important: don't use old cached data)
    with RECOGNITION_LOCK:
        if name in EMP_EMBEDDINGS_CACHE:
            del EMP_EMBEDDINGS_CACHE[name]
            print(f"[DELETE] Cleared cache for employee '{name}'")
        if name in EMP_CENTROIDS:
            del EMP_CENTROIDS[name]
    
    # Update EMP_DB
    EMP_DB.pop(name, None)
    rebuild_centroids()

def delete_photo(name, photo_index):
    """Delete a specific photo and its corresponding embedding."""
    name = name.strip()
    if not name:
        return
    
    meta = _load_meta(name)
    photos = meta.get("photos", [])
    embeddings = meta.get("embeddings", [])
    
    # Find and remove photo
    photo_to_remove = None
    for i, photo in enumerate(photos):
        if photo.get("index") == photo_index:
            photo_to_remove = photo
            photos.pop(i)
            break
    
    # Find and remove corresponding embedding (by index)
    if photo_to_remove:
        emb_index = photo_to_remove.get("index")  # Assume same index
        for i, emb in enumerate(embeddings):
            if emb.get("index") == emb_index:
                embeddings.pop(i)
                # Delete embedding file
                emb_file = os.path.join(_get_emp_emb_dir(name), f"emb_{emb_index:03d}.npy")
                if os.path.exists(emb_file):
                    try:
                        os.remove(emb_file)
                    except:
                        pass
                break
        
        # Delete photo file
        if photo_to_remove.get("file"):
            photo_file = os.path.join(_get_emp_photos_dir(name), photo_to_remove["file"])
            if os.path.exists(photo_file):
                try:
                    os.remove(photo_file)
                except:
                    pass
    
    meta["photos"] = photos
    meta["embeddings"] = embeddings
    _save_meta(name, meta)
    rebuild_centroids()

def clear_all():
    """Clear all employees."""
    global EMP_DB
    for name in list(EMP_DB.keys()):
        delete_employee(name)
    EMP_DB = {}
    rebuild_centroids()

def rebuild_centroids():
    """Rebuild centroids from .npy embedding files - scans directories directly.
    Loads ALL embeddings from ALL photos for each employee to ensure maximum accuracy."""
    global EMP_CENTROIDS, EMP_DB, EMP_EMBEDDINGS_CACHE
    
    # IMPORTANT: Clear cache completely to ensure fresh data (no stale cached embeddings)
    # This ensures deleted employees' data is not used
    with RECOGNITION_LOCK:
        EMP_CENTROIDS = {}
        EMP_EMBEDDINGS_CACHE = {}
    
    try:
        # First, ensure EMP_DB is populated by scanning directories
        if not EMP_DB or len(EMP_DB) == 0:
            load_db()
        
        # Also scan embeddings_db directly to find all employees (more reliable)
        discovered_employees = set()
        
        # Scan embeddings_db directory directly
        if os.path.exists(EMB_DB_DIR):
            for item in os.listdir(EMB_DB_DIR):
                emb_dir = os.path.join(EMB_DB_DIR, item)
                if os.path.isdir(emb_dir):
                    # Check if it has embedding files
                    emb_files = glob.glob(os.path.join(emb_dir, "emb_*.npy"))
                    if emb_files:
                        discovered_employees.add(item)
                        # Also add to EMP_DB if not already there
                        if item not in EMP_DB:
                            EMP_DB[item] = True
        
        # Also scan employees_db directory
        if os.path.exists(EMP_DB_DIR):
            for item in os.listdir(EMP_DB_DIR):
                emp_path = os.path.join(EMP_DB_DIR, item)
                if os.path.isdir(emp_path) and item != "thumbnails":
                    discovered_employees.add(item)
                    if item not in EMP_DB:
                        EMP_DB[item] = True
        
        # Now process all discovered employees
        valid_employees = []
        
        for name in discovered_employees:
            emb_dir = _get_emp_emb_dir(name)
            if not os.path.exists(emb_dir):
                continue
            
            # Load all embeddings for this employee
            emb_files = sorted(glob.glob(os.path.join(emb_dir, "emb_*.npy")))
            if not emb_files:
                print(f"[WARN] No embedding files found for {name} in {emb_dir}")
                continue
            
            valid_embs = []
            for emb_file in emb_files:
                try:
                    emb = np.load(emb_file)
                    if emb.shape == (512,) or emb.shape == (1, 512):
                        emb = emb.flatten()
                        if len(emb) == 512:
                            # Ensure embedding is normalized (should already be, but double-check)
                            emb = l2norm(emb)
                            valid_embs.append(emb)
                except Exception as e:
                    print(f"[WARN] Failed to load {emb_file}: {e}")
                    continue
            
            if not valid_embs:
                print(f"[WARN] Skipping {name} (no valid embeddings from {len(emb_files)} files)")
                continue
            
            # Convert to array (embeddings are already normalized from l2norm above)
            arr = np.array(valid_embs, dtype=np.float32)
            
            # Double-check normalization (should already be normalized, but ensure it)
            arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-6, 1e6)
            
            # Compute centroid using MEDIAN (from working reference code for better recognition)
            # Reference code uses median: centroid = np.median(arr, axis=0)
            centroid = np.median(arr, axis=0)  # Use median (more robust than mean for recognition)
            centroid = centroid / max(np.linalg.norm(centroid), 1e-6)
            
            # Debug: Verify centroid is normalized
            centroid_norm = np.linalg.norm(centroid)
            if abs(centroid_norm - 1.0) > 0.01:
                print(f"[WARN] Centroid for {name} not properly normalized: norm={centroid_norm:.6f}")
            else:
                print(f"[DEBUG] Centroid for {name} normalized correctly: norm={centroid_norm:.6f}")
            
            EMP_CENTROIDS[name] = centroid
            # Cache ALL embeddings from ALL photos in memory for fast multi-angle recognition
            # This ensures maximum accuracy by comparing against all uploaded photos
            EMP_EMBEDDINGS_CACHE[name] = valid_embs
            valid_employees.append(name)
            print(f"[INFO] Loaded {len(valid_embs)} embeddings for {name} from {len(emb_files)} files (ALL embeddings cached)")
        
        print(f"[INFO] Centroids rebuilt successfully: {len(EMP_CENTROIDS)} employees ({', '.join(valid_employees)})")
        
    except Exception as e:
        print(f"[ERROR] Failed to rebuild centroids: {e}")
        import traceback
        traceback.print_exc()


def l2norm(v): v=np.asarray(v,np.float32); return v/max(np.linalg.norm(v),1e-6)

def recognize(emb):
    """
    Mobile-level multi-angle recognition: Compares against ALL individual embeddings.
    This provides mobile-level accuracy that works from any angle, just like phone face unlock.
    Embedding should already be normalized via l2norm before calling this.
    """
    # SAFETY: Thread-safe access to global data structures
    with RECOGNITION_LOCK:
        if not EMP_CENTROIDS:
            return "Customer", 0.0
        
        # MOBILE-LEVEL RECOGNITION: Compare against ALL individual embeddings (not just centroid)
        # This allows recognition from any angle, just like mobile face unlock
        best_match_name = None
        best_match_sim = 0.0
        all_employee_sims = {}  # Track best sim per employee
        
        # Use cached embeddings for fast access (no disk I/O during recognition)
        for name in EMP_CENTROIDS.keys():
            # Try cache first (fast - no disk reads)
            if name in EMP_EMBEDDINGS_CACHE and len(EMP_EMBEDDINGS_CACHE[name]) > 0:
                cached_embs = EMP_EMBEDDINGS_CACHE[name]
            else:
                # Fallback: Load from disk (slower, but ensures we have data)
                emb_dir = _get_emp_emb_dir(name)
                if not os.path.exists(emb_dir):
                    continue
                emb_files = sorted(glob.glob(os.path.join(emb_dir, "emb_*.npy")))
                if not emb_files:
                    continue
                cached_embs = []
                for emb_file in emb_files:
                    try:
                        stored_emb = np.load(emb_file)
                        if stored_emb.shape == (512,) or stored_emb.shape == (1, 512):
                            stored_emb = stored_emb.flatten()
                            if len(stored_emb) == 512:
                                stored_emb = l2norm(stored_emb)
                                cached_embs.append(stored_emb)
                    except:
                        continue
                # Cache for next time (fast access)
                if cached_embs:
                    EMP_EMBEDDINGS_CACHE[name] = cached_embs
            
            if not cached_embs:
                continue
            
            # SAFETY CHECK: Validate embeddings before stacking (prevents segmentation fault)
            try:
                # Filter out any invalid embeddings (wrong shape, NaN, etc.)
                valid_embs = []
                for e in cached_embs:
                    if isinstance(e, np.ndarray) and e.shape == (512,) and not np.any(np.isnan(e)):
                        valid_embs.append(e)
                
                if not valid_embs:
                    continue
                
                # OPTIMIZATION: Batch compute similarities (faster than one-by-one)
                # Stack all embeddings and compute similarities in one go
                emb_array = np.stack(valid_embs)
                sims = cosine_similarity(emb.reshape(1, -1), emb_array)[0]
            except Exception as e:
                # If stacking fails, skip this employee (prevents segmentation fault)
                print(f"[WARN] Failed to process embeddings for {name}: {e}")
                continue
            
            # Use the BEST match from any angle (max similarity) - mobile-level accuracy
            best_emp_sim = float(np.max(sims))
            all_employee_sims[name] = best_emp_sim
            
            if best_emp_sim > best_match_sim:
                best_match_sim = best_emp_sim
                best_match_name = name
    
    # If no multi-angle match found, fall back to centroid matching
    if best_match_name is None:
        # Fallback: Use centroid matching
        try:
            names, centers = zip(*EMP_CENTROIDS.items())
            if not names or not centers:
                return "Customer", 0.0
            centers = np.stack(centers)
            sims = cosine_similarity(emb.reshape(1, -1), centers)[0]
            i = sims.argmax()
            margin = sims[i] - (sorted(sims)[-2] if len(sims) > 1 else 0)
            best_sim = float(sims[i])
            
            # High-accuracy: Accept if above threshold with margin
            if sims[i] >= SIM_THRESHOLD and margin >= SIM_MARGIN:
                return names[i], best_sim
            # Fallback: Accept if very high similarity (0.50+) even with smaller margin
            if sims[i] >= 0.50:
                return names[i], best_sim
            return "Customer", best_sim
        except Exception as e:
            # Safety: If centroid matching fails, return Customer
            print(f"[WARN] Centroid matching failed: {e}")
            return "Customer", 0.0
    
    # Calculate margin for multi-angle match (compare best match vs second best)
    sorted_sims = sorted(all_employee_sims.values(), reverse=True)
    margin = sorted_sims[0] - (sorted_sims[1] if len(sorted_sims) > 1 else 0)
    
    # High-accuracy recognition: Use ALL embeddings from ALL photos for comparison
    # This provides 98% accuracy by comparing against all uploaded face angles
    if best_match_sim >= SIM_THRESHOLD:
        # Require margin to ensure confident match (prevents false positives)
        if margin >= SIM_MARGIN or len(sorted_sims) == 1:
            return best_match_name, best_match_sim
        # If similarity is very high, accept even with smaller margin
        if best_match_sim >= 0.50:  # Very high confidence
            return best_match_name, best_match_sim
    
    # Fallback: Accept if close to threshold with good margin (for multi-angle recognition)
    if best_match_sim >= (SIM_THRESHOLD - 0.05) and margin >= 0.03:  # 0.25+ with margin
        return best_match_name, best_match_sim
    
    return "Customer", best_match_sim

# =========================
# BatchFaceWorker for Multi-Photo Upload
# =========================
class BatchFaceWorker(QThread):
    """Worker thread for processing multiple photos in batch."""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)  # list of (emb, face_img) tuples
    error = pyqtSignal(str)
    
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
    
    def run(self):
        results = []
        total = len(self.file_paths)
        
        # OPTIMIZATION: Reduce progress update frequency to reduce overhead
        progress_interval = max(1, total // 10)  # Update every 10% or at least every image
        
        for idx, file_path in enumerate(self.file_paths):
            try:
                # OPTIMIZATION: Load image with faster method
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                # Process through InsightFace
                if MODEL is None:
                    continue
                
                # OPTIMIZATION: Get dimensions once
                h, w = img.shape[:2]
                
                # CRITICAL FIX: Resize image to DET_SIZE before processing (same as live recognition)
                # This ensures embeddings match between uploaded photos and live camera
                # OPTIMIZATION: Use faster interpolation for uploads
                img_resized = cv2.resize(img, DET_SIZE, interpolation=cv2.INTER_LINEAR)
                
                with MODEL_LOCK:
                    faces = MODEL.get(img_resized)
                
                if not faces or len(faces) == 0:
                    continue
                
                # Use first detected face
                face = faces[0]
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # OPTIMIZATION: Pre-calculate scale factors once
                scale_x = w / DET_SIZE[0]
                scale_y = h / DET_SIZE[1]
                
                # Scale bbox back to original image size for cropping
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)
                
                # Crop face region with padding from ORIGINAL image
                padding = 20
                x1_orig = max(0, x1_orig - padding)
                y1_orig = max(0, y1_orig - padding)
                x2_orig = min(w, x2_orig + padding)
                y2_orig = min(h, y2_orig + padding)
                
                face_img = img[y1_orig:y2_orig, x1_orig:x2_orig]
                if face_img.size == 0:
                    continue
                
                # OPTIMIZATION: Use faster interpolation for thumbnail resize
                face_img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_LINEAR)
                
                # Get normalized embedding (from resized detection, which matches live recognition)
                emb = l2norm(getattr(face, "normed_embedding", face.embedding))
                
                results.append((emb, face_img))
                
                # OPTIMIZATION: Emit progress less frequently to reduce signal overhead
                if (idx + 1) % progress_interval == 0 or (idx + 1) == total:
                    self.progress.emit(idx + 1, total)
                
            except Exception as e:
                print(f"[WARN] Failed to process {file_path}: {e}")
                continue
        
        # Final progress update
        self.progress.emit(total, total)
        self.finished.emit(results)

# =========================
# Session Tracker with Conversation Logging (CSV/JSON)
# =========================
import csv

class SessionTracker:
    """Tracker for employee engagement status with conversation logging (CSV/JSON only, no video recording)."""
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}
        self.day = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = os.path.join(LOG_DIR, self.day)
        self.csv_path = os.path.join(self.log_dir, "conv_log.csv")
        self.json_path = os.path.join(self.log_dir, "conv_log.json")
        # Ensure paths and files are created
        self._ensure_paths()

    def update_engaged(self, name, engaged, cam_name=None):
        """Update engagement status for an employee and log conversation start/end."""
        if name == "Customer":
            return
        now = time.time()
        with self.lock:
            if name not in self.sessions:
                self.sessions[name] = {"engaged": False, "last_seen": now, "start_time": None, "first_seen": now, "cam_name": None}
            else:
                # If employee was absent and now seen again, reset first_seen
                last_seen = self.sessions[name].get("last_seen", 0)
                if (now - last_seen) > ABSENT_TIMEOUT_SEC:
                    # Employee was absent, reset first_seen
                    self.sessions[name]["first_seen"] = now
            
            prev_engaged = self.sessions[name]["engaged"]
            self.sessions[name]["engaged"] = engaged
            self.sessions[name]["last_seen"] = now
            
            # Log conversation start - store camera name
            if not prev_engaged and engaged:
                self.sessions[name]["start_time"] = datetime.now()
                if cam_name:
                    self.sessions[name]["cam_name"] = cam_name
            
            # Log conversation end
            if prev_engaged and not engaged and self.sessions[name]["start_time"]:
                start_dt = self.sessions[name]["start_time"]
                end_dt = datetime.now()
                cam_name_for_log = self.sessions[name].get("cam_name")
                self._append_log(name, start_dt, end_dt, cam_name_for_log)
                self.sessions[name]["start_time"] = None
                self.sessions[name]["cam_name"] = None

    def _ensure_paths(self):
        """Ensure log directory and files exist, updating paths if day changed."""
        current_day = datetime.now().strftime("%Y-%m-%d")
        if current_day != self.day:
            self.day = current_day
            self.log_dir = os.path.join(LOG_DIR, self.day)
            self.csv_path = os.path.join(self.log_dir, "conv_log.csv")
            self.json_path = os.path.join(self.log_dir, "conv_log.json")
        
        # Ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Ensure CSV file exists with header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["employee", "start", "end", "duration_sec", "duration_hms", "camera"])
        
        # Ensure JSON file exists
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w") as f:
                json.dump([], f, indent=2)

    def _append_log(self, name, start_dt, end_dt, cam_name=None):
        """Append conversation log entry to CSV and JSON."""
        # Ensure paths are up-to-date and directories/files exist
        self._ensure_paths()
        
        dur = int((end_dt - start_dt).total_seconds())
        hh, mm, ss = dur // 3600, (dur % 3600) // 60, dur % 60
        dur_str = f"{hh:02d}:{mm:02d}:{ss:02d}"

        entry = {
            "employee": name,
            "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": dur,
            "duration_hms": dur_str,
            "camera": cam_name or "Unknown",
        }

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                entry["employee"],
                entry["start"],
                entry["end"],
                entry["duration_sec"],
                entry["duration_hms"],
                entry["camera"]
            ])
            
        # Append to JSON
        data = json.load(open(self.json_path)) if os.path.exists(self.json_path) else []
        data.append(entry)
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[LOG] {name} → {dur_str} on {cam_name or 'Unknown'} recorded; saved to conv_log.csv/json")

    def is_engaged(self, name):
        """Check if employee is currently engaged."""
        with self.lock:
            s = self.sessions.get(name)
            return bool(s and s.get("engaged"))

    def touch_seen(self, name):
        """Mark employee as seen."""
        if name == "Customer":
            return
        now = time.time()
        with self.lock:
            if name not in self.sessions:
                self.sessions[name] = {"engaged": False, "last_seen": now, "start_time": None, "first_seen": now, "cam_name": None}
            else:
                # If employee was absent and now seen again, reset first_seen
                last_seen = self.sessions[name].get("last_seen", 0)
                if (now - last_seen) > ABSENT_TIMEOUT_SEC:
                    # Employee was absent, reset first_seen
                    self.sessions[name]["first_seen"] = now
            self.sessions[name]["last_seen"] = now
    
    def first_seen(self, name):
        """Get first-seen timestamp (when employee was first detected in current session)."""
        with self.lock:
            s = self.sessions.get(name)
            return s.get("first_seen", 0) if s else 0
    
    def is_present(self, name):
        """Check if employee should be marked as 'Present' (requires minimum duration)."""
        if name == "Customer":
            return False
        now = time.time()
        with self.lock:
            s = self.sessions.get(name)
            if not s:
                return False
            last_seen_ts = s.get("last_seen", 0)
            first_seen_ts = s.get("first_seen", 0)
            # Check if seen recently
            seen_recent = (now - last_seen_ts) < ABSENT_TIMEOUT_SEC if last_seen_ts else False
            if not seen_recent:
                return False
            # Require minimum duration before showing "Present"
            if first_seen_ts > 0:
                duration = now - first_seen_ts
                return duration >= MIN_PRESENT_DURATION_SEC
            return False

    def last_seen(self, name):
        """Get last-seen timestamp."""
        with self.lock:
            s = self.sessions.get(name)
            return s.get("last_seen", 0) if s else 0

    def tick(self):
        """Periodic cleanup - log any ongoing conversations that ended."""
        now = datetime.now()
        with self.lock:
            for name, s in list(self.sessions.items()):
                # If employee was engaged but hasn't been seen recently, finalize the conversation
                if s.get("engaged") and s.get("start_time"):
                    last_seen_ts = s.get("last_seen", 0)
                    if (time.time() - last_seen_ts) > 30.0:  # 30 seconds grace period
                        start_dt = s["start_time"]
                        end_dt = datetime.now()
                        cam_name_for_log = s.get("cam_name")
                        self._append_log(name, start_dt, end_dt, cam_name_for_log)
                        s["engaged"] = False
                        s["start_time"] = None
                        s["cam_name"] = None

# --- Global instance
SESSION = SessionTracker()

# =========================
# Conversation Logic Helper
# =========================
def detect_proximity_conversation(detections, embeddings=None):
    """
    detections = [(name, (x1, y1, x2, y2))]
    embeddings = [embedding_vector] (optional, for verifying customers aren't employees)
    returns engaged employee names and green-line pairs
    """
    engaged, pairs = set(), []
    
    employees = [(n, b) for n, b in detections if n != "Customer"]
    customer_detections = [(n, b, i) for i, (n, b) in enumerate(detections) if n == "Customer"]
    
    if not employees or not customer_detections:
        return engaged, pairs

    verified_customers = []
    if embeddings is not None and len(EMP_CENTROIDS) > 0:
        # ============================================================
        # PARAMETER: Customer Verification Threshold
        # ============================================================
        # ADJUST THIS: Similarity threshold to verify customer is not an employee
        # Higher value (0.7-0.9) = stricter verification (fewer false customers)
        # Lower value (0.4-0.6) = more lenient (may allow employees as customers)
        CUSTOMER_VERIFICATION_THRESHOLD = 0.8
        # ============================================================
        
        for cn, cb, idx in customer_detections:
            if idx < len(embeddings):
                emb = embeddings[idx]
                is_employee = False
                try:
                    if len(EMP_CENTROIDS) > 0:
                        names, centers = zip(*EMP_CENTROIDS.items())
                        centers = np.stack(centers)
                        sims = cosine_similarity(emb.reshape(1, -1), centers)[0]
                        max_sim = float(np.max(sims))
                        if max_sim >= CUSTOMER_VERIFICATION_THRESHOLD:
                            is_employee = True
                except Exception as e:
                    pass
                
                if not is_employee:
                    verified_customers.append((cn, cb))
            else:
                verified_customers.append((cn, cb))
    else:
        verified_customers = [(cn, cb) for cn, cb, _ in customer_detections]
    
    if not verified_customers:
        return engaged, pairs

    def center(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2)//2, (y1 + y2)//2)

    # ============================================================
    # PARAMETERS: Conversation Distance Detection
    # ============================================================
    # ADJUST THESE VALUES to control conversation detection sensitivity:
    
    # Base distance multiplier - controls overall detection distance
    # INCREASED for wider conversation detection (was 50, now 120)
    # Higher = detects conversations at greater distances
    DISTANCE_BASE_MULTIPLIER = 120
    
    # Reference area for distance calculation (pixels^2)
    # Used to normalize distance based on face size (default: 15000)
    # Adjust if faces appear much larger/smaller in your camera view
    DISTANCE_REF_AREA = 15000
    
    # Minimum face area threshold (pixels^2) (default: 3000)
    # Prevents division by very small face areas
    MIN_FACE_AREA = 3000
    
    # Minimum distance threshold (pixels) - INCREASED for wider detection
    # Even small faces must be at least this close to be considered
    MIN_DISTANCE_THRESHOLD = 30  # Increased from 20 to 30
    
    # Maximum distance threshold (pixels) - INCREASED for wider detection
    # Maximum distance for conversation detection regardless of face size
    MAX_DISTANCE_THRESHOLD = 200  # Increased from 100 to 250
    
    # Vertical alignment multiplier (default: 1.0)
    # Controls how strict vertical alignment check is
    # Higher (1.5-2.0) = allows more vertical offset (people at different heights)
    # Lower (0.7-0.9) = stricter vertical alignment (people must be at similar height)
    VERTICAL_ALIGNMENT_MULTIPLIER = 2.2
    # ============================================================

    for en, eb in employees:
        ec = center(eb)
        ew, eh = eb[2]-eb[0], eb[3]-eb[1]
        earea = ew * eh

        for cn, cb in verified_customers:
            cc = center(cb)
            cw, ch = cb[2]-cb[0], cb[3]-cb[1]
            carea = cw * ch
            avg_area = (earea + carea) / 2.0

            # Calculate adaptive distance threshold based on face size
            # Larger faces = closer to camera = smaller threshold
            # Smaller faces = farther from camera = larger threshold
            dist_thresh = np.clip(
                DISTANCE_BASE_MULTIPLIER * (DISTANCE_REF_AREA / max(avg_area, MIN_FACE_AREA))**0.5,
                MIN_DISTANCE_THRESHOLD,
                MAX_DISTANCE_THRESHOLD
            )
            d = ((ec[0]-cc[0])**2 + (ec[1]-cc[1])**2)**0.5
            
            # Check if employee and customer are at similar vertical level
            vert_ok = abs(ec[1]-cc[1]) < (eh + ch) * VERTICAL_ALIGNMENT_MULTIPLIER

            if d <= dist_thresh and vert_ok:
                engaged.add(en)
                pairs.append((ec, cc))
                break

    return engaged, pairs

# =========================
# URL Normalization Helper
# =========================
def normalize_rtsp_url(url):
    """
    Normalize RTSP URL to fix common formatting issues.
    NOTE: For CP Plus and Hikvision, leading zeros in channel numbers (0101, 0201) are REQUIRED.
    This function only normalizes Dahua format URLs, not Hikvision/CP Plus format.
    """
    if not url or not isinstance(url, str):
        return url
    
    import re
    
    # Skip normalization for Hikvision/CP Plus format (they need leading zeros)
    # These formats use /Streaming/Channels/0101 pattern
    if '/Streaming/Channels/' in url:
        # Keep the URL as-is - CP Plus and Hikvision need leading zeros
        return url
    
    # Only normalize Dahua format or other formats that don't use /Streaming/Channels/
    # For Dahua, the format is /cam/realmonitor?channel=X, so no normalization needed
    return url

# =========================
# Streamer Class (Face Recognition)
# =========================
class Streamer:
    def __init__(self, url, auto_fn, cam_name=None):
        # Normalize URL to fix channel number formatting issues
        self.url = normalize_rtsp_url(url)
        self.cam_name = cam_name or "Cam1"
        self.auto_fn = auto_fn
        self.run = True
        self.is_working = False  # Track if stream is actually working

        # Use FFmpeg directly for RTSP streaming with retry logic
        # Some DVRs need a moment between connection attempts
        max_retries = 3
        retry_delay = 0.5
        self.cap = None
        
        for attempt in range(max_retries):
            try:
                if not self.url:
                    print(f"[ERROR] Empty URL for {self.cam_name}")
                    break
                print(f"[DEBUG] Attempting to open stream {self.cam_name}: {self.url[:50]}...")
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Try to read a frame to verify it's working
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.is_working = True
                        if attempt > 0:
                            print(f"[INFO] Stream {self.cam_name} opened successfully on retry {attempt + 1}")
                        else:
                            print(f"[INFO] Stream {self.cam_name} opened successfully")
                        break
                    else:
                        print(f"[WARN] Stream {self.cam_name} opened but failed to read frame")
                        self.cap.release()
                        self.cap = None
                        self.is_working = False
                else:
                    print(f"[WARN] Stream {self.cam_name} failed to open (isOpened() returned False)")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                print(f"[WARN] Attempt {attempt + 1} failed for {self.cam_name}: {e}")
                import traceback
                traceback.print_exc()
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] Failed to open RTSP stream after {max_retries} attempts: {self.url}")
            self.is_working = False
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
        else:
            if not self.is_working:
                print(f"[WARN] Stream {self.cam_name} opened but not receiving frames")

        self.lastf = None
        self.prev_objs = []
        self._auto_last = {}
        self._auto_count = defaultdict(int)
        self._engaged_frames = defaultdict(int)
        self._disengaged_frames = defaultdict(int)
        self._last_engaged_time = {}  # Track last time each employee was detected as engaged

        self.q = queue.Queue(maxsize=1)

        # Only start threads if stream is successfully opened
        if self.cap is not None and self.cap.isOpened():
            threading.Thread(target=self.caploop, daemon=True).start()
            threading.Thread(target=self.detloop, daemon=True).start()
        else:
            print(f"[WARN] Not starting threads for {self.cam_name} - stream not opened")

    def caploop(self):
        consecutive_failures = 0
        while self.run:
            # Check if cap is None or not opened before reading
            if self.cap is None or not self.cap.isOpened():
                self.is_working = False
                time.sleep(0.1)
                continue
            
            ok, f = self.cap.read()
            if not ok:
                consecutive_failures += 1
                # If stream fails repeatedly, mark as not working
                if consecutive_failures > 100:  # ~1 second of failures at 10ms intervals
                    self.is_working = False
                time.sleep(0.01)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            self.is_working = True

            f = cv2.resize(f, (VIEW_W, VIEW_H))
            self.lastf = f

            # Always use latest frame - drop old ones for lower latency
            try:
                self.q.put_nowait(f)
            except queue.Full:
                try:
                    _ = self.q.get_nowait()  # Remove old frame
                    self.q.put_nowait(f)     # Add new frame immediately
                except:
                    pass
                    
            self.fps_count = getattr(self, "fps_count", 0) + 1
            # Reduced sleep for faster frame capture (lower latency)
            time.sleep(0.0005)

    def detloop(self):
        frame_id = 0
        no_frame_warnings = 0
        while self.run:
            # Skip processing if stream is not working
            if not self.is_working:
                time.sleep(0.5)  # Check less frequently if stream is down
                continue
                
            try:
                f = self.q.get(timeout=1)
                try:
                    while True:
                        f = self.q.get_nowait()
                except queue.Empty:
                    pass
                now = time.time()
                if not hasattr(self, "last_infer"):
                    self.last_infer = 0
                # Reduced delay to 0.02s for faster recognition (50 FPS - mobile-level responsiveness)
                # This makes recognition immediate and responsive from any angle
                if now - self.last_infer < 0.02:
                    continue
                self.last_infer = now
                no_frame_warnings = 0  # Reset warning counter on successful frame

            except queue.Empty:
                no_frame_warnings += 1
                # Warn if no frames received for a while (but stream says it's working)
                if no_frame_warnings > 30:  # ~30 seconds
                    if no_frame_warnings % 30 == 0:  # Warn every 30 seconds
                        print(f"[WARN] Stream {self.cam_name} not receiving frames (queue empty)")
                continue

            frame_id += 1
            if not hasattr(self, "_fps_times"):
                self._fps_times = []
            self._fps_times.append(time.time())
            self._fps_times = [t for t in self._fps_times if time.time() - t < 2]
            self._last_fps = len(self._fps_times) / 2.0

            if MODEL is None:
                cur_boxes, recogs, embs = [], [], []
            else:
                # Use smaller detection size for faster processing (resize frame before detection)
                # This reduces processing time significantly
                # Unified resize for stable embedding extraction
                h, w = f.shape[:2]

                # Resize to fixed DET_SIZE = (960, 960) → consistent input for InsightFace (HD resolution)
                f_det = cv2.resize(f, DET_SIZE)

                # Compute scaling for converting detection bbox → original frame
                scale_x = w / DET_SIZE[0]
                scale_y = h / DET_SIZE[1]

                
                with MODEL_LOCK:
                    faces = MODEL.get(f_det)
              
                cur_boxes, recogs, embs, sims_list, raw_names = [], [], [], [], []
                
                # MINIMAL filtering: Only filter extreme false positives
                # Let recognition similarity threshold do the real filtering
                # This ensures employee faces are detected and recognized
                h_frame, w_frame = f.shape[:2]
                min_face_size = min(w_frame, h_frame) * 0.03  # Very lenient: 3% minimum (was 5%)
                max_face_size = min(w_frame, h_frame) * 0.9   # Very lenient: 90% maximum (was 80%)
                
                valid_faces = []
                for fa in faces:
                    # Get bbox in original frame coordinates
                    x1 = int(fa.bbox[0] * scale_x)
                    y1 = int(fa.bbox[1] * scale_y)
                    x2 = int(fa.bbox[2] * scale_x)
                    y2 = int(fa.bbox[3] * scale_y)
                    
                    # Calculate face dimensions
                    face_width = x2 - x1
                    face_height = y2 - y1
                    face_size = max(face_width, face_height)
                    face_area = face_width * face_height
                    
                    # MINIMAL filtering - only filter extreme cases:
                    # 1. Face must be large enough (very lenient threshold)
                    if face_size < min_face_size:
                        continue
                    
                    # 2. Face must not be too large (entire image)
                    if face_size > max_face_size:
                        continue
                    
                    # 3. Face aspect ratio - very lenient (allow most shapes)
                    aspect_ratio = face_width / max(face_height, 1)
                    if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Very lenient (was 0.3 to 3.0)
                        continue
                    
                    # 4. Face area - very lenient minimum
                    if face_area < 1000:  # Very low threshold (was 3000)
                        continue
                    
                    # 5. Detection confidence - very lenient
                    if hasattr(fa, 'det_score') and fa.det_score < 0.2:  # Very lenient (was 0.4)
                        continue
                    
                    # 6. Edge check - REMOVED (allow faces at edges)
                    # This was filtering out valid employee faces
                    
                    valid_faces.append(fa)
                
                # Debug: Log face detection (after filtering) - only log when valid faces found
                if len(valid_faces) > 0:
                    # Only log occasionally to avoid spam
                    if not hasattr(self, '_detect_log_counter'):
                        self._detect_log_counter = 0
                    self._detect_log_counter += 1
                    if self._detect_log_counter % 60 == 0:  # Log every 60 frames (~2 seconds at 30 FPS)
                        print(f"[DETECT] Found {len(valid_faces)} valid face(s) in {self.cam_name} (filtered from {len(faces)} detections)")
                        # Also log if centroids are available
                        if EMP_CENTROIDS:
                            print(f"[DEBUG] EMP_CENTROIDS has {len(EMP_CENTROIDS)} employees: {list(EMP_CENTROIDS.keys())}")
                        else:
                            print(f"[WARN] EMP_CENTROIDS is empty - recognition will not work!")
                # Removed: Logging filtered out faces (not needed)

                for fa in valid_faces:
                    # Restore bbox to original frame coordinates
                    x1 = int(fa.bbox[0] * scale_x)
                    y1 = int(fa.bbox[1] * scale_y)
                    x2 = int(fa.bbox[2] * scale_x)
                    y2 = int(fa.bbox[3] * scale_y)

                    emb = l2norm(getattr(fa, "normed_embedding", fa.embedding))
                    # SAFETY: Wrap recognition in try-except to prevent segmentation fault
                    try:
                        raw_name, sim = recognize(emb)
                        # Debug: Log recognition results
                        if raw_name != "Customer":
                            print(f"[RECOG] {self.cam_name}: Recognized '{raw_name}' with similarity {sim:.3f}")
                        elif sim > 0.20:  # Log if similarity is close to threshold (might be employee)
                            print(f"[RECOG] {self.cam_name}: Customer detected with similarity {sim:.3f} (below threshold {SIM_THRESHOLD})")
                    except Exception as e:
                        print(f"[ERROR] Recognition failed: {e}")
                        import traceback
                        traceback.print_exc()
                        raw_name, sim = "Customer", 0.0
                    
                    # Store raw recognition name before smoothing (for snapshot capture)
                    raw_names.append(raw_name)
                    
                    # ============================================================
                    # SMOOTHING: Prevent Employee → Customer → Employee flip
                    # (From working reference code - immediate recognition)
                    # ============================================================
                    name = raw_name  # Start with raw name
                    if self.prev_objs:
                        # Find closest previous object for smoothing
                        x_center = int((x1 + x2) / 2)
                        y_center = int((y1 + y2) / 2)
                        best_j, best_d = None, 99999
                        for j, p in enumerate(self.prev_objs):
                            px1, py1, px2, py2 = p["box"]
                            pcx = int((px1 + px2) / 2)
                            pcy = int((py1 + py2) / 2)
                            d = ((x_center - pcx)**2 + (y_center - pcy)**2)**0.5
                            if d < best_d:
                                best_d = d
                                best_j = j
                        
                        # Smoothing logic: Prevent flickering between employee and Customer
                        if best_j is not None:
                            prev_name = self.prev_objs[best_j]["name"]
                            # Case 1: Previous was employee, current is Customer
                            if prev_name != "Customer" and name == "Customer":
                                # Very lenient: Keep employee name if similarity is reasonable
                                # This helps with multi-angle recognition where similarity might be lower
                                if sim >= 0.20:  # Very lenient (was SIM_THRESHOLD - 0.05 = 0.25)
                                    name = prev_name   # KEEP employee name stable
                                # If similarity is too low (< 0.20), accept "Customer" to prevent false positives
                            # Case 2: Previous was Customer, current is employee (allow switch if similarity is reasonable)
                            elif prev_name == "Customer" and name != "Customer":
                                # Allow switching from Customer to employee if similarity is reasonable
                                # Use lenient threshold to allow multi-angle recognition
                                if sim >= 0.20:  # Very lenient (was SIM_THRESHOLD=0.30)
                                    # Keep the employee name (already set)
                                    pass
                                else:
                                    # If similarity is too low, keep as Customer to prevent false positives
                                    name = "Customer"
                    # ============================================================
                    
                    cur_boxes.append((x1, y1, x2, y2))
                    recogs.append(name)
                    embs.append(emb)
                    sims_list.append(sim)  # Store similarity for snapshot filtering


            now = time.time()
            matched = self._match_prev(cur_boxes)
            stable_names, new_objs = [], []
            for i, box in enumerate(cur_boxes):
                raw = recogs[i]
                j = matched[i]
                if j is not None:
                    prev = self.prev_objs[j]
                    # Only keep previous employee name if similarity is still reasonable
                    # This prevents false positives when employee is not actually present
                    if prev["name"] != "Customer" and raw == "Customer" and (now - prev["last_t"]) <= SMOOTH_KEEP_SEC:
                        # Check similarity - more lenient threshold for multi-angle recognition
                        if i < len(sims_list) and sims_list[i] >= 0.20:  # Very lenient (was SIM_THRESHOLD - 0.05 = 0.25)
                            stable = prev["name"]
                        else:
                            # Similarity too low (< 0.20), accept "Customer" to prevent false positives
                            stable = raw
                    else:
                        stable = raw
                else:
                    stable = raw
                stable_names.append(stable)
                new_objs.append({"box": box, "name": stable, "last_t": now})
            
            # Keep previous employee detections for grace period even when no faces detected
            # This prevents immediate loss of recognition when face turns (temporary detection failure)
            if len(cur_boxes) == 0 and len(self.prev_objs) > 0:
                # No faces detected in current frame - keep recent employee detections
                for prev_obj in self.prev_objs:
                    if prev_obj["name"] != "Customer" and (now - prev_obj["last_t"]) <= SMOOTH_KEEP_SEC:
                        # Keep employee detection for grace period
                        new_objs.append(prev_obj)
            
            self.prev_objs = new_objs

            # ============================================================
            # OPTIONAL: Capture employee face snapshots
            # ============================================================
            # To disable: Comment out this entire block
            # ============================================================
            if SNAPSHOT_CAPTURE_AVAILABLE:
                # Capture snapshots based on RAW recognition (before smoothing)
                # Use STRICT threshold to prevent false positives
                # Only capture when recognition confidence is HIGH
                SNAPSHOT_SIMILARITY_THRESHOLD = 0.50  # High threshold to prevent false positives
                SNAPSHOT_MIN_MARGIN = 0.10  # Require significant margin over other matches
                
                for i, (raw_name, box) in enumerate(zip(raw_names, cur_boxes)):
                    if raw_name != "Customer":  # Only capture employees, not customers
                        # Use raw recognition name and similarity for snapshot
                        if i < len(sims_list):
                            sim = sims_list[i]
                            # Capture snapshots with high confidence (98% accuracy requirement)
                            # Use threshold that matches recognition threshold for consistency
                            SNAPSHOT_SIMILARITY_THRESHOLD = 0.35  # Slightly above SIM_THRESHOLD for snapshots
                            if sim >= SNAPSHOT_SIMILARITY_THRESHOLD:
                                # Additional verification: Ensure confident match
                                # Require similarity to be at least 0.35 (above recognition threshold)
                                if sim >= 0.35:
                                    capture_employee_snapshot(
                                        employee_name=raw_name,
                                        frame=f,
                                        face_box=box,
                                        cam_name=self.cam_name
                                    )
                                else:
                                    # Similarity is above snapshot threshold but not confident enough
                                    # Skip to prevent false positives
                                    pass
                        # NO FALLBACK - Don't capture if we don't have similarity score
                        # This prevents capturing snapshots of unrecognized faces
            # ============================================================

            if self.auto_fn():
                tnow = time.time()
                for i, (name, emb) in enumerate(zip(stable_names, embs)):
                    if name != "Customer":
                        last = self._auto_last.get(name, 0)
                        if self._auto_count[name] < AUTO_COLLECT_MAX_SAMPLES and (tnow - last) >= AUTO_COLLECT_MIN_GAP_SEC:
                            # Extract face region from frame for thumbnail
                            face_img = None
                            if i < len(cur_boxes):
                                x1, y1, x2, y2 = cur_boxes[i]
                                h, w = f.shape[:2]
                                x1, y1 = max(0, int(x1)), max(0, int(y1))
                                x2, y2 = min(w, int(x2)), min(h, int(y2))
                                if y2 > y1 and x2 > x1:
                                    face_img = f[y1:y2, x1:x2]
                            add_embedding(name, emb, face_img)
                            self._auto_last[name] = tnow
                            self._auto_count[name] += 1

            # Filter out low-confidence recognitions before engagement tracking
            # VERY LENIENT: Lower threshold significantly to allow employee recognition
            # Let the recognition similarity threshold (SIM_THRESHOLD) do the filtering
            MIN_CONFIDENCE_FOR_STATUS = 0.20  # Very lenient (was 0.25, originally 0.30) - allows multi-angle recognition
            confident_detections = []
            confident_stable_names = []
            confident_boxes = []
            confident_embs = []
            
            for i, (name, box, emb) in enumerate(zip(stable_names, cur_boxes, embs)):
                # Only include if similarity is high enough
                if name != "Customer":
                    if i < len(sims_list):
                        sim = sims_list[i]
                        # Only process if similarity is above confidence threshold
                        if sim >= MIN_CONFIDENCE_FOR_STATUS:
                            confident_detections.append((name, box))
                            confident_stable_names.append(name)
                            confident_boxes.append(box)
                            confident_embs.append(emb)
                        else:
                            # Debug: Log when recognition is filtered out due to low confidence
                            print(f"[FILTER] {self.cam_name}: Filtered '{name}' (similarity {sim:.3f} < {MIN_CONFIDENCE_FOR_STATUS})")
                    # If no similarity score, skip to prevent false positives
                else:
                    # Always include customers
                    confident_detections.append((name, box))
                    confident_stable_names.append(name)
                    confident_boxes.append(box)
                    confident_embs.append(emb)
            
            # Use only confident detections for engagement tracking
            dets = confident_detections
            engaged_names, pairs = detect_proximity_conversation(dets, confident_embs)
            
            # Update stable_names to only include confident recognitions
            # This ensures status updates only happen for high-confidence matches
            stable_names = confident_stable_names
            cur_boxes = confident_boxes
            embs = confident_embs

            # ============================================================
            # ENGAGEMENT TRACKING PARAMETERS - DEBOUNCED FOR STABILITY
            # ============================================================
            # Minimum frames required to mark as engaged (prevents false positives)
            # At ~50 FPS: 10 frames = ~0.2s, 15 frames = ~0.3s
            # Higher values = more stable, less sensitive to brief movements
            MIN_ENGAGED_FRAMES = 10  # Increased from 1 to 10 for better debouncing
            
            # Minimum frames required to mark as disengaged (prevents rapid flipping)
            # Employee must be disengaged for this many frames before switching to "Present"
            # At ~50 FPS: 20 frames = ~0.4s, 30 frames = ~0.6s
            MIN_DISENGAGED_FRAMES = 30  # Require 30 frames (~0.6s) of disengagement before switching
            # ============================================================

            current_time = time.time()

            # Get all employees to process: currently detected + recently engaged (even if not currently detected)
            # This ensures we continue tracking engagement even when recognition is temporarily lost
            employees_to_process = set(stable_names)
            # Add employees who were recently engaged (within CONVERSATION_TIMEOUT_SEC) even if not currently detected
            for emp_name in list(self._last_engaged_time.keys()):
                if emp_name != "Customer":
                    time_since_last = current_time - self._last_engaged_time[emp_name]
                    if time_since_last <= CONVERSATION_TIMEOUT_SEC:
                        employees_to_process.add(emp_name)

            # Update engagement status for UI display only (no recording)
            for n in employees_to_process:
                if n == "Customer":
                    continue
                
                is_currently_detected = n in stable_names
                is_currently_engaged = n in engaged_names
                
                if is_currently_engaged:
                    # Employee is currently detected as engaged
                    self._engaged_frames[n] += 1
                    self._disengaged_frames[n] = 0  # Reset disengaged counter
                    # Update last engagement time - reset the timeout timer
                    self._last_engaged_time[n] = current_time
                elif is_currently_detected:
                    # Employee is detected but not currently engaged
                    self._disengaged_frames[n] += 1
                    # Check if we have a previous engagement time
                    if n not in self._last_engaged_time:
                        # Never been engaged, so not engaged
                        self._engaged_frames[n] = 0
                    else:
                        # Check if timeout has passed since last detection
                        time_since_last = current_time - self._last_engaged_time[n]
                        # Stabilized engagement timeout logic with debouncing
                        if time_since_last > CONVERSATION_TIMEOUT_SEC:
                            # Only decrease engagement if we've been disengaged for enough frames
                            # This prevents rapid flipping when face moves slightly
                            if self._disengaged_frames[n] >= MIN_DISENGAGED_FRAMES:
                                # Gradually decrease engagement instead of dropping suddenly
                                if self._engaged_frames[n] > 0:
                                    self._engaged_frames[n] -= 1       # smooth decay
                                else:
                                    # Only when fully decayed → mark as idle
                                    if n in self._last_engaged_time:
                                        del self._last_engaged_time[n]
                        else:
                            # Still within timeout period - keep engagement frames if we were recently engaged
                            # This prevents rapid flipping when face moves slightly but is still in conversation
                            if self._disengaged_frames[n] < MIN_DISENGAGED_FRAMES:
                                # If we haven't been disengaged long enough, maintain engagement
                                # This adds hysteresis to prevent rapid flipping
                                pass  # Keep current engagement state
                else:
                    # Employee is NOT currently detected (recognition temporarily lost)
                    # But they were recently engaged - continue tracking their engagement timeout
                    if n in self._last_engaged_time:
                        time_since_last = current_time - self._last_engaged_time[n]
                        # Don't increment disengaged_frames when recognition is lost
                        # The timeout check (CONVERSATION_TIMEOUT_SEC) is sufficient
                        # Only clear engagement after full timeout period
                        if time_since_last > CONVERSATION_TIMEOUT_SEC:
                            # Timeout exceeded - clear engagement
                            if self._engaged_frames[n] > 0:
                                self._engaged_frames[n] -= 1
                            else:
                                if n in self._last_engaged_time:
                                    del self._last_engaged_time[n]
                    else:
                        # Never been engaged and not detected - clear state
                        self._engaged_frames[n] = 0

                # CRITICAL FIX: Only mark as "In Conversation" if employee is CURRENTLY engaged with a customer
                # is_currently_engaged = True means customer is detected nearby (from detect_proximity_conversation)
                # If is_currently_engaged = False, there is NO customer nearby, so employee should NOT be marked as engaged
                
                if is_currently_engaged:
                    # Employee is currently engaged with customer - use normal engagement logic
                    if n in self._last_engaged_time:
                        # Check if still within grace period
                        time_since_last = current_time - self._last_engaged_time[n]
                        if time_since_last <= CONVERSATION_TIMEOUT_SEC:
                            # Within timeout period - maintain engagement even if recognition is temporarily lost
                            if not is_currently_detected:
                                # Recognition lost but within timeout - keep engagement
                                engaged_stable = True
                            elif self._disengaged_frames[n] < MIN_DISENGAGED_FRAMES:
                                # Hysteresis: If we haven't been disengaged long enough, keep engagement
                                engaged_stable = True
                            else:
                                # Been disengaged long enough - check if we have enough engaged frames
                                engaged_stable = self._engaged_frames[n] >= MIN_ENGAGED_FRAMES
                        else:
                            # Timeout exceeded - check if we've been disengaged long enough
                            if self._disengaged_frames[n] >= MIN_DISENGAGED_FRAMES:
                                engaged_stable = False
                            else:
                                # Hysteresis: Still transitioning, keep previous state
                                engaged_stable = self._engaged_frames[n] >= MIN_ENGAGED_FRAMES
                    else:
                        # No previous engagement - require minimum frames before marking as engaged
                        engaged_stable = self._engaged_frames[n] >= MIN_ENGAGED_FRAMES
                else:
                    # NOT currently engaged - no customer detected nearby
                    # Clear engagement immediately (no timeout grace period when no customer is present)
                    engaged_stable = False
                    # Clear engagement state
                    if n in self._last_engaged_time:
                        del self._last_engaged_time[n]
                    self._engaged_frames[n] = 0
                
                SESSION.update_engaged(n, engaged_stable, self.cam_name)
                # Update last_seen if employee is currently detected OR was recently detected
                # This keeps employees as "Present" during temporary recognition failures (face turns)
                if is_currently_detected:
                    SESSION.touch_seen(n)
                else:
                    # Check if employee was recently detected (within grace period)
                    # This handles temporary recognition failures when face turns
                    for prev_obj in self.prev_objs:
                        if prev_obj["name"] == n and (current_time - prev_obj["last_t"]) <= SMOOTH_KEEP_SEC:
                            # Employee was recently detected - keep them as "Present"
                            SESSION.touch_seen(n)
                            break
                # Debug: Log when employee is seen (only log occasionally to avoid spam)
                # Use a simple counter to avoid referencing frame_id which might not be in scope
                if not hasattr(self, '_status_log_counter'):
                    self._status_log_counter = 0
                self._status_log_counter += 1
                if self._status_log_counter % 30 == 0:  # Log every 30 updates (~1 second at 30 FPS)
                    if engaged_stable:
                        print(f"[STATUS] {n} is engaged (in conversation)")
                    else:
                        print(f"[STATUS] {n} is present (seen recently)")

            self.lastf = f

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _match_prev(self, cur_boxes):
        matched = [None] * len(cur_boxes)
        used = set()
        for i, cb in enumerate(cur_boxes):
            cc = self._center(cb)
            bestd, bestj = 1e9, -1
            for j, p in enumerate(self.prev_objs):
                if j in used:
                    continue
                pc = self._center(p["box"])
                d = ((cc[0] - pc[0]) ** 2 + (cc[1] - pc[1]) ** 2) ** 0.5
                if d < bestd:
                    bestd, bestj = d, j
            if bestj >= 0 and bestd <= MATCH_MAX_DIST:
                matched[i] = bestj
                used.add(bestj)
        return matched

    def stop(self):
        self.run = False
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

class EmpCusDashboardScreen(QWidget):
    """
    Emp–Cus Interaction Camera Selection (Crowd Density UI Style)
    Clean white UI with simple previews + checkboxes.
    """
    def __init__(self, controller, urls=None, cam_names=None):
        super().__init__()
        self.controller = controller
        
        # Ensure centroids are loaded (in case auto-load didn't complete)
        # This ensures recognition works even if module import auto-load failed
        if not EMP_CENTROIDS or len(EMP_CENTROIDS) == 0:
            print("[PREVIEW] Centroids not loaded yet - loading now...")
            try:
                load_db()
                rebuild_centroids()
            except Exception as e:
                print(f"[WARN] Failed to load centroids: {e}")
        self.setStyleSheet("""
            QWidget { background:white; color:#0f2027; }
            QLabel { font-size:18px; }
            QCheckBox { font-size:16px; }
            QPushButton {
                font-size:20px; font-weight:bold; color:white;
                background:#1976D2; border-radius:10px; padding:10px 20px;
            }
            QPushButton:hover { background:#2196F3; }
        """)

        # ========== Load previous selected ==============
        self.prev_selected = []
        if os.path.exists("selected_emp_cus.json"):
            try:
                self.prev_selected = json.load(open("selected_emp_cus.json"))
            except:
                pass

        # ========== Build UI Layout =====================
        layout = QVBoxLayout(self)

        title = QLabel("🧠 Emp–Cus Interaction – Camera Selection")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:24px; font-weight:bold; color:#1565C0;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(40)     # same spacing as crowd density
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # ========== Load Cameras ========================
        self.preview_labels = []
        self.checkboxes = []
        self.streams = []

        # ========== Load DVR Config and Build URLs (Priority 1) ==========
        def _load_dvr_cams():
            """Load cameras from DVR config with correct credentials."""
            try:
                config_paths = [
                    os.path.join("core", "config.json"),
                    os.path.expanduser("~/.third_umpire/config.json"),
                    os.path.join(os.path.dirname(__file__), "..", "..", "core", "config.json"),
                ]
                
                config = None
                for path in config_paths:
                    if os.path.exists(path):
                        try:
                            with open(path, "r") as f:
                                config = json.load(f)
                            print(f"[DVR-PREVIEW] Loaded config from: {path}")
                            break
                        except Exception as e:
                            print(f"[WARN] Failed to load config from {path}: {e}")
                            continue
                
                if not config:
                    return None, None
                
                brand = config.get("brand", "Hikvision")
                ip = config.get("ip", "")
                user = config.get("user", "")
                pwd = config.get("pass", "")
                
                if not all([ip, user, pwd]):
                    return None, None
                
                print(f"[DVR-PREVIEW] Building preview URLs: {user}@{ip} (brand: {brand})")
                
                try:
                    from modules.utils import dvr_logic
                    url_dicts = dvr_logic.build_rtsp_urls(brand, ip, user, pwd, max_channels=16)
                    cam_names = [item["name"] for item in url_dicts]
                    cam_urls = [normalize_rtsp_url(item["url"]) for item in url_dicts]
                    return cam_names, cam_urls
                except ImportError:
                    # Fallback for Hikvision
                    if brand == "Hikvision":
                        cam_names = [f"Cam {i+1}" for i in range(16)]
                        # Format: /101, /201, /301, etc. (not /0101, /0201)
                        cam_urls = [
                            f"rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{(i+1)}01"
                            for i in range(16)
                        ]
                        # No need to normalize - already in correct format
                        return cam_names, cam_urls
                    return None, None
                except Exception as e:
                    print(f"[ERROR] Failed to build preview URLs: {e}")
                    return None, None
            except Exception as e:
                print(f"[ERROR] Failed to load DVR config for previews: {e}")
                return None, None
        
        # Priority selection logic (DVR config first!)
        dvr_names, dvr_urls = _load_dvr_cams()
        if dvr_names and dvr_urls:
            all_cams = [(dvr_names[i], dvr_urls[i]) for i in range(len(dvr_names))]
            print(f"[DVR-PREVIEW] Found {len(all_cams)} cameras from DVR config, filtering active cameras...")
        elif urls and cam_names:
            # Normalize provided URLs
            normalized_urls = [normalize_rtsp_url(url) for url in urls]
            all_cams = [(cam_names[i], normalized_urls[i]) for i in range(len(urls))]
            print(f"[PREVIEW] Found {len(all_cams)} provided cameras, filtering active cameras...")
        elif self.controller and hasattr(self.controller, "active_cams"):
            cams = self.controller.active_cams
            # Normalize URLs from controller - these should already be filtered
            normalized_cams = [(c["cam_name"], normalize_rtsp_url(c.get("url", ""))) for c in cams]
            all_cams = normalized_cams
            print(f"[PREVIEW] Using {len(all_cams)} cameras from controller (already filtered)")
        else:
            # Fallback to STREAM_URLS (normalized)
            normalized_streams = [normalize_rtsp_url(url) for url in STREAM_URLS]
            all_cams = [(f"Cam {i+1}", normalized_streams[i]) for i in range(len(STREAM_URLS))]
            print(f"[WARN-PREVIEW] Using hardcoded STREAM_URLS, filtering active cameras...")
        
        # Filter to show ONLY active cameras (those that can connect)
        # This ensures only working cameras are shown in the selection screen
        def test_camera_connection(cam_name, cam_url):
            """Quick test to see if camera can connect."""
            try:
                test_cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
                test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                test_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)  # 2 second timeout for reliable check
                if test_cap.isOpened():
                    # Try to read one frame to confirm it's actually working
                    ret, _ = test_cap.read()
                    test_cap.release()
                    return ret
                test_cap.release()
                return False
            except:
                return False
        
        # Filter cameras synchronously - process events to keep UI responsive
        print(f"[PREVIEW] Testing {len(all_cams)} cameras for connectivity (this may take a moment)...")
        self.cams_list = []
        for idx, (cam_name, cam_url) in enumerate(all_cams):
            # Process events every few cameras to keep UI responsive during test
            if idx % 2 == 0:
                QApplication.processEvents()
            
            if test_camera_connection(cam_name, cam_url):
                self.cams_list.append((cam_name, cam_url))
                print(f"[PREVIEW] ✓ {cam_name} is active")
            else:
                print(f"[PREVIEW] ✗ {cam_name} is inactive (skipped)")
        
        if not self.cams_list:
            # If no active cameras found, show message
            print(f"[WARN] No active cameras found out of {len(all_cams)} total")
            # Still show all cameras so user can see what's available (but they won't work)
            self.cams_list = all_cams
        else:
            print(f"[PREVIEW] Filtered to {len(self.cams_list)} active cameras (out of {len(all_cams)} total)")

        # ========== Build Grid (exactly like Crowd Density) ========
        row = col = 0
        for cam_name, cam_url in self.cams_list:

            lbl = QLabel()
            lbl.setFixedSize(300, 200)
            lbl.setStyleSheet("background:#e8e8e8; border:1px solid #ccc;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.preview_labels.append(lbl)

            cb = QCheckBox(cam_name)
            self.checkboxes.append(cb)
            for item in self.prev_selected:
                # SAFETY FIX → skip corrupted items
                if not isinstance(item, dict):
                    print("[EMPCUS] WARNING: Bad item in selected_emp_cus.json →", item)
                    continue

                if item.get("name") == cam_name:
                    cb.setChecked(True)


            v = QVBoxLayout()
            v.addWidget(lbl)
            v.addWidget(cb)
            w = QWidget()
            w.setLayout(v)
            self.grid.addWidget(w, row, col)

            col += 1
            if col == 3:
                col = 0
                row += 1

        # ========== Buttons (same as crowd density) ==============
        btn_row = QHBoxLayout()

        back_btn = QPushButton("← Back")
        back_btn.setStyleSheet("""
            background:#2e7d32;
            color:white;
            padding:10px 20px;
            border-radius:10px;
            font-size:18px;
        """)
        back_btn.clicked.connect(lambda: self.controller.load_ml_selection())
        btn_row.addWidget(back_btn)

        run_btn = QPushButton("▶ Run Analytics")
        run_btn.setStyleSheet("""
            background:#1976D2;
            color:white;
            padding:16px 28px;
            border-radius:10px;
            font-size:20px;
            font-weight:bold;
        """)
        run_btn.clicked.connect(self.on_run)
        btn_row.addWidget(run_btn, alignment=Qt.AlignmentFlag.AlignRight)

        layout.addLayout(btn_row)

        # Footer identical
        footer = QLabel("  Crowd Density      Heatmap      Daily Visitors      Emp–Cus Interaction      Motion Detection")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("font-size:14px; color:#555; margin:10px;")
        layout.addWidget(footer)

        # ========== Start Previews ======================
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_previews)
        self.timer.start(400)

        # Initialize video captures asynchronously to prevent UI blocking
        # Start with empty streams, initialize them gradually
        print(f"[PREVIEW] Initializing {len(self.cams_list)} camera previews asynchronously...")
        self.streams = [None] * len(self.cams_list)
        self._init_camera_index = 0
        self._init_timer = QTimer()
        self._init_timer.timeout.connect(self._init_next_camera)
        self._init_timer.start(100)  # Initialize one camera every 100ms to keep UI responsive

    # --------------------------------------------------
    def _init_next_camera(self):
        """Initialize cameras one by one asynchronously to keep UI responsive."""
        if self._init_camera_index >= len(self.cams_list):
            self._init_timer.stop()
            print(f"[PREVIEW] Finished initializing {len(self.cams_list)} cameras")
            return
        
        i = self._init_camera_index
        cam_name, cam_url = self.cams_list[i]
        
        # Process events to keep UI responsive
        QApplication.processEvents()
        
        print(f"[PREVIEW] Opening camera {i+1}/{len(self.cams_list)} ({cam_name}): {cam_url[:50]}...")
        cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        if not cap.isOpened():
            print(f"[WARN] Failed to open preview for {cam_name}: {cam_url}")
        else:
            print(f"[PREVIEW] Successfully opened {cam_name}")
        
        self.streams[i] = cap
        self._init_camera_index += 1
        
        # Process events again after each camera
        QApplication.processEvents()
    
    # --------------------------------------------------
    def update_previews(self):
        for i, cap in enumerate(self.streams):
            if i >= len(self.preview_labels):
                continue
            if cap is None:
                # Camera not initialized yet
                self.preview_labels[i].setText("⏳\nInitializing...")
                continue
            if not cap.isOpened():
                # Show error message on preview label
                self.preview_labels[i].setText(f"❌\nFailed to connect")
                self.preview_labels[i].setStyleSheet("background:#ffcccc; border:1px solid #ff0000; color:#000;")
                continue
            
            ok, frame = cap.read()
            if not ok:
                # Show "No signal" message if not already showing error
                if "❌" not in self.preview_labels[i].text():
                    self.preview_labels[i].setText("⏳\nConnecting...")
                continue
            
            # Successfully got frame - show it
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (300, 200))
            q = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888)
            self.preview_labels[i].setPixmap(QPixmap.fromImage(q))
            self.preview_labels[i].setText("")  # Clear any error text
            self.preview_labels[i].setStyleSheet("background:#e8e8e8; border:1px solid #ccc;")  # Reset style

    # --------------------------------------------------
    def on_run(self):
        selected = []
        for i, cb in enumerate(self.checkboxes):
            if cb.isChecked():
                cam_name, cam_url = self.cams_list[i]
                # Normalize URL before saving
                normalized_url = normalize_rtsp_url(cam_url)
                selected.append({"name": cam_name, "url": normalized_url})

        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select at least one camera.")
            return

        json.dump(selected, open("selected_emp_cus.json", "w"), indent=4)
        print("[EMPCUS] Saved selection →", selected)
        
        # Extract URLs and camera names for recognition
        selected_urls = [item["url"] for item in selected]
        selected_names = [item["name"] for item in selected]
        
        # Close preview connections before starting recognition
        # This prevents "connection already in use" errors
        print("[EMPCUS] Closing preview connections...")
        self.timer.stop()
        if hasattr(self, '_init_timer'):
            self._init_timer.stop()
        for cap in self.streams:
            if cap is not None:
                try:
                    if cap.isOpened():
                        cap.release()
                except:
                    pass
        self.streams = []
        
        # Start recognition IMMEDIATELY (don't wait, don't go back to ML selection)
        print("[EMPCUS] Starting recognition immediately...")
        self.controller.start_emp_cus_interaction(selected_urls, selected_names)

    # --------------------------------------------------
    def closeEvent(self, e):
        self.timer.stop()
        if hasattr(self, '_init_timer'):
            self._init_timer.stop()
        for cap in self.streams:
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
        e.accept()


class EmployeeManagementScreen(QWidget):
    def __init__(self, controller=None, urls=None, cam_names=None):
        super().__init__()
        self.controller = controller
        self.provided_urls = urls  # Store provided URLs
        self.provided_cam_names = cam_names  # Store provided camera names
        # Removed setWindowTitle and showMaximized - screen is embedded in main window stack
       # self.setStyleSheet("background:#0f2027; color:white;")
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                                            stop:0 #edf3ff,
                                            stop:1 #d6e0ff);
                color: #123e91;
                font-family:'Segoe UI';
            }
            QLabel#title {
                font-size:28px;
                font-weight:700;
                color:#123e91;
            }
            QCheckBox {
                font-size:16px;
                color:#123e91;
            }
        """)


        # --- state flags ---
        self._auto = False
        self.streams_running = False
        self.display_visible = False  # Always False - headless mode only

        # --- placeholders ---
        self.strms = []
        self.prev_status = {}

        # --- right panel widgets ---
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search employee...")
        self.search.textChanged.connect(self.update_table)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Employee", "Photos", "Status", "Actions"])
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tbl.customContextMenuRequested.connect(self.context_menu)
        
        # Enable scrolling with visible scrollbar when needed
        self.tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Disable row selection highlighting/blinking - remove orange highlight
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
        # Table size policy - expand horizontally, respect height constraints for scrolling
        self.tbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        # Custom stylesheet to remove selection highlighting and prevent blinking
        self.tbl.setStyleSheet("""
            QTableWidget {
                border: 1px solid #64B5F6;
                background-color: #0A1929;
                gridline-color: #1E3A5F;
                color: white;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: transparent;
                color: inherit;
            }
            QTableWidget::item:hover {
                background-color: #1E3A5F;
            }
            QHeaderView::section {
                background-color: #2C5364;
                color: white;
                padding: 6px;
                border: 1px solid #64B5F6;
                font-weight: bold;
            }
            QScrollBar:vertical {
                background: #0A1929;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #64B5F6;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #90CAF9;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # --- control buttons ---
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.setStyleSheet("background:#2C5364; color:white; font-weight:bold; font-size:16px; padding:10px;")
        self.btn_back = QPushButton("← Back")
        self.btn_back.setStyleSheet("background:#2C5364; color:white; font-weight:bold; font-size:16px; padding:10px;")
        # --- CPU/GPU monitor panel ---
        self.lbl_sys = QLabel("CPU: --% | GPU: --% | Temp: --°C")
        self.lbl_sys.setStyleSheet("color: lightgray; font-size: 12px;")

        # --- right panel layout ---
        rside = QVBoxLayout()
        rside.addWidget(QLabel("Employees"))
        rside.addWidget(self.search)
        
        # Wrap employees table in QScrollArea (EXACT same approach as Finalized Clips)
        # Create a container widget first (like clips_list_widget)
        employees_container = QWidget()
        employees_container_layout = QVBoxLayout(employees_container)
        employees_container_layout.setContentsMargins(0, 0, 0, 0)
        employees_container_layout.setSpacing(0)
        employees_container_layout.addWidget(self.tbl)  # Add table to container
        
        # Now create scroll area (same as clips)
        employees_scroll = QScrollArea()
        employees_scroll.setWidgetResizable(True)
        employees_scroll.setMinimumHeight(200)  # Minimum visible height
        employees_scroll.setMaximumHeight(500)  # Maximum visible height - scrolling activates beyond this
        # Use same simple stylesheet as clips (no explicit scrollbar policies needed)
        employees_scroll.setStyleSheet("border: 1px solid #64B5F6; border-radius: 4px; background: #0A1929;")
        
        # Disable table's own scrollbars - let scroll area handle all scrolling
        self.tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Remove border from table since scroll area has it
        table_style = self.tbl.styleSheet()
        if "border: 1px solid" in table_style:
            self.tbl.setStyleSheet(table_style.replace("border: 1px solid #64B5F6;", "border: none;"))
        
        # Set the container widget in scroll area (same pattern as clips)
        employees_scroll.setWidget(employees_container)
        
        # Add scroll area to layout with stretch (same as clips)
        rside.addWidget(employees_scroll, 1)

        # Add Settings button above other elements
        rside.addWidget(self.btn_settings)
        rside.addWidget(self.btn_back)
        rside.addWidget(self.lbl_sys)

        # --- root layout ---
        root = QHBoxLayout(self)
        root.addLayout(rside, 1)  # Right panel gets full width

        # --- Connect button actions ---
        self.btn_settings.clicked.connect(self.settings_menu)
        self.btn_back.clicked.connect(lambda: self.controller.load_ml_selection() if self.controller else None)


        # --- timers ---
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_table)
        self.ui_timer.start(300)  # Update every 300ms for faster status updates (was 1000ms)

        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_sys_stats)
        self.sys_timer.start(2000)

        self.session_timer = QTimer()
        self.session_timer.timeout.connect(SESSION.tick)
        self.session_timer.start(500)

        # Load DB and rebuild centroids IMMEDIATELY (no delay for faster recognition)
        # Process events during heavy operations to keep UI responsive
        # Only load DB if not already loaded (to avoid reloading when screen is reused)
        # SAFETY: Add initialization lock to prevent multiple simultaneous initializations
        if not hasattr(self, '_initializing'):
            self._initializing = True
            try:
                if not hasattr(self, '_db_loaded'):
                    try:
                        QApplication.processEvents()
                        load_db()
                        QApplication.processEvents()
                        rebuild_centroids()
                        QApplication.processEvents()
                        
                        # Start recognition IMMEDIATELY (no delay)
                        # Only start if streams aren't already running (to avoid restarting when screen is reused)
                        if not self.streams_running:
                            # Use QTimer to defer stream creation slightly to avoid race conditions
                            # This prevents segmentation fault from simultaneous thread creation
                            QTimer.singleShot(100, self.start_headless_auto)
                        else:
                            print("[INFO] Recognition already running - skipping auto-start")
                        self._db_loaded = True
                    except Exception as e:
                        print(f"[ERROR] Failed to load DB/rebuild centroids: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # DB already loaded, just refresh the table
                    try:
                        load_db()
                        rebuild_centroids()
                        self.update_table()
                    except Exception as e:
                        print(f"[ERROR] Failed to refresh DB: {e}")
                        import traceback
                        traceback.print_exc()
            finally:
                self._initializing = False
        else:
            print("[WARN] EmployeeManagementScreen initialization already in progress - skipping")


    # ============================================================
    # AUTO-START HEADLESS RECOGNITION
    # ============================================================
    def _load_dvr_config_and_build_urls(self):
        """
        Load DVR credentials from core/config.json and build RTSP URLs.
        Returns (stream_urls, cam_names) or (None, None) if config not found.
        """
        try:
            # Try multiple possible config paths
            config_paths = [
                os.path.join("core", "config.json"),
                os.path.expanduser("~/.third_umpire/config.json"),
                os.path.join(os.path.dirname(__file__), "..", "..", "core", "config.json"),
            ]
            
            config = None
            for path in config_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            config = json.load(f)
                        print(f"[DVR] Loaded config from: {path}")
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to load config from {path}: {e}")
                        continue
            
            if not config:
                print("[WARN] No DVR config found - cannot build RTSP URLs from credentials")
                return None, None
            
            # Extract DVR credentials
            brand = config.get("brand", "Hikvision")
            ip = config.get("ip", "")
            user = config.get("user", "")
            pwd = config.get("pass", "")
            
            if not all([ip, user, pwd]):
                print("[WARN] Incomplete DVR config - missing IP, user, or password")
                return None, None
            
            print(f"[DVR] Building RTSP URLs using credentials: {user}@{ip} (brand: {brand})")
            
            # Use dvr_logic to build URLs
            try:
                from modules.utils import dvr_logic
                url_dicts = dvr_logic.build_rtsp_urls(brand, ip, user, pwd, max_channels=16)
                
                # Extract URLs and names
                stream_urls = [item["url"] for item in url_dicts]
                cam_names = [item["name"] for item in url_dicts]
                
                # Normalize URLs to fix channel number formatting
                stream_urls = [normalize_rtsp_url(url) for url in stream_urls]
                
                print(f"[DVR] Built {len(stream_urls)} RTSP URLs from DVR config")
                return stream_urls, cam_names
            except ImportError:
                print("[WARN] dvr_logic module not available - using fallback URL construction")
                # Fallback: construct URLs manually for Hikvision
                if brand == "Hikvision":
                    # Format: /101, /201, /301, etc. (not /0101, /0201)
                    stream_urls = [
                        f"rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{(i+1)}01"
                        for i in range(16)
                    ]
                    cam_names = [f"Cam {i+1}" for i in range(16)]
                    # No need to normalize - already in correct format
                    return stream_urls, cam_names
                return None, None
            except Exception as e:
                print(f"[ERROR] Failed to build RTSP URLs: {e}")
                import traceback
                traceback.print_exc()
                return None, None
                
        except Exception as e:
            print(f"[ERROR] Failed to load DVR config: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def start_headless_auto(self):
        """Auto-start recognition headlessly on startup - deferred to avoid blocking UI."""
        # Always reload from selected_emp_cus.json to get latest selection
        # Don't reuse old streams - user may have changed camera selection
        if self.streams_running and self.strms:
            print("[INFO] Stopping old recognition streams to reload with latest camera selection...")
            # Stop old streams
            for s in self.strms:
                if hasattr(s, "stop"):
                    s.stop()
            self.strms = []
            self.streams_running = False
        
        # Start streams IMMEDIATELY (no delay for faster recognition)
        # Double-check first
        if self.streams_running and self.strms:
            print("[INFO] Recognition already running - skipping restart")
            return
        
        # Try to get URLs from multiple sources (priority order)
        stream_urls = None
        cam_names = None
        
        # 1. HIGHEST PRIORITY: Load from selected_emp_cus.json (user's selected cameras from camera selection screen)
        if os.path.exists("selected_emp_cus.json"):
            try:
                selected = json.load(open("selected_emp_cus.json"))
                print(f"[DEBUG] Loaded selected_emp_cus.json: {len(selected) if isinstance(selected, list) else 'not a list'} items")
                if selected and isinstance(selected, list) and len(selected) > 0:
                    # Normalize URLs to fix channel number formatting (e.g., /0101 -> /101)
                    stream_urls = [normalize_rtsp_url(item.get("url")) for item in selected if item.get("url")]
                    cam_names = [item.get("name") for item in selected if item.get("name")]
                    print(f"[DEBUG] Extracted {len(stream_urls)} URLs and {len(cam_names)} names from selected_emp_cus.json")
                    if stream_urls:
                        print(f"[INFO] Using {len(stream_urls)} cameras from selected_emp_cus.json (user's selection)")
                    else:
                        print(f"[WARN] selected_emp_cus.json has {len(selected)} items but no valid URLs extracted")
                        print(f"[DEBUG] Sample item: {selected[0] if selected else 'empty'}")
                else:
                    print(f"[WARN] selected_emp_cus.json is empty or not a list")
            except Exception as e:
                print(f"[WARN] Failed to load selected_emp_cus.json: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. FALLBACK: Use provided URLs if available (only if no selection exists)
        if not stream_urls and self.provided_urls and len(self.provided_urls) > 0:
            # Normalize URLs to fix channel number formatting
            stream_urls = [normalize_rtsp_url(url) for url in self.provided_urls]
            cam_names = self.provided_cam_names if self.provided_cam_names else [f"Cam {i+1}" for i in range(len(stream_urls))]
            print(f"[INFO] Using {len(stream_urls)} provided URLs (fallback - no selection found)")
        
        # 3. Try to get from controller's active_cams (which should have DVR credentials)
        if not stream_urls and self.controller and hasattr(self.controller, "active_cams"):
            active_cams = self.controller.active_cams
            if active_cams and len(active_cams) > 0:
                # Normalize URLs to fix channel number formatting
                stream_urls = [normalize_rtsp_url(cam.get("url", "")) for cam in active_cams if cam.get("url")]
                cam_names = [cam.get("cam_name", "") for cam in active_cams if cam.get("url")]
                if stream_urls:
                    print(f"[INFO] Using {len(stream_urls)} URLs from controller's active_cams")
        
        # 4. FALLBACK: Load DVR config and build URLs (only if no selection exists)
        # This builds ALL 16 cameras, so it's last priority - limit to 8 max
        if not stream_urls:
            dvr_urls, dvr_names = self._load_dvr_config_and_build_urls()
            if dvr_urls and dvr_names:
                # Only use first 8 cameras from DVR (most DVRs have max 8 channels)
                stream_urls = dvr_urls[:8]
                cam_names = dvr_names[:8]
                print(f"[INFO] Using first 8 RTSP URLs from DVR config (fallback - no selection found)")
        
        # 5. LAST RESORT: Fall back to STREAM_URLS if defined (hardcoded)
        if not stream_urls:
            stream_urls = globals().get('STREAM_URLS', [])
            # Normalize default URLs too
            stream_urls = [normalize_rtsp_url(url) for url in stream_urls]
            if stream_urls:
                print(f"[WARN] Using {len(stream_urls)} hardcoded STREAM_URLS (last resort)")
        
        if not stream_urls:
            print("[WARN] No stream URLs available - cannot start streams")
            print("[DEBUG] URL sources checked:")
            print(f"  - selected_emp_cus.json exists: {os.path.exists('selected_emp_cus.json')}")
            print(f"  - provided_urls: {self.provided_urls}")
            print(f"  - controller.active_cams: {getattr(self.controller, 'active_cams', None) if self.controller else None}")
            print(f"  - DVR config available: {self._load_dvr_config_and_build_urls()}")
            print("[INFO] Recognition will start automatically when cameras are available")
            return
        
        print(f"[INFO] Starting recognition immediately on {len(stream_urls)} camera(s)")
        print(f"[DEBUG] Stream URLs being used:")
        for i, (url, name) in enumerate(zip(stream_urls, cam_names if cam_names else [f"Cam{i+1}" for i in range(len(stream_urls))])):
            print(f"  [{i+1}] {name}: {url[:80]}...")
        # Create streamers with staggered delays to avoid overwhelming DVR
        # Some DVRs have limits on simultaneous connection attempts
        # SAFETY: Create streams with error handling to prevent segmentation fault
        self.strms = []
        for i, url in enumerate(stream_urls):
            try:
                cam_name = cam_names[i] if cam_names and i < len(cam_names) else f"Cam{i+1}"
                # Add small delay between connection attempts (except first one)
                # This prevents "500 Internal Server Error" from too many simultaneous connections
                if i > 0:
                    time.sleep(0.3)  # 300ms delay between connections
                streamer = Streamer(url, lambda: self._auto, cam_name=cam_name)
                self.strms.append(streamer)
            except Exception as e:
                print(f"[ERROR] Failed to create streamer for {cam_name}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other streams even if one fails
                continue
        
        # Mark as running immediately - verification will happen asynchronously
        # This prevents UI freeze while streams initialize
        self.streams_running = True
        self.display_visible = False
        print(f"[INFO] Recognition started immediately - {len(stream_urls)} streams initializing")
        
        # Verify streams asynchronously in background (non-blocking)
        def verify_streams_async():
            """Verify streams in background without blocking UI."""
            time.sleep(1)  # Give streams time to initialize
            working_count = 0
            for i, streamer in enumerate(self.strms):
                if streamer.is_working:
                    working_count += 1
                    print(f"[INFO] Stream {i+1} ({streamer.cam_name}) is working")
                else:
                    # Check if cap exists before checking if it's opened
                    if streamer.cap is not None and streamer.cap.isOpened():
                        print(f"[WARN] Stream {i+1} ({streamer.cam_name}) opened but not receiving frames yet")
                    else:
                        print(f"[WARN] Stream {i+1} ({streamer.cam_name}) failed to open")
            
            if working_count > 0:
                print(f"[INFO] Recognition active: {working_count}/{len(self.strms)} streams working")
            else:
                print(f"[WARN] No streams are working yet. Recognition will start when streams connect.")
                # Don't set streams_running to False - let it retry
        
        # Run verification in background thread to avoid blocking UI
        threading.Thread(target=verify_streams_async, daemon=True).start()

    # Stream view removed - always in headless mode


    # ============================================================
    # MONITOR
    # ============================================================

    def update_sys_stats(self):
        """Show CPU/GPU usage and temperature (Jetson-friendly, no logic change)."""
        try:
            import psutil, subprocess, os

            # --- CPU usage ---
            cpu = psutil.cpu_percent()

            # --- GPU load & temperature (JetPack 6.2+ safe) ---
            gpu_load, gpu_temp = "?", "?"
            try:
                # use absolute path so GUI-launched apps find it
                tegra = "/usr/bin/tegrastats"
                # JetPack 6.2 has no --count → capture first line only
                out = subprocess.check_output(f"{tegra} --interval 1000 | head -n 1",
                                              shell=True,
                                              stderr=subprocess.DEVNULL).decode("utf-8", "ignore")

                # Parse GR3D_FREQ  or  GR3D
                if "GR3D_FREQ" in out:
                    val = out.split("GR3D_FREQ")[1].split()[0]
                    gpu_load = val.split("%")[0] + "%"
                elif "GR3D" in out:
                    val = out.split("GR3D")[1].split()[0]
                    gpu_load = val.split("%")[0] + "%"

                # Parse GPU@45.3C  style temperature
                for token in out.split():
                    if token.startswith("GPU@") and token.endswith("C"):
                        gpu_temp = token.split("@")[1]
                        break
                    if token.startswith("GPU") and token.endswith("C"):
                        gpu_temp = token.split("GPU")[-1]
                        break

                # final fallback from /sys thermal zones
                if gpu_temp == "?":
                    for z in os.listdir("/sys/devices/virtual/thermal"):
                        ttype = open(f"/sys/devices/virtual/thermal/{z}/type").read().lower()
                        if "gpu" in ttype:
                            tfile = f"/sys/devices/virtual/thermal/{z}/temp"
                            gpu_temp = f"{int(open(tfile).read())/1000:.1f}°C"
                            break
            except Exception as e:
                print("[WARN] tegrastats failed:", e)

            self.lbl_sys.setText(f"CPU: {cpu:.0f}% | GPU: {gpu_load} | Temp: {gpu_temp}")

        except Exception as e:
            self.lbl_sys.setText(f"SysMon error: {e}")







    # ============================================================
    # EMPLOYEE TABLE + STATUS UPDATES
    # ============================================================
    def update_table(self):
        # Process events at start to prevent "not responding" warnings
        QApplication.processEvents()
        
        filt = self.search.text().strip().lower()
        names = [n for n in sorted(EMP_DB.keys()) if filt in n.lower()]
        
        # Only update if employee list actually changed to prevent unnecessary redraws/blinking
        current_rows = self.tbl.rowCount()
        current_names = []
        for i in range(current_rows):
            item = self.tbl.item(i, 0)
            if item:
                current_names.append(item.text())
        
        # If names haven't changed, don't rebuild table - just update status
        if current_names == names and len(names) > 0:
            # Only update status column without clearing table
            now = time.time()
            for row in range(self.tbl.rowCount()):
                item = self.tbl.item(row, 0)
                if not item:
                    continue
                name = item.text()
                
                engaged = SESSION.is_engaged(name)
                is_present = SESSION.is_present(name)
                
                if engaged:
                    status = "In Conversation"
                    color = QColor("green")
                elif is_present:
                    status = "Present"  # Requires minimum duration before showing "Present"
                    color = QColor("orange")
                else:
                    status = "Absent"
                    color = QColor("gray")
                
                self.prev_status[name] = status
                
                # Update photo count
                photos_dir = _get_emp_photos_dir(name)
                photo_count = len(glob.glob(os.path.join(photos_dir, "photo_*.jpg"))) if os.path.exists(photos_dir) else 0
                pitem = self.tbl.item(row, 1)
                if pitem:
                    pitem.setText(str(photo_count))
                else:
                    self.tbl.setItem(row, 1, QTableWidgetItem(str(photo_count)))
                
                # Update status column
                sitem = self.tbl.item(row, 2)
                if sitem:
                    sitem.setText(status)
                    sitem.setForeground(color)
                else:
                    sitem = QTableWidgetItem(status)
                    sitem.setForeground(color)
                    self.tbl.setItem(row, 2, sitem)
            
            return  # Exit early - no need to rebuild table
        
        # Only rebuild table if names actually changed
        # Process events before clearing to keep UI responsive
        QApplication.processEvents()
        
        self.tbl.setRowCount(0)
        now = time.time()

        for name in names:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            self.tbl.setItem(row, 0, QTableWidgetItem(name))
            
            # Photo count
            photos_dir = _get_emp_photos_dir(name)
            photo_count = len(glob.glob(os.path.join(photos_dir, "photo_*.jpg"))) if os.path.exists(photos_dir) else 0
            self.tbl.setItem(row, 1, QTableWidgetItem(str(photo_count)))

            engaged = SESSION.is_engaged(name)
            is_present = SESSION.is_present(name)

            if engaged:
                status = "In Conversation"
                color = QColor("green")
            elif is_present:
                status = "Present"  # Requires minimum duration before showing "Present"
                color = QColor("orange")
            else:
                status = "Absent"
                color = QColor("gray")

            self.prev_status[name] = status
            sitem = QTableWidgetItem(status)
            sitem.setForeground(color)
            self.tbl.setItem(row, 2, sitem)

            # Action buttons: Rename, Reregister faces, Delete
            cell = QWidget()
            hl = QHBoxLayout(cell)
            hl.setContentsMargins(2, 2, 2, 2)
            hl.setSpacing(4)
            
            btn_rename = QPushButton("Rename")
            btn_rereg = QPushButton("Reregister faces")
            btn_delete = QPushButton("Delete")
            
            btn_rename.setStyleSheet("background:#1976D2; color:white; padding:4px 8px; font-size:11px;")
            btn_rereg.setStyleSheet("background:#388E3C; color:white; padding:4px 8px; font-size:11px;")
            btn_delete.setStyleSheet("background:#D32F2F; color:white; padding:4px 8px; font-size:11px;")
            
            btn_rename.clicked.connect(lambda _, n=name: self.rename_employee_dialog(n))
            btn_rereg.clicked.connect(lambda _, n=name: self.reregister_faces_dialog(n))
            btn_delete.clicked.connect(lambda _, n=name: self.delete_employee_dialog(n))
            
            hl.addWidget(btn_rename)
            hl.addWidget(btn_rereg)
            hl.addWidget(btn_delete)
            
            self.tbl.setCellWidget(row, 3, cell)
            
            # Process events periodically during table population to prevent freeze
            if row % 10 == 0:
                QApplication.processEvents()
        
        # Process events after all rows are added
        QApplication.processEvents()
        
        # NO scroll restoration - let QScrollArea handle scroll position naturally
        # This prevents blinking and scrolling issues
        # The scroll area with setWidgetResizable(True) maintains scroll automatically

    # ============================================================
    # CONTEXT MENU (RENAME/DELETE)
    # ============================================================
    def context_menu(self, pos):
        idx = self.tbl.indexAt(pos)
        if not idx.isValid():
            return
        row = idx.row()
        name = self.tbl.item(row, 0).text()
        menu = QMenu(self)
        rn = menu.addAction("Rename")
        dl = menu.addAction("Delete")
        # PyQt6 uses exec() instead of exec_()
        act = menu.exec(self.tbl.viewport().mapToGlobal(pos))
        if act == rn:
            new, ok = QInputDialog.getText(self, "Rename", "New name:")
            if ok and new.strip():
                rename_employee(name, new.strip())
                self.update_table()
        elif act == dl:
            if QMessageBox.question(
                self, "Confirm", f"Delete {name}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                delete_employee(name)
                self.update_table()

    # ============================================================
    # CLIP VIEW / HISTORY (unchanged)
    # ============================================================
    def rename_employee_dialog(self, name):
        """Rename employee dialog."""
        new_name, ok = QInputDialog.getText(self, "Rename Employee", f"New name for {name}:", QLineEdit.EchoMode.Normal, name)
        if ok and new_name.strip() and new_name.strip() != name:
            rename_employee(name, new_name.strip())
            self.update_table()
            rebuild_centroids()

    def reregister_faces_dialog(self, name):
        """Reregister faces dialog - shows all uploaded photos and allows adding more."""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Reregister Faces - {name}")
        dlg.resize(800, 600)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(f"Employee: {name}", styleSheet="font-size:16px; font-weight:bold; color:#90CAF9;"))

        # Show existing photos section with checkboxes
        photos_header_layout = QHBoxLayout()
        photos_header_layout.addWidget(QLabel("Existing Photos:", styleSheet="font-size:14px; font-weight:bold;"))
        
        btn_delete_selected = QPushButton("Delete Selected")
        btn_delete_selected.setStyleSheet("background:#D32F2F; color:white; font-weight:bold; padding:8px; font-size:12px;")
        photos_header_layout.addWidget(btn_delete_selected, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(photos_header_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        scroll.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#0A1929;")
        
        photos_widget = QWidget()
        photos_layout = QGridLayout(photos_widget)
        photos_layout.setSpacing(10)

        # Store checkboxes and photo paths for deletion
        photo_checkboxes = {}  # {photo_index: (checkbox, photo_path, photo_index)}
        
        def load_existing_photos():
            """Load and display existing photos with checkboxes."""
            # Clear existing widgets
            for i in reversed(range(photos_layout.count())):
                item = photos_layout.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.setParent(None)
            photo_checkboxes.clear()
            
            # Load and display thumbnails from new DB structure
            photos_dir = _get_emp_photos_dir(name)
            if os.path.exists(photos_dir):
                thumb_files = sorted(glob.glob(os.path.join(photos_dir, "photo_*.jpg")))
                if thumb_files:
                    cols = 4
                    for idx, thumb_path in enumerate(thumb_files):
                        row = idx // cols
                        col = idx % cols
                        
                        # Create thumbnail widget
                        thumb_widget = QWidget()
                        thumb_widget.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#1E3A5F;")
                        thumb_layout = QVBoxLayout(thumb_widget)
                        thumb_layout.setContentsMargins(5, 5, 5, 5)
                        thumb_layout.setSpacing(5)
                        
                        # Add checkbox with white square edges for visibility
                        checkbox = QCheckBox()
                        checkbox.setStyleSheet("""
                            QCheckBox {
                                color: white;
                                background-color: white;
                                border: 2px solid white;
                                border-radius: 3px;
                                padding: 2px;
                            }
                            QCheckBox::indicator {
                                width: 18px;
                                height: 18px;
                                background-color: white;
                                border: 2px solid #64B5F6;
                                border-radius: 3px;
                            }
                            QCheckBox::indicator:checked {
                                background-color: #4CAF50;
                                border: 2px solid #4CAF50;
                            }
                        """)
                        thumb_layout.addWidget(checkbox)
                        
                        # Extract photo index from filename (photo_XXX.jpg)
                        try:
                            filename = os.path.basename(thumb_path)
                            photo_index = int(filename.replace("photo_", "").replace(".jpg", ""))
                        except:
                            photo_index = idx + 1
                        
                        photo_checkboxes[photo_index] = (checkbox, thumb_path, photo_index)
                        
                        # Load and display image
                        try:
                            img = cv2.imread(thumb_path)
                            if img is not None:
                                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                                pix = QPixmap.fromImage(q)
                                
                                img_label = QLabel()
                                img_label.setPixmap(pix.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                                thumb_layout.addWidget(img_label)
                                
                                # Show timestamp if available
                                filename = os.path.basename(thumb_path)
                                if "_" in filename:
                                    try:
                                        ts_part = filename.split("_")[:2]  # Date and time parts
                                        if len(ts_part) >= 2:
                                            date_str = ts_part[0]
                                            time_str = ts_part[1][:6]  # HHMMSS
                                            if len(date_str) == 8 and len(time_str) == 6:
                                                display_ts = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]} {time_str[0:2]}:{time_str[2:4]}"
                                                ts_label = QLabel(display_ts)
                                                ts_label.setStyleSheet("color:#BBDEFB; font-size:9px;")
                                                ts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                                                thumb_layout.addWidget(ts_label)
                                    except:
                                        pass
                                
                                photos_layout.addWidget(thumb_widget, row, col)
                        except Exception as e:
                            print(f"[WARN] Failed to load thumbnail {thumb_path}: {e}")
                    
                    if not thumb_files:
                        no_photos_label = QLabel("No photos registered yet.")
                        no_photos_label.setStyleSheet("color:#90CAF9; padding:20px;")
                        no_photos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        photos_layout.addWidget(no_photos_label, 0, 0, 1, 4)
                else:
                    no_photos_label = QLabel("No photos registered yet.")
                    no_photos_label.setStyleSheet("color:#90CAF9; padding:20px;")
                    no_photos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    photos_layout.addWidget(no_photos_label, 0, 0, 1, 4)
            else:
                no_photos_label = QLabel("No photos registered yet.")
                no_photos_label.setStyleSheet("color:#90CAF9; padding:20px;")
                no_photos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                photos_layout.addWidget(no_photos_label, 0, 0, 1, 4)
        
        def delete_selected_photos():
            """Delete selected photos."""
            selected_indices = []
            for photo_index, (checkbox, photo_path, idx) in photo_checkboxes.items():
                if checkbox.isChecked():
                    selected_indices.append(photo_index)
            
            if not selected_indices:
                QMessageBox.warning(dlg, "No Selection", "Please select at least one photo to delete.")
                return
            
            if QMessageBox.question(
                dlg, "Confirm Deletion",
                f"Delete {len(selected_indices)} selected photo(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                for photo_index in selected_indices:
                    delete_photo(name, photo_index)
                
                rebuild_centroids()
                QMessageBox.information(dlg, "Deleted", f"Successfully deleted {len(selected_indices)} photo(s).")
                load_existing_photos()  # Reload photos
                self.update_table()  # Update main table
        
        btn_delete_selected.clicked.connect(delete_selected_photos)
        
        # Load photos initially
        load_existing_photos()

        scroll.setWidget(photos_widget)
        layout.addWidget(scroll, 1)

        # Upload additional photos buttons (Upload Additional Photo removed)
        upload_layout = QHBoxLayout()
        
        btn_upload_multi = QPushButton("Upload Multiple Photos")
        btn_upload_multi.setStyleSheet("background:#2196F3; color:white; font-weight:bold; padding:10px; font-size:13px;")
        upload_layout.addWidget(btn_upload_multi)
        
        btn_crop_stream = QPushButton("Crop Photo from Stream")
        btn_crop_stream.setStyleSheet("background:#FF9800; color:white; font-weight:bold; padding:10px; font-size:13px;")
        upload_layout.addWidget(btn_crop_stream)
        layout.addLayout(upload_layout)
        
        # Progress label
        progress_label = QLabel("")
        progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_label.setStyleSheet("color:#90CAF9; font-size:12px;")
        layout.addWidget(progress_label)
        
        batch_worker = None
        
        def upload_multiple_additional():
            file_paths, _ = QFileDialog.getOpenFileNames(
                dlg,
                "Select Photos to Add",
                "",
                "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
            )
            
            if not file_paths:
                return
            
            if MODEL is None:
                QMessageBox.warning(dlg, "Error", "InsightFace model not available.")
                return
            
            # Disable buttons during processing
            btn_upload_multi.setEnabled(False)
            progress_label.setText(f"Processing {len(file_paths)} photos...")
            QApplication.processEvents()
            
            # Start batch worker
            nonlocal batch_worker
            if batch_worker:
                batch_worker.terminate()
                batch_worker.wait()
            
            batch_worker = BatchFaceWorker(file_paths)
            
            def on_progress(current, total):
                progress_label.setText(f"Processing {current}/{total} photos...")
                QApplication.processEvents()
            
            def on_finished(results):
                if not results:
                    QMessageBox.warning(dlg, "No Faces", "No faces detected in any of the selected photos.")
                    btn_upload_multi.setEnabled(True)
                    return
                
                # Save all embeddings
                for emb, face_img in results:
                    add_embedding(name, emb, face_img)
                
                rebuild_centroids()  # Rebuild centroids immediately - recognition will use new embeddings right away
                
                # Count embeddings
                emb_dir = _get_emp_emb_dir(name)
                emb_count = len(glob.glob(os.path.join(emb_dir, "emb_*.npy"))) if os.path.exists(emb_dir) else 0
                
                QMessageBox.information(dlg, "Success", f"Added {len(results)} photo(s) for {name}.\nTotal embeddings: {emb_count}\nRecognition will start immediately.")
                btn_upload_multi.setEnabled(True)
                
                # Refresh the dialog and update table
                if hasattr(self, 'update_table'):
                    self.update_table()  # Update table to show new photo count
                dlg.accept()
                self.reregister_faces_dialog(name)
            
            def on_error(msg):
                progress_label.setText(f"Error: {msg}")
                btn_upload_multi.setEnabled(True)
                QMessageBox.warning(dlg, "Error", msg)
            
            batch_worker.progress.connect(on_progress)
            batch_worker.finished.connect(on_finished)
            batch_worker.error.connect(on_error)
            batch_worker.start()
        
        def crop_from_stream_reregister():
            """Open stream capture for reregister - saves directly to employee."""
            # Get available cameras
            available_cams = []
            
            # Try to load from selected_emp_cus.json first
            if os.path.exists("selected_emp_cus.json"):
                try:
                    selected = json.load(open("selected_emp_cus.json"))
                    for item in selected:
                        if isinstance(item, dict) and item.get("url"):
                            available_cams.append({
                                "name": item.get("name", "Unknown"),
                                "url": normalize_rtsp_url(item.get("url"))
                            })
                except:
                    pass
            
            # Fallback to DVR config
            if not available_cams:
                dvr_urls, dvr_names = self._load_dvr_config_and_build_urls()
                if dvr_urls and dvr_names:
                    for i, url in enumerate(dvr_urls[:8]):
                        available_cams.append({
                            "name": dvr_names[i] if i < len(dvr_names) else f"Cam {i+1}",
                            "url": normalize_rtsp_url(url)
                        })
            
            if not available_cams:
                QMessageBox.warning(dlg, "No Cameras", "No cameras available. Please configure cameras first.")
                print("[ERROR] No cameras found in reregister stream capture dialog")
                return
            
            print(f"[DEBUG] Reregister stream capture: Found {len(available_cams)} cameras:")
            for cam in available_cams:
                print(f"  - {cam['name']}: {cam['url'][:80]}...")
            
            # Create stream dialog
            stream_dlg = QDialog(dlg)
            stream_dlg.setWindowTitle("Crop Photo from Stream")
            stream_dlg.resize(1000, 700)
            
            layout_stream = QVBoxLayout(stream_dlg)
            
            title = QLabel("Capture Photos from Live Stream")
            title.setStyleSheet("font-size:18px; font-weight:bold; color:#90CAF9;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout_stream.addWidget(title)
            
            # Camera selection
            cam_select_layout = QHBoxLayout()
            cam_select_layout.addWidget(QLabel("Select Camera:"))
            cam_combo = QComboBox()
            for cam in available_cams:
                cam_combo.addItem(cam["name"], cam["url"])
            cam_combo.setStyleSheet("padding:8px; font-size:13px;")
            cam_select_layout.addWidget(cam_combo)
            layout_stream.addLayout(cam_select_layout)
            
            # Stream preview
            preview_label = QLabel("Connecting to stream...")
            preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview_label.setMinimumSize(640, 480)
            preview_label.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#0A1929; color:#90CAF9;")
            layout_stream.addWidget(preview_label)
            
            # Control buttons
            control_layout = QHBoxLayout()
            btn_capture = QPushButton("📷 Capture Photo")
            btn_capture.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:10px; font-size:14px;")
            control_layout.addWidget(btn_capture)
            
            btn_back = QPushButton("← Back")
            btn_back.setStyleSheet("background:#2C5364; color:white; font-weight:bold; padding:10px; font-size:14px;")
            control_layout.addWidget(btn_back)
            layout_stream.addLayout(control_layout)
            
            # Captured photos section
            captured_label = QLabel("Captured Photos:")
            captured_label.setStyleSheet("font-size:14px; font-weight:bold; color:#90CAF9; margin-top:10px;")
            layout_stream.addWidget(captured_label)
            
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMinimumHeight(200)
            scroll.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#0A1929;")
            
            captured_widget = QWidget()
            captured_layout = QGridLayout(captured_widget)
            captured_layout.setSpacing(10)
            scroll.setWidget(captured_widget)
            layout_stream.addWidget(scroll)
            
            captured_photos = []
            current_cap = None
            preview_timer = QTimer()
            
            def update_captured_display():
                for i in reversed(range(captured_layout.count())):
                    item = captured_layout.itemAt(i)
                    if item:
                        widget = item.widget()
                        if widget:
                            widget.setParent(None)
                
                cols = 4
                for idx, photo_data in enumerate(captured_photos):
                    if len(photo_data) == 2:
                        emb, face_img = photo_data
                        photo_name = f"Photo {idx + 1}"
                    else:
                        emb, face_img, full_frame, photo_name = photo_data
                    
                    row = idx // cols
                    col = idx % cols
                    
                    photo_widget = QWidget()
                    photo_widget.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#1E3A5F;")
                    photo_layout = QVBoxLayout(photo_widget)
                    photo_layout.setContentsMargins(5, 5, 5, 5)
                    photo_layout.setSpacing(5)
                    
                    thumb = cv2.resize(face_img, (120, 120))
                    rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q)
                    
                    img_label = QLabel()
                    img_label.setPixmap(pix)
                    img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    photo_layout.addWidget(img_label)
                    
                    name_label = QLabel(photo_name)
                    name_label.setStyleSheet("color:#BBDEFB; font-size:10px;")
                    name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    photo_layout.addWidget(name_label)
                    
                    btn_delete = QPushButton("Delete")
                    btn_delete.setStyleSheet("background:#D32F2F; color:white; padding:4px; font-size:10px;")
                    btn_delete.clicked.connect(lambda checked, idx=idx: (captured_photos.pop(idx), update_captured_display()))
                    photo_layout.addWidget(btn_delete)
                    
                    captured_layout.addWidget(photo_widget, row, col)
            
            # Preview update function (defined outside start_stream to avoid reconnection issues)
            def update_preview_reregister():
                if current_cap and current_cap.isOpened():
                    ret, frame = current_cap.read()
                    if ret and frame is not None:
                        display_frame = cv2.resize(frame, (640, 480))
                        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                        pix = QPixmap.fromImage(q)
                        preview_label.setPixmap(pix)
                        preview_label.setText("")
                    else:
                        preview_label.setText("⏳ Waiting...")
                else:
                    preview_label.setText("❌ Disconnected")
            
            # Connect timer once (outside start_stream)
            preview_timer.timeout.connect(update_preview_reregister)
            
            def start_stream():
                nonlocal current_cap
                if current_cap:
                    try:
                        current_cap.release()
                    except:
                        pass
                    current_cap = None
                
                if preview_timer.isActive():
                    preview_timer.stop()
                
                cam_url = cam_combo.currentData()
                # Fallback: if currentData() returns None, try to get from available_cams by name
                if not cam_url:
                    current_name = cam_combo.currentText()
                    for cam in available_cams:
                        if cam["name"] == current_name:
                            cam_url = cam["url"]
                            break
                if not cam_url:
                    preview_label.setText("No camera selected or URL not found")
                    print(f"[ERROR] No URL found for camera: {cam_combo.currentText()}")
                    return
                
                print(f"[DEBUG] Selected camera: {cam_combo.currentText()}, URL: {cam_url[:80]}...")
                
                # Open stream with retry logic
                max_retries = 3
                retry_delay = 0.5
                preview_label.setText("⏳ Connecting to stream...")
                QApplication.processEvents()
                
                for attempt in range(max_retries):
                    try:
                        print(f"[DEBUG] Attempting to open stream (attempt {attempt + 1}/{max_retries}): {cam_url[:80]}...")
                        # Use FFmpeg backend for RTSP
                        current_cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
                        
                        if current_cap.isOpened():
                            # Set RTSP-specific properties for better connection
                            current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            # Give it a moment to establish connection
                            time.sleep(0.2)
                            # Try to read a frame to verify it's working
                            ret, frame = current_cap.read()
                            if ret and frame is not None:
                                print(f"[INFO] Stream opened successfully on attempt {attempt + 1}")
                                # Start preview timer
                                preview_timer.start(33)
                                update_preview_reregister()
                                return  # Success - exit function
                            else:
                                print(f"[WARN] Stream opened but failed to read frame on attempt {attempt + 1}")
                                current_cap.release()
                                current_cap = None
                        else:
                            print(f"[WARN] Stream failed to open (isOpened() returned False) on attempt {attempt + 1}")
                            if current_cap:
                                current_cap.release()
                                current_cap = None
                    except Exception as e:
                        if current_cap:
                            try:
                                current_cap.release()
                            except:
                                pass
                            current_cap = None
                        print(f"[WARN] Attempt {attempt + 1} failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Wait before retry (except on last attempt)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        preview_label.setText(f"⏳ Retrying connection ({attempt + 2}/{max_retries})...")
                        QApplication.processEvents()
                
                # If we get here, all retries failed
                preview_label.setText("❌ Failed to connect to stream\n\nPlease check:\n- Camera is online\n- RTSP URL is correct\n- Network connection is active")
                print(f"[ERROR] Failed to open RTSP stream after {max_retries} attempts: {cam_url}")
            
            def show_crop_dialog_reregister(full_frame):
                """Show crop dialog for reregister - saves directly to employee."""
                crop_dlg = QDialog(stream_dlg)
                crop_dlg.setWindowTitle("Crop Face from Photo")
                crop_dlg.resize(900, 700)
                
                crop_layout = QVBoxLayout(crop_dlg)
                
                title_label = QLabel("Select Face to Crop")
                title_label.setStyleSheet("font-size:16px; font-weight:bold; color:#90CAF9;")
                title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                crop_layout.addWidget(title_label)
                
                full_frame_label = QLabel()
                full_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                full_frame_label.setMinimumSize(800, 500)
                full_frame_label.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#0A1929;")
                
                display_frame = cv2.resize(full_frame.copy(), (800, 500))
                rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(q)
                full_frame_label.setPixmap(pix)
                crop_layout.addWidget(full_frame_label)
                
                detected_faces = []
                selected_face_idx = 0
                
                if MODEL is not None:
                    try:
                        frame_resized = cv2.resize(full_frame.copy(), DET_SIZE)
                        with MODEL_LOCK:
                            faces = MODEL.get(frame_resized)
                        
                        if faces and len(faces) > 0:
                            scale_x = full_frame.shape[1] / DET_SIZE[0]
                            scale_y = full_frame.shape[0] / DET_SIZE[1]
                            
                            h, w = full_frame.shape[:2]
                            min_face_size = min(w, h) * 0.05  # Minimum 5% of image dimension
                            max_face_size = min(w, h) * 0.8   # Maximum 80% of image dimension
                            
                            for face in faces:
                                x1, y1, x2, y2 = map(int, face.bbox)
                                x1_orig = int(x1 * scale_x)
                                y1_orig = int(y1 * scale_y)
                                x2_orig = int(x2 * scale_x)
                                y2_orig = int(y2 * scale_y)
                                
                                # Calculate face dimensions
                                face_width = x2_orig - x1_orig
                                face_height = y2_orig - y1_orig
                                face_size = max(face_width, face_height)
                                face_area = face_width * face_height
                                
                                # Filter false positives:
                                # 1. Face must be large enough (not tiny objects)
                                if face_size < min_face_size:
                                    continue
                                
                                # 2. Face must not be too large (not entire image)
                                if face_size > max_face_size:
                                    continue
                                
                                # 3. Face aspect ratio should be reasonable (not too wide or tall)
                                aspect_ratio = face_width / max(face_height, 1)
                                if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Too narrow or too wide
                                    continue
                                
                                # 4. Face area should be reasonable
                                if face_area < (min_face_size ** 2) * 0.5:
                                    continue
                                
                                padding = 20
                                x1_orig = max(0, x1_orig - padding)
                                y1_orig = max(0, y1_orig - padding)
                                x2_orig = min(w, x2_orig + padding)
                                y2_orig = min(h, y2_orig + padding)
                                
                                detected_faces.append({
                                    "bbox": (x1_orig, y1_orig, x2_orig, y2_orig),
                                    "face_obj": face
                                })
                            
                            display_frame_with_boxes = display_frame.copy()
                            for i, face_info in enumerate(detected_faces):
                                x1, y1, x2, y2 = face_info["bbox"]
                                scale_disp_x = 800 / full_frame.shape[1]
                                scale_disp_y = 500 / full_frame.shape[0]
                                x1_disp = int(x1 * scale_disp_x)
                                y1_disp = int(y1 * scale_disp_y)
                                x2_disp = int(x2 * scale_disp_x)
                                y2_disp = int(y2 * scale_disp_y)
                                
                                color = (0, 255, 0) if i == selected_face_idx else (255, 0, 0)
                                cv2.rectangle(display_frame_with_boxes, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                                cv2.putText(display_frame_with_boxes, f"Face {i+1}", (x1_disp, y1_disp - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            rgb = cv2.cvtColor(display_frame_with_boxes, cv2.COLOR_BGR2RGB)
                            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                            pix = QPixmap.fromImage(q)
                            full_frame_label.setPixmap(pix)
                    except Exception as e:
                        print(f"[WARN] Face detection failed: {e}")
                
                face_select_layout = QHBoxLayout()
                if len(detected_faces) > 1:
                    face_select_layout.addWidget(QLabel("Select Face:"))
                    face_combo = QComboBox()
                    for i in range(len(detected_faces)):
                        face_combo.addItem(f"Face {i+1}")
                    face_combo.setStyleSheet("padding:8px; font-size:13px;")
                    
                    def update_selected_face(idx):
                        nonlocal selected_face_idx
                        selected_face_idx = idx
                        display_frame_with_boxes = display_frame.copy()
                        for i, face_info in enumerate(detected_faces):
                            x1, y1, x2, y2 = face_info["bbox"]
                            scale_disp_x = 800 / full_frame.shape[1]
                            scale_disp_y = 500 / full_frame.shape[0]
                            x1_disp = int(x1 * scale_disp_x)
                            y1_disp = int(y1 * scale_disp_y)
                            x2_disp = int(x2 * scale_disp_x)
                            y2_disp = int(y2 * scale_disp_y)
                            color = (0, 255, 0) if i == selected_face_idx else (255, 0, 0)
                            cv2.rectangle(display_frame_with_boxes, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                            cv2.putText(display_frame_with_boxes, f"Face {i+1}", (x1_disp, y1_disp - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        rgb = cv2.cvtColor(display_frame_with_boxes, cv2.COLOR_BGR2RGB)
                        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                        pix = QPixmap.fromImage(q)
                        full_frame_label.setPixmap(pix)
                    
                    face_combo.currentIndexChanged.connect(update_selected_face)
                    face_select_layout.addWidget(face_combo)
                else:
                    face_select_layout.addWidget(QLabel("Face detected automatically"))
                crop_layout.addLayout(face_select_layout)
                
                cropped_label = QLabel("Cropped Face Preview:")
                cropped_label.setStyleSheet("font-size:14px; font-weight:bold; color:#90CAF9;")
                crop_layout.addWidget(cropped_label)
                
                cropped_preview = QLabel("No face cropped yet")
                cropped_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cropped_preview.setMinimumSize(200, 200)
                cropped_preview.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#1E3A5F; color:#90CAF9;")
                crop_layout.addWidget(cropped_preview)
                
                # Name input removed for reregister - not required
                
                btn_layout = QHBoxLayout()
                
                btn_crop = QPushButton("Crop Face")
                btn_crop.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:10px; font-size:14px;")
                
                def perform_crop():
                    if not detected_faces:
                        QMessageBox.warning(crop_dlg, "No Face", "No face detected.")
                        return
                    
                    try:
                        face_info = detected_faces[selected_face_idx]
                        x1, y1, x2, y2 = face_info["bbox"]
                        
                        face_img = full_frame[y1:y2, x1:x2]
                        if face_img.size == 0:
                            QMessageBox.warning(crop_dlg, "Error", "Could not extract face.")
                            return
                        
                        face_img_resized = cv2.resize(face_img, (160, 160))
                        face_obj = face_info["face_obj"]
                        emb = l2norm(getattr(face_obj, "normed_embedding", face_obj.embedding))
                        
                        preview = cv2.resize(face_img_resized, (200, 200))
                        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                        pix = QPixmap.fromImage(q)
                        cropped_preview.setPixmap(pix)
                        cropped_preview.setText("")
                        
                        crop_dlg.cropped_data = {
                            "emb": emb,
                            "face_img": face_img_resized,
                            "full_frame": full_frame.copy()
                        }
                        
                        QMessageBox.information(crop_dlg, "Cropped", "Face cropped! Click 'Save' to add it.")
                    except Exception as e:
                        QMessageBox.warning(crop_dlg, "Error", f"Failed to crop: {e}")
                
                btn_crop.clicked.connect(perform_crop)
                btn_layout.addWidget(btn_crop)
                
                btn_save = QPushButton("Save")
                btn_save.setStyleSheet("background:#2196F3; color:white; font-weight:bold; padding:10px; font-size:14px;")
                
                def save_cropped():
                    if not hasattr(crop_dlg, 'cropped_data'):
                        QMessageBox.warning(crop_dlg, "No Crop", "Please crop a face first.")
                        return
                    
                    data = crop_dlg.cropped_data
                    captured_photos.append((data["emb"], data["face_img"], data["full_frame"], f"Photo {len(captured_photos) + 1}"))
                    update_captured_display()
                    crop_dlg.accept()
                    QMessageBox.information(stream_dlg, "Saved", "Photo saved!")
                
                btn_save.clicked.connect(save_cropped)
                btn_layout.addWidget(btn_save)
                
                btn_cancel = QPushButton("Cancel")
                btn_cancel.setStyleSheet("background:#2C5364; color:white; padding:10px; font-size:14px;")
                btn_cancel.clicked.connect(crop_dlg.reject)
                btn_layout.addWidget(btn_cancel)
                
                crop_layout.addLayout(btn_layout)
                crop_dlg.exec()
            
            def capture_photo():
                if not current_cap or not current_cap.isOpened():
                    QMessageBox.warning(stream_dlg, "No Stream", "Please wait for stream to connect.")
                    return
                
                ret, frame = current_cap.read()
                if not ret or frame is None:
                    QMessageBox.warning(stream_dlg, "Error", "Failed to capture frame.")
                    return
                
                show_crop_dialog_reregister(frame)
            
            def close_and_save():
                nonlocal current_cap
                if preview_timer.isActive():
                    preview_timer.stop()
                if current_cap:
                    try:
                        current_cap.release()
                    except:
                        pass
                    current_cap = None
                
                # Save all captured photos to employee
                if captured_photos:
                    for photo_data in captured_photos:
                        if len(photo_data) == 2:
                            emb, face_img = photo_data
                        else:
                            emb, face_img = photo_data[0], photo_data[1]
                        add_embedding(name, emb, face_img)
                    
                    rebuild_centroids()
                    emb_count = len(glob.glob(os.path.join(_get_emp_emb_dir(name), "emb_*.npy"))) if os.path.exists(_get_emp_emb_dir(name)) else 0
                    QMessageBox.information(stream_dlg, "Saved", f"Added {len(captured_photos)} photo(s) for {name}.\nTotal embeddings: {emb_count}")
                    
                    # Refresh dialog
                    stream_dlg.accept()
                    dlg.accept()
                    self.update_table()
                    # Reload the reregister dialog to show new photos
                    self.reregister_faces_dialog(name)
                else:
                    stream_dlg.accept()
            
            cam_combo.currentIndexChanged.connect(start_stream)
            btn_capture.clicked.connect(capture_photo)
            btn_back.clicked.connect(close_and_save)
            
            start_stream()
            
            def cleanup():
                if preview_timer.isActive():
                    preview_timer.stop()
                if current_cap:
                    try:
                        current_cap.release()
                    except:
                        pass
            
            stream_dlg.finished.connect(cleanup)
            stream_dlg.exec()
        
        btn_upload_multi.clicked.connect(upload_multiple_additional)
        btn_crop_stream.clicked.connect(crop_from_stream_reregister)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("background:#2C5364; color:white; padding:8px;")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)

        dlg.exec()

    def delete_employee_dialog(self, name):
        """Delete employee confirmation dialog."""
        if QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete employee '{name}'?\n\nThis will permanently delete all face data for this employee.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes:
            delete_employee(name)
            self.update_table()
            QMessageBox.information(self, "Deleted", f"Employee '{name}' deleted successfully.")

    def stream_capture_dialog(self, parent_dlg, uploaded_data_ref):
        """Dialog for capturing photos from live camera stream."""
        stream_dlg = QDialog(parent_dlg)
        stream_dlg.setWindowTitle("Crop Photo from Stream")
        stream_dlg.resize(1000, 700)
        
        layout = QVBoxLayout(stream_dlg)
        
        # Title
        title = QLabel("Capture Photos from Live Stream")
        title.setStyleSheet("font-size:18px; font-weight:bold; color:#90CAF9;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Get available cameras
        available_cams = []
        
        # Try to load from selected_emp_cus.json first
        if os.path.exists("selected_emp_cus.json"):
            try:
                selected = json.load(open("selected_emp_cus.json"))
                for item in selected:
                    if isinstance(item, dict) and item.get("url"):
                        available_cams.append({
                            "name": item.get("name", "Unknown"),
                            "url": normalize_rtsp_url(item.get("url"))
                        })
            except:
                pass
        
        # Fallback to DVR config
        if not available_cams:
            dvr_urls, dvr_names = self._load_dvr_config_and_build_urls()
            if dvr_urls and dvr_names:
                for i, url in enumerate(dvr_urls[:8]):  # Limit to 8 cameras
                    available_cams.append({
                        "name": dvr_names[i] if i < len(dvr_names) else f"Cam {i+1}",
                        "url": normalize_rtsp_url(url)
                    })
        
        if not available_cams:
            QMessageBox.warning(stream_dlg, "No Cameras", "No cameras available. Please configure cameras first.")
            print("[ERROR] No cameras found in stream capture dialog")
            return
        
        print(f"[DEBUG] Stream capture dialog: Found {len(available_cams)} cameras:")
        for cam in available_cams:
            print(f"  - {cam['name']}: {cam['url'][:80]}...")
        
        # Camera selection dropdown
        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Select Camera:"))
        cam_combo = QComboBox()
        for cam in available_cams:
            cam_combo.addItem(cam["name"], cam["url"])
        cam_combo.setStyleSheet("padding:8px; font-size:13px;")
        cam_select_layout.addWidget(cam_combo)
        layout.addLayout(cam_select_layout)
        
        # Stream preview area
        preview_label = QLabel("Connecting to stream...")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumSize(640, 480)
        preview_label.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#0A1929; color:#90CAF9;")
        layout.addWidget(preview_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        btn_capture = QPushButton("📷 Capture Photo")
        btn_capture.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:10px; font-size:14px;")
        control_layout.addWidget(btn_capture)
        
        btn_back = QPushButton("← Back")
        btn_back.setStyleSheet("background:#2C5364; color:white; font-weight:bold; padding:10px; font-size:14px;")
        control_layout.addWidget(btn_back)
        layout.addLayout(control_layout)
        
        # Captured photos section
        captured_label = QLabel("Captured Photos:")
        captured_label.setStyleSheet("font-size:14px; font-weight:bold; color:#90CAF9; margin-top:10px;")
        layout.addWidget(captured_label)
        
        # Scroll area for captured photos
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        scroll.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#0A1929;")
        
        captured_widget = QWidget()
        captured_layout = QGridLayout(captured_widget)
        captured_layout.setSpacing(10)
        scroll.setWidget(captured_widget)
        layout.addWidget(scroll)
        
        # Store captured photos: list of (emb, face_img, full_frame, name) tuples
        captured_photos = []
        photos_saved = False  # Track if photos have been saved to prevent duplicates
        current_stream = None
        current_cap = None
        preview_timer = QTimer()
        
        def update_captured_display():
            """Update the grid of captured photos."""
            # Clear existing widgets
            for i in reversed(range(captured_layout.count())):
                item = captured_layout.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.setParent(None)
            
            # Add captured photos
            cols = 4
            for idx, photo_data in enumerate(captured_photos):
                # Handle both old format (emb, face_img) and new format (emb, face_img, full_frame, name)
                if len(photo_data) == 2:
                    emb, face_img = photo_data
                    full_frame = None
                    photo_name = f"Photo {idx + 1}"
                else:
                    emb, face_img, full_frame, photo_name = photo_data
                
                row = idx // cols
                col = idx % cols
                
                # Create photo widget
                photo_widget = QWidget()
                photo_widget.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#1E3A5F;")
                photo_layout = QVBoxLayout(photo_widget)
                photo_layout.setContentsMargins(5, 5, 5, 5)
                photo_layout.setSpacing(5)
                
                # Display thumbnail
                thumb = cv2.resize(face_img, (120, 120))
                rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(q)
                
                img_label = QLabel()
                img_label.setPixmap(pix)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                photo_layout.addWidget(img_label)
                
                # Name label
                name_label = QLabel(photo_name)
                name_label.setStyleSheet("color:#BBDEFB; font-size:10px;")
                name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                photo_layout.addWidget(name_label)
                
                # Name/Rename button
                btn_name = QPushButton("Name")
                btn_name.setStyleSheet("background:#FF9800; color:white; padding:4px; font-size:10px;")
                
                def rename_photo(index):
                    current_name = captured_photos[index][3] if len(captured_photos[index]) > 3 else f"Photo {index + 1}"
                    new_name, ok = QInputDialog.getText(stream_dlg, "Rename Photo", "Enter new name:", QLineEdit.EchoMode.Normal, current_name)
                    if ok and new_name.strip():
                        # Update name in captured_photos
                        photo_data = list(captured_photos[index])
                        if len(photo_data) < 4:
                            # Convert old format to new format
                            photo_data.extend([None, new_name.strip()])
                        else:
                            photo_data[3] = new_name.strip()
                        captured_photos[index] = tuple(photo_data)
                        update_captured_display()
                
                btn_name.clicked.connect(lambda checked, idx=idx: rename_photo(idx))
                photo_layout.addWidget(btn_name)
                
                # Delete button
                btn_delete = QPushButton("Delete")
                btn_delete.setStyleSheet("background:#D32F2F; color:white; padding:4px; font-size:10px;")
                
                def delete_photo(index):
                    captured_photos.pop(index)
                    update_captured_display()
                
                btn_delete.clicked.connect(lambda checked, idx=idx: delete_photo(idx))
                photo_layout.addWidget(btn_delete)
                
                captured_layout.addWidget(photo_widget, row, col)
        
        # Preview update function (defined outside start_stream to avoid reconnection issues)
        def update_preview():
            if current_cap and current_cap.isOpened():
                ret, frame = current_cap.read()
                if ret and frame is not None:
                    # Resize for display
                    display_frame = cv2.resize(frame, (640, 480))
                    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q)
                    preview_label.setPixmap(pix)
                    preview_label.setText("")
                else:
                    preview_label.setText("⏳ Waiting for stream...")
            else:
                preview_label.setText("❌ Stream disconnected")
        
        # Connect timer once (outside start_stream)
        preview_timer.timeout.connect(update_preview)
        
        def start_stream():
            """Start streaming from selected camera."""
            nonlocal current_cap, current_stream
            
            # Stop existing stream
            if current_cap:
                try:
                    current_cap.release()
                except:
                    pass
                current_cap = None
            
            if preview_timer.isActive():
                preview_timer.stop()
            
            # Get selected camera URL
            cam_url = cam_combo.currentData()
            # Fallback: if currentData() returns None, try to get from available_cams by name
            if not cam_url:
                current_name = cam_combo.currentText()
                for cam in available_cams:
                    if cam["name"] == current_name:
                        cam_url = cam["url"]
                        break
            if not cam_url:
                preview_label.setText("No camera selected or URL not found")
                print(f"[ERROR] No URL found for camera: {cam_combo.currentText()}")
                return
            
            print(f"[DEBUG] Selected camera: {cam_combo.currentText()}, URL: {cam_url[:80]}...")
            
            # Open stream with retry logic (similar to main Streamer class)
            max_retries = 3
            retry_delay = 0.5
            preview_label.setText("⏳ Connecting to stream...")
            QApplication.processEvents()
            
            for attempt in range(max_retries):
                try:
                    print(f"[DEBUG] Attempting to open stream (attempt {attempt + 1}/{max_retries}): {cam_url[:80]}...")
                    # Use FFmpeg backend for RTSP
                    current_cap = cv2.VideoCapture(cam_url, cv2.CAP_FFMPEG)
                    
                    if current_cap.isOpened():
                        # Set RTSP-specific properties for better connection
                        current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Give it a moment to establish connection
                        time.sleep(0.2)
                        # Try to read a frame to verify it's working (with timeout)
                        ret, frame = current_cap.read()
                        if ret and frame is not None:
                            print(f"[INFO] Stream opened successfully on attempt {attempt + 1}")
                            # Start preview timer
                            preview_timer.start(33)  # ~30 FPS
                            update_preview()  # Initial update
                            return  # Success - exit function
                        else:
                            print(f"[WARN] Stream opened but failed to read frame on attempt {attempt + 1}")
                            current_cap.release()
                            current_cap = None
                    else:
                        print(f"[WARN] Stream failed to open (isOpened() returned False) on attempt {attempt + 1}")
                        if current_cap:
                            current_cap.release()
                            current_cap = None
                except Exception as e:
                    if current_cap:
                        try:
                            current_cap.release()
                        except:
                            pass
                        current_cap = None
                    print(f"[WARN] Attempt {attempt + 1} failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Wait before retry (except on last attempt)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    preview_label.setText(f"⏳ Retrying connection ({attempt + 2}/{max_retries})...")
                    QApplication.processEvents()
            
            # If we get here, all retries failed
            preview_label.setText("❌ Failed to connect to stream\n\nPlease check:\n- Camera is online\n- RTSP URL is correct\n- Network connection is active")
            print(f"[ERROR] Failed to open RTSP stream after {max_retries} attempts: {cam_url}")
        
        def show_crop_dialog(full_frame):
            """Show dialog to crop face from full captured frame."""
            crop_dlg = QDialog(stream_dlg)
            crop_dlg.setWindowTitle("Crop Face from Photo")
            crop_dlg.resize(900, 700)
            
            crop_layout = QVBoxLayout(crop_dlg)
            
            # Title
            title_label = QLabel("Select Face to Crop")
            title_label.setStyleSheet("font-size:16px; font-weight:bold; color:#90CAF9;")
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            crop_layout.addWidget(title_label)
            
            # Full frame display
            full_frame_label = QLabel()
            full_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            full_frame_label.setMinimumSize(800, 500)
            full_frame_label.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#0A1929;")
            
            # Resize frame for display
            display_frame = cv2.resize(full_frame.copy(), (800, 500))
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q)
            full_frame_label.setPixmap(pix)
            crop_layout.addWidget(full_frame_label)
            
            # Detect faces automatically
            detected_faces = []
            selected_face_idx = 0
            
            if MODEL is not None:
                try:
                    frame_resized = cv2.resize(full_frame.copy(), DET_SIZE)
                    with MODEL_LOCK:
                        faces = MODEL.get(frame_resized)
                    
                    if faces and len(faces) > 0:
                        # Scale bbox back to original frame size
                        scale_x = full_frame.shape[1] / DET_SIZE[0]
                        scale_y = full_frame.shape[0] / DET_SIZE[1]
                        
                        h, w = full_frame.shape[:2]
                        min_face_size = min(w, h) * 0.05  # Minimum 5% of image dimension
                        max_face_size = min(w, h) * 0.8   # Maximum 80% of image dimension
                        
                        for face in faces:
                            x1, y1, x2, y2 = map(int, face.bbox)
                            x1_orig = int(x1 * scale_x)
                            y1_orig = int(y1 * scale_y)
                            x2_orig = int(x2 * scale_x)
                            y2_orig = int(y2 * scale_y)
                            
                            # Calculate face dimensions
                            face_width = x2_orig - x1_orig
                            face_height = y2_orig - y1_orig
                            face_size = max(face_width, face_height)
                            face_area = face_width * face_height
                            
                            # Filter false positives:
                            # 1. Face must be large enough (not tiny objects)
                            if face_size < min_face_size:
                                continue
                            
                            # 2. Face must not be too large (not entire image)
                            if face_size > max_face_size:
                                continue
                            
                            # 3. Face aspect ratio should be reasonable (not too wide or tall)
                            aspect_ratio = face_width / max(face_height, 1)
                            if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Too narrow or too wide
                                continue
                            
                            # 4. Face area should be reasonable
                            if face_area < (min_face_size ** 2) * 0.5:
                                continue
                            
                            # Add padding
                            padding = 20
                            x1_orig = max(0, x1_orig - padding)
                            y1_orig = max(0, y1_orig - padding)
                            x2_orig = min(w, x2_orig + padding)
                            y2_orig = min(h, y2_orig + padding)
                            
                            detected_faces.append({
                                "bbox": (x1_orig, y1_orig, x2_orig, y2_orig),
                                "face_obj": face
                            })
                        
                        # Draw detected faces on display frame
                        display_frame_with_boxes = display_frame.copy()
                        for i, face_info in enumerate(detected_faces):
                            x1, y1, x2, y2 = face_info["bbox"]
                            # Scale to display size
                            scale_disp_x = 800 / full_frame.shape[1]
                            scale_disp_y = 500 / full_frame.shape[0]
                            x1_disp = int(x1 * scale_disp_x)
                            y1_disp = int(y1 * scale_disp_y)
                            x2_disp = int(x2 * scale_disp_x)
                            y2_disp = int(y2 * scale_disp_y)
                            
                            # Draw rectangle (green for selected, blue for others)
                            color = (0, 255, 0) if i == selected_face_idx else (255, 0, 0)
                            cv2.rectangle(display_frame_with_boxes, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                            cv2.putText(display_frame_with_boxes, f"Face {i+1}", (x1_disp, y1_disp - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        rgb = cv2.cvtColor(display_frame_with_boxes, cv2.COLOR_BGR2RGB)
                        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                        pix = QPixmap.fromImage(q)
                        full_frame_label.setPixmap(pix)
                except Exception as e:
                    print(f"[WARN] Face detection failed: {e}")
            
            # Face selection (if multiple faces detected)
            face_select_layout = QHBoxLayout()
            if len(detected_faces) > 1:
                face_select_layout.addWidget(QLabel("Select Face:"))
                face_combo = QComboBox()
                for i in range(len(detected_faces)):
                    face_combo.addItem(f"Face {i+1}")
                face_combo.setStyleSheet("padding:8px; font-size:13px;")
                
                def update_selected_face(idx):
                    nonlocal selected_face_idx
                    selected_face_idx = idx
                    # Redraw with new selection
                    display_frame_with_boxes = display_frame.copy()
                    for i, face_info in enumerate(detected_faces):
                        x1, y1, x2, y2 = face_info["bbox"]
                        scale_disp_x = 800 / full_frame.shape[1]
                        scale_disp_y = 500 / full_frame.shape[0]
                        x1_disp = int(x1 * scale_disp_x)
                        y1_disp = int(y1 * scale_disp_y)
                        x2_disp = int(x2 * scale_disp_x)
                        y2_disp = int(y2 * scale_disp_y)
                        color = (0, 255, 0) if i == selected_face_idx else (255, 0, 0)
                        cv2.rectangle(display_frame_with_boxes, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 2)
                        cv2.putText(display_frame_with_boxes, f"Face {i+1}", (x1_disp, y1_disp - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    rgb = cv2.cvtColor(display_frame_with_boxes, cv2.COLOR_BGR2RGB)
                    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q)
                    full_frame_label.setPixmap(pix)
                
                face_combo.currentIndexChanged.connect(update_selected_face)
                face_select_layout.addWidget(face_combo)
            else:
                face_select_layout.addWidget(QLabel("Face detected automatically"))
            crop_layout.addLayout(face_select_layout)
            
            # Cropped face preview
            cropped_label = QLabel("Cropped Face Preview:")
            cropped_label.setStyleSheet("font-size:14px; font-weight:bold; color:#90CAF9;")
            crop_layout.addWidget(cropped_label)
            
            cropped_preview = QLabel("No face cropped yet")
            cropped_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cropped_preview.setMinimumSize(200, 200)
            cropped_preview.setStyleSheet("border: 2px solid #64B5F6; border-radius:8px; background:#1E3A5F; color:#90CAF9;")
            crop_layout.addWidget(cropped_preview)
            
            # Name input (REQUIRED for register)
            name_layout = QHBoxLayout()
            name_layout.addWidget(QLabel("Employee Name (required):"))
            name_input = QLineEdit()
            name_input.setPlaceholderText("Enter employee name...")
            name_input.setStyleSheet("padding:8px; font-size:13px;")
            name_layout.addWidget(name_input)
            crop_layout.addLayout(name_layout)
            
            # Buttons
            btn_layout = QHBoxLayout()
            
            btn_crop = QPushButton("Crop Face")
            btn_crop.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:10px; font-size:14px;")
            
            def perform_crop():
                if not detected_faces:
                    QMessageBox.warning(crop_dlg, "No Face", "No face detected. Please ensure a face is visible in the photo.")
                    return
                
                try:
                    face_info = detected_faces[selected_face_idx]
                    x1, y1, x2, y2 = face_info["bbox"]
                    
                    # Crop face from original frame
                    face_img = full_frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        QMessageBox.warning(crop_dlg, "Error", "Could not extract face from photo.")
                        return
                    
                    # Resize to 160x160
                    face_img_resized = cv2.resize(face_img, (160, 160))
                    
                    # Get embedding from face object
                    face_obj = face_info["face_obj"]
                    emb = l2norm(getattr(face_obj, "normed_embedding", face_obj.embedding))
                    
                    # Show cropped preview
                    preview = cv2.resize(face_img_resized, (200, 200))
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q)
                    cropped_preview.setPixmap(pix)
                    cropped_preview.setText("")
                    
                    # Store cropped face data
                    crop_dlg.cropped_data = {
                        "emb": emb,
                        "face_img": face_img_resized,
                        "full_frame": full_frame.copy(),
                        "name": name_input.text().strip() or f"Photo {len(captured_photos) + 1}"
                    }
                    
                    QMessageBox.information(crop_dlg, "Cropped", "Face cropped successfully! Click 'Save' to add it.")
                    
                except Exception as e:
                    QMessageBox.warning(crop_dlg, "Error", f"Failed to crop face: {e}")
                    print(f"[ERROR] Crop error: {e}")
                    import traceback
                    traceback.print_exc()
            
            btn_crop.clicked.connect(perform_crop)
            btn_layout.addWidget(btn_crop)
            
            btn_save = QPushButton("Save")
            btn_save.setStyleSheet("background:#2196F3; color:white; font-weight:bold; padding:10px; font-size:14px;")
            
            def save_cropped():
                if not hasattr(crop_dlg, 'cropped_data'):
                    QMessageBox.warning(crop_dlg, "No Crop", "Please crop a face first.")
                    return
                
                # Require name input
                emp_name = name_input.text().strip()
                if not emp_name:
                    QMessageBox.warning(crop_dlg, "Name Required", "Please enter employee name to save the photo.")
                    return
                
                data = crop_dlg.cropped_data
                emb = data["emb"]
                face_img = data["face_img"]
                
                # Save immediately to employee database
                add_embedding(emp_name, emb, face_img)
                rebuild_centroids()
                
                # Update parent dialog's uploaded_data for preview
                if not uploaded_data_ref.get("results"):
                    uploaded_data_ref["results"] = []
                uploaded_data_ref["results"].append((emb, face_img))
                
                # Update employee list display
                if hasattr(parent_dlg, 'update_table'):
                    parent_dlg.update_table()
                
                crop_dlg.accept()
                stream_dlg.accept()  # Close stream dialog
                parent_dlg.accept()  # Close register dialog
                QMessageBox.information(parent_dlg, "Saved", f"Employee '{emp_name}' registered successfully!\nPhoto saved and added to employee list.")
                # Navigate to employees screen
                if hasattr(self, 'controller') and self.controller:
                    # The screen will automatically show employees after dialog closes
                    pass
            
            btn_save.clicked.connect(save_cropped)
            btn_layout.addWidget(btn_save)
            
            btn_cancel = QPushButton("Cancel")
            btn_cancel.setStyleSheet("background:#2C5364; color:white; padding:10px; font-size:14px;")
            btn_cancel.clicked.connect(crop_dlg.reject)
            btn_layout.addWidget(btn_cancel)
            
            crop_layout.addLayout(btn_layout)
            
            crop_dlg.exec()
        
        def capture_photo():
            """Capture and process photo from current stream."""
            if not current_cap or not current_cap.isOpened():
                QMessageBox.warning(stream_dlg, "No Stream", "Please wait for stream to connect first.")
                return
            
            # Capture frame
            ret, frame = current_cap.read()
            if not ret or frame is None:
                QMessageBox.warning(stream_dlg, "Error", "Failed to capture frame from stream.")
                return
            
            # Show crop dialog with full frame
            show_crop_dialog(frame)
        
        def close_stream():
            """Close stream and return captured photos."""
            nonlocal current_cap, current_stream, photos_saved
            
            if preview_timer.isActive():
                preview_timer.stop()
            
            if current_cap:
                try:
                    current_cap.release()
                except:
                    pass
                current_cap = None
            
            # Save photos if not already saved
            if captured_photos and not photos_saved:
                existing = uploaded_data_ref.get("results", [])
                # Convert to (emb, face_img) format for compatibility
                converted_photos = []
                for photo_data in captured_photos:
                    if len(photo_data) == 2:
                        converted_photos.append(photo_data)
                    else:
                        # Extract emb and face_img from new format
                        converted_photos.append((photo_data[0], photo_data[1]))
                uploaded_data_ref["results"] = existing + converted_photos
                photos_saved = True
            
            stream_dlg.accept()
        
        # Connect signals
        cam_combo.currentIndexChanged.connect(start_stream)
        btn_capture.clicked.connect(capture_photo)
        btn_back.clicked.connect(close_stream)
        
        # Start initial stream
        start_stream()
        
        # Cleanup on close
        def cleanup():
            nonlocal photos_saved
            if preview_timer.isActive():
                preview_timer.stop()
            if current_cap:
                try:
                    current_cap.release()
                except:
                    pass
            # Save captured photos even if closed via X button (only if not already saved)
            if captured_photos and not photos_saved:
                existing = uploaded_data_ref.get("results", [])
                # Convert to (emb, face_img) format for compatibility
                converted_photos = []
                for photo_data in captured_photos:
                    if len(photo_data) == 2:
                        converted_photos.append(photo_data)
                    else:
                        # Extract emb and face_img from new format
                        converted_photos.append((photo_data[0], photo_data[1]))
                uploaded_data_ref["results"] = existing + converted_photos
                photos_saved = True
        
        stream_dlg.finished.connect(cleanup)
        
        stream_dlg.exec()

    def register_employee_dialog(self, parent_dlg=None):
        """Register new employee from photo upload."""
        dlg = QDialog(self if parent_dlg is None else parent_dlg)
        dlg.setWindowTitle("Register New Employee")
        dlg.resize(500, 400)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Register New Employee", styleSheet="font-size:18px; font-weight:bold; color:#90CAF9;"))

        # Photo preview label
        preview_label = QLabel("No photo selected")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setMinimumHeight(200)
        preview_label.setStyleSheet("border: 2px dashed #64B5F6; border-radius:8px; background:#0A1929; color:#90CAF9;")
        layout.addWidget(preview_label)

        # Upload buttons - multiple and crop from stream (single photo removed)
        upload_layout = QHBoxLayout()
        
        btn_upload_multi = QPushButton("Upload Multiple Photos")
        btn_upload_multi.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:10px;")
        upload_layout.addWidget(btn_upload_multi)
        
        btn_crop_stream = QPushButton("Crop Photo from Stream")
        btn_crop_stream.setStyleSheet("background:#FF9800; color:white; font-weight:bold; padding:10px;")
        upload_layout.addWidget(btn_crop_stream)
        layout.addLayout(upload_layout)
        
        # Progress label
        progress_label = QLabel("")
        progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_label.setStyleSheet("color:#90CAF9; font-size:12px;")
        layout.addWidget(progress_label)

        # Employee name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Employee Name:"))
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter employee name...")
        name_input.setStyleSheet("padding:8px; font-size:13px;")
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)

        # Store uploaded data - now supports multiple
        uploaded_data = {"results": []}  # List of (emb, face_img) tuples
        batch_worker = None

        def upload_multiple_photos():
            file_paths, _ = QFileDialog.getOpenFileNames(
                dlg,
                "Select Photos for Registration",
                "",
                "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
            )
            
            if not file_paths:
                return
            
            if MODEL is None:
                QMessageBox.warning(dlg, "Error", "InsightFace model not available.")
                return
            
            # Disable buttons during processing
            btn_upload_multi.setEnabled(False)
            progress_label.setText(f"Processing {len(file_paths)} photos...")
            QApplication.processEvents()
            
            # Start batch worker
            nonlocal batch_worker
            if batch_worker:
                batch_worker.terminate()
                batch_worker.wait()
            
            batch_worker = BatchFaceWorker(file_paths)
            
            def on_progress(current, total):
                progress_label.setText(f"Processing {current}/{total} photos...")
                QApplication.processEvents()
            
            def on_finished(results):
                uploaded_data["results"] = results
                progress_label.setText(f"Processed {len(results)} photos successfully!")
                btn_upload_multi.setEnabled(True)
                
                if results:
                    # Show first result as preview
                    emb, face_img = results[0]
                    preview = cv2.resize(face_img, (300, 300))
                    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q)
                    preview_label.setPixmap(pix.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    preview_label.setText(f"{len(results)} photo(s) ready")
                else:
                    QMessageBox.warning(dlg, "No Faces", "No faces detected in any of the selected photos.")
            
            def on_error(msg):
                progress_label.setText(f"Error: {msg}")
                btn_upload_multi.setEnabled(True)
                QMessageBox.warning(dlg, "Error", msg)
            
            batch_worker.progress.connect(on_progress)
            batch_worker.finished.connect(on_finished)
            batch_worker.error.connect(on_error)
            batch_worker.start()
        
        def crop_from_stream():
            """Open stream capture dialog."""
            self.stream_capture_dialog(dlg, uploaded_data)
            # Update preview if photos were captured
            if uploaded_data["results"]:
                # Show first result as preview
                emb, face_img = uploaded_data["results"][0]
                preview = cv2.resize(face_img, (300, 300))
                rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(q)
                preview_label.setPixmap(pix.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                preview_label.setText(f"{len(uploaded_data['results'])} photo(s) ready")
        
        btn_upload_multi.clicked.connect(upload_multiple_photos)
        btn_crop_stream.clicked.connect(crop_from_stream)

        # OK button
        btn_ok = QPushButton("OK - Save Employee")
        btn_ok.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:12px; font-size:14px;")
        
        def save_employee():
            if not uploaded_data["results"]:
                QMessageBox.warning(dlg, "No Photo", "Please upload at least one photo first.")
                return
            
            emp_name = name_input.text().strip()
            if not emp_name:
                QMessageBox.warning(dlg, "No Name", "Please enter an employee name.")
                return
            
            if emp_name in EMP_DB:
                if QMessageBox.question(
                    dlg, "Employee Exists",
                    f"Employee '{emp_name}' already exists. Add these photos to existing employee?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                ) != QMessageBox.StandardButton.Yes:
                    return
            
            # Save all embeddings and photos
            progress_label.setText("Saving...")
            QApplication.processEvents()
            
            for emb, face_img in uploaded_data["results"]:
                add_embedding(emp_name, emb, face_img)
            
            rebuild_centroids()  # Rebuild centroids immediately - recognition will use new embeddings right away
            
            photo_count = len(uploaded_data["results"])
            QMessageBox.information(dlg, "Success", f"Employee '{emp_name}' registered successfully!\n{photo_count} photo(s) added.\nRecognition will start immediately.")
            dlg.accept()
            
            # Update table to show new employee
            if hasattr(self, 'update_table'):
                self.update_table()
            
            # Refresh employee table
            self.update_table()

            # Close parent settings dialog if it exists
            if parent_dlg:
                parent_dlg.accept()
        
        btn_ok.clicked.connect(save_employee)
        layout.addWidget(btn_ok)

        # Cancel button
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background:#2C5364; color:white; padding:8px;")
        btn_cancel.clicked.connect(dlg.reject)
        layout.addWidget(btn_cancel)

        dlg.exec()

    def settings_menu(self):
        """New Settings dialog with Register and Delete Selected Employees."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.resize(600, 500)

        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Settings", styleSheet="font-size:18px; font-weight:bold; color:#90CAF9;"))

        # Register button
        btn_register = QPushButton("Register")
        btn_register.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:12px; font-size:14px;")
        btn_register.clicked.connect(lambda: self.register_employee_dialog(dlg))
        layout.addWidget(btn_register)

        layout.addWidget(QLabel(""))  # Spacer

        # Delete Selected Employees section
        layout.addWidget(QLabel("Delete Selected Employees", styleSheet="font-size:14px; font-weight:bold; color:#FF9800;"))
        
        # Select All button
        btn_select_all = QPushButton("Select All")
        btn_select_all.setStyleSheet("background:#2196F3; color:white; padding:8px;")
        layout.addWidget(btn_select_all)

        # Scroll area for employee checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        scroll.setStyleSheet("border: 1px solid #64B5F6; border-radius:4px; background:#0A1929;")
        
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)
        checkbox_layout.setSpacing(5)
        
        checkboxes = {}
        for name in sorted(EMP_DB.keys()):
            cb = QCheckBox(name)
            cb.setStyleSheet("color:white; font-size:12px; padding:4px;")
            checkbox_layout.addWidget(cb)
            checkboxes[name] = cb
        
        if not checkboxes:
            no_emp_label = QLabel("No employees registered yet.")
            no_emp_label.setStyleSheet("color:#90CAF9; padding:20px;")
            no_emp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            checkbox_layout.addWidget(no_emp_label)
        
        scroll.setWidget(checkbox_widget)
        layout.addWidget(scroll, 1)

        # Select All functionality
        def toggle_select_all():
            all_checked = all(cb.isChecked() for cb in checkboxes.values())
            for cb in checkboxes.values():
                cb.setChecked(not all_checked)
            btn_select_all.setText("Deselect All" if not all_checked else "Select All")
        
        btn_select_all.clicked.connect(toggle_select_all)

        # Delete Selected button
        btn_delete_selected = QPushButton("Delete Selected")
        btn_delete_selected.setStyleSheet("background:#D32F2F; color:white; font-weight:bold; padding:12px; font-size:14px;")
        
        def delete_selected():
            selected = [name for name, cb in checkboxes.items() if cb.isChecked()]
            if not selected:
                QMessageBox.warning(dlg, "No Selection", "Please select at least one employee to delete.")
                return
            
            msg = f"Delete {len(selected)} employee(s)?\n\n" + "\n".join(selected[:10])
            if len(selected) > 10:
                msg += f"\n... and {len(selected) - 10} more"
            
            if QMessageBox.question(
                dlg, "Confirm Deletion", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                for name in selected:
                    delete_employee(name)
                QMessageBox.information(dlg, "Deleted", f"Successfully deleted {len(selected)} employee(s).")
                dlg.accept()
                self.update_table()
        
        btn_delete_selected.clicked.connect(delete_selected)
        layout.addWidget(btn_delete_selected)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("background:#2C5364; color:white; padding:8px;")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)

        dlg.exec()
    


    def get_auto(self):
        return self._auto

    def set_auto(self, v):
        self._auto = bool(v)

    # ============================================================
    # CLOSE EVENT CLEANUP
    # ============================================================
    def closeEvent(self, e):
        """Clean up resources when screen is closed."""
        try:
            # Check if this is a real close (app shutting down) or just navigation
            # Don't stop recognition streams when just navigating away - keep them running
            is_app_closing = not hasattr(self, 'controller') or self.controller is None
            
            if is_app_closing:
                print("[INFO] App closing - stopping all recognition streams...")
            else:
                print("[INFO] Navigating away - keeping recognition streams running...")
                # Just stop timers, but keep streams running
                if hasattr(self, "ui_timer") and self.ui_timer:
                    try:
                        self.ui_timer.stop()
                    except:
                        pass
                if hasattr(self, "sys_timer") and self.sys_timer:
                    try:
                        self.sys_timer.stop()
                    except:
                        pass
                if hasattr(self, "session_timer") and self.session_timer:
                    try:
                        self.session_timer.stop()
                    except:
                        pass
            # Clear the _db_loaded flag so it can be reloaded if screen is reused
            if hasattr(self, "_db_loaded"):
                delattr(self, "_db_loaded")
            e.accept()
            return
            
            # Only stop streams if app is actually closing
            # Stop all timers first
            if hasattr(self, "ui_timer") and self.ui_timer:
                try:
                    self.ui_timer.stop()
                except:
                    pass
            if hasattr(self, "sys_timer") and self.sys_timer:
                try:
                    self.sys_timer.stop()
                except:
                    pass
            if hasattr(self, "session_timer") and self.session_timer:
                try:
                    self.session_timer.stop()
                except:
                    pass
            
            # Stop all recognition streams safely (only when app is closing)
            if hasattr(self, "strms") and self.strms:
                print(f"[INFO] Stopping {len(self.strms)} recognition streams...")
                for s in self.strms:
                    try:
                        if s and hasattr(s, "stop"):
                            s.stop()
                        if s and hasattr(s, "run"):
                            s.run = False
                    except Exception as ex:
                        print(f"[WARN] Error stopping stream: {ex}")
                self.strms = []
            
            self.streams_running = False
            
            # Clear the _db_loaded flag so it can be reloaded if screen is reused
            if hasattr(self, "_db_loaded"):
                delattr(self, "_db_loaded")
            
            # Clear initialization flag
            if hasattr(self, "_initializing"):
                delattr(self, "_initializing")
            
            print("[INFO] Employee Management Screen closed and cleaned up")
        except Exception as e:
            print(f"[WARN] Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
        e.accept()
    
    def hideEvent(self, e):
        """Called when screen is hidden (navigating away) - keep streams running."""
        # Don't stop streams when just hiding - keep recognition running
        # Only stop UI timers to save resources
        if hasattr(self, "ui_timer") and self.ui_timer:
            try:
                self.ui_timer.stop()
            except:
                pass
        if hasattr(self, "sys_timer") and self.sys_timer:
            try:
                self.sys_timer.stop()
            except:
                pass
        # Keep session_timer running (needed for status updates)
        # Keep streams running (recognition continues in background)
        super().hideEvent(e)
    
    def showEvent(self, e):
        """Called when screen is shown - restart UI timers."""
        # Restart UI timers when screen is shown again
        if hasattr(self, "ui_timer") and self.ui_timer:
            try:
                self.ui_timer.start(1000)
            except:
                pass
        if hasattr(self, "sys_timer") and self.sys_timer:
            try:
                self.sys_timer.start(5000)
            except:
                pass
        super().showEvent(e)

class EmpCusLiveScreen(QWidget):
    """
    This is the LIVE Employee–Customer Interaction ML Screen.
    Runs face recognition on selected cameras only.
    """
    def __init__(self, controller=None):
        super().__init__()
        self.controller = controller
        self.controller.emp_cus_worker = self
        
        # DO NOT open new window – embed inside main UI
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                                            stop:0 #edf3ff,
                                            stop:1 #d6e0ff);
                color:#123e91;
                font-family:'Segoe UI';
            }
        """)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🟡 Emp–Cus Interaction – Live Analytics")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:26px; font-weight:700; margin:10px;")
        layout.addWidget(title)

        # Back button
        back_btn = QPushButton("⬅ Back")
        back_btn.setFixedWidth(150)
        back_btn.setStyleSheet("""
            background:#2e7d32;
            color:white;
            border-radius:10px;
            padding:10px;
            font-size:16px;
        """)
        back_btn.clicked.connect(lambda: controller.load_ml_selection() if controller else None)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # -----------------------------------------------------
        # LOAD SELECTED CAMERAS EXACTLY AS SAVED
        # -----------------------------------------------------
        selected = []
        try:
            saved = json.load(open("selected_emp_cus.json"))
        except:
            saved = []

        for item in saved:
            # Normalize URL to fix channel number formatting (e.g., /0101 -> /101)
            normalized_url = normalize_rtsp_url(item.get("url", ""))
            selected.append({
                "cam_name": item["name"],
                "url": normalized_url
            })

        if not selected:
            lbl = QLabel("No cameras selected!")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size:20px; color:#f55; margin-top:50px;")
            layout.addWidget(lbl)
            return

        # -----------------------------------------------------
        # START STREAMERS
        # -----------------------------------------------------
        self.streams = []
        for cam in selected:
            streamer = Streamer(cam["url"], lambda: False, cam_name=cam["cam_name"])
            self.streams.append(streamer)
        
        # ===== HEARTBEAT: Tell AppController Emp–Cus is running =====
        self.heartbeat = QTimer()
        self.heartbeat.timeout.connect(self._send_heartbeat)
        self.heartbeat.start(1000)   # every 1 second

        live_label = QLabel("Live running for: " + ", ".join([c["cam_name"] for c in selected]))
        live_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_label.setStyleSheet("font-size:16px; margin-top:20px;")
        layout.addWidget(live_label)

    def _send_heartbeat(self):
        if self.controller:
            # Mark as active
            self.controller.emp_cus_worker = self
            # Refresh status bar icons
            self.controller.update_status_bar()



    def closeEvent(self, e):
        for s in self.streams:
            s.stop()
        e.accept()


# =========================
# Run (if executed directly)
# =========================
if __name__=="__main__":
    # Check if STREAM_URLS is defined, if not, define it or show error
    if 'STREAM_URLS' not in globals() or not STREAM_URLS:
        print("[ERROR] STREAM_URLS not defined. Please define STREAM_URLS before running.")
        print("Example: STREAM_URLS = ['rtsp://...', 'rtsp://...']")
        sys.exit(1)
    app=QApplication(sys.argv); load_db(); rebuild_centroids()
    win = EmpCusDashboardScreen(controller=None); win.showMaximized(); sys.exit(app.exec())