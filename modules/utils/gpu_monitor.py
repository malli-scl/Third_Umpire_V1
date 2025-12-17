#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU & Performance Monitor (Extended)
------------------------------------
• Tracks CUDA memory usage (allocated / reserved)
• Aggregates per-model TOPS and GFLOPS usage
• Logs compute utilization every 3 minutes
• Provides GPU/CPU temp + FPS for AppController status bar
• Safe on Jetson / desktop / fallback to mock data
"""

import os, time, threading, torch
from datetime import datetime

# ============================================================
#  Continuous GPU Memory Monitor Thread
# ============================================================

def gpu_monitor(stop_event, interval=5):
    """Continuously log GPU memory every few seconds."""
    while not stop_event.is_set():
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                reserved = torch.cuda.memory_reserved(0) / 1024**2
                print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
            except Exception as e:
                print(f"[WARN] GPU monitor error: {e}")
        time.sleep(interval)

gpu_stop_event = threading.Event()
gpu_thread = threading.Thread(target=gpu_monitor, args=(gpu_stop_event,), daemon=True)
gpu_thread.start()

# ============================================================
#  Performance Aggregation Helper (TOPS / GFLOPS)
# ============================================================

GLOBAL_PERF = {}  # model_name → {"per_cam": {cam: tops}, "total": total_tops}

def update_performance_stats(model_name: str, per_cam_tops: dict):
    """Update TOPS & GFLOPS stats per model and print summary."""
    global GLOBAL_PERF

    if not hasattr(update_performance_stats, "_last_updates"):
        update_performance_stats._last_updates = {}

    now = time.time()
    last_update = update_performance_stats._last_updates.get(model_name, 0)
    if now - last_update < 180:  # every 3 minutes
        return
    update_performance_stats._last_updates[model_name] = now

    total_tops = sum(per_cam_tops.values())
    GLOBAL_PERF[model_name] = {"per_cam": per_cam_tops, "total": total_tops}

    grand_total_tops = sum(v["total"] for v in GLOBAL_PERF.values())
    grand_total_gflops = grand_total_tops * 1e3

    print("\n========== COMPUTE UTILIZATION ==========")
    for mdl, info in GLOBAL_PERF.items():
        print(f"{mdl} Model:")
        for cam, tops in info["per_cam"].items():
            print(f"  ├── {cam}: {tops:.3f} TOPS | {tops*1e3:.1f} GFLOPS")
        print(f"  └── TOTAL: {info['total']:.3f} TOPS | {info['total']*1e3:.1f} GFLOPS\n")
    print("-----------------------------------------")
    print(f"GRAND TOTAL → {grand_total_tops:.3f} TOPS | {grand_total_gflops:.1f} GFLOPS")
    print("=========================================\n")

    # Optional UI update hook
    try:
        from core.app_controller import win
        if hasattr(win, 'result_screen'):
            lines = []
            for mdl, info in GLOBAL_PERF.items():
                lines.append(f"{mdl}: {info['total']:.3f} TOPS | {info['total']*1e3:.1f} GFLOPS")
            lines.append(f"TOTAL: {grand_total_tops:.3f} TOPS | {grand_total_gflops:.1f} GFLOPS")
            text = "⚙ Performance Summary\n" + "\n".join(lines)
            win.result_screen.lbl_perf.setText(text)
    except Exception:
        pass


# ============================================================
#  Periodic Utilization Logger
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "compute_utilization.txt")

def _print_periodic_utilization():
    """Print and log consolidated compute utilization summary every 3 minutes."""
    global GLOBAL_PERF
    threading.Timer(180.0, _print_periodic_utilization).start()
    if not GLOBAL_PERF:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n[{now}] ========= PERIODIC COMPUTE UTILIZATION =========="]

    grand_total_tops = 0.0
    for mdl, info in GLOBAL_PERF.items():
        total = info["total"]
        grand_total_tops += total
        lines.append(f"{mdl} Model:")
        for cam, tops in info["per_cam"].items():
            lines.append(f"  ├── {cam}: {tops:.3f} TOPS | {tops*1e3:.1f} GFLOPS")
        lines.append(f"  └── TOTAL: {total:.3f} TOPS | {total*1e3:.1f} GFLOPS\n")

    lines.append("-----------------------------------------")
    lines.append(f"GRAND TOTAL → {grand_total_tops:.3f} TOPS | "
                 f"{grand_total_tops*1e3:.1f} GFLOPS | {(grand_total_tops*1e12):.2e} FLOPs")
    lines.append("=========================================\n")

    summary_text = "\n".join(lines)
    print(summary_text)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(summary_text)
    except Exception as e:
        print(f"[WARN] Could not write utilization log: {e}")

# ✅ Kick-start periodic logging once
_print_periodic_utilization()


# ============================================================
#  Helper Functions for AppController Status Bar
# ============================================================

def get_gpu_temp():
    """Return GPU temperature (Jetson / NVIDIA / fallback)."""
    try:
        # Jetson device path
        for path in [
            "/sys/devices/gpu.0/temp",
            "/sys/devices/virtual/thermal/thermal_zone1/temp",
            "/sys/class/thermal/thermal_zone1/temp",
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    t = int(f.read().strip()) / 1000
                    if 20 < t < 100:
                        return round(t, 1)

        # Fallback: infer temp from memory use
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**2
            return round(40 + min(mem / 50, 30), 1)
    except Exception:
        pass
    return None


def get_cpu_temp():
    """Return CPU temperature (generic Linux thermal zone)."""
    try:
        for zone in os.listdir("/sys/class/thermal"):
            if zone.startswith("thermal_zone"):
                path = f"/sys/class/thermal/{zone}/temp"
                if os.path.exists(path):
                    with open(path) as f:
                        t = int(f.read().strip()) / 1000
                        if 20 < t < 100:
                            return round(t, 1)
    except Exception:
        pass
    return None


def get_gpu_fps():
    """Return approximate FPS based on GPU load or fallback heuristic."""
    try:
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(0) / 1024**2
            return int(15 + min(mem / 50, 30))  # heuristic
    except Exception:
        pass
    return None
