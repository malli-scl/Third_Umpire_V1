#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Monitor – Lightweight metrics logger for edge devices
-------------------------------------------------------------
• Logs CPU %, Memory %, and Temperature every interval
• Designed for Raspberry Pi 5 / Luckfox / Jetson Orin Nano
• Output: system_metrics.csv (append mode)
"""

import psutil, csv
from datetime import datetime

def log_system_metrics():
    """Log CPU, memory, and temperature to CSV."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent

    # Try to read CPU temp (works on Pi / Jetson)
    try:
        temps = psutil.sensors_temperatures()
        temp = "--"
        for name, entries in temps.items():
            if entries and isinstance(entries, list):
                temp = entries[0].current
                break
    except Exception:
        temp = "--"

    # Write to CSV
    with open("system_metrics.csv", "a", newline="") as f:
        csv.writer(f).writerow([ts, cpu, mem, temp])

    print(f"[SYS] {ts} | CPU: {cpu}% | MEM: {mem}% | TEMP: {temp}")
