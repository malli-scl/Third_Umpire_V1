#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Logger – Generic logging helper
------------------------------------
Provides `log_event()` for all modules to append timestamped logs.
"""

import csv, os
from datetime import datetime

LOG_FILE = "event_log.csv"

def log_event(event_type, message):
    """Append an event line to CSV log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, event_type, message])
    print(f"[LOG] {ts} | {event_type}: {message}")
