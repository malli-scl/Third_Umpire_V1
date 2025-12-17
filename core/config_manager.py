#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Config Manager (SAFE MODE)
--------------------------------
This version removes ALL automatic validation during login.
The DVR will only be tested using the credentials the user enters.
"""

import os, json

CONFIG_PATH = os.path.expanduser("~/ThirdUmpire_Analytics/config.json")


# ==========================================================
#   SAVE CONFIG
# ==========================================================
def save_config(data: dict):
    """Save DVR credentials."""
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[CONFIG] Saved DVR config to {CONFIG_PATH}")
    except Exception as e:
        print(f"[CONFIG] Save failed: {e}")


# ==========================================================
#   LOAD CONFIG
# ==========================================================
def load_config() -> dict | None:
    """Load DVR config (does not validate)."""
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        print("[CONFIG] Loaded stored DVR config.")
        return data
    except Exception as e:
        print(f"[CONFIG] Load failed: {e}")
        return None


# ==========================================================
#   VALIDATE CONFIG — DISABLED
# ==========================================================
def validate_config(cfg: dict) -> bool:
    """
    Disabled to prevent old password from interfering with login.
    """
    print("[CONFIG] Skipping saved config auto-validation.")
    return False


# ==========================================================
#   AUTO RETEST — DISABLED
# ==========================================================
def auto_retest_dvr(cfg: dict, detect_func):
    """
    Disabled so DVR IP is not auto-retested during login.
    We always use the user-entered IP.
    """
    print("[RETEST] Auto-retest disabled.")
    return None
