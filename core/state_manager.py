#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Third Umpire – State Manager
-----------------------------
Handles state persistence for ML models to enable resume after reboot.
"""

import os
import json
from typing import Dict, List, Optional

STATE_FILE = os.path.expanduser("~/.ThirdUmpire_Analytics/app_state.json")


def save_model_state(model_name: str, urls: List[str], cam_names: List[str]):
    """
    Save running model state.
    
    Args:
        model_name: Name of the model (e.g., "Crowd Density", "HeatMap")
        urls: List of RTSP URLs
        cam_names: List of camera names
    """
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    
    state = load_all_state()
    
    state["running_models"][model_name] = {
        "urls": urls,
        "cam_names": cam_names,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print(f"[STATE] Saved state for {model_name}")
    except Exception as e:
        print(f"[STATE] Failed to save state: {e}")


def remove_model_state(model_name: str):
    """
    Remove model from running state.
    
    Args:
        model_name: Name of the model to remove
    """
    state = load_all_state()
    
    if model_name in state["running_models"]:
        del state["running_models"][model_name]
        
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
            print(f"[STATE] Removed state for {model_name}")
        except Exception as e:
            print(f"[STATE] Failed to remove state: {e}")


def load_all_state() -> Dict:
    """Load all saved state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[STATE] Failed to load state: {e}")
    
    return {
        "running_models": {},
        "last_saved": None
    }


def get_running_models() -> Dict[str, Dict]:
    """Get dictionary of running models with their URLs and camera names."""
    state = load_all_state()
    return state.get("running_models", {})


def clear_all_state():
    """Clear all saved state (useful for testing or reset)."""
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
            print("[STATE] Cleared all state")
        except Exception as e:
            print(f"[STATE] Failed to clear state: {e}")

