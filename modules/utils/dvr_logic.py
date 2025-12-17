#!/usr/bin/env python3
# -- coding: utf-8 --
"""
DVR Logic – RTSP URL Builder & Parallel 16-Channel Tester (Fast)
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;0"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
try:
    cv2.setLogLevel(0)   # Disable OpenCV/FFmpeg RTSP logs safely
except:
    pass


import time
import socket
import ipaddress
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

RTSP_PATTERNS = {
    "Hikvision": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{ch:02d}01",
    "Dahua":     "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={ch}&subtype=0",
    "CP Plus":   "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/{ch:02d}01",  # CP Plus uses Hikvision format with leading zeros
    "Samsung":   "rtsp://{user}:{pwd}@{ip}:554/onvif/profile{ch}/media.smp",
}

def build_rtsp_urls(brand: str, ip: str, user: str, pwd: str, max_channels: int = 16):
    """
    Build a list of RTSP URL + name pairs (list of dicts)
    for compatibility with DashboardScreen.
    
    CRITICAL: Passwords with @ symbol MUST be URL-encoded for RTSP URLs.
    The @ symbol is the separator between credentials and host in RTSP URLs.
    Without encoding, passwords containing @ break URL parsing.
    """
    pattern = RTSP_PATTERNS.get(brand)
    if not pattern:
        raise ValueError(f"Unsupported DVR brand: {brand}")

    # URL-encode username and password to handle special characters
    # This is REQUIRED for passwords containing @ symbol
    # Without encoding, the @ in password is interpreted as credential/host separator
    encoded_user = quote(user, safe='')
    encoded_pwd = quote(pwd, safe='')

    urls = []
    # CP Plus and Hikvision use 1-based channel numbering (channels 1-16)
    # Format: /Streaming/Channels/0101, /Streaming/Channels/0201, etc.
    for ch in range(1, max_channels + 1):
        url = pattern.format(user=encoded_user, pwd=encoded_pwd, ip=ip, ch=ch)
        urls.append({"url": url, "name": f"Cam {ch}"})

    print(f"[INFO] Generated {len(urls)} RTSP URLs for {brand} DVR ({ip})")
    print(f"[INFO] Sample URL (Cam 1): {urls[0]['url'] if urls else 'N/A'}")
    return urls

def _check_single(url, timeout=3.0):
    """Test single RTSP URL connection. Returns (url, status) where status is True/False."""
    cap = None
    try:
        # Set RTSP options via environment for better connection handling
        # These options help with authentication and connection stability
        # Increased timeout values for CP Plus compatibility
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;10000000|"  # Increased from 5000000 to 10000000 (10 seconds)
            "max_delay;1000000|"  # Increased from 500000 to 1000000
            "buffersize;10000000|"
            "loglevel;0"
        )
        
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start = time.time()
        ok = False
        
        # Give more time for RTSP authentication and connection
        # CP Plus may need more time to establish connection
        while time.time() - start < timeout:
            if cap.isOpened():
                # Try to read a frame to confirm it's actually working
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    ok = True
                    break
            time.sleep(0.5)  # Increased from 0.3 to 0.5 seconds for better reliability
        
        if cap:
            cap.release()
        
        return url, ok
    except Exception as e:
        if cap:
            try:
                cap.release()
            except:
                pass
        return url, False

def test_rtsp_urls(urls, timeout=3.0, max_workers=8):
    """
    Parallel test of RTSP URLs. Returns list[(url, status)].
    Increased default timeout to 3.0 seconds for better CP Plus compatibility.
    """
    results = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_check_single, url, timeout): i for i, url in enumerate(urls)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                url, ok = future.result()
                results[idx] = (url, ok)
            except Exception:
                results[idx] = (urls[idx], False)
    return results

# ============================
# DVR IP Auto-Detection Function
# ============================
def auto_detect_dvr_ip():
    """Auto-detect DVR IP address and return it, or None if not found"""
    print("[DVR] Auto-detecting DVR IP...")
    
    # Common DVR ports
    common_ports = [554, 80, 8000, 443, 8080, 8001, 8002, 8554, 1935]
    timeout = 0.2
    max_workers = 100
    
    # Blacklist of IPs to exclude from DVR detection
    blacklisted_ips = [""]  # Add IPs to block here
    
    # Common networks to scan
    common_networks = ["192.168.1.0/24", "192.168.0.0/24", "172.16.0.0/24", "10.0.0.0/24"]
    
    def scan_port(ip, port, timeout):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            result = s.connect_ex((ip, port))
            s.close()
            return result == 0
        except Exception:
            return False

    def scan_ip(ip):
        ip_str = str(ip)
        
        # Skip blacklisted IPs
        if ip_str in blacklisted_ips:
            return None
            
        found_ports = []
        for port in common_ports:
            if scan_port(ip_str, port, timeout):
                found_ports.append(port)
        # Only return IP if it has RTSP port 554 (DVR requirement)
        return ip_str if 554 in found_ports else None

    # Scan all networks in parallel
    results = []
    total_hosts = sum(len(list(ipaddress.IPv4Network(net, strict=False).hosts())) for net in common_networks)
    print(f"[DVR] Scanning {total_hosts} hosts for DVR devices...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for network in common_networks:
            for ip in ipaddress.IPv4Network(network, strict=False).hosts():
                futures.append(executor.submit(scan_ip, ip))
        
        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    results.append(result)
            except Exception:
                pass

    if not results:
        print("[DVR] No DVR devices found")
        return None

    # Sort by network priority: 172.16.0.x > 192.168.1.x > 192.168.0.x > 10.0.0.x
    def sort_key(ip):
        if ip.startswith("172.16.0."):
            return (1, ipaddress.IPv4Address(ip))
        elif ip.startswith("192.168.1."):
            return (2, ipaddress.IPv4Address(ip))
        elif ip.startswith("192.168.0."):
            return (3, ipaddress.IPv4Address(ip))
        elif ip.startswith("10.0.0."):
            return (4, ipaddress.IPv4Address(ip))
        else:
            return (5, ipaddress.IPv4Address(ip))
    
    results.sort(key=sort_key)
    detected_ip = results[0]
    print(f"[DVR] DVR IP detected: {detected_ip}")
    return detected_ip

