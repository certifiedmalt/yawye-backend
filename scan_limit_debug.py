#!/usr/bin/env python3
"""
Focused test for scan limit issue
"""

import requests
import json

BASE_URL = "https://eatwise-scan.preview.emergentagent.com/api"
TEST_USER_EMAIL = "testuser@foodscan.com"
TEST_USER_PASSWORD = "SecurePass123!"
NUTELLA_BARCODE = "3017620422003"

def test_scan_limit_debug():
    # Login first
    login_payload = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    if response.status_code != 200:
        print(f"Login failed: {response.text}")
        return
    
    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Check current profile
    profile_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if profile_response.status_code == 200:
        profile = profile_response.json()
        print(f"Current daily scans: {profile.get('daily_scans', 'unknown')}")
        print(f"Subscription tier: {profile.get('subscription_tier', 'unknown')}")
    
    # Try scanning multiple times
    scan_payload = {"barcode": NUTELLA_BARCODE}
    
    for i in range(8):  # Try 8 scans
        print(f"\nAttempting scan #{i+1}")
        response = requests.post(f"{BASE_URL}/scan", json=scan_payload, headers=headers, timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Scan successful")
        elif response.status_code == 403:
            print(f"❌ Scan blocked: {response.text}")
            break
        else:
            print(f"❌ Unexpected error: {response.text}")
            break
        
        # Check profile after each scan
        profile_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
        if profile_response.status_code == 200:
            profile = profile_response.json()
            print(f"Daily scans after scan #{i+1}: {profile.get('daily_scans', 'unknown')}")

if __name__ == "__main__":
    test_scan_limit_debug()