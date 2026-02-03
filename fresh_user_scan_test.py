#!/usr/bin/env python3
"""
Test scan limit with fresh user
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "https://upf-scanner.preview.emergentagent.com/api"
FRESH_USER_EMAIL = f"freshuser{int(time.time())}@foodscan.com"
TEST_USER_PASSWORD = "SecurePass123!"
TEST_USER_NAME = "Fresh Test User"
NUTELLA_BARCODE = "3017620422003"

def test_fresh_user_scan_limit():
    print(f"Testing with fresh user: {FRESH_USER_EMAIL}")
    
    # Register fresh user
    register_payload = {
        "email": FRESH_USER_EMAIL,
        "password": TEST_USER_PASSWORD,
        "name": TEST_USER_NAME
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    if response.status_code != 200:
        print(f"Registration failed: {response.text}")
        return False
    
    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Fresh user registered successfully")
    
    # Check initial profile
    profile_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if profile_response.status_code == 200:
        profile = profile_response.json()
        print(f"Initial daily scans: {profile.get('daily_scans', 'unknown')}")
        print(f"Subscription tier: {profile.get('subscription_tier', 'unknown')}")
    
    # Try scanning exactly 6 times (limit should be 5)
    scan_payload = {"barcode": NUTELLA_BARCODE}
    successful_scans = 0
    
    for i in range(6):
        print(f"\nAttempting scan #{i+1}")
        response = requests.post(f"{BASE_URL}/scan", json=scan_payload, headers=headers, timeout=30)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            successful_scans += 1
            print(f"✅ Scan #{i+1} successful")
        elif response.status_code == 403:
            print(f"❌ Scan #{i+1} blocked: {response.text}")
            if successful_scans == 5:
                print("✅ SCAN LIMIT TEST PASSED: Correctly blocked 6th scan after 5 successful scans")
                return True
            else:
                print(f"❌ SCAN LIMIT TEST FAILED: Blocked after only {successful_scans} scans")
                return False
        else:
            print(f"❌ Unexpected error on scan #{i+1}: {response.text}")
            return False
        
        # Check profile after each successful scan
        if response.status_code == 200:
            profile_response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
            if profile_response.status_code == 200:
                profile = profile_response.json()
                print(f"Daily scans after scan #{i+1}: {profile.get('daily_scans', 'unknown')}")
        
        time.sleep(1)  # Small delay between scans
    
    print(f"❌ SCAN LIMIT TEST FAILED: Completed {successful_scans} scans without hitting limit")
    return False

if __name__ == "__main__":
    success = test_fresh_user_scan_limit()
    if not success:
        exit(1)