#!/usr/bin/env python3
"""
Additional backend tests for favorites and subscription upgrade
"""

import requests
import json
import time

BASE_URL = "https://nutrition-launch.preview.emergentagent.com/api"
TEST_USER_EMAIL = "testuser@foodscan.com"
TEST_USER_PASSWORD = "SecurePass123!"
NUTELLA_BARCODE = "3017620422003"

def test_additional_endpoints():
    print("Testing Additional Backend Endpoints")
    print("=" * 50)
    
    # Login
    login_payload = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    if response.status_code != 200:
        print(f"❌ Login failed: {response.text}")
        return
    
    token = response.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Login successful")
    
    # Test Favorites Management
    print("\n1. Testing Favorites Management")
    print("-" * 30)
    
    # Add to favorites
    fav_payload = {"product_id": NUTELLA_BARCODE}
    response = requests.post(f"{BASE_URL}/favorites/add", json=fav_payload, headers=headers)
    if response.status_code == 200:
        print("✅ Add to favorites successful")
    else:
        print(f"❌ Add to favorites failed: {response.status_code} - {response.text}")
    
    # Get favorites
    response = requests.get(f"{BASE_URL}/favorites", headers=headers)
    if response.status_code == 200:
        favorites = response.json().get("favorites", [])
        print(f"✅ Get favorites successful: {len(favorites)} favorites found")
    else:
        print(f"❌ Get favorites failed: {response.status_code} - {response.text}")
    
    # Remove from favorites
    response = requests.delete(f"{BASE_URL}/favorites/remove/{NUTELLA_BARCODE}", headers=headers)
    if response.status_code == 200:
        print("✅ Remove from favorites successful")
    else:
        print(f"❌ Remove from favorites failed: {response.status_code} - {response.text}")
    
    # Test Subscription Upgrade
    print("\n2. Testing Subscription Upgrade")
    print("-" * 30)
    
    response = requests.post(f"{BASE_URL}/subscription/upgrade", headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data.get("subscription_tier") == "premium":
            print("✅ Subscription upgrade successful")
        else:
            print(f"❌ Subscription upgrade response invalid: {data}")
    else:
        print(f"❌ Subscription upgrade failed: {response.status_code} - {response.text}")
    
    # Verify upgrade by checking profile
    response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
    if response.status_code == 200:
        profile = response.json()
        if profile.get("subscription_tier") == "premium":
            print("✅ Profile shows premium subscription")
        else:
            print(f"❌ Profile still shows: {profile.get('subscription_tier')}")
    
    # Test unlimited scans for premium user
    print("\n3. Testing Premium Unlimited Scans")
    print("-" * 30)
    
    scan_payload = {"barcode": NUTELLA_BARCODE}
    
    # Try 3 more scans (user already had 5, should work with premium)
    for i in range(3):
        response = requests.post(f"{BASE_URL}/scan", json=scan_payload, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"✅ Premium scan #{i+1} successful")
        else:
            print(f"❌ Premium scan #{i+1} failed: {response.status_code} - {response.text}")
        time.sleep(1)

if __name__ == "__main__":
    test_additional_endpoints()