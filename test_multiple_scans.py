#!/usr/bin/env python3
"""
Additional test for multiple scans with known working barcodes
"""

import requests
import time

# Get backend URL
BASE_URL = "https://eatwhatyouare.preview.emergentagent.com"
API_BASE = f"{BASE_URL}/api"

def test_multiple_scans_with_known_barcodes():
    """Test multiple scans with barcodes known to exist in Open Food Facts"""
    
    # First register a new user for clean testing
    test_email = f"test.user+{int(time.time())}@example.com"
    test_data = {
        "email": test_email,
        "password": "TestPass123!",
        "name": "Test User"
    }
    
    # Register
    response = requests.post(f"{API_BASE}/auth/register", json=test_data, timeout=10)
    if response.status_code != 200:
        print(f"❌ Registration failed: {response.text}")
        return False
    
    auth_token = response.json()["token"]
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Known working barcodes from Open Food Facts
    known_barcodes = [
        "3017620422003",  # Nutella (already used in first scan)
        "8000500037560",  # Nutella 400g
        "3017620425035",  # Nutella B-ready
        "3017620429743",  # Nutella & GO!
        "3017620422010",  # Nutella 750g
    ]
    
    successful_scans = 0
    hit_limit = False
    
    print(f"Testing multiple scans with user: {test_data['name']}")
    
    for i, barcode in enumerate(known_barcodes):
        scan_data = {"barcode": barcode}
        try:
            print(f"Attempting scan {i+1} with barcode: {barcode}")
            response = requests.post(f"{API_BASE}/scan", json=scan_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                successful_scans += 1
                print(f"✅ Scan {i+1} successful: {data.get('product_name', 'Unknown')}")
            elif response.status_code == 403:
                print(f"🛑 Hit scan limit after {successful_scans} scans (expected for free tier)")
                hit_limit = True
                break
            elif response.status_code == 404:
                print(f"⚠️  Barcode {barcode} not found in Open Food Facts")
            else:
                print(f"❌ Scan {i+1} failed: {response.status_code} - {response.text}")
            
            time.sleep(1)  # Small delay between requests
            
        except Exception as e:
            print(f"❌ Scan {i+1} exception: {e}")
            break
    
    print(f"\nResults: {successful_scans} successful scans")
    if hit_limit:
        print("✅ Free tier limit working correctly")
    
    return successful_scans > 0

if __name__ == "__main__":
    success = test_multiple_scans_with_known_barcodes()
    print(f"\nMultiple scans test: {'PASS' if success else 'FAIL'}")