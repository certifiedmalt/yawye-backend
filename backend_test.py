#!/usr/bin/env python3
"""
Backend API Testing for "You Are What You Eat" Barcode Scanning App
Tests all backend endpoints in priority order as specified in review request.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "https://upf-scanner.preview.emergentagent.com/api"
TEST_USER_EMAIL = "testuser@foodscan.com"
TEST_USER_PASSWORD = "SecurePass123!"
TEST_USER_NAME = "Food Scanner Test User"
NUTELLA_BARCODE = "3017620422003"  # Good test case as specified

class BackendTester:
    def __init__(self):
        self.base_url = BASE_URL
        self.token = None
        self.user_id = None
        self.test_results = []
        
    def log_test(self, test_name, success, details, response_data=None):
        """Log test results"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        if response_data and not success:
            print(f"   Response: {response_data}")
        print()

    def test_health_check(self):
        """Test 1: Health Check - GET /api/health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_test("Health Check", True, "Backend is healthy and responding")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response format: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, f"Connection error: {str(e)}")
            return False

    def test_user_registration(self):
        """Test 2: User Registration - POST /api/auth/register"""
        try:
            payload = {
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD,
                "name": TEST_USER_NAME
            }
            
            response = requests.post(
                f"{self.base_url}/auth/register",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "token" in data and "user" in data:
                    self.token = data["token"]
                    self.user_id = data["user"]["id"]
                    user_info = data["user"]
                    
                    # Validate user object structure
                    required_fields = ["id", "email", "name", "subscription_tier"]
                    missing_fields = [field for field in required_fields if field not in user_info]
                    
                    if missing_fields:
                        self.log_test("User Registration", False, 
                                    f"Missing user fields: {missing_fields}", data)
                        return False
                    
                    if user_info["email"] != TEST_USER_EMAIL or user_info["name"] != TEST_USER_NAME:
                        self.log_test("User Registration", False, 
                                    "User data mismatch", data)
                        return False
                        
                    self.log_test("User Registration", True, 
                                f"User registered successfully. ID: {self.user_id}, Tier: {user_info['subscription_tier']}")
                    return True
                else:
                    self.log_test("User Registration", False, 
                                "Missing token or user in response", data)
                    return False
            elif response.status_code == 400:
                # User might already exist, try to continue with login
                self.log_test("User Registration", False, 
                            f"Registration failed (user may exist): {response.text}")
                return False
            else:
                self.log_test("User Registration", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("User Registration", False, f"Request error: {str(e)}")
            return False

    def test_user_login(self):
        """Test 3: User Login - POST /api/auth/login"""
        try:
            payload = {
                "email": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            }
            
            response = requests.post(
                f"{self.base_url}/auth/login",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "token" in data and "user" in data:
                    self.token = data["token"]
                    self.user_id = data["user"]["id"]
                    user_info = data["user"]
                    
                    # Validate user object structure
                    required_fields = ["id", "email", "name", "subscription_tier"]
                    missing_fields = [field for field in required_fields if field not in user_info]
                    
                    if missing_fields:
                        self.log_test("User Login", False, 
                                    f"Missing user fields: {missing_fields}", data)
                        return False
                    
                    self.log_test("User Login", True, 
                                f"Login successful. Token received, User ID: {self.user_id}")
                    return True
                else:
                    self.log_test("User Login", False, 
                                "Missing token or user in response", data)
                    return False
            else:
                self.log_test("User Login", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("User Login", False, f"Request error: {str(e)}")
            return False

    def test_get_user_profile(self):
        """Test 4: Get User Profile - GET /api/auth/me"""
        if not self.token:
            self.log_test("Get User Profile", False, "No authentication token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                f"{self.base_url}/auth/me",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["id", "email", "name", "subscription_tier", "daily_scans"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Get User Profile", False, 
                                f"Missing profile fields: {missing_fields}", data)
                    return False
                
                self.log_test("Get User Profile", True, 
                            f"Profile retrieved. Daily scans: {data['daily_scans']}, Tier: {data['subscription_tier']}")
                return True
            else:
                self.log_test("Get User Profile", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Get User Profile", False, f"Request error: {str(e)}")
            return False

    def test_product_scanning(self):
        """Test 5: Product Scanning - POST /api/scan (CRITICAL FEATURE)"""
        if not self.token:
            self.log_test("Product Scanning", False, "No authentication token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {"barcode": NUTELLA_BARCODE}
            
            print(f"Testing product scan with Nutella barcode: {NUTELLA_BARCODE}")
            response = requests.post(
                f"{self.base_url}/scan",
                json=payload,
                headers=headers,
                timeout=30  # Longer timeout for AI analysis
            )
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["product_name", "brands", "ingredients_text", "analysis"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_test("Product Scanning", False, 
                                f"Missing response fields: {missing_fields}", data)
                    return False
                
                # Validate AI analysis structure
                analysis = data.get("analysis", {})
                analysis_fields = ["harmful_ingredients", "beneficial_ingredients", "overall_score", "recommendation"]
                missing_analysis = [field for field in analysis_fields if field not in analysis]
                
                if missing_analysis:
                    self.log_test("Product Scanning", False, 
                                f"Missing AI analysis fields: {missing_analysis}", data)
                    return False
                
                # Check if AI analysis has meaningful content
                harmful = analysis.get("harmful_ingredients", [])
                beneficial = analysis.get("beneficial_ingredients", [])
                score = analysis.get("overall_score", 0)
                
                # Validate harmful ingredients structure
                if harmful:
                    for ingredient in harmful[:2]:  # Check first 2
                        required_harm_fields = ["name", "health_risk", "severity", "study_reference"]
                        missing_harm = [field for field in required_harm_fields if field not in ingredient]
                        if missing_harm:
                            self.log_test("Product Scanning", False, 
                                        f"Harmful ingredient missing fields: {missing_harm}", ingredient)
                            return False
                
                # Validate beneficial ingredients structure  
                if beneficial:
                    for ingredient in beneficial[:2]:  # Check first 2
                        required_benefit_fields = ["name", "health_benefit", "study_reference"]
                        missing_benefit = [field for field in required_benefit_fields if field not in ingredient]
                        if missing_benefit:
                            self.log_test("Product Scanning", False, 
                                        f"Beneficial ingredient missing fields: {missing_benefit}", ingredient)
                            return False
                
                # Check score range
                if not (1 <= score <= 10):
                    self.log_test("Product Scanning", False, 
                                f"Invalid health score: {score} (should be 1-10)", data)
                    return False
                
                self.log_test("Product Scanning", True, 
                            f"Scan successful! Product: {data['product_name']}, Score: {score}/10, "
                            f"Harmful: {len(harmful)}, Beneficial: {len(beneficial)}")
                
                # Log some analysis details for verification
                if harmful:
                    print(f"   Sample harmful ingredient: {harmful[0]['name']} - {harmful[0]['health_risk']}")
                if beneficial:
                    print(f"   Sample beneficial ingredient: {beneficial[0]['name']} - {beneficial[0]['health_benefit']}")
                    
                return True
            elif response.status_code == 404:
                self.log_test("Product Scanning", False, 
                            f"Product not found or no ingredients: {response.text}")
                return False
            elif response.status_code == 403:
                self.log_test("Product Scanning", False, 
                            f"Scan limit reached: {response.text}")
                return False
            else:
                self.log_test("Product Scanning", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Product Scanning", False, f"Request error: {str(e)}")
            return False

    def test_scan_history(self):
        """Test 6: Scan History - GET /api/scans/history"""
        if not self.token:
            self.log_test("Scan History", False, "No authentication token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.get(
                f"{self.base_url}/scans/history",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "scans" in data:
                    scans = data["scans"]
                    if len(scans) > 0:
                        # Validate scan structure
                        scan = scans[0]
                        required_fields = ["user_id", "barcode", "product_name", "analysis", "scanned_at"]
                        missing_fields = [field for field in required_fields if field not in scan]
                        
                        if missing_fields:
                            self.log_test("Scan History", False, 
                                        f"Missing scan fields: {missing_fields}", scan)
                            return False
                    
                    self.log_test("Scan History", True, 
                                f"History retrieved successfully. {len(scans)} scans found")
                    return True
                else:
                    self.log_test("Scan History", False, 
                                "Missing 'scans' field in response", data)
                    return False
            else:
                self.log_test("Scan History", False, 
                            f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Scan History", False, f"Request error: {str(e)}")
            return False

    def test_daily_scan_limit(self):
        """Test 7: Daily Scan Limit (Free tier) - Multiple scans to test limit"""
        if not self.token:
            self.log_test("Daily Scan Limit", False, "No authentication token available")
            return False
            
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            payload = {"barcode": NUTELLA_BARCODE}
            
            # Try to scan 6 times (free limit is 5)
            successful_scans = 0
            for i in range(6):
                response = requests.post(
                    f"{self.base_url}/scan",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    successful_scans += 1
                elif response.status_code == 403:
                    # This is expected after 5 scans
                    if successful_scans >= 5:
                        self.log_test("Daily Scan Limit", True, 
                                    f"Scan limit enforced correctly after {successful_scans} scans. 6th scan blocked with 403.")
                        return True
                    else:
                        self.log_test("Daily Scan Limit", False, 
                                    f"Unexpected 403 after only {successful_scans} scans")
                        return False
                else:
                    self.log_test("Daily Scan Limit", False, 
                                f"Unexpected response on scan {i+1}: HTTP {response.status_code}")
                    return False
                
                # Small delay between scans
                time.sleep(1)
            
            # If we get here, limit wasn't enforced
            self.log_test("Daily Scan Limit", False, 
                        f"Scan limit not enforced - completed {successful_scans} scans without restriction")
            return False
            
        except Exception as e:
            self.log_test("Daily Scan Limit", False, f"Request error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests in priority order"""
        print("=" * 80)
        print("BACKEND API TESTING - You Are What You Eat App")
        print("=" * 80)
        print(f"Testing against: {self.base_url}")
        print(f"Test user: {TEST_USER_EMAIL}")
        print(f"Test barcode: {NUTELLA_BARCODE} (Nutella)")
        print("=" * 80)
        print()
        
        # Test sequence as specified in review request
        tests = [
            ("Health Check", self.test_health_check),
            ("User Registration", self.test_user_registration),
            ("User Login", self.test_user_login),
            ("Get User Profile", self.test_get_user_profile),
            ("Product Scanning (CRITICAL)", self.test_product_scanning),
            ("Scan History", self.test_scan_history),
            ("Daily Scan Limit", self.test_daily_scan_limit)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"Running: {test_name}")
            print("-" * 40)
            if test_func():
                passed += 1
            print()
        
        # Summary
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED - See details above")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test']}: {result['details']}")
        
        return passed, total

if __name__ == "__main__":
    tester = BackendTester()
    passed, total = tester.run_all_tests()
    
    # Exit with error code if tests failed
    if passed != total:
        exit(1)