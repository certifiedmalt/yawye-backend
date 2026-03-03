#!/usr/bin/env python3
"""
Backend API Test Suite for You Are What You Eat App
Tests auth flow, scanning, gamification, quiz, and assistant endpoints
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# Get backend URL from frontend .env file
def get_backend_url():
    try:
        with open('/app/frontend/.env', 'r') as f:
            for line in f:
                if line.startswith('EXPO_PUBLIC_BACKEND_URL='):
                    return line.split('=', 1)[1].strip()
    except Exception as e:
        print(f"Error reading frontend .env: {e}")
    return "https://ingredient-checker-10.preview.emergentagent.com"

BASE_URL = get_backend_url()
API_BASE = f"{BASE_URL}/api"

print(f"Testing backend at: {API_BASE}")

class TestResults:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        
    def add_result(self, test_name, success, details=""):
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        if not success:
            self.failed_tests.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details:
            print(f"   Details: {details}")
    
    def summary(self):
        total = len(self.results)
        passed = total - len(self.failed_tests)
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {passed}/{total} tests passed")
        print(f"{'='*60}")
        
        if self.failed_tests:
            print("\nFAILED TESTS:")
            for test in self.failed_tests:
                print(f"❌ {test['test']}: {test['details']}")
        
        return len(self.failed_tests) == 0

# Global test state
test_results = TestResults()
auth_token = None
user_data = None

def test_health_check():
    """Test basic health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                test_results.add_result("Health Check", True, "Backend is healthy")
                return True
            else:
                test_results.add_result("Health Check", False, f"Unexpected response: {data}")
        else:
            test_results.add_result("Health Check", False, f"Status code: {response.status_code}")
    except Exception as e:
        test_results.add_result("Health Check", False, f"Exception: {str(e)}")
    return False

def test_user_registration():
    """Test user registration endpoint"""
    global auth_token, user_data
    
    # Use realistic test data
    test_email = f"sarah.johnson+{int(time.time())}@example.com"
    test_data = {
        "email": test_email,
        "password": "SecurePass123!",
        "name": "Sarah Johnson"
    }
    
    try:
        response = requests.post(f"{API_BASE}/auth/register", json=test_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "token" in data and "user" in data:
                auth_token = data["token"]
                user_data = data["user"]
                test_results.add_result("User Registration", True, f"User created: {user_data['name']}")
                return True
            else:
                test_results.add_result("User Registration", False, f"Missing token or user in response: {data}")
        else:
            test_results.add_result("User Registration", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("User Registration", False, f"Exception: {str(e)}")
    return False

def test_user_login():
    """Test user login endpoint"""
    global auth_token, user_data
    
    if not user_data:
        test_results.add_result("User Login", False, "No user data from registration")
        return False
    
    login_data = {
        "email": user_data["email"],
        "password": "SecurePass123!"
    }
    
    try:
        response = requests.post(f"{API_BASE}/auth/login", json=login_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "token" in data and "user" in data:
                auth_token = data["token"]  # Update token
                test_results.add_result("User Login", True, f"Login successful for {data['user']['name']}")
                return True
            else:
                test_results.add_result("User Login", False, f"Missing token or user in response: {data}")
        else:
            test_results.add_result("User Login", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("User Login", False, f"Exception: {str(e)}")
    return False

def test_auth_me():
    """Test /api/auth/me endpoint"""
    if not auth_token:
        test_results.add_result("Auth Me", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    try:
        response = requests.get(f"{API_BASE}/auth/me", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["id", "email", "name", "subscription_tier", "daily_scans"]
            if all(field in data for field in required_fields):
                test_results.add_result("Auth Me", True, f"Profile retrieved: {data['name']}, tier: {data['subscription_tier']}, scans: {data['daily_scans']}")
                return True
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Auth Me", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Auth Me", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Auth Me", False, f"Exception: {str(e)}")
    return False

def test_product_scan():
    """Test product scanning with Open Food Facts integration"""
    if not auth_token:
        test_results.add_result("Product Scan", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Test with Nutella barcode (known to work with Open Food Facts)
    scan_data = {"barcode": "3017620422003"}  # Nutella
    
    try:
        response = requests.post(f"{API_BASE}/scan", json=scan_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["product_name", "brands", "ingredients_text", "analysis"]
            if all(field in data for field in required_fields):
                analysis = data["analysis"]
                if "overall_score" in analysis and "harmful_ingredients" in analysis:
                    test_results.add_result("Product Scan", True, 
                        f"Scanned: {data['product_name']}, Score: {analysis.get('overall_score')}/10, "
                        f"Harmful ingredients: {len(analysis.get('harmful_ingredients', []))}")
                    return True
                else:
                    test_results.add_result("Product Scan", False, f"Incomplete analysis data: {analysis}")
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Product Scan", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Product Scan", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Product Scan", False, f"Exception: {str(e)}")
    return False

def test_multiple_scans():
    """Test multiple scans to verify free tier limits"""
    if not auth_token:
        test_results.add_result("Multiple Scans", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Test different barcodes
    barcodes = [
        "8901030895559",  # Maggi noodles
        "8901030895566",  # Another Maggi variant
        "8901030895573",  # Another variant
        "8901030895580",  # Another variant
    ]
    
    successful_scans = 0
    
    for i, barcode in enumerate(barcodes):
        scan_data = {"barcode": barcode}
        try:
            response = requests.post(f"{API_BASE}/scan", json=scan_data, headers=headers, timeout=30)
            if response.status_code == 200:
                successful_scans += 1
            elif response.status_code == 403:
                # Hit the limit
                break
            time.sleep(1)  # Small delay between requests
        except Exception as e:
            print(f"Scan {i+1} failed: {e}")
            break
    
    if successful_scans > 0:
        test_results.add_result("Multiple Scans", True, f"Successfully performed {successful_scans} additional scans")
        return True
    else:
        test_results.add_result("Multiple Scans", False, "No successful scans")
        return False

def test_gamification_stats():
    """Test gamification stats endpoint"""
    if not auth_token:
        test_results.add_result("Gamification Stats", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    try:
        response = requests.get(f"{API_BASE}/gamification/stats", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["current_streak", "level", "xp", "daily_quests", "total_scans"]
            if all(field in data for field in required_fields):
                test_results.add_result("Gamification Stats", True, 
                    f"Level: {data['level']}, XP: {data['xp']}, Streak: {data['current_streak']}, "
                    f"Total scans: {data['total_scans']}")
                return True
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Gamification Stats", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Gamification Stats", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Gamification Stats", False, f"Exception: {str(e)}")
    return False

def test_update_streak():
    """Test streak update endpoint"""
    if not auth_token:
        test_results.add_result("Update Streak", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    try:
        response = requests.post(f"{API_BASE}/gamification/update-streak", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["current_streak", "total_scans"]
            if all(field in data for field in required_fields):
                test_results.add_result("Update Streak", True, 
                    f"Streak updated: {data['current_streak']}, Total scans: {data['total_scans']}")
                return True
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Update Streak", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Update Streak", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Update Streak", False, f"Exception: {str(e)}")
    return False

def test_daily_quiz():
    """Test daily quiz endpoint"""
    if not auth_token:
        test_results.add_result("Daily Quiz", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    try:
        response = requests.get(f"{API_BASE}/quiz/daily", headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["id", "question", "options"]
            if all(field in data for field in required_fields):
                test_results.add_result("Daily Quiz", True, 
                    f"Quiz question retrieved: {data['question'][:50]}...")
                return data  # Return question data for answer test
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Daily Quiz", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Daily Quiz", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Daily Quiz", False, f"Exception: {str(e)}")
    return None

def test_quiz_answer(question_data):
    """Test quiz answer submission"""
    if not auth_token or not question_data:
        test_results.add_result("Quiz Answer", False, "No auth token or question data available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Submit answer (choose first option)
    answer_data = {
        "question_id": question_data["id"],
        "answer": "0"  # First option
    }
    
    try:
        response = requests.post(f"{API_BASE}/quiz/answer", json=answer_data, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["correct", "explanation"]
            if all(field in data for field in required_fields):
                result = "correct" if data["correct"] else "incorrect"
                test_results.add_result("Quiz Answer", True, 
                    f"Answer submitted ({result}), XP earned: {data.get('xp_earned', 'N/A')}")
                return True
            else:
                missing = [f for f in required_fields if f not in data]
                test_results.add_result("Quiz Answer", False, f"Missing fields: {missing}")
        else:
            test_results.add_result("Quiz Answer", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Quiz Answer", False, f"Exception: {str(e)}")
    return False

def test_complete_quest():
    """Test quest completion endpoint"""
    if not auth_token:
        test_results.add_result("Complete Quest", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Try to complete a quest
    quest_id = "scan_3_products"
    
    try:
        response = requests.post(f"{API_BASE}/gamification/complete-quest?quest_id={quest_id}", 
                               headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "success" in data:
                if data["success"]:
                    test_results.add_result("Complete Quest", True, 
                        f"Quest completed, XP earned: {data.get('xp_earned', 'N/A')}")
                else:
                    test_results.add_result("Complete Quest", True, 
                        f"Quest endpoint working (already completed): {data.get('message', '')}")
                return True
            else:
                test_results.add_result("Complete Quest", False, f"Unexpected response format: {data}")
        else:
            test_results.add_result("Complete Quest", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Complete Quest", False, f"Exception: {str(e)}")
    return False

def test_assistant_chat():
    """Test AI assistant chat endpoint"""
    if not auth_token:
        test_results.add_result("Assistant Chat", False, "No auth token available")
        return False
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Ask a simple nutrition question
    chat_data = {
        "message": "What are ultra-processed foods and why should I avoid them?",
        "conversation_history": []
    }
    
    try:
        response = requests.post(f"{API_BASE}/assistant/chat", json=chat_data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                response_text = data["response"]
                test_results.add_result("Assistant Chat", True, 
                    f"AI response received ({len(response_text)} chars): {response_text[:100]}...")
                return True
            else:
                test_results.add_result("Assistant Chat", False, f"Empty or missing response: {data}")
        else:
            test_results.add_result("Assistant Chat", False, f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        test_results.add_result("Assistant Chat", False, f"Exception: {str(e)}")
    return False

def main():
    """Run all backend tests in realistic flow"""
    print("Starting Backend API Test Suite")
    print(f"Target: {API_BASE}")
    print("="*60)
    
    # Test sequence following realistic user flow
    
    # 1. Health check
    test_health_check()
    
    # 2. Authentication flow
    test_user_registration()
    test_user_login()
    test_auth_me()
    
    # 3. Product scanning (respecting free tier limits)
    test_product_scan()
    test_multiple_scans()
    
    # 4. Gamification endpoints
    test_gamification_stats()
    test_update_streak()
    
    # 5. Quiz system
    question_data = test_daily_quiz()
    test_quiz_answer(question_data)
    test_complete_quest()
    
    # 6. AI Assistant
    test_assistant_chat()
    
    # Summary
    all_passed = test_results.summary()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)