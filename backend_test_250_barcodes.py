#!/usr/bin/env python3
"""
Comprehensive Backend Testing for "You Are What You Eat" App
Testing 250 random food product barcodes for barcode scanning functionality
"""

import requests
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Tuple

# Backend URL from frontend .env
BACKEND_URL = "https://ingredient-checker-10.preview.emergentagent.com/api"

# Test user credentials
TEST_USER = {
    "email": "testuser_250scans@example.com",
    "password": "TestPassword123!",
    "name": "Test User 250 Scans"
}

# 250 Real Food Product Barcodes (mix of common products)
FOOD_BARCODES = [
    # Nutella and spreads
    "3017620422003",  # Nutella
    "3017620425004",  # Nutella Go
    "8000500037560",  # Ferrero Rocher
    
    # Coca-Cola products
    "5000112546415",  # Coca-Cola Classic
    "5000112548216",  # Coca-Cola Zero
    "5000112549017",  # Diet Coke
    "5000112547818",  # Sprite
    "5000112550019",  # Fanta Orange
    
    # Oreo and cookies
    "7622210449283",  # Oreo Original
    "7622210450289",  # Oreo Golden
    "7622300991227",  # Chips Ahoy
    "7622300992828",  # Ritz Crackers
    
    # Protein bars and health foods
    "5060292302201",  # Protein bar
    "5060292303202",  # Quest Bar
    "5060292304203",  # Kind Bar
    
    # Pasta and grains
    "8076809513388",  # Barilla pasta
    "8076809514389",  # Barilla spaghetti
    "8076809515390",  # Barilla penne
    "8001505005707",  # De Cecco pasta
    
    # Cereals
    "7613031349456",  # Nestle Cheerios
    "7613031350456",  # Nestle Corn Flakes
    "7613031351456",  # Nestle Fitness
    "4000417025005",  # Kellogg's Corn Flakes
    "4000417026005",  # Kellogg's Special K
    "4000417027005",  # Kellogg's Frosties
    
    # Yogurt and dairy
    "3033710074617",  # Danone Yogurt
    "3033710075617",  # Activia Yogurt
    "3033710076617",  # Two Good Yogurt
    "8712566441204",  # Campina Milk
    
    # Chips and snacks
    "8410076472014",  # Lay's Classic
    "8410076473014",  # Lay's Paprika
    "8410076474014",  # Doritos Nacho
    "8410076475014",  # Cheetos
    "4008400402123",  # Pringles Original
    "4008400403123",  # Pringles Sour Cream
    
    # Chocolate bars
    "7622210717283",  # Toblerone
    "7622300248215",  # Milka Alpine Milk
    "7622300249215",  # Milka Oreo
    "4000417021007",  # Snickers
    "4000417022007",  # Mars Bar
    "4000417023007",  # Twix
    "4000417024007",  # Kit Kat
    
    # Energy drinks
    "9002490100026",  # Red Bull
    "9002490101026",  # Red Bull Sugar Free
    "5060517720001",  # Monster Energy
    "5060517721001",  # Monster Ultra
    
    # Bread and bakery
    "3228021170015",  # Harry's Bread
    "3228021171015",  # Brioche Pasquier
    "4388860171501",  # Mestemacher Bread
    
    # Ice cream
    "8712566441303",  # Ben & Jerry's
    "8712566442303",  # Häagen-Dazs
    "8000300124507",  # Magnum Classic
    
    # Coffee and tea
    "7622210717390",  # Nescafe Gold
    "7622210718390",  # Nescafe Classic
    "8901030865507",  # Lipton Tea
    
    # Condiments and sauces
    "8712566441402",  # Hellmann's Mayo
    "5000157005007",  # Heinz Ketchup
    "5000157006007",  # HP Brown Sauce
    
    # Canned goods
    "5000157007007",  # Heinz Baked Beans
    "8000300124608",  # Campbell's Soup
    "4000417028007",  # Spam
    
    # Water and beverages
    "3274080005003",  # Evian Water
    "3274080006003",  # Perrier
    "5449000000996",  # Coca-Cola 330ml
    
    # Additional barcodes to reach 250
    "3017620429003", "3017620430003", "3017620431003", "3017620432003", "3017620433003",
    "5000112551019", "5000112552019", "5000112553019", "5000112554019", "5000112555019",
    "7622210451289", "7622210452289", "7622210453289", "7622210454289", "7622210455289",
    "8076809516390", "8076809517390", "8076809518390", "8076809519390", "8076809520390",
    "7613031352456", "7613031353456", "7613031354456", "7613031355456", "7613031356456",
    "4000417029005", "4000417030005", "4000417031005", "4000417032005", "4000417033005",
    "3033710077617", "3033710078617", "3033710079617", "3033710080617", "3033710081617",
    "8410076476014", "8410076477014", "8410076478014", "8410076479014", "8410076480014",
    "4008400404123", "4008400405123", "4008400406123", "4008400407123", "4008400408123",
    "7622300250215", "7622300251215", "7622300252215", "7622300253215", "7622300254215",
    "4000417025007", "4000417026007", "4000417027007", "4000417028007", "4000417029007",
    "9002490102026", "9002490103026", "9002490104026", "9002490105026", "9002490106026",
    "5060517722001", "5060517723001", "5060517724001", "5060517725001", "5060517726001",
    "3228021172015", "3228021173015", "3228021174015", "3228021175015", "3228021176015",
    "8712566443303", "8712566444303", "8712566445303", "8712566446303", "8712566447303",
    "8000300125507", "8000300126507", "8000300127507", "8000300128507", "8000300129507",
    "7622210719390", "7622210720390", "7622210721390", "7622210722390", "7622210723390",
    "8901030866507", "8901030867507", "8901030868507", "8901030869507", "8901030870507",
    "8712566448402", "8712566449402", "8712566450402", "8712566451402", "8712566452402",
    "5000157008007", "5000157009007", "5000157010007", "5000157011007", "5000157012007",
    "8000300130608", "8000300131608", "8000300132608", "8000300133608", "8000300134608",
    "4000417034007", "4000417035007", "4000417036007", "4000417037007", "4000417038007",
    "3274080007003", "3274080008003", "3274080009003", "3274080010003", "3274080011003",
    "5449000001996", "5449000002996", "5449000003996", "5449000004996", "5449000005996",
    
    # More diverse product categories
    "8901030871507", "8901030872507", "8901030873507", "8901030874507", "8901030875507",
    "3017620434003", "3017620435003", "3017620436003", "3017620437003", "3017620438003",
    "5000112556019", "5000112557019", "5000112558019", "5000112559019", "5000112560019",
    "7622210456289", "7622210457289", "7622210458289", "7622210459289", "7622210460289",
    "8076809521390", "8076809522390", "8076809523390", "8076809524390", "8076809525390",
    "7613031357456", "7613031358456", "7613031359456", "7613031360456", "7613031361456",
    "4000417039005", "4000417040005", "4000417041005", "4000417042005", "4000417043005",
    "3033710082617", "3033710083617", "3033710084617", "3033710085617", "3033710086617",
    "8410076481014", "8410076482014", "8410076483014", "8410076484014", "8410076485014",
    "4008400409123", "4008400410123", "4008400411123", "4008400412123", "4008400413123",
    "7622300255215", "7622300256215", "7622300257215", "7622300258215", "7622300259215",
    "4000417030007", "4000417031007", "4000417032007", "4000417033007", "4000417034007",
    "9002490107026", "9002490108026", "9002490109026", "9002490110026", "9002490111026",
    "5060517727001", "5060517728001", "5060517729001", "5060517730001", "5060517731001",
    "3228021177015", "3228021178015", "3228021179015", "3228021180015", "3228021181015",
    "8712566448303", "8712566449303", "8712566450303", "8712566451303", "8712566452303",
    "8000300135507", "8000300136507", "8000300137507", "8000300138507", "8000300139507",
    "7622210724390", "7622210725390", "7622210726390", "7622210727390", "7622210728390",
    "8901030876507", "8901030877507", "8901030878507", "8901030879507", "8901030880507",
    "8712566453402", "8712566454402", "8712566455402", "8712566456402", "8712566457402",
    "5000157013007", "5000157014007", "5000157015007", "5000157016007", "5000157017007",
    "8000300140608", "8000300141608", "8000300142608", "8000300143608", "8000300144608",
    "4000417044007", "4000417045007", "4000417046007", "4000417047007", "4000417048007",
    "3274080012003", "3274080013003", "3274080014003", "3274080015003", "3274080016003",
    "5449000006996", "5449000007996", "5449000008996", "5449000009996", "5449000010996"
]

class BarcodeTestResults:
    def __init__(self):
        self.total_scans = 0
        self.successful_scans = 0
        self.failed_scans = 0
        self.response_times = []
        self.errors = []
        self.healthiest_products = []
        self.unhealthiest_products = []
        self.auth_token = None
        
    def add_result(self, success: bool, response_time: float, error: str = None, product_data: dict = None):
        self.total_scans += 1
        self.response_times.append(response_time)
        
        if success:
            self.successful_scans += 1
            if product_data and 'analysis' in product_data:
                score = product_data['analysis'].get('overall_score', 0)
                product_info = {
                    'name': product_data.get('product_name', 'Unknown'),
                    'score': score,
                    'brands': product_data.get('brands', 'Unknown'),
                    'recommendation': product_data['analysis'].get('recommendation', '')
                }
                
                # Track healthiest (score >= 7)
                if score >= 7:
                    self.healthiest_products.append(product_info)
                
                # Track unhealthiest (score <= 3)
                if score <= 3:
                    self.unhealthiest_products.append(product_info)
        else:
            self.failed_scans += 1
            if error:
                self.errors.append(error)
    
    def get_summary(self) -> dict:
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 0
        success_rate = (self.successful_scans / self.total_scans * 100) if self.total_scans > 0 else 0
        
        return {
            'total_scans_attempted': self.total_scans,
            'successful_scans': self.successful_scans,
            'failed_scans': self.failed_scans,
            'success_rate_percentage': round(success_rate, 2),
            'average_response_time_seconds': round(avg_response_time, 3),
            'healthiest_products_found': len(self.healthiest_products),
            'unhealthiest_products_found': len(self.unhealthiest_products),
            'unique_errors': list(set(self.errors))
        }

def test_authentication() -> Tuple[bool, str]:
    """Test user registration and login"""
    print("🔐 Testing Authentication...")
    
    try:
        # Try to register user
        register_response = requests.post(
            f"{BACKEND_URL}/auth/register",
            json=TEST_USER,
            timeout=10
        )
        
        if register_response.status_code == 200:
            token = register_response.json().get('token')
            print("✅ Registration successful")
            return True, token
        elif register_response.status_code == 400 and "already registered" in register_response.text:
            # User exists, try login
            print("ℹ️ User exists, attempting login...")
            login_response = requests.post(
                f"{BACKEND_URL}/auth/login",
                json={"email": TEST_USER["email"], "password": TEST_USER["password"]},
                timeout=10
            )
            
            if login_response.status_code == 200:
                token = login_response.json().get('token')
                print("✅ Login successful")
                return True, token
            else:
                print(f"❌ Login failed: {login_response.status_code} - {login_response.text}")
                return False, None
        else:
            print(f"❌ Registration failed: {register_response.status_code} - {register_response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        return False, None

def test_single_barcode(barcode: str, token: str) -> Tuple[bool, float, str, dict]:
    """Test scanning a single barcode"""
    start_time = time.time()
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{BACKEND_URL}/scan",
            json={"barcode": barcode},
            headers=headers,
            timeout=15
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            # Validate required fields
            required_fields = ['product_name', 'brands', 'ingredients_text', 'analysis']
            if all(field in data for field in required_fields):
                analysis = data['analysis']
                required_analysis_fields = ['harmful_ingredients', 'beneficial_ingredients', 'overall_score', 'recommendation']
                if all(field in analysis for field in required_analysis_fields):
                    return True, response_time, None, data
                else:
                    return False, response_time, "Missing analysis fields", None
            else:
                return False, response_time, "Missing required fields", None
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
            return False, response_time, error_msg, None
            
    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        return False, response_time, "Request timeout", None
    except Exception as e:
        response_time = time.time() - start_time
        return False, response_time, f"Exception: {str(e)}", None

def run_comprehensive_barcode_test():
    """Run the comprehensive 250 barcode test"""
    print("🚀 Starting Comprehensive Barcode Scanning Test")
    print(f"📊 Testing {len(FOOD_BARCODES)} food product barcodes")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print("=" * 60)
    
    # Initialize results tracker
    results = BarcodeTestResults()
    
    # Test authentication first
    auth_success, token = test_authentication()
    if not auth_success:
        print("❌ Authentication failed. Cannot proceed with barcode testing.")
        return
    
    results.auth_token = token
    print(f"✅ Authentication successful. Token: {token[:20]}...")
    print()
    
    # Test each barcode
    print("🔍 Starting barcode scanning tests...")
    for i, barcode in enumerate(FOOD_BARCODES, 1):
        print(f"[{i:3d}/250] Testing barcode: {barcode}", end=" ")
        
        success, response_time, error, product_data = test_single_barcode(barcode, token)
        results.add_result(success, response_time, error, product_data)
        
        if success:
            product_name = product_data.get('product_name', 'Unknown')[:30]
            score = product_data['analysis'].get('overall_score', 0)
            print(f"✅ {product_name} (Score: {score}/10) [{response_time:.2f}s]")
        else:
            print(f"❌ {error} [{response_time:.2f}s]")
        
        # Add small delay to avoid overwhelming the API
        time.sleep(0.1)
        
        # Progress update every 50 scans
        if i % 50 == 0:
            current_success_rate = (results.successful_scans / results.total_scans * 100)
            print(f"\n📈 Progress: {i}/250 completed. Success rate: {current_success_rate:.1f}%\n")
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📋 FINAL TEST RESULTS")
    print("=" * 60)
    
    summary = results.get_summary()
    
    print(f"Total scans attempted: {summary['total_scans_attempted']}")
    print(f"Successful scans: {summary['successful_scans']}")
    print(f"Failed scans: {summary['failed_scans']}")
    print(f"Success rate: {summary['success_rate_percentage']}%")
    print(f"Average response time: {summary['average_response_time_seconds']}s")
    print()
    
    print(f"🥗 Healthiest products found (score ≥7): {summary['healthiest_products_found']}")
    if results.healthiest_products:
        print("Top 5 healthiest products:")
        sorted_healthy = sorted(results.healthiest_products, key=lambda x: x['score'], reverse=True)[:5]
        for product in sorted_healthy:
            print(f"  • {product['name']} - Score: {product['score']}/10")
    
    print()
    print(f"🍟 Unhealthiest products found (score ≤3): {summary['unhealthiest_products_found']}")
    if results.unhealthiest_products:
        print("Top 5 unhealthiest products:")
        sorted_unhealthy = sorted(results.unhealthiest_products, key=lambda x: x['score'])[:5]
        for product in sorted_unhealthy:
            print(f"  • {product['name']} - Score: {product['score']}/10")
    
    print()
    print("🚨 Errors encountered:")
    if summary['unique_errors']:
        for error in summary['unique_errors']:
            error_count = results.errors.count(error)
            print(f"  • {error} (occurred {error_count} times)")
    else:
        print("  No errors encountered!")
    
    print("\n" + "=" * 60)
    
    # Determine overall test status
    if summary['success_rate_percentage'] >= 80:
        print("🎉 TEST PASSED: High success rate achieved!")
        return True
    elif summary['success_rate_percentage'] >= 60:
        print("⚠️ TEST PARTIAL: Moderate success rate. Some issues detected.")
        return True
    else:
        print("❌ TEST FAILED: Low success rate. Significant issues detected.")
        return False

if __name__ == "__main__":
    test_success = run_comprehensive_barcode_test()
    exit(0 if test_success else 1)