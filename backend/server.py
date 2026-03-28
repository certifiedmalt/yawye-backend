from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import requests
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from openai import AsyncOpenAI
import asyncio
import time
import logging
import random
import hashlib
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import resend
import urllib.parse
import base64

# Setup logging for analytics
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("barcode_scanner")

load_dotenv()

app = FastAPI()

# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again."}
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiter for auth endpoints
rate_limit_store: Dict[str, list] = {}
RATE_LIMIT_WINDOW = 300  # 5 minutes
RATE_LIMIT_MAX = 10  # max attempts per window

def check_rate_limit(key: str) -> bool:
    """Returns True if request is allowed, False if rate limited"""
    now = time.time()
    if key not in rate_limit_store:
        rate_limit_store[key] = []
    # Clean old entries
    rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[key]) >= RATE_LIMIT_MAX:
        return False
    rate_limit_store[key].append(now)
    return True

# MongoDB
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
scans_collection = db["scans"]
favorites_collection = db["favorites"]
product_cache_collection = db["product_cache"]  # New: Cache for faster lookups
scan_analytics_collection = db["scan_analytics"]  # New: Analytics tracking

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "yawye-prod-secret-k3y-2026-x9m2p7q4")
ALGORITHM = "HS256"

# Email Configuration (Resend)
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "re_AQX2aqrW_F2SZfwdfcgufvnNYuLNHCzFa")

async def send_reset_email(to_email: str, reset_code: str):
    """Send password reset code via Resend"""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set - email sending disabled")
        return False
    
    try:
        resend.api_key = RESEND_API_KEY

        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; padding: 24px; background: #0c0c0c; color: #ffffff; border-radius: 12px;">
            <h2 style="color: #00e676; text-align: center;">You Are What You Eat</h2>
            <p style="color: #ffffff;">Hi there,</p>
            <p style="color: #cccccc;">You requested a password reset. Here's your code:</p>
            <div style="background: #1a1a1a; border-radius: 8px; padding: 20px; text-align: center; margin: 20px 0;">
                <span style="font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #00e676;">{reset_code}</span>
            </div>
            <p style="color: #888888; font-size: 14px;">This code expires in 15 minutes.</p>
            <p style="color: #888888; font-size: 14px;">If you didn't request this, you can safely ignore this email.</p>
            <hr style="border: 1px solid #333333; margin: 20px 0;">
            <p style="color: #666666; font-size: 12px; text-align: center;">You Are What You Eat - Scan smarter, eat better.</p>
        </div>
        """

        params = {
            "from": "You Are What You Eat <onboarding@resend.dev>",
            "to": [to_email],
            "subject": "Your Password Reset Code",
            "html": html_content,
        }

        # Send in a thread to not block async
        def _send():
            return resend.Emails.send(params)
        
        result = await asyncio.get_event_loop().run_in_executor(None, _send)
        logger.info(f"Reset email sent to {to_email} via Resend: {result}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email via Resend to {to_email}: {e}")
        return False

# Open Food Facts API (Global + UK-specific)
OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product"
OFF_UK_API_URL = "https://uk.openfoodfacts.org/api/v2/product"
# Brocade.io - Open barcode database (free, no key needed)
BROCADE_API_URL = "https://www.brocade.io/api/items"
# Backup API - UPC Item DB
UPC_API_URL = "https://api.upcitemdb.com/prod/trial/lookup"
# USDA FoodData Central API (Free)
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1"
USDA_API_KEY = os.getenv("USDA_API_KEY", "8j33kUA9sULTBKp9NVrNZj0YYmpwSbFc4ZDvbxTc")
# FatSecret API (Free tier - 5000 calls/month)
FATSECRET_API_URL = "https://platform.fatsecret.com/rest/server.api"
FATSECRET_CLIENT_ID = os.getenv("FATSECRET_CLIENT_ID", "")
FATSECRET_CLIENT_SECRET = os.getenv("FATSECRET_CLIENT_SECRET", "")

# LLM Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cache settings
CACHE_EXPIRY_DAYS = 30  # Cache products for 30 days

# Analytics tracking
async def log_scan_analytics(barcode: str, success: bool, source: str, response_time: float, error: str = None):
    """Log scan analytics for monitoring"""
    try:
        await scan_analytics_collection.insert_one({
            "barcode": barcode,
            "success": success,
            "source": source,  # "cache", "openfoodfacts", "upcitemdb"
            "response_time_ms": int(response_time * 1000),
            "error": error,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"SCAN: barcode={barcode} success={success} source={source} time={response_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")

# Product caching functions
async def get_cached_product(barcode: str) -> Optional[Dict[str, Any]]:
    """Get product from cache if exists and not expired"""
    try:
        cached = await product_cache_collection.find_one({"barcode": barcode})
        if cached:
            # Check if cache is still valid
            cache_age = datetime.utcnow() - cached.get("cached_at", datetime.utcnow())
            if cache_age.days < CACHE_EXPIRY_DAYS:
                logger.info(f"CACHE HIT: {barcode}")
                return cached
            else:
                logger.info(f"CACHE EXPIRED: {barcode}")
        return None
    except Exception as e:
        logger.error(f"Cache lookup error: {e}")
        return None

async def cache_product(barcode: str, product_data: Dict[str, Any]):
    """Cache product data for faster future lookups"""
    try:
        cache_doc = {
            "barcode": barcode,
            "product_name": product_data.get("product_name"),
            "brands": product_data.get("brands"),
            "ingredients_text": product_data.get("ingredients_text"),
            "image_url": product_data.get("image_url"),
            "analysis": product_data.get("analysis"),
            "cached_at": datetime.utcnow(),
            "source": product_data.get("source", "openfoodfacts")
        }
        await product_cache_collection.update_one(
            {"barcode": barcode},
            {"$set": cache_doc},
            upsert=True
        )
        logger.info(f"CACHED: {barcode}")
    except Exception as e:
        logger.error(f"Cache save error: {e}")

# Retry logic for API calls
def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 15) -> Optional[requests.Response]:
    """Fetch URL with retry logic and exponential backoff"""
    headers = {"User-Agent": "YAWYE-App/1.0 (contact@yawye.app)"}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers=headers)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                return response  # Product genuinely not found
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error on attempt {attempt + 1}: {e}")
        
        # Exponential backoff
        if attempt < max_retries - 1:
            wait_time = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
            time.sleep(wait_time)
    
    return None

# Fetch from Open Food Facts
def fetch_from_openfoodfacts(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from Open Food Facts API"""
    start_time = time.time()
    try:
        response = fetch_with_retry(f"{OFF_API_URL}/{barcode}.json", max_retries=2, timeout=10)
        
        if response and response.status_code == 200:
            data = response.json()
            if data.get("status") == 1:
                product = data.get("product", {})
                return {
                    "product_name": product.get("product_name", "Unknown Product"),
                    "brands": product.get("brands", "Unknown Brand"),
                    "ingredients_text": product.get("ingredients_text", ""),
                    "image_url": product.get("image_url", ""),
                    "source": "openfoodfacts",
                    "fetch_time": time.time() - start_time
                }
    except Exception as e:
        logger.error(f"Open Food Facts error: {e}")
    
    return None

# Fetch from USDA FoodData Central (Free API)
def fetch_from_usda(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from USDA FoodData Central"""
    start_time = time.time()
    try:
        # Search by UPC/GTIN code
        search_url = f"{USDA_API_URL}/foods/search?query={barcode}&dataType=Branded&pageSize=1&api_key={USDA_API_KEY}"
        response = fetch_with_retry(search_url, max_retries=2, timeout=15)
        
        if response and response.status_code == 200:
            data = response.json()
            foods = data.get("foods", [])
            if foods:
                food = foods[0]
                # Extract ingredients from USDA format
                ingredients = food.get("ingredients", "")
                
                return {
                    "product_name": food.get("description", "Unknown Product"),
                    "brands": food.get("brandOwner", food.get("brandName", "Unknown Brand")),
                    "ingredients_text": ingredients,
                    "image_url": "",  # USDA doesn't provide images
                    "source": "usda",
                    "fetch_time": time.time() - start_time,
                    "nutrition": {
                        "calories": next((n.get("value") for n in food.get("foodNutrients", []) if n.get("nutrientName") == "Energy"), None),
                        "protein": next((n.get("value") for n in food.get("foodNutrients", []) if n.get("nutrientName") == "Protein"), None),
                        "fat": next((n.get("value") for n in food.get("foodNutrients", []) if "Total lipid" in n.get("nutrientName", "")), None),
                        "carbs": next((n.get("value") for n in food.get("foodNutrients", []) if "Carbohydrate" in n.get("nutrientName", "")), None),
                    }
                }
        
        # Also try searching by product name extracted from barcode lookup
        logger.info(f"USDA: No direct barcode match for {barcode}, trying alternate search")
        
    except Exception as e:
        logger.error(f"USDA API error: {e}")
    
    return None

# Fetch from FatSecret API (Free tier - 5000 calls/month)
def fetch_from_fatsecret(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from FatSecret API"""
    start_time = time.time()
    
    # FatSecret requires OAuth 1.0 authentication
    if not FATSECRET_CLIENT_ID or not FATSECRET_CLIENT_SECRET:
        logger.debug("FatSecret credentials not configured")
        return None
    
    try:
        import hashlib
        import hmac
        import urllib.parse
        
        # OAuth 1.0 signature base
        oauth_params = {
            "oauth_consumer_key": FATSECRET_CLIENT_ID,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_nonce": ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16)),
            "oauth_version": "1.0",
            "method": "food.find_id_for_barcode",
            "barcode": barcode,
            "format": "json"
        }
        
        # Create signature base string
        sorted_params = sorted(oauth_params.items())
        param_string = urllib.parse.urlencode(sorted_params)
        base_string = f"GET&{urllib.parse.quote(FATSECRET_API_URL, safe='')}&{urllib.parse.quote(param_string, safe='')}"
        
        # Sign with HMAC-SHA1
        signing_key = f"{FATSECRET_CLIENT_SECRET}&"
        signature = hmac.new(
            signing_key.encode('utf-8'),
            base_string.encode('utf-8'),
            hashlib.sha1
        ).digest()
        
        import base64
        oauth_params["oauth_signature"] = base64.b64encode(signature).decode('utf-8')
        
        # Make request
        url = f"{FATSECRET_API_URL}?{urllib.parse.urlencode(oauth_params)}"
        response = requests.get(url, timeout=10, headers={"User-Agent": "YAWYE-App/1.0 (contact@yawye.app)"})
        
        if response.status_code == 200:
            data = response.json()
            if "food_id" in data:
                food_id = data["food_id"]["value"]
                
                # Get food details
                oauth_params["method"] = "food.get.v2"
                oauth_params["food_id"] = food_id
                oauth_params["oauth_timestamp"] = str(int(time.time()))
                oauth_params["oauth_nonce"] = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))
                del oauth_params["barcode"]
                
                # Re-sign
                sorted_params = sorted(oauth_params.items())
                param_string = urllib.parse.urlencode(sorted_params)
                base_string = f"GET&{urllib.parse.quote(FATSECRET_API_URL, safe='')}&{urllib.parse.quote(param_string, safe='')}"
                signature = hmac.new(signing_key.encode('utf-8'), base_string.encode('utf-8'), hashlib.sha1).digest()
                oauth_params["oauth_signature"] = base64.b64encode(signature).decode('utf-8')
                
                detail_url = f"{FATSECRET_API_URL}?{urllib.parse.urlencode(oauth_params)}"
                detail_response = requests.get(detail_url, timeout=10, headers={"User-Agent": "YAWYE-App/1.0 (contact@yawye.app)"})
                
                if detail_response.status_code == 200:
                    food_data = detail_response.json().get("food", {})
                    return {
                        "product_name": food_data.get("food_name", "Unknown Product"),
                        "brands": food_data.get("brand_name", "Unknown Brand"),
                        "ingredients_text": "",  # FatSecret doesn't provide ingredients in free tier
                        "image_url": "",
                        "source": "fatsecret",
                        "fetch_time": time.time() - start_time
                    }
                    
    except Exception as e:
        logger.error(f"FatSecret API error: {e}")
    
    return None

# Fetch from UPC Item DB (backup)
def fetch_from_upcitemdb(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from UPC Item DB as backup"""
    start_time = time.time()
    try:
        response = fetch_with_retry(f"{UPC_API_URL}?upc={barcode}", max_retries=2, timeout=10)
        
        if response and response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            if items:
                item = items[0]
                # UPC Item DB doesn't have ingredients, but has basic info
                return {
                    "product_name": item.get("title", "Unknown Product"),
                    "brands": item.get("brand", "Unknown Brand"),
                    "ingredients_text": item.get("description", ""),  # May be empty
                    "image_url": item.get("images", [""])[0] if item.get("images") else "",
                    "source": "upcitemdb",
                    "fetch_time": time.time() - start_time
                }
    except Exception as e:
        logger.error(f"UPC Item DB error: {e}")
    
    return None

def fetch_from_off_search(barcode: str) -> Optional[Dict[str, Any]]:
    """Search Open Food Facts by barcode using their search API as a fallback"""
    start_time = time.time()
    try:
        # Try the v2 search endpoint which sometimes finds products the direct lookup misses
        search_url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={barcode}&search_simple=1&action=process&json=1&page_size=1"
        headers = {"User-Agent": "YAWYE-App/1.0 (contact@yawye.app)"}
        response = requests.get(search_url, timeout=10, headers=headers)
        
        if response and response.status_code == 200:
            data = response.json()
            products = data.get("products", [])
            if products:
                product = products[0]
                return {
                    "product_name": product.get("product_name", "Unknown Product"),
                    "brands": product.get("brands", "Unknown Brand"),
                    "ingredients_text": product.get("ingredients_text", ""),
                    "image_url": product.get("image_url", ""),
                    "source": "openfoodfacts_search",
                    "fetch_time": time.time() - start_time
                }
    except Exception as e:
        logger.error(f"OFF Search error: {e}")
    
    return None

def fetch_from_off_uk(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from UK-specific Open Food Facts database"""
    start_time = time.time()
    try:
        response = fetch_with_retry(f"{OFF_UK_API_URL}/{barcode}.json", max_retries=2, timeout=10)
        
        if response and response.status_code == 200:
            data = response.json()
            if data.get("status") == 1:
                product = data.get("product", {})
                name = product.get("product_name", "")
                if name:
                    return {
                        "product_name": name,
                        "brands": product.get("brands", "Unknown Brand"),
                        "ingredients_text": product.get("ingredients_text", ""),
                        "image_url": product.get("image_url", ""),
                        "source": "openfoodfacts_uk",
                        "fetch_time": time.time() - start_time
                    }
    except Exception as e:
        logger.error(f"OFF UK error: {e}")
    return None

def fetch_from_brocade(barcode: str) -> Optional[Dict[str, Any]]:
    """Fetch product data from Brocade.io open barcode database"""
    start_time = time.time()
    try:
        # Pad barcode to 14 digits (GTIN format)
        gtin = barcode.zfill(14)
        headers = {"User-Agent": "YAWYE-App/1.0 (contact@yawye.app)"}
        response = requests.get(f"{BROCADE_API_URL}/{gtin}", timeout=8, headers=headers)
        
        if response and response.status_code == 200:
            data = response.json()
            name = data.get("name", "")
            if name:
                # Brocade stores ingredients in properties
                props = data.get("properties", {})
                ingredients = props.get("ingredients", "")
                return {
                    "product_name": name,
                    "brands": data.get("brand_name", "Unknown Brand"),
                    "ingredients_text": ingredients,
                    "image_url": "",
                    "source": "brocade",
                    "fetch_time": time.time() - start_time
                }
    except Exception as e:
        logger.error(f"Brocade.io error: {e}")
    return None

# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    email: EmailStr
    code: str
    new_password: str

class ScanRequest(BaseModel):
    barcode: str

class FavoriteRequest(BaseModel):
    product_id: str

class QuizAnswerRequest(BaseModel):
    question_id: str
    answer: str

# Helper Functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

async def identify_product_by_barcode(client, barcode: str) -> dict:
    """When no food database has this barcode, use AI to identify and analyze the product"""
    try:
        prompt = f"""You are a food product expert with encyclopedic knowledge of barcodes and commercial food products worldwide.

A user scanned barcode: {barcode}

Barcode prefixes indicate the country of origin:
- 50xxxxx = United Kingdom
- 40-44xxx = Germany
- 30-37xxx = France
- 80-83xxx = Italy/Spain
- 00-09xxx = USA/Canada
- 87xxxxx = Netherlands
- 93-94xxx = Australia/NZ

TASK: Identify what product this barcode belongs to. If you recognize it, provide the product name, brand, typical ingredients, and full health analysis. If you don't recognize the exact barcode, say so honestly but still provide your best guess based on the barcode prefix (country) and any patterns you recognize.

Respond with JSON only:
{{
  "identified_product": "product name or 'Unknown product'",
  "identified_brand": "brand name or 'Unknown'",
  "typical_ingredients": "comma-separated list of typical ingredients for this product",
  "harmful_ingredients": [
    {{"name": "ingredient", "health_impact": "explanation", "severity": "high/medium/low", "processing_level": "NOVA level", "research_summary": "citation", "study_link": "pubmed link"}}
  ],
  "beneficial_ingredients": [
    {{"name": "ingredient", "health_benefit": "explanation", "benefit_type": "type", "key_nutrients": "list", "processing_level": "NOVA level", "research_summary": "citation", "study_link": "link"}}
  ],
  "carcinogens_found": [
    {{"name": "chemical", "iarc_group": "Group classification", "cancer_types": "linked cancers", "explanation": "how it causes harm", "source": "reference"}}
  ],
  "chemical_breakdown": [
    {{"name": "E-number or chemical", "common_name": "actual name", "purpose": "why used", "health_concern": "risk summary", "banned_in": "countries or empty"}}
  ],
  "healthier_alternatives": [
    {{"product_type": "what to buy instead", "example_brands": "brand examples", "why_better": "reason", "score_estimate": "X/10"}}
  ],
  "shocking_facts": [
    {{"fact": "alarming but TRUE fact", "ingredient": "which ingredient"}}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage estimate",
  "processing_category": "Whole Food/Minimally Processed/Processed/Ultra-Processed",
  "recommendation": "actionable advice",
  "ingredients_estimated": true,
  "confidence": "high/medium/low"
}}"""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food product identification expert. You know thousands of commercial food products and their barcodes. Identify the product and provide health analysis. Be honest about your confidence level. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        result["analysis_note"] = "Product identified by AI - not found in food databases"
        return result
    except Exception as e:
        logger.error(f"AI barcode identification error: {e}")
        return None

async def analyze_product_by_name(client, product_name: str) -> dict:
    """When no ingredients list is available, use AI knowledge to analyze the product by name"""
    try:
        prompt = f"""You are a food science expert. The product "{product_name}" was scanned but no ingredient list was found in the database.

Based on your knowledge of this product (or similar products with this name), provide your best analysis. If you recognize the product, analyze its typical ingredients. If not, indicate that clearly.

CRITICAL: Do NOT default to 5/10. Analyze the product honestly:
- Alcohol products: 1-3/10
- Sugary drinks, crisps, sweets: 1-4/10
- Processed ready meals: 3-5/10
- Mixed items with some wholesome ingredients: 5-7/10
- Whole/natural foods: 7-10/10

Respond with JSON only:
{{
  "harmful_ingredients": [
    {{"name": "ingredient", "health_impact": "explanation", "severity": "high/medium/low", "processing_level": "NOVA level", "research_summary": "citation", "study_link": "pubmed link"}}
  ],
  "beneficial_ingredients": [
    {{"name": "ingredient", "health_benefit": "explanation", "benefit_type": "type", "key_nutrients": "list", "processing_level": "NOVA level", "research_summary": "citation", "study_link": "link"}}
  ],
  "carcinogens_found": [
    {{"name": "chemical", "iarc_group": "Group classification", "cancer_types": "linked cancers", "explanation": "how it causes harm", "source": "reference"}}
  ],
  "chemical_breakdown": [
    {{"name": "E-number or chemical", "common_name": "actual name", "purpose": "why used", "health_concern": "risk summary", "banned_in": "countries or empty"}}
  ],
  "healthier_alternatives": [
    {{"product_type": "what to buy instead", "example_brands": "brand examples", "why_better": "reason", "score_estimate": "X/10"}}
  ],
  "shocking_facts": [
    {{"fact": "A single alarming but TRUE fact about an ingredient in this product. Focus on contradictions, bans, industrial uses, or comparisons that shock consumers. E.g. 'Banned in cosmetics but still in your food', 'Same cancer classification as tobacco', 'Used in industrial paint removal'.", "ingredient": "which ingredient"}}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage estimate",
  "processing_category": "Whole Food/Minimally Processed/Processed/Ultra-Processed",
  "recommendation": "actionable advice",
  "ingredients_estimated": true,
  "confidence": "high/medium/low"
}}"""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food science expert with encyclopedic knowledge of commercial food products worldwide. When you recognize a product, provide detailed analysis based on its typical formulation. Be honest about uncertainty. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        result["analysis_note"] = "Based on AI knowledge - no ingredient list was available from the database"
        return result
    except Exception as e:
        logger.error(f"AI product-by-name analysis error: {e}")
        return {
            "harmful_ingredients": [],
            "beneficial_ingredients": [],
            "overall_score": 0,
            "upf_score": "Unknown",
            "processing_category": "Unknown",
            "recommendation": f"Could not analyze '{product_name}'. No ingredient data available. Check the packaging directly.",
            "analysis_note": "Analysis failed - no ingredient data available"
        }

async def analyze_ingredients_with_ai(product_name: str, ingredients: str) -> dict:
    """Analyze ingredients using OpenAI GPT-4o with focus on ultra-processed foods (UPFs)"""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # If no ingredients provided, use product-name-based analysis
        if not ingredients or ingredients.strip() == "":
            return await analyze_product_by_name(client, product_name)
        
        prompt = f"""You are a food science expert specializing in ultra-processed foods (UPFs), carcinogens, and nutritional health risks.

Analyze this product: {product_name}

Ingredients: {ingredients}

MANDATORY SCORE CAPS (override all other scoring):
- ALCOHOL (beer, wine, spirits, cider, cocktails, alcopops): ALWAYS 1-3/10. Group 1 carcinogen. Flag liver damage, cancer risk, addiction, empty calories.
- PROCESSED MEAT (bacon, sausages, ham, salami, hot dogs, pepperoni, chorizo, deli meat, corned beef, pate, meat pies with nitrites): ALWAYS 1-4/10. Group 1 carcinogen — same classification as tobacco and asbestos. Flag colorectal cancer, stomach cancer.
- RED MEAT (beef, lamb, pork — unprocessed): Max 5/10. Group 2A probable carcinogen. Flag colorectal cancer risk.
- HIGH SUGAR products (soft drinks, energy drinks, sweets, candy): ALWAYS 1-4/10.
- HIGH CAFFEINE (energy drinks with >150mg caffeine): Flag cardiac risk.
- Any product with 3+ carcinogens from the list below: Max 3/10.

CARCINOGENS & CHEMICALS TO FLAG:
Group 1 (CONFIRMED carcinogens — same certainty as tobacco):
- Alcohol/ethanol
- Processed meat (via nitrites/nitrates forming nitrosamines)
- Aflatoxins (found in improperly stored grains/nuts)

Group 2A (PROBABLE carcinogens):
- Acrylamide (formed in fried/baked starchy foods — crisps, chips, toast, biscuits)
- Red meat (beef, lamb, pork)
- Glyphosate residues (pesticide traces in non-organic grains)
- Glycidyl esters / glycidol (formed when palm oil is refined at high temperatures)
- Very hot beverages (>65C)

Group 2B (POSSIBLE carcinogens):
- BHA / butylated hydroxyanisole (E320) — preservative in cereals, snacks
- Titanium dioxide (E171) — whitening agent, BANNED in EU since 2022
- Aspartame (E951) — artificial sweetener
- 4-MEI / 4-methylimidazole (in caramel coloring E150d) — found in cola, soy sauce, dark beers
- Carbon black (E153)
- Lead (trace contamination)
- Styrene (from polystyrene packaging leaching)
- Red 3 / Erythrosine (E127) — banned in cosmetics, still in food
- Allura Red / Red 40 (E129)
- Sunset Yellow / Yellow 6 (E110)

Endocrine disruptors:
- BPA / Bisphenol A (from can linings, plastic packaging)
- Phthalates (from plastic food packaging)
- PFAS / forever chemicals (from microwave popcorn bags, fast food wrappers)

Other dangerous chemicals:
- Sodium nitrite (E250) — forms nitrosamines, colorectal cancer
- Potassium bromate (E924) — flour treatment, BANNED in EU/UK/Canada/Brazil
- Propylparaben (E217) — preservative, endocrine disruptor
- TBHQ (E319) — preservative linked to tumors in animal studies
- BHT / butylated hydroxytoluene (E321) — preservative
- Sodium benzoate (E211) — when combined with vitamin C/citric acid forms BENZENE (known carcinogen)
- Phosphoric acid (E338) — in cola, erodes bones and teeth
- Brominated vegetable oil / BVO — BANNED in EU, still in some US drinks
- Tartrazine / Yellow 5 (E102) — linked to hyperactivity, banned for children in EU
- Carrageenan (E407) — intestinal inflammation

HARMFUL UPF ingredients: seed oils (sunflower, rapeseed, soybean), emulsifiers (E471/E472/polysorbate 80), artificial sweeteners (sucralose, acesulfame K), preservatives, artificial colors, modified starches, hydrogenated/partially hydrogenated oils, added sugars, high fructose corn syrup, MSG (E621), maltodextrin, palm oil, dextrose, invert sugar syrup.

BENEFICIAL: proteins, vitamins, minerals, fiber, healthy fats (olive oil, avocado, nuts), omega-3, probiotics, whole grains, antioxidants, polyphenols, iron, calcium, zinc.

Respond with JSON only:
{{
  "harmful_ingredients": [
    {{"name": "ingredient", "health_impact": "2-3 sentences explaining what this does to the body", "severity": "high/medium/low", "processing_level": "NOVA 4", "research_summary": "study citation", "study_link": "pubmed link"}}
  ],
  "beneficial_ingredients": [
    {{"name": "ingredient", "health_benefit": "2-3 sentences", "benefit_type": "protein/vitamin/fiber", "key_nutrients": "list", "processing_level": "NOVA 1", "research_summary": "citation", "study_link": "link"}}
  ],
  "carcinogens_found": [
    {{"name": "chemical/ingredient name", "iarc_group": "Group 1/2A/2B/Endocrine Disruptor", "cancer_types": "types of cancer linked", "explanation": "1-2 sentences on how it causes harm", "source": "WHO/IARC/EFSA reference"}}
  ],
  "chemical_breakdown": [
    {{"name": "E-number or chemical name", "common_name": "what it actually is", "purpose": "why its in the product", "health_concern": "1 sentence risk summary", "banned_in": "list countries where banned, or empty"}}
  ],
  "healthier_alternatives": [
    {{"product_type": "what to look for instead", "example_brands": "2-3 specific brand examples if possible", "why_better": "1 sentence explaining why this is healthier", "score_estimate": "estimated score out of 10"}}
  ],
  "shocking_facts": [
    {{"fact": "A single alarming but TRUE fact about an ingredient in this product. Focus on contradictions, bans, industrial uses, or comparisons that would shock a consumer. Examples: 'This dye is banned in cosmetics but still allowed in your food', 'This preservative shares a cancer classification with tobacco', 'This ingredient is used in industrial paint removal', 'Banned in 30+ countries but legal in the US/UK'. Make each fact specific to THIS product's actual ingredients.", "ingredient": "which ingredient this fact is about"}}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage of ingredients that are ultra-processed",
  "processing_category": "Whole Food/Minimally Processed/Processed/Ultra-Processed",
  "recommendation": "actionable advice on whether to consume and what to switch to"
}}

STRICT SCORING: 8-10 whole/minimally processed foods only. 5-7 mixed. 1-4 ultra-processed. Alcohol ALWAYS 1-3. Processed meat ALWAYS 1-4. Products with ANY Group 1 carcinogen MUST score no higher than 4."""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food science expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"AI Analysis error: {e}")
        return {
            "harmful_ingredients": [],
            "beneficial_ingredients": [],
            "overall_score": 5,
            "upf_score": "0%",
            "processing_category": "Unknown",
            "recommendation": "Unable to analyze ingredients at this time."
        }

# Routes
@app.get("/api/download/icon")
async def download_icon():
    icon_path = "/app/frontend/assets/images/icon.png"
    return FileResponse(icon_path, media_type="image/png", filename="you-are-what-you-eat-icon.png")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/auth/register")
async def register(user: UserRegister):
    # Check if user exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user.password)
    user_doc = {
        "email": user.email,
        "name": user.name,
        "password": hashed_password,
        "subscription_tier": "free",
        "total_scans": 0,
        "created_at": datetime.utcnow()
    }
    result = await users_collection.insert_one(user_doc)
    
    # Create token
    token = create_access_token({"sub": str(result.inserted_id)})
    
    return {
        "token": token,
        "user": {
            "id": str(result.inserted_id),
            "email": user.email,
            "name": user.name,
            "subscription_tier": "free"
        }
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    # Rate limiting
    if not check_rate_limit(f"login:{user.email}"):
        raise HTTPException(status_code=429, detail="Too many login attempts. Please wait 5 minutes.")
    
    # Find user
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_access_token({"sub": str(db_user["_id"])})
    
    return {
        "token": token,
        "user": {
            "id": str(db_user["_id"]),
            "email": db_user["email"],
            "name": db_user["name"],
            "subscription_tier": db_user.get("subscription_tier", "free")
        }
    }

# Password Reset - Request Code
@app.post("/api/auth/forgot-password")
async def forgot_password(req: PasswordResetRequest):
    # Rate limiting
    if not check_rate_limit(f"reset:{req.email}"):
        raise HTTPException(status_code=429, detail="Too many reset attempts. Please wait 5 minutes.")
    
    user = await users_collection.find_one({"email": req.email})
    if not user:
        # Don't reveal if email exists - always return success
        return {"message": "If an account exists with that email, a reset code has been sent."}
    
    # Generate 6-digit code
    code = str(random.randint(100000, 999999))
    expires = datetime.utcnow() + timedelta(minutes=15)
    
    # Store reset code in user document
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"reset_code": code, "reset_code_expires": expires}}
    )
    
    # Try to send email
    email_sent = await send_reset_email(req.email, code)
    
    if email_sent:
        return {"message": "A reset code has been sent to your email."}
    else:
        # Fallback: return code in response if email not configured
        return {
            "message": "Reset code generated.",
            "reset_code": code
        }

# Password Reset - Confirm with Code
@app.post("/api/auth/reset-password")
async def reset_password(req: PasswordResetConfirm):
    user = await users_collection.find_one({"email": req.email})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid reset request")
    
    # Check code
    stored_code = user.get("reset_code")
    code_expires = user.get("reset_code_expires")
    
    if not stored_code or not code_expires:
        raise HTTPException(status_code=400, detail="No reset code found. Please request a new one.")
    
    if datetime.utcnow() > code_expires:
        raise HTTPException(status_code=400, detail="Reset code has expired. Please request a new one.")
    
    if req.code != stored_code:
        raise HTTPException(status_code=400, detail="Invalid reset code")
    
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    # Update password and clear reset code
    hashed = get_password_hash(req.new_password)
    await users_collection.update_one(
        {"_id": user["_id"]},
        {"$set": {"password": hashed}, "$unset": {"reset_code": "", "reset_code_expires": ""}}
    )
    
    # Return token so user is auto-logged in
    token = create_access_token({"sub": str(user["_id"])})
    return {
        "message": "Password reset successfully",
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "subscription_tier": user.get("subscription_tier", "free")
        }
    }

# Delete Account
@app.delete("/api/auth/delete-account")
async def delete_account(current_user = Depends(get_current_user)):
    user_id = current_user["_id"]
    # Delete user data
    await users_collection.delete_one({"_id": user_id})
    # Delete related data
    await favorites_collection.delete_many({"user_id": str(user_id)})
    await scan_analytics_collection.delete_many({"user_id": str(user_id)})
    return {"message": "Account deleted successfully"}

@app.get("/api/auth/me")
async def get_me(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "name": current_user["name"],
        "subscription_tier": current_user.get("subscription_tier", "free"),
        "total_scans": current_user.get("total_scans", 0)
    }

@app.post("/api/scan")
async def scan_product(scan_req: ScanRequest, current_user = Depends(get_current_user)):
    """
    Improved barcode scanning with:
    - Local caching for instant results
    - Retry logic with exponential backoff
    - Multiple API sources (Open Food Facts + UPC Item DB backup)
    - Analytics tracking
    """
    start_time = time.time()
    barcode = scan_req.barcode.strip()
    
    # Check subscription limits - 5 TOTAL lifetime scans for free users
    subscription_tier = current_user.get("subscription_tier", "free")
    total_scans = current_user.get("total_scans", 0)
    
    if subscription_tier == "free" and total_scans >= 5:
        raise HTTPException(
            status_code=403,
            detail="Free scan limit reached (5 scans). Upgrade to premium for unlimited scans."
        )
    
    # STEP 1: Check cache first for instant results
    cached = await get_cached_product(barcode)
    if cached and cached.get("analysis"):
        response_time = time.time() - start_time
        await log_scan_analytics(barcode, True, "cache", response_time)
        # Increment scan count
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"total_scans": 1}}
        )
        result = {
            "product_name": cached.get("product_name", "Unknown"),
            "brands": cached.get("brands", ""),
            "ingredients_text": cached.get("ingredients_text", ""),
            "image_url": cached.get("image_url", ""),
            "analysis": cached["analysis"],
            "source": "cache",
            "response_time_ms": int(response_time * 1000)
        }
        return {k: v for k, v in result.items() if k != "_id"}
    
    # STEP 2: Parallel API calls with smart routing based on barcode prefix
    # UK/EU barcodes start with 50/40-44, US barcodes start with 0
    barcode_prefix = barcode[:2] if len(barcode) >= 2 else ""
    
    # Run all API sources in parallel using ThreadPoolExecutor
    
    product_data = None
    source = "none"
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        # Smart ordering: prioritize based on barcode region
        if barcode_prefix in ['50', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '87', '90', '93', '94']:
            # EU/UK barcode — OFF + OFF UK first, plus all others in parallel
            futures = {
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_off_uk, barcode): "openfoodfacts_uk",
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
                executor.submit(fetch_from_brocade, barcode): "brocade",
            }
        elif barcode_prefix in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']:
            # US/Canada barcode — USDA first, then others in parallel
            futures = {
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
                executor.submit(fetch_from_brocade, barcode): "brocade",
            }
        else:
            # Other regions — try all simultaneously
            futures = {
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_off_uk, barcode): "openfoodfacts_uk",
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
                executor.submit(fetch_from_brocade, barcode): "brocade",
            }
        
        # Collect ALL results, prioritize ones WITH ingredients
        results_with_ingredients = []
        results_without_ingredients = []
        
        try:
            for future in as_completed(futures, timeout=15):
                src = futures[future]
                try:
                    result = future.result()
                    if result:
                        if result.get("ingredients_text"):
                            results_with_ingredients.append((result, src))
                            logger.info(f"{src}: Found product WITH ingredients")
                        else:
                            results_without_ingredients.append((result, src))
                            logger.info(f"{src}: Found product WITHOUT ingredients")
                except Exception as e:
                    logger.warning(f"{src} error: {e}")
        except TimeoutError:
            logger.warning(f"Some API sources timed out for {barcode}, using partial results")
        
        # Prefer results WITH ingredients
        if results_with_ingredients:
            product_data, source = results_with_ingredients[0]
            logger.info(f"Using {source} (has ingredients)")
        elif results_without_ingredients:
            product_data, source = results_without_ingredients[0]
            logger.info(f"Using {source} (no ingredients available)")
    
    # STEP 3: If parallel calls all failed, try additional fallbacks
    if not product_data:
        logger.info(f"All parallel sources failed for {barcode}, trying OFF search + FatSecret")
        # Try OFF search API (different endpoint, sometimes finds products the direct lookup misses)
        product_data = fetch_from_off_search(barcode)
        if product_data:
            source = "openfoodfacts_search"
        else:
            product_data = fetch_from_fatsecret(barcode)
            source = "fatsecret"
    
    # STEP 4: If all sources fail, return 404
    if not product_data:
        response_time = time.time() - start_time
        await log_scan_analytics(barcode, False, "none", response_time, "Product not found in any database")
        raise HTTPException(status_code=404, detail="Product not found. Try scanning again or entering the barcode manually.")
    
    # STEP 5: Check if we have ingredients (required for analysis)
    ingredients_text = product_data.get("ingredients_text", "")
    if not ingredients_text:
        # No ingredients from database - use AI to analyze by product name
        logger.info(f"No ingredients for {barcode}, using AI product-name analysis")
        try:
            analysis = await analyze_ingredients_with_ai(
                product_data.get("product_name", "Unknown"),
                ""  # Empty ingredients triggers product-by-name analysis
            )
        except Exception as e:
            logger.error(f"AI name-based analysis failed: {e}")
            analysis = {
                "harmful_ingredients": [],
                "beneficial_ingredients": [],
                "overall_score": 0,
                "recommendation": "No ingredient information available. Check the packaging for details.",
                "analysis_note": "No ingredient data available from any source"
            }
        
        response_time = time.time() - start_time
        await log_scan_analytics(barcode, True, source, response_time, "No ingredients - AI name analysis")
        
        product_data["analysis"] = analysis
        await cache_product(barcode, product_data)
        
        # Save scan
        scan_doc = {
            "user_id": str(current_user["_id"]),
            "barcode": barcode,
            "product_name": product_data.get("product_name"),
            "brands": product_data.get("brands"),
            "ingredients_text": "",
            "image_url": product_data.get("image_url"),
            "analysis": analysis,
            "scanned_at": datetime.utcnow(),
            "source": source
        }
        await scans_collection.insert_one(scan_doc)
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"total_scans": 1}}
        )
        
        return {
            "product_name": product_data.get("product_name"),
            "brands": product_data.get("brands"),
            "ingredients_text": "",
            "image_url": product_data.get("image_url"),
            "analysis": analysis,
            "warning": "Ingredients not found in database - analysis based on AI product knowledge"
        }
    
    # STEP 6: Analyze ingredients with AI
    try:
        analysis = await analyze_ingredients_with_ai(
            product_data.get("product_name", "Unknown"),
            ingredients_text
        )
    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        analysis = {
            "harmful_ingredients": [],
            "beneficial_ingredients": [],
            "overall_score": 5,
            "recommendation": "Analysis temporarily unavailable. Please try again.",
            "nova_group": None,
            "additives": []
        }
    
    # STEP 6.5: Cache the result for future lookups
    product_data["analysis"] = analysis
    await cache_product(barcode, product_data)
    
    # STEP 7: Save to user's scan history
    scan_doc = {
        "user_id": str(current_user["_id"]),
        "barcode": barcode,
        "product_name": product_data.get("product_name"),
        "brands": product_data.get("brands"),
        "ingredients_text": ingredients_text,
        "image_url": product_data.get("image_url"),
        "analysis": analysis,
        "scanned_at": datetime.utcnow(),
        "source": source
    }
    await scans_collection.insert_one(scan_doc)
    
    # Update user's total scan count
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$inc": {"total_scans": 1}}
    )
    
    # Update daily quest progress
    user_id = str(current_user["_id"])
    gamification = await db["gamification"].find_one({"user_id": user_id})
    if gamification:
        daily_quests = gamification.get("daily_quests", {})
        xp_earned = 0
        
        # Quest 1: Scan 3 products
        if "scan_3_products" in daily_quests and not daily_quests["scan_3_products"]["completed"]:
            current_progress = daily_quests["scan_3_products"].get("progress", 0) + 1
            daily_quests["scan_3_products"]["progress"] = current_progress
            if current_progress >= 3:
                daily_quests["scan_3_products"]["completed"] = True
                xp_earned += daily_quests["scan_3_products"]["xp"]
        
        # Quest 2: Find a healthy product (8+/10)
        if "find_healthy_product" in daily_quests and not daily_quests["find_healthy_product"]["completed"]:
            overall_score = analysis.get("overall_score", 0)
            if overall_score >= 8:
                daily_quests["find_healthy_product"]["completed"] = True
                daily_quests["find_healthy_product"]["progress"] = 1
                xp_earned += daily_quests["find_healthy_product"]["xp"]
        
        # Update gamification data
        await db["gamification"].update_one(
            {"user_id": user_id},
            {
                "$set": {"daily_quests": daily_quests},
                "$inc": {"xp": xp_earned}
            }
        )
    
    # Log analytics
    response_time = time.time() - start_time
    await log_scan_analytics(barcode, True, source, response_time)
    
    return {
        "product_name": product_data.get("product_name"),
        "brands": product_data.get("brands"),
        "ingredients_text": ingredients_text,
        "image_url": product_data.get("image_url"),
        "analysis": analysis,
        "source": source,
        "response_time_ms": int(response_time * 1000)
    }

# Background AI analysis task storage
_pending_analyses = {}

@app.post("/api/scan/quick")
async def scan_product_quick(scan_req: ScanRequest, current_user = Depends(get_current_user)):
    """
    Stage 1: Quick product lookup - returns product name/image fast.
    If cached with full analysis, returns everything immediately.
    If not cached, fetches product data from ALL food DBs and starts background AI analysis.
    """
    barcode = scan_req.barcode.strip()
    logger.info(f"Quick scan request for barcode: {barcode}")
    
    # Check subscription limits
    subscription_tier = current_user.get("subscription_tier", "free")
    total_scans = current_user.get("total_scans", 0)
    if subscription_tier == "free" and total_scans >= 5:
        raise HTTPException(status_code=403, detail="Free scan limit reached (5 scans). Upgrade to premium for unlimited scans.")
    
    # Check cache first - if ANY data exists (with or without analysis), use it
    cached = await product_cache_collection.find_one({"barcode": barcode}, {"_id": 0})
    if cached and cached.get("analysis"):
        await users_collection.update_one({"_id": current_user["_id"]}, {"$inc": {"total_scans": 1}})
        logger.info(f"Cache hit (complete) for {barcode}")
        return {
            "status": "complete",
            "product_name": cached.get("product_name"),
            "brands": cached.get("brands", ""),
            "ingredients_text": cached.get("ingredients_text", ""),
            "image_url": cached.get("image_url", ""),
            "analysis": cached.get("analysis"),
            "source": cached.get("source", "cache")
        }
    
    # If product data is cached but analysis is still pending, return analyzing
    if cached and cached.get("product_name") and not cached.get("analysis_error"):
        logger.info(f"Cache hit (pending analysis) for {barcode}")
        return {
            "status": "analyzing",
            "product_name": cached.get("product_name"),
            "brands": cached.get("brands", ""),
            "ingredients_text": cached.get("ingredients_text", ""),
            "image_url": cached.get("image_url", ""),
            "analysis": None,
            "source": cached.get("source", "cache")
        }
    
    # Check if a previous analysis failed - clear it so we retry
    if cached and cached.get("analysis_error"):
        await product_cache_collection.delete_one({"barcode": barcode})
    
    # Not in cache - do a quick product lookup from ALL sources (no AI yet)
    product_data = None
    source = "none"
    
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
            executor.submit(fetch_from_off_uk, barcode): "openfoodfacts_uk",
            executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
            executor.submit(fetch_from_usda, barcode): "usda",
            executor.submit(fetch_from_brocade, barcode): "brocade",
            executor.submit(fetch_from_off_search, barcode): "openfoodfacts_search",
            executor.submit(fetch_from_fatsecret, barcode): "fatsecret",
        }
        
        results_with_ingredients = []
        results_without_ingredients = []
        
        try:
            for future in as_completed(futures, timeout=12):
                src = futures[future]
                try:
                    result = future.result()
                    if result:
                        if result.get("ingredients_text"):
                            results_with_ingredients.append((result, src))
                            logger.info(f"Quick scan {src}: found WITH ingredients")
                        else:
                            results_without_ingredients.append((result, src))
                            logger.info(f"Quick scan {src}: found WITHOUT ingredients")
                except Exception as e:
                    logger.warning(f"Quick scan {src} error: {e}")
        except TimeoutError:
            logger.warning(f"Quick scan: some sources timed out for {barcode}")
        
        if results_with_ingredients:
            product_data, source = results_with_ingredients[0]
        elif results_without_ingredients:
            product_data, source = results_without_ingredients[0]
    
    if not product_data:
        # NO database has this barcode — use AI as last resort to identify and analyze
        logger.info(f"Quick scan: no DB match for {barcode}, falling back to AI identification")
        product_data = {"product_name": f"Product (barcode {barcode})", "brands": "", "ingredients_text": "", "image_url": ""}
        source = "ai_identification"
    
    # Increment scan count
    await users_collection.update_one({"_id": current_user["_id"]}, {"$inc": {"total_scans": 1}})
    
    # Start background AI analysis
    ingredients_text = product_data.get("ingredients_text", "")
    product_name = product_data.get("product_name", "Unknown")
    
    # IMMEDIATELY cache the product data (without analysis) to prevent duplicate lookups
    await product_cache_collection.update_one(
        {"barcode": barcode},
        {"$set": {
            "barcode": barcode,
            "product_name": product_name,
            "brands": product_data.get("brands", ""),
            "ingredients_text": ingredients_text,
            "image_url": product_data.get("image_url", ""),
            "cached_at": datetime.utcnow(),
            "source": source
        }},
        upsert=True
    )
    
    # Save scan to history (one entry per barcode per user per minute to avoid duplicates)
    recent_scan = await scans_collection.find_one({
        "user_id": str(current_user["_id"]),
        "barcode": barcode,
        "scanned_at": {"$gte": datetime.utcnow() - timedelta(minutes=1)}
    })
    
    if recent_scan:
        scan_id = str(recent_scan["_id"])
    else:
        scan_doc = {
            "user_id": str(current_user["_id"]),
            "barcode": barcode,
            "product_name": product_name,
            "brands": product_data.get("brands", ""),
            "ingredients_text": ingredients_text,
            "image_url": product_data.get("image_url", ""),
            "analysis": None,
            "scanned_at": datetime.utcnow(),
            "source": source
        }
        scan_insert = await scans_collection.insert_one(scan_doc)
        scan_id = str(scan_insert.inserted_id)
    
    async def run_background_analysis():
        try:
            logger.info(f"Background AI analysis starting for {barcode} ({product_name})")
            
            if source == "ai_identification":
                # No database had this product — ask AI to identify it from the barcode
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                ai_result = await identify_product_by_barcode(client, barcode)
                if ai_result:
                    # AI identified the product — update the product info
                    identified_name = ai_result.pop("identified_product", product_name)
                    identified_brand = ai_result.pop("identified_brand", "")
                    typical_ingredients = ai_result.pop("typical_ingredients", "")
                    analysis = ai_result
                    
                    cache_data = {
                        "barcode": barcode,
                        "product_name": identified_name,
                        "brands": identified_brand,
                        "ingredients_text": typical_ingredients,
                        "image_url": "",
                        "analysis": analysis,
                        "cached_at": datetime.utcnow(),
                        "source": "ai_identification"
                    }
                else:
                    # AI also couldn't identify — store a generic result
                    analysis = {
                        "harmful_ingredients": [],
                        "beneficial_ingredients": [],
                        "overall_score": 0,
                        "upf_score": "Unknown",
                        "processing_category": "Unknown",
                        "recommendation": "This product could not be identified. Please check the barcode or enter the product name manually.",
                        "analysis_note": "Product not recognized by any source"
                    }
                    cache_data = {
                        "barcode": barcode,
                        "product_name": f"Unknown Product ({barcode})",
                        "brands": "",
                        "ingredients_text": "",
                        "image_url": "",
                        "analysis": analysis,
                        "cached_at": datetime.utcnow(),
                        "source": "ai_identification"
                    }
            else:
                # Normal path — product found in database, just need AI analysis
                analysis = await analyze_ingredients_with_ai(product_name, ingredients_text)
                cache_data = {
                    "barcode": barcode,
                    "product_name": product_name,
                    "brands": product_data.get("brands", ""),
                    "ingredients_text": ingredients_text,
                    "image_url": product_data.get("image_url", ""),
                    "analysis": analysis,
                    "cached_at": datetime.utcnow(),
                    "source": source
                }
            
            await product_cache_collection.update_one(
                {"barcode": barcode},
                {"$set": cache_data},
                upsert=True
            )
            
            # Update the scan history entry with the analysis
            await scans_collection.update_one(
                {"_id": ObjectId(scan_id)},
                {"$set": {
                    "analysis": analysis,
                    "product_name": cache_data.get("product_name", product_name)
                }}
            )
            
            logger.info(f"Background analysis complete for {barcode}: score {analysis.get('overall_score')}")
        except Exception as e:
            logger.error(f"Background analysis FAILED for {barcode}: {e}")
            # Mark the error in cache so the status endpoint can report it
            await product_cache_collection.update_one(
                {"barcode": barcode},
                {"$set": {
                    "barcode": barcode,
                    "product_name": product_name,
                    "brands": product_data.get("brands", ""),
                    "ingredients_text": ingredients_text,
                    "image_url": product_data.get("image_url", ""),
                    "analysis_error": str(e),
                    "cached_at": datetime.utcnow(),
                    "source": source
                }},
                upsert=True
            )
    
    # Fire and forget the background analysis
    asyncio.create_task(run_background_analysis())
    
    logger.info(f"Quick scan returning 'analyzing' for {barcode} ({product_name}) from {source}")
    return {
        "status": "analyzing",
        "product_name": product_name,
        "brands": product_data.get("brands", ""),
        "ingredients_text": ingredients_text,
        "image_url": product_data.get("image_url", ""),
        "analysis": None,
        "source": source
    }

@app.get("/api/scan/status/{barcode}")
async def scan_status(barcode: str, current_user = Depends(get_current_user)):
    """
    Stage 2: Poll for analysis results. Returns full data when AI analysis is complete.
    Also reports errors so the frontend doesn't poll forever.
    """
    cached = await product_cache_collection.find_one({"barcode": barcode}, {"_id": 0})
    if cached:
        if cached.get("analysis"):
            return {
                "status": "complete",
                "product_name": cached.get("product_name"),
                "brands": cached.get("brands", ""),
                "ingredients_text": cached.get("ingredients_text", ""),
                "image_url": cached.get("image_url", ""),
                "analysis": cached.get("analysis"),
                "source": cached.get("source", "cache")
            }
        if cached.get("analysis_error"):
            return {
                "status": "error",
                "error": "AI analysis failed. Please try scanning again.",
                "product_name": cached.get("product_name"),
                "brands": cached.get("brands", ""),
            }
    
    return {"status": "analyzing"}

@app.get("/api/analytics/scans")
async def get_scan_analytics():
    """Get scan analytics summary for monitoring"""
    try:
        # Get stats from the last 24 hours
        since = datetime.utcnow() - timedelta(hours=24)
        
        pipeline = [
            {"$match": {"timestamp": {"$gte": since}}},
            {"$group": {
                "_id": None,
                "total_scans": {"$sum": 1},
                "successful_scans": {"$sum": {"$cond": ["$success", 1, 0]}},
                "cache_hits": {"$sum": {"$cond": [{"$eq": ["$source", "cache"]}, 1, 0]}},
                "avg_response_time_ms": {"$avg": "$response_time_ms"},
                "openfoodfacts_count": {"$sum": {"$cond": [{"$eq": ["$source", "openfoodfacts"]}, 1, 0]}},
                "upcitemdb_count": {"$sum": {"$cond": [{"$eq": ["$source", "upcitemdb"]}, 1, 0]}}
            }}
        ]
        
        result = await scan_analytics_collection.aggregate(pipeline).to_list(1)
        
        if result:
            stats = result[0]
            total = stats.get("total_scans", 0)
            successful = stats.get("successful_scans", 0)
            cache_hits = stats.get("cache_hits", 0)
            
            return {
                "period": "last_24_hours",
                "total_scans": total,
                "successful_scans": successful,
                "success_rate": round((successful / total * 100), 1) if total > 0 else 0,
                "cache_hit_rate": round((cache_hits / total * 100), 1) if total > 0 else 0,
                "avg_response_time_ms": round(stats.get("avg_response_time_ms", 0), 0),
                "sources": {
                    "cache": cache_hits,
                    "openfoodfacts": stats.get("openfoodfacts_count", 0),
                    "upcitemdb": stats.get("upcitemdb_count", 0)
                }
            }
        
        return {
            "period": "last_24_hours",
            "total_scans": 0,
            "message": "No scan data available"
        }
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return {"error": str(e)}

@app.delete("/api/admin/cache")
async def clear_product_cache(current_user = Depends(get_current_user)):
    """Clear the product cache - use after updating AI prompts"""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    result = await product_cache_collection.delete_many({})
    return {"message": f"Cleared {result.deleted_count} cached products"}

@app.delete("/api/cache/clear")
async def clear_cache_with_key(key: str = ""):
    """Clear product cache with secret key (no auth required)"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    result = await product_cache_collection.delete_many({})
    return {"message": f"Cleared {result.deleted_count} cached products"}

@app.post("/api/admin/cache_insert")
async def admin_cache_insert(request: Request, key: str = ""):
    """Admin endpoint to directly insert product data into cache for pre-warming"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    body = await request.json()
    barcode = body.get("barcode")
    if not barcode:
        raise HTTPException(status_code=400, detail="barcode required")
    product_data = {
        "barcode": barcode,
        "product_name": body.get("product_name", "Unknown"),
        "brands": body.get("brands", ""),
        "ingredients_text": body.get("ingredients_text", ""),
        "image_url": body.get("image_url", ""),
        "analysis": body.get("analysis"),
        "cached_at": datetime.utcnow(),
        "source": body.get("source", "admin_prewarm")
    }
    await product_cache_collection.update_one(
        {"barcode": barcode},
        {"$set": product_data},
        upsert=True
    )
    return {"status": "ok", "barcode": barcode}

@app.get("/api/admin/cache_count")
async def admin_cache_count(key: str = ""):
    """Get number of cached products"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    count = await product_cache_collection.count_documents({})
    return {"cached_products": count}

@app.get("/api/admin/user_stats")
async def admin_user_stats(key: str = ""):
    """Get user and subscriber statistics"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    total_users = await users_collection.count_documents({})
    premium_users = await users_collection.count_documents({"subscription_tier": "premium"})
    free_users = total_users - premium_users
    premium_list = []
    async for u in users_collection.find({"subscription_tier": "premium"}, {"_id": 0, "email": 1, "name": 1, "subscription_tier": 1, "total_scans": 1}):
        premium_list.append(u)
    return {
        "total_users": total_users,
        "premium_subscribers": premium_users,
        "free_users": free_users,
        "premium_user_details": premium_list
    }

@app.post("/api/admin/reset_password")
async def admin_reset_password(request: Request, key: str = ""):
    """Admin endpoint to reset a user's password"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    body = await request.json()
    email = body.get("email", "")
    new_password = body.get("new_password", "")
    if not email or not new_password:
        raise HTTPException(status_code=400, detail="email and new_password required")
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    hashed = get_password_hash(new_password)
    # Update all possible password field names
    update_fields = {"password": hashed, "password_hash": hashed}
    await users_collection.update_one({"email": email}, {"$set": update_fields})
    # Return field names for debugging
    user_fields = [k for k in user.keys() if k != "_id"]
    return {"status": "ok", "email": email, "fields": user_fields}

@app.post("/api/admin/prewarm")
async def admin_prewarm(request: Request, key: str = ""):
    """Pre-warm cache by analyzing a product by name using AI (no barcode lookup needed)"""
    if key != "yawye2024clear":
        raise HTTPException(status_code=403, detail="Invalid key")
    body = await request.json()
    barcode = body.get("barcode", "")
    product_name = body.get("product_name", "")
    if not barcode or not product_name:
        raise HTTPException(status_code=400, detail="barcode and product_name required")
    # Check if already cached
    existing = await product_cache_collection.find_one({"barcode": barcode})
    if existing and existing.get("analysis"):
        return {"status": "already_cached", "barcode": barcode}
    # Use AI to analyze by product name
    try:
        analysis = await analyze_ingredients_with_ai(product_name, "")
        product_data = {
            "barcode": barcode,
            "product_name": product_name,
            "brands": "",
            "ingredients_text": "",
            "image_url": "",
            "analysis": analysis,
            "cached_at": datetime.utcnow(),
            "source": "ai_prewarm"
        }
        await product_cache_collection.update_one(
            {"barcode": barcode},
            {"$set": product_data},
            upsert=True
        )
        return {"status": "cached", "barcode": barcode, "score": analysis.get("overall_score")}
    except Exception as e:
        logger.error(f"Prewarm error for {product_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scans/history")
async def get_scan_history(current_user = Depends(get_current_user)):
    scans = await scans_collection.find(
        {"user_id": str(current_user["_id"])}
    ).sort("scanned_at", -1).limit(50).to_list(50)
    
    for scan in scans:
        scan["_id"] = str(scan["_id"])
    
    return {"scans": scans}

@app.post("/api/favorites/add")
async def add_favorite(fav_req: FavoriteRequest, current_user = Depends(get_current_user)):
    # Check if already favorited
    existing = await favorites_collection.find_one({
        "user_id": str(current_user["_id"]),
        "product_id": fav_req.product_id
    })
    
    if existing:
        return {"message": "Already in favorites"}
    
    # Get scan details
    scan = await scans_collection.find_one({"barcode": fav_req.product_id})
    if not scan:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Add to favorites
    fav_doc = {
        "user_id": str(current_user["_id"]),
        "product_id": fav_req.product_id,
        "product_name": scan.get("product_name"),
        "image_url": scan.get("image_url"),
        "added_at": datetime.utcnow()
    }
    await favorites_collection.insert_one(fav_doc)
    
    return {"message": "Added to favorites"}

@app.delete("/api/favorites/remove/{product_id}")
async def remove_favorite(product_id: str, current_user = Depends(get_current_user)):
    result = await favorites_collection.delete_one({
        "user_id": str(current_user["_id"]),
        "product_id": product_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Favorite not found")
    
    return {"message": "Removed from favorites"}

@app.get("/api/favorites")
async def get_favorites(current_user = Depends(get_current_user)):
    favorites = await favorites_collection.find(
        {"user_id": str(current_user["_id"])}
    ).sort("added_at", -1).to_list(100)
    
    for fav in favorites:
        fav["_id"] = str(fav["_id"])
    
    return {"favorites": favorites}

@app.post("/api/subscription/upgrade")
async def upgrade_subscription(current_user = Depends(get_current_user)):
    """Upgrade user to premium after successful RevenueCat purchase"""
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"subscription_tier": "premium"}}
    )
    return {"message": "Upgraded to premium", "subscription_tier": "premium"}

@app.post("/api/webhooks/revenuecat")
async def revenuecat_webhook(request: Request):
    """RevenueCat webhook to sync subscription status automatically"""
    try:
        # Verify authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header != "Jmaster1986!":
            print(f"RevenueCat webhook: Invalid auth header")
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        body = await request.json()
        event = body.get("event", {})
        event_type = event.get("type", "")
        app_user_id = event.get("app_user_id", "")
        
        print(f"RevenueCat webhook: {event_type} for user {app_user_id}")
        
        # Handle subscription events
        if event_type in ["INITIAL_PURCHASE", "RENEWAL", "PRODUCT_CHANGE", "UNCANCELLATION"]:
            # User subscribed or renewed
            if ObjectId.is_valid(app_user_id):
                await users_collection.update_one(
                    {"_id": ObjectId(app_user_id)},
                    {"$set": {"subscription_tier": "premium"}}
                )
                print(f"RevenueCat: Upgraded {app_user_id} to premium")
            
        elif event_type in ["CANCELLATION", "EXPIRATION", "BILLING_ISSUE"]:
            # Subscription ended
            if ObjectId.is_valid(app_user_id):
                await users_collection.update_one(
                    {"_id": ObjectId(app_user_id)},
                    {"$set": {"subscription_tier": "free"}}
                )
                print(f"RevenueCat: Downgraded {app_user_id} to free")
            
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"RevenueCat webhook error: {e}")
        return {"status": "error", "message": str(e)}

# Gamification Endpoints

@app.get("/api/gamification/stats")
async def get_gamification_stats(current_user = Depends(get_current_user)):
    """Get user's gamification stats (streaks, quests, badges, level)"""
    user_id = str(current_user["_id"])
    
    # Get or create gamification data
    gamification = await db["gamification"].find_one({"user_id": user_id})
    
    if not gamification:
        # Initialize gamification data
        gamification = {
            "user_id": user_id,
            "current_streak": 0,
            "longest_streak": 0,
            "last_scan_date": None,
            "total_scans": 0,
            "level": 1,
            "xp": 0,
            "badges": [],
            "daily_quests": {
                "scan_3_products": {"completed": False, "progress": 0, "xp": 10},
                "find_healthy_product": {"completed": False, "progress": 0, "xp": 25},
                "use_assistant": {"completed": False, "progress": 0, "xp": 20},
            },
            "last_quest_reset": datetime.utcnow(),
            "quiz_streak": 0,
            "quiz_correct_answers": 0,
            "quiz_total_answers": 0,
        }
        await db["gamification"].insert_one(gamification)
    
    # Reset daily quests if needed
    last_reset = gamification.get("last_quest_reset", datetime.utcnow())
    if datetime.utcnow() - last_reset > timedelta(days=1):
        gamification["daily_quests"] = {
            "scan_3_products": {"completed": False, "progress": 0, "xp": 10},
            "find_healthy_product": {"completed": False, "progress": 0, "xp": 25},
            "use_assistant": {"completed": False, "progress": 0, "xp": 20},
        }
        gamification["last_quest_reset"] = datetime.utcnow()
        await db["gamification"].update_one(
            {"user_id": user_id},
            {"$set": {"daily_quests": gamification["daily_quests"], "last_quest_reset": datetime.utcnow()}}
        )
    
    # Calculate level from XP
    xp = gamification.get("xp", 0)
    level = 1 + (xp // 100)  # Level up every 100 XP
    
    gamification["_id"] = str(gamification["_id"])
    gamification["level"] = level
    
    return gamification

@app.post("/api/gamification/update-streak")
async def update_streak(current_user = Depends(get_current_user)):
    """Update user's scan streak"""
    user_id = str(current_user["_id"])
    
    gamification = await db["gamification"].find_one({"user_id": user_id})
    
    if not gamification:
        return await get_gamification_stats(current_user)
    
    now = datetime.utcnow()
    last_scan = gamification.get("last_scan_date")
    current_streak = gamification.get("current_streak", 0)
    
    if last_scan:
        last_scan_date = last_scan.date() if isinstance(last_scan, datetime) else datetime.fromisoformat(str(last_scan)).date()
        today = now.date()
        
        if last_scan_date == today:
            # Already scanned today
            pass
        elif last_scan_date == today - timedelta(days=1):
            # Consecutive day
            current_streak += 1
        else:
            # Streak broken
            current_streak = 1
    else:
        current_streak = 1
    
    longest_streak = max(gamification.get("longest_streak", 0), current_streak)
    total_scans = gamification.get("total_scans", 0) + 1
    
    # Update database
    await db["gamification"].update_one(
        {"user_id": user_id},
        {
            "$set": {
                "current_streak": current_streak,
                "longest_streak": longest_streak,
                "last_scan_date": now,
                "total_scans": total_scans,
            }
        }
    )
    
    # Check for streak badges
    new_badges = []
    badges = gamification.get("badges", [])
    
    if current_streak >= 3 and "streak_3" not in badges:
        new_badges.append({"id": "streak_3", "name": "3-Day Warrior", "icon": "🔥"})
    if current_streak >= 7 and "streak_7" not in badges:
        new_badges.append({"id": "streak_7", "name": "Week Champion", "icon": "⭐"})
    if current_streak >= 30 and "streak_30" not in badges:
        new_badges.append({"id": "streak_30", "name": "Monthly Master", "icon": "💎"})
    
    if new_badges:
        all_badges = badges + [b["id"] for b in new_badges]
        await db["gamification"].update_one(
            {"user_id": user_id},
            {"$set": {"badges": all_badges}}
        )
    
    return {
        "current_streak": current_streak,
        "new_badges": new_badges,
        "total_scans": total_scans,
    }

@app.post("/api/gamification/complete-quest")
async def complete_quest(quest_id: str, current_user = Depends(get_current_user)):
    """Mark a daily quest as completed"""
    user_id = str(current_user["_id"])
    
    gamification = await db["gamification"].find_one({"user_id": user_id})
    if not gamification:
        return {"error": "Gamification data not found"}
    
    daily_quests = gamification.get("daily_quests", {})
    
    if quest_id in daily_quests and not daily_quests[quest_id]["completed"]:
        daily_quests[quest_id]["completed"] = True
        daily_quests[quest_id]["progress"] = 1
        
        # Award XP
        xp_reward = daily_quests[quest_id]["xp"]
        new_xp = gamification.get("xp", 0) + xp_reward
        
        await db["gamification"].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "daily_quests": daily_quests,
                    "xp": new_xp,
                }
            }
        )
        
        return {"success": True, "xp_earned": xp_reward, "total_xp": new_xp}
    
    return {"success": False, "message": "Quest already completed or not found"}

# Quiz data
QUIZ_QUESTIONS = [
    {
        "id": "q1",
        "question": "Which additive is linked to gut inflammation?",
        "options": ["Citric Acid", "Emulsifiers (E471)", "Vitamin C", "Salt"],
        "correct": 1,
        "explanation": "Emulsifiers like E471 can disrupt the gut microbiome and increase inflammation markers."
    },
    {
        "id": "q2",
        "question": "What does NOVA 4 classification mean?",
        "options": ["Whole foods", "Processed ingredients", "Processed foods", "Ultra-processed foods"],
        "correct": 3,
        "explanation": "NOVA 4 represents ultra-processed foods with industrial formulations and additives."
    },
    {
        "id": "q3",
        "question": "Which is NOT an ultra-processed food marker?",
        "options": ["Maltodextrin", "Whole wheat flour", "Modified starches", "Artificial sweeteners"],
        "correct": 1,
        "explanation": "Whole wheat flour is minimally processed (NOVA 1-2), while the others are UPF markers."
    },
    {
        "id": "q4",
        "question": "What health risk is associated with palm oil?",
        "options": ["Increased vitamins", "Inflammation", "Better digestion", "Stronger bones"],
        "correct": 1,
        "explanation": "Palm oil consumption is linked to increased inflammation and metabolic dysfunction."
    },
    {
        "id": "q5",
        "question": "Which preservative is commonly found in soft drinks?",
        "options": ["Vitamin E", "Sodium benzoate", "Calcium", "Iron"],
        "correct": 1,
        "explanation": "Sodium benzoate is a common preservative in sodas and can form benzene under certain conditions."
    },
]

@app.get("/api/quiz/daily")
async def get_daily_quiz(current_user = Depends(get_current_user)):
    """Get today's quiz question"""
    # Rotate question based on day of year
    day_of_year = datetime.utcnow().timetuple().tm_yday
    question_index = day_of_year % len(QUIZ_QUESTIONS)
    question = QUIZ_QUESTIONS[question_index].copy()
    
    # Remove correct answer from response
    correct_answer = question.pop("correct")
    question.pop("explanation")
    
    return question

@app.post("/api/quiz/answer")
async def submit_quiz_answer(answer_req: QuizAnswerRequest, current_user = Depends(get_current_user)):
    """Submit quiz answer and get result"""
    user_id = str(current_user["_id"])
    
    # Find the question
    question = next((q for q in QUIZ_QUESTIONS if q["id"] == answer_req.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Check answer
    is_correct = int(answer_req.answer) == question["correct"]
    
    # Update gamification stats
    gamification = await db["gamification"].find_one({"user_id": user_id})
    if gamification:
        quiz_correct = gamification.get("quiz_correct_answers", 0)
        quiz_total = gamification.get("quiz_total_answers", 0)
        quiz_streak = gamification.get("quiz_streak", 0)
        
        if is_correct:
            quiz_correct += 1
            quiz_streak += 1
            xp_reward = 15
        else:
            quiz_streak = 0
            xp_reward = 5  # Participation XP
        
        quiz_total += 1
        new_xp = gamification.get("xp", 0) + xp_reward
        
        await db["gamification"].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "quiz_correct_answers": quiz_correct,
                    "quiz_total_answers": quiz_total,
                    "quiz_streak": quiz_streak,
                    "xp": new_xp,
                }
            }
        )
        
        return {
            "correct": is_correct,
            "explanation": question["explanation"],
            "xp_earned": xp_reward,
            "quiz_streak": quiz_streak,
            "accuracy": round((quiz_correct / quiz_total) * 100, 1) if quiz_total > 0 else 0,
        }
    
    return {"correct": is_correct, "explanation": question["explanation"]}

# AI Assistant endpoint
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[dict] = []

@app.post("/api/assistant/chat")
async def assistant_chat(chat_req: ChatRequest, current_user = Depends(get_current_user)):
    """AI Health Assistant - Educational information only"""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Update daily quest progress for using assistant
        user_id = str(current_user["_id"])
        gamification = await db["gamification"].find_one({"user_id": user_id})
        if gamification:
            daily_quests = gamification.get("daily_quests", {})
            if "use_assistant" in daily_quests and not daily_quests["use_assistant"]["completed"]:
                daily_quests["use_assistant"]["completed"] = True
                daily_quests["use_assistant"]["progress"] = 1
                xp_earned = daily_quests["use_assistant"]["xp"]
                await db["gamification"].update_one(
                    {"user_id": user_id},
                    {
                        "$set": {"daily_quests": daily_quests},
                        "$inc": {"xp": xp_earned}
                    }
                )
        
        system_message = """You are a health education assistant for "You Are What You Eat" app. 

CRITICAL RULES:
1. You provide EDUCATIONAL information, NOT personalized medical advice
2. You CAN explain what medical conditions ARE and what causes them (educational)
3. You CANNOT diagnose, prescribe treatments, or give personalized medical advice
4. Always add a brief disclaimer: "This is educational information - consult a healthcare professional for personal medical advice"
5. Keep responses informative but concise (2-3 paragraphs max)
6. Be helpful and educational - don't refuse legitimate questions about health topics

WHAT YOU CAN DO (Educational):
- Explain what conditions like diabetes, heart disease, Alzheimer's, cancer ARE
- Explain what CAUSES these conditions (diet, lifestyle, genetics)
- Explain how nutrition and diet relate to health conditions
- Discuss research linking UPFs/ingredients to health outcomes
- Explain what metabolic dysfunction, insulin resistance, inflammation ARE
- Discuss how specific ingredients affect the body
- General nutrition and food science education

WHAT YOU CANNOT DO (Medical Advice):
- Diagnose someone with a condition
- Prescribe treatments or medications
- Create personalized diet plans for treating medical conditions
- Tell someone to stop taking medications
- Give specific dosage recommendations"""

        # Build conversation messages for OpenAI
        messages = [{"role": "system", "content": system_message}]
        
        recent_history = chat_req.conversation_history[-10:] if chat_req.conversation_history else []
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": chat_req.message})
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return {"response": response.choices[0].message.content.strip()}
        
    except Exception as e:
        print(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from assistant")

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import mimetypes

@app.delete("/api/marketing/video/{filename}")
async def delete_marketing_video(filename: str):
    safe_name = os.path.basename(filename)
    file_path = f"/app/marketing/{safe_name}"
    if os.path.exists(file_path) and (safe_name.endswith(".mp4") or safe_name.endswith(".png")):
        os.remove(file_path)
        return {"message": f"Deleted {safe_name}"}
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/marketing/file/{filename}")
async def serve_marketing_file(filename: str):
    safe_name = os.path.basename(filename)
    file_path = f"/app/marketing/{safe_name}"
    if os.path.exists(file_path):
        media_type = "image/png" if safe_name.endswith(".png") else "video/mp4"
        return FileResponse(file_path, media_type=media_type, filename=safe_name)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/marketing")
async def marketing_catalog():
    """Serve the full marketing assets viewer with videos and images"""
    marketing_dir = "/app/marketing"
    video_cards = ""
    appstore_cards = ""
    if os.path.exists(marketing_dir):
        for f in sorted(os.listdir(marketing_dir)):
            if f.endswith(".mp4"):
                size_mb = round(os.path.getsize(os.path.join(marketing_dir, f)) / (1024*1024), 1)
                is_new = "ad_v2" in f
                badge = '<span class="badge badge-new">NEW</span>' if is_new else ''
                name = f.replace('.mp4','').replace('_',' ').title()
                video_cards += f'''<div class="card" id="card-{f}" data-testid="video-card-{f}" data-url="/api/marketing/video/{f}" data-name="{name}" data-type="video">
                <div class="select-check" onclick="toggleSelect(this)" data-testid="select-{f}"></div>
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/{f}" type="video/mp4"></video>
                <div class="card-info">
                    {badge}
                    <h3>{name}</h3>
                    <div class="meta">1280x720 | 8s | {size_mb}MB</div>
                    <div class="btn-row">
                        <a class="btn save-btn" href="/api/marketing/video/{f}" download data-testid="save-{f}">Save</a>
                        <button class="btn share-btn" onclick="shareAsset('/api/marketing/video/{f}', '{name}')" data-testid="share-{f}">Share</button>
                        <button class="btn del-btn" onclick="deleteVideo('{f}')" data-testid="delete-{f}">Delete</button>
                    </div>
                </div>
            </div>'''
            elif f.startswith("APPSTORE_") and f.endswith(".png"):
                size_kb = round(os.path.getsize(os.path.join(marketing_dir, f)) / 1024)
                name = f.replace('.png','').replace('_',' ').replace('APPSTORE ', '').title()
                appstore_cards += f'''<div class="card" id="card-{f}" data-testid="screenshot-card-{f}" data-url="/api/marketing/file/{f}" data-name="{name}" data-type="image">
                <div class="select-check" onclick="toggleSelect(this)" data-testid="select-{f}"></div>
                <img src="/api/marketing/file/{f}" style="width:100%;border-radius:8px;" loading="lazy" />
                <div class="card-info">
                    <h3>{name}</h3>
                    <div class="meta">1284x2778 | {size_kb}KB | App Store Ready</div>
                    <div class="btn-row">
                        <a class="btn save-btn" href="/api/marketing/file/{f}" download="{f}" data-testid="save-{f}">Save</a>
                        <button class="btn share-btn" onclick="shareAsset('/api/marketing/file/{f}', '{name}')" data-testid="share-{f}">Share</button>
                    </div>
                </div>
            </div>'''
            elif f.startswith("IPAD_APPSTORE_") and f.endswith(".png"):
                size_kb = round(os.path.getsize(os.path.join(marketing_dir, f)) / 1024)
                name = f.replace('.png','').replace('_',' ').replace('IPAD APPSTORE ', 'iPad: ').title()
                appstore_cards += f'''<div class="card" id="card-{f}" data-testid="screenshot-card-{f}" data-url="/api/marketing/file/{f}" data-name="{name}" data-type="image">
                <div class="select-check" onclick="toggleSelect(this)" data-testid="select-{f}"></div>
                <img src="/api/marketing/file/{f}" style="width:100%;border-radius:8px;" loading="lazy" />
                <div class="card-info">
                    <span class="badge badge-new">iPAD</span>
                    <h3>{name}</h3>
                    <div class="meta">2048x2732 | {size_kb}KB | iPad 13" Ready</div>
                    <div class="btn-row">
                        <a class="btn save-btn" href="/api/marketing/file/{f}" download="{f}" data-testid="save-{f}">Save</a>
                        <button class="btn share-btn" onclick="shareAsset('/api/marketing/file/{f}', '{name}')" data-testid="share-{f}">Share</button>
                    </div>
                </div>
            </div>'''
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YAWYE Marketing Assets</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a0a; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 40px 24px; padding-bottom: 120px; }
        h1 { font-size: 32px; color: #4CAF50; text-align: center; margin-bottom: 8px; }
        .subtitle { color: #888; text-align: center; margin-bottom: 48px; }
        h2 { font-size: 22px; color: #fff; margin-bottom: 8px; border-bottom: 2px solid #4CAF50; display: inline-block; padding-bottom: 6px; }
        .section { margin-bottom: 48px; }
        .desc { color: #888; font-size: 14px; margin: 8px 0 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 24px; }
        .card { background: #1a1a1a; border-radius: 16px; overflow: hidden; border: 1px solid #333; transition: opacity 0.4s, transform 0.4s, border-color 0.2s; position: relative; }
        .card:hover { border-color: #4CAF50; }
        .card.selected { border-color: #2196F3; box-shadow: 0 0 0 2px rgba(33,150,243,0.3); }
        .card.removing { opacity: 0; transform: scale(0.9); }
        .card img { width: 100%; height: auto; }
        .card video { width: 100%; height: auto; background: #000; }
        .card-info { padding: 16px; }
        .card-info h3 { font-size: 15px; color: #fff; margin-bottom: 4px; }
        .meta { font-size: 12px; color: #888; margin-bottom: 10px; }
        .badge { font-size: 12px; color: #4CAF50; background: rgba(76,175,80,0.1); padding: 4px 10px; border-radius: 8px; display: inline-block; margin-bottom: 8px; }
        .badge-new { background: rgba(255,215,0,0.15); color: #FFD700; }
        .btn-row { display: flex; gap: 8px; margin-top: 4px; }
        .btn { display: inline-flex; align-items: center; justify-content: center; padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; border: none; text-decoration: none; transition: all 0.2s; }
        .save-btn { background: #4CAF50; color: #fff; }
        .save-btn:hover { background: #45a049; transform: translateY(-1px); }
        .share-btn { background: transparent; color: #2196F3; border: 1px solid #2196F3; }
        .share-btn:hover { background: #2196F3; color: #fff; transform: translateY(-1px); }
        .del-btn { background: transparent; color: #FF5252; border: 1px solid #FF5252; }
        .del-btn:hover { background: #FF5252; color: #fff; transform: translateY(-1px); }
        .img-btn-row { display: flex; gap: 8px; margin-top: 8px; }
        .toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%) translateY(100px); background: #1a1a1a; color: #fff; padding: 12px 24px; border-radius: 12px; border: 1px solid #333; font-size: 14px; z-index: 999; opacity: 0; transition: all 0.3s; }
        .toast.show { transform: translateX(-50%) translateY(0); opacity: 1; }
        .toast.success { border-color: #4CAF50; }
        .toast.error { border-color: #FF5252; }
        .select-check { position: absolute; top: 12px; left: 12px; width: 28px; height: 28px; border-radius: 50%; border: 2px solid rgba(255,255,255,0.4); background: rgba(0,0,0,0.5); cursor: pointer; z-index: 10; display: flex; align-items: center; justify-content: center; transition: all 0.2s; font-size: 14px; color: transparent; }
        .select-check:hover { border-color: #2196F3; background: rgba(33,150,243,0.2); }
        .select-check.checked { background: #2196F3; border-color: #2196F3; color: #fff; }
        .bulk-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a1a; border-top: 2px solid #2196F3; padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; z-index: 1000; transform: translateY(100%); transition: transform 0.3s; }
        .bulk-bar.visible { transform: translateY(0); }
        .bulk-bar .bulk-info { font-size: 15px; font-weight: 600; color: #fff; }
        .bulk-bar .bulk-info span { color: #2196F3; }
        .bulk-bar .bulk-actions { display: flex; gap: 10px; align-items: center; }
        .bulk-bar .bulk-actions .btn { padding: 10px 22px; font-size: 14px; }
        .select-all-btn { background: transparent; color: #aaa; border: 1px solid #555; padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s; margin-left: 12px; }
        .select-all-btn:hover { color: #fff; border-color: #2196F3; }
        .section-header { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>You Are What You Eat</h1>
    <p class="subtitle">Marketing Assets Library</p>
    <div id="toast" class="toast"></div>

    <div class="section">
        <div class="section-header">
            <h2>Video Clips</h2>
            <button class="select-all-btn" onclick="selectAllInSection('video-grid')" data-testid="select-all-videos">Select All</button>
        </div>
        <p class="desc">AI-generated video clips (Sora 2). Save to download or delete to remove.</p>
        <div class="grid" id="video-grid">
            """ + video_cards + """
        </div>
    </div>

    <div class="section">
        <div class="section-header">
            <h2>App Store Screenshots</h2>
            <button class="select-all-btn" onclick="selectAllInSection('screenshot-grid')" data-testid="select-all-screenshots">Select All</button>
        </div>
        <p class="desc">Real app screenshots for iPhone (1284x2778) and iPad (2048x2732). Save and upload directly to App Store Connect.</p>
        <div class="grid" id="screenshot-grid">
            """ + appstore_cards + """
        </div>
    </div>

    <div class="section">
        <h2>App Screenshots (Phone Mockups)</h2>
        <p class="desc">AI-generated phone mockups matching the real app UI.</p>
        <div class="grid">
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bfa301dc173e39940ba3dc39168891b7a822285cbe7db660e9c91b838f839688.png" alt="Dashboard">
                <div class="card-info"><h3>Dashboard Screen</h3><div class="meta">Play Store / Social Posts</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bfa301dc173e39940ba3dc39168891b7a822285cbe7db660e9c91b838f839688.png" download="dashboard.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bfa301dc173e39940ba3dc39168891b7a822285cbe7db660e9c91b838f839688.png', 'Dashboard Screen')" data-testid="share-dashboard">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/c00f5806aaf51dec2d3dbd443d45b030eca647ff35ba31bea0ba3783865a64f1.png" alt="Scan">
                <div class="card-info"><h3>Barcode Scanning Screen</h3><div class="meta">Play Store / Social Posts</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/c00f5806aaf51dec2d3dbd443d45b030eca647ff35ba31bea0ba3783865a64f1.png" download="scan.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/c00f5806aaf51dec2d3dbd443d45b030eca647ff35ba31bea0ba3783865a64f1.png', 'Barcode Scanning')" data-testid="share-scan">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/291b690b2118fbd86a99e1f474e1914e815596d5cdfe3e4b688cc4e919410dbb.png" alt="Unhealthy">
                <div class="card-info"><h3>Result: Unhealthy (3/10)</h3><div class="meta">Ad Creative - Problem</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/291b690b2118fbd86a99e1f474e1914e815596d5cdfe3e4b688cc4e919410dbb.png" download="unhealthy_result.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/291b690b2118fbd86a99e1f474e1914e815596d5cdfe3e4b688cc4e919410dbb.png', 'Unhealthy Result')" data-testid="share-unhealthy">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/1614e64b0d3a205d014c507c591f8a87356cb7eeae9eb1888176d7c35dc95e38.png" alt="Healthy">
                <div class="card-info"><h3>Result: Healthy (9/10)</h3><div class="meta">Ad Creative - Solution</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/1614e64b0d3a205d014c507c591f8a87356cb7eeae9eb1888176d7c35dc95e38.png" download="healthy_result.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/1614e64b0d3a205d014c507c591f8a87356cb7eeae9eb1888176d7c35dc95e38.png', 'Healthy Result')" data-testid="share-healthy">Share</button></div></div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Ad Banners & Creatives</h2>
        <p class="desc">Ready-to-use promotional images for social media and ads.</p>
        <div class="grid">
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/4b4486bd747f67380d17b2b87913c12b5c83e9bd24072e205eb822648772b0f1.png" alt="Banner">
                <div class="card-info"><h3>Promo Banner (Wide)</h3><div class="meta">Facebook / Google Ads</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/4b4486bd747f67380d17b2b87913c12b5c83e9bd24072e205eb822648772b0f1.png" download="promo_banner.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/4b4486bd747f67380d17b2b87913c12b5c83e9bd24072e205eb822648772b0f1.png', 'Promo Banner')" data-testid="share-banner">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/a6dd97679a6afd03ba228468a49be2754d2adbc4df3bb27c7a8a57ac97ad7bb7.png" alt="Comparison">
                <div class="card-info"><h3>Before vs After</h3><div class="meta">Social / Ad Creative</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/a6dd97679a6afd03ba228468a49be2754d2adbc4df3bb27c7a8a57ac97ad7bb7.png" download="comparison.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/a6dd97679a6afd03ba228468a49be2754d2adbc4df3bb27c7a8a57ac97ad7bb7.png', 'Before vs After')" data-testid="share-comparison">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bd83b3e39253e9ff29ec2d57986e544cb6b8df2c3df7174eefc5b02c9a64f2dd.png" alt="Lifestyle">
                <div class="card-info"><h3>Lifestyle Shopping</h3><div class="meta">Instagram Posts</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bd83b3e39253e9ff29ec2d57986e544cb6b8df2c3df7174eefc5b02c9a64f2dd.png" download="lifestyle.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bd83b3e39253e9ff29ec2d57986e544cb6b8df2c3df7174eefc5b02c9a64f2dd.png', 'Lifestyle Shopping')" data-testid="share-lifestyle">Share</button></div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/af0b4e2fe3e045458d158fb0a670dad23536de7c26732176d00043cbda500171.png" alt="IG Story">
                <div class="card-info"><h3>SCAN. SCORE. KNOW.</h3><div class="meta">Instagram / TikTok Stories</div>
                <div class="img-btn-row"><a class="btn save-btn" href="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/af0b4e2fe3e045458d158fb0a670dad23536de7c26732176d00043cbda500171.png" download="ig_story.png">Save</a><button class="btn share-btn" onclick="shareAsset('https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/af0b4e2fe3e045458d158fb0a670dad23536de7c26732176d00043cbda500171.png', 'SCAN SCORE KNOW')" data-testid="share-igstory">Share</button></div></div>
            </div>
        </div>
    </div>

    <div class="bulk-bar" id="bulk-bar" data-testid="bulk-action-bar">
        <div class="bulk-info"><span id="select-count">0</span> items selected</div>
        <div class="bulk-actions">
            <button class="btn" style="background:transparent;color:#aaa;border:1px solid #555;" onclick="clearSelection()" data-testid="clear-selection-btn">Clear</button>
            <button class="btn save-btn" onclick="downloadSelected()" data-testid="download-selected-btn">Download All</button>
            <button class="btn share-btn" onclick="shareSelected()" data-testid="share-selected-btn">Share All</button>
        </div>
    </div>

    <script>
        const selected = new Set();

        function toggleSelect(el) {
            const card = el.closest('.card');
            const url = card.dataset.url;
            if (el.classList.contains('checked')) {
                el.classList.remove('checked');
                el.textContent = '';
                card.classList.remove('selected');
                selected.delete(card);
            } else {
                el.classList.add('checked');
                el.textContent = '\\u2713';
                card.classList.add('selected');
                selected.add(card);
            }
            updateBulkBar();
        }

        function updateBulkBar() {
            const bar = document.getElementById('bulk-bar');
            const count = document.getElementById('select-count');
            count.textContent = selected.size;
            bar.classList.toggle('visible', selected.size > 0);
        }

        function selectAllInSection(gridId) {
            const grid = document.getElementById(gridId);
            const checks = grid.querySelectorAll('.select-check');
            const allSelected = Array.from(checks).every(c => c.classList.contains('checked'));
            checks.forEach(el => {
                const card = el.closest('.card');
                if (allSelected) {
                    el.classList.remove('checked');
                    el.textContent = '';
                    card.classList.remove('selected');
                    selected.delete(card);
                } else {
                    el.classList.add('checked');
                    el.textContent = '\\u2713';
                    card.classList.add('selected');
                    selected.add(card);
                }
            });
            updateBulkBar();
        }

        function clearSelection() {
            document.querySelectorAll('.select-check.checked').forEach(el => {
                el.classList.remove('checked');
                el.textContent = '';
                el.closest('.card').classList.remove('selected');
            });
            selected.clear();
            updateBulkBar();
        }

        async function downloadSelected() {
            if (selected.size === 0) return;
            showToast('Starting download of ' + selected.size + ' items...', 'success');
            let delay = 0;
            selected.forEach(card => {
                const url = card.dataset.url;
                const name = card.dataset.name || 'asset';
                setTimeout(() => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = url.split('/').pop();
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }, delay);
                delay += 300;
            });
        }

        async function shareSelected() {
            if (selected.size === 0) return;
            const urls = [];
            selected.forEach(card => {
                const url = card.dataset.url;
                if (url.startsWith('http')) urls.push(url);
                else urls.push(window.location.origin + url);
            });
            const text = 'YAWYE Marketing Assets:\\n' + urls.join('\\n');
            if (navigator.share) {
                try {
                    await navigator.share({ title: 'YAWYE Assets (' + selected.size + ' items)', text: text });
                } catch(e) { if (e.name !== 'AbortError') copyToClipboard(text); }
            } else {
                copyToClipboard(text);
            }
        }

        function showToast(msg, type) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.className = 'toast show ' + type;
            setTimeout(() => t.className = 'toast', 3000);
        }

        async function shareAsset(url, title) {
            const fullUrl = window.location.origin + url;
            if (url.startsWith('http')) {
                var shareUrl = url;
            } else {
                var shareUrl = fullUrl;
            }
            if (navigator.share) {
                try {
                    await navigator.share({ title: 'You Are What You Eat - ' + title, text: 'Check out this marketing asset for YAWYE', url: shareUrl });
                } catch(e) { if (e.name !== 'AbortError') copyToClipboard(shareUrl); }
            } else {
                copyToClipboard(shareUrl);
            }
        }

        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(() => {
                showToast('Link copied to clipboard', 'success');
            }).catch(() => {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                showToast('Link copied to clipboard', 'success');
            });
        }

        async function deleteVideo(filename) {
            if (!confirm('Delete "' + filename.replace(/_/g,' ') + '"? This cannot be undone.')) return;
            const card = document.getElementById('card-' + filename);
            try {
                const res = await fetch('/api/marketing/video/' + filename, { method: 'DELETE' });
                if (res.ok) {
                    selected.delete(card);
                    updateBulkBar();
                    card.classList.add('removing');
                    setTimeout(() => card.remove(), 400);
                    showToast('Deleted ' + filename, 'success');
                } else {
                    showToast('Failed to delete', 'error');
                }
            } catch(e) {
                showToast('Error: ' + e.message, 'error');
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/api/marketing/video/{filename}")
async def serve_marketing_video(filename: str):
    safe_name = os.path.basename(filename)
    file_path = f"/app/marketing/{safe_name}"
    if os.path.exists(file_path) and safe_name.endswith(".mp4"):
        return FileResponse(file_path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")

@app.get("/api/marketing/videos")
async def list_marketing_videos():
    marketing_dir = "/app/marketing"
    videos = []
    if os.path.exists(marketing_dir):
        for f in sorted(os.listdir(marketing_dir)):
            if f.endswith(".mp4"):
                size_mb = round(os.path.getsize(os.path.join(marketing_dir, f)) / (1024*1024), 1)
                videos.append({"filename": f, "size_mb": size_mb, "url": f"/api/marketing/video/{f}"})
    return {"videos": videos}

# ============ WEBSITE PAGES ============
@app.get("/", response_class=HTMLResponse)
async def website_home():
    with open("/app/website-public/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/privacy-policy", response_class=HTMLResponse)
async def website_privacy():
    with open("/app/website-public/privacy-policy.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/terms-of-service", response_class=HTMLResponse)
async def website_terms():
    with open("/app/website-public/terms-of-service.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/support", response_class=HTMLResponse)
async def website_support():
    with open("/app/website-public/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# Serve static files (images, assets)
import os
if os.path.exists("/app/backend/static"):
    app.mount("/api/static", StaticFiles(directory="/app/backend/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
