from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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

# Open Food Facts API
OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product"
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
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
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
        response = fetch_with_retry(f"{OFF_API_URL}/{barcode}.json", max_retries=3, timeout=20)
        
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
        response = requests.get(url, timeout=10)
        
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
                detail_response = requests.get(detail_url, timeout=10)
                
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

async def analyze_ingredients_with_ai(product_name: str, ingredients: str) -> dict:
    """Analyze ingredients using OpenAI GPT-4o with focus on ultra-processed foods (UPFs)"""
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""You are a food science expert specializing in ultra-processed foods (UPFs).

Analyze these ingredients from {product_name}:

{ingredients}

Identify harmful UPF ingredients AND beneficial whole food nutrients.

HARMFUL: seed oils, emulsifiers, artificial sweeteners, preservatives, artificial colors, modified starches, hydrogenated oils, added sugars.

BENEFICIAL: proteins, vitamins, fiber, healthy fats, probiotics.

Respond with JSON only:
{{
  "harmful_ingredients": [
    {{"name": "ingredient", "health_impact": "2-3 sentences", "severity": "high/medium/low", "processing_level": "NOVA 4", "research_summary": "study citation", "study_link": "pubmed link"}}
  ],
  "beneficial_ingredients": [
    {{"name": "ingredient", "health_benefit": "2-3 sentences", "benefit_type": "protein/vitamin/fiber", "key_nutrients": "list", "processing_level": "NOVA 1", "research_summary": "citation", "study_link": "link"}}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage",
  "processing_category": "Whole Food/Minimally Processed/Processed/Ultra-Processed",
  "recommendation": "actionable advice"
}}

Score 8-10 for whole foods, 5-7 for mixed, 1-4 for ultra-processed."""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food science expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
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
    
    # CACHE DISABLED - Always fetch fresh results for accuracy
    # Each scan will get the latest AI analysis
    
    # STEP 1: Parallel API calls with smart routing based on barcode prefix
    # UK/EU barcodes start with 50/40-44, US barcodes start with 0
    barcode_prefix = barcode[:2] if len(barcode) >= 2 else ""
    
    # Run all API sources in parallel using ThreadPoolExecutor
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    product_data = None
    source = "none"
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Smart ordering: prioritize based on barcode region
        if barcode_prefix in ['50', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '87', '90', '93', '94']:
            # EU/UK barcode — Open Food Facts first, then others in parallel
            futures = {
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
            }
        elif barcode_prefix in ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']:
            # US/Canada barcode — USDA first, then others in parallel
            futures = {
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
            }
        else:
            # Other regions — try all simultaneously
            futures = {
                executor.submit(fetch_from_openfoodfacts, barcode): "openfoodfacts",
                executor.submit(fetch_from_usda, barcode): "usda",
                executor.submit(fetch_from_upcitemdb, barcode): "upcitemdb",
            }
        
        # Collect ALL results, prioritize ones WITH ingredients
        results_with_ingredients = []
        results_without_ingredients = []
        
        for future in as_completed(futures, timeout=20):
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
        
        # Prefer results WITH ingredients
        if results_with_ingredients:
            product_data, source = results_with_ingredients[0]
            logger.info(f"Using {source} (has ingredients)")
        elif results_without_ingredients:
            product_data, source = results_without_ingredients[0]
            logger.info(f"Using {source} (no ingredients available)")
    
    # STEP 3: If parallel calls all failed, try FatSecret as last resort
    if not product_data:
        logger.info(f"All parallel sources failed for {barcode}, trying FatSecret")
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
        # Still return basic info even without ingredients
        response_time = time.time() - start_time
        await log_scan_analytics(barcode, True, source, response_time, "No ingredients")
        
        return {
            "product_name": product_data.get("product_name"),
            "brands": product_data.get("brands"),
            "ingredients_text": "",
            "image_url": product_data.get("image_url"),
            "analysis": {
                "harmful_ingredients": [],
                "beneficial_ingredients": [],
                "overall_score": 5,
                "recommendation": "No ingredient information available for this product. Check the packaging for details.",
                "nova_group": None,
                "additives": []
            },
            "warning": "No ingredients found - limited analysis available"
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
    
    # CACHE DISABLED - No longer caching results
    product_data["analysis"] = analysis
    
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

@app.get("/api/marketing")
async def marketing_catalog():
    """Serve the full marketing assets viewer with videos and images"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YAWYE Marketing Assets</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a0a; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 40px 24px; }
        h1 { font-size: 32px; color: #4CAF50; text-align: center; margin-bottom: 8px; }
        .subtitle { color: #888; text-align: center; margin-bottom: 48px; }
        h2 { font-size: 22px; color: #fff; margin-bottom: 8px; border-bottom: 2px solid #4CAF50; display: inline-block; padding-bottom: 6px; }
        .section { margin-bottom: 48px; }
        .desc { color: #888; font-size: 14px; margin: 8px 0 24px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 24px; }
        .card { background: #1a1a1a; border-radius: 16px; overflow: hidden; border: 1px solid #333; }
        .card:hover { border-color: #4CAF50; }
        .card img { width: 100%; height: auto; }
        .card video { width: 100%; height: auto; background: #000; }
        .card-info { padding: 16px; }
        .card-info h3 { font-size: 15px; color: #fff; margin-bottom: 4px; }
        .meta { font-size: 12px; color: #888; margin-bottom: 8px; }
        .badge { font-size: 12px; color: #4CAF50; background: rgba(76,175,80,0.1); padding: 4px 10px; border-radius: 8px; display: inline-block; margin-bottom: 8px; }
        .badge-new { background: rgba(255,215,0,0.15); color: #FFD700; }
        .dl-btn { display: inline-block; background: #4CAF50; color: #fff; text-decoration: none; padding: 6px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; }
        .dl-btn:hover { background: #45a049; }
    </style>
</head>
<body>
    <h1>You Are What You Eat</h1>
    <p class="subtitle">Marketing Assets Library</p>

    <div class="section">
        <h2>Video Clips</h2>
        <p class="desc">AI-generated video clips (Sora 2). Click play to watch, or right-click to save.</p>
        <div class="grid">
            <div class="card">
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/app_score_reveal_unhealthy.mp4" type="video/mp4"></video>
                <div class="card-info">
                    <span class="badge badge-new">NEW</span>
                    <h3>Score Reveal: Unhealthy Product</h3>
                    <div class="meta">1280x720 | 8s | Phone showing 3/10 red score</div>
                    <a class="dl-btn" href="/api/marketing/video/app_score_reveal_unhealthy.mp4" download>Download</a>
                </div>
            </div>
            <div class="card">
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/app_dashboard_hero.mp4" type="video/mp4"></video>
                <div class="card-info">
                    <span class="badge badge-new">NEW</span>
                    <h3>App Dashboard Hero Shot</h3>
                    <div class="meta">1280x720 | 8s | Phone on kitchen counter showing app</div>
                    <a class="dl-btn" href="/api/marketing/video/app_dashboard_hero.mp4" download>Download</a>
                </div>
            </div>
            <div class="card">
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/yawye_ad_clip1_problem.mp4" type="video/mp4"></video>
                <div class="card-info">
                    <h3>The Problem: Confused at Labels</h3>
                    <div class="meta">1280x720 | 8s | Person reading ingredient labels</div>
                    <a class="dl-btn" href="/api/marketing/video/yawye_ad_clip1_problem.mp4" download>Download</a>
                </div>
            </div>
            <div class="card">
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/yawye_ad_clip2_solution.mp4" type="video/mp4"></video>
                <div class="card-info">
                    <h3>The Solution: Scanning with App</h3>
                    <div class="meta">1280x720 | 8s | Person confidently scanning product</div>
                    <a class="dl-btn" href="/api/marketing/video/yawye_ad_clip2_solution.mp4" download>Download</a>
                </div>
            </div>
            <div class="card">
                <video controls playsinline preload="metadata"><source src="/api/marketing/video/yawye_ad_clip2_couple.mp4" type="video/mp4"></video>
                <div class="card-info">
                    <h3>Couple Shopping Together</h3>
                    <div class="meta">1280x720 | 8s | Couple scanning a product</div>
                    <a class="dl-btn" href="/api/marketing/video/yawye_ad_clip2_couple.mp4" download>Download</a>
                </div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>App Screenshots (Phone Mockups)</h2>
        <p class="desc">AI-generated phone mockups matching the real app UI. Right-click to save.</p>
        <div class="grid">
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bfa301dc173e39940ba3dc39168891b7a822285cbe7db660e9c91b838f839688.png" alt="Dashboard">
                <div class="card-info"><h3>Dashboard Screen</h3><div class="meta">Play Store / Social Posts</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/c00f5806aaf51dec2d3dbd443d45b030eca647ff35ba31bea0ba3783865a64f1.png" alt="Scan">
                <div class="card-info"><h3>Barcode Scanning Screen</h3><div class="meta">Play Store / Social Posts</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/291b690b2118fbd86a99e1f474e1914e815596d5cdfe3e4b688cc4e919410dbb.png" alt="Unhealthy">
                <div class="card-info"><h3>Result: Unhealthy (3/10)</h3><div class="meta">Ad Creative - Problem</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/1614e64b0d3a205d014c507c591f8a87356cb7eeae9eb1888176d7c35dc95e38.png" alt="Healthy">
                <div class="card-info"><h3>Result: Healthy (9/10)</h3><div class="meta">Ad Creative - Solution</div></div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Ad Banners & Creatives</h2>
        <p class="desc">Ready-to-use promotional images for social media and ads.</p>
        <div class="grid">
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/4b4486bd747f67380d17b2b87913c12b5c83e9bd24072e205eb822648772b0f1.png" alt="Banner">
                <div class="card-info"><h3>Promo Banner (Wide)</h3><div class="meta">Facebook / Google Ads</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/a6dd97679a6afd03ba228468a49be2754d2adbc4df3bb27c7a8a57ac97ad7bb7.png" alt="Comparison">
                <div class="card-info"><h3>Before vs After</h3><div class="meta">Social / Ad Creative</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/bd83b3e39253e9ff29ec2d57986e544cb6b8df2c3df7174eefc5b02c9a64f2dd.png" alt="Lifestyle">
                <div class="card-info"><h3>Lifestyle Shopping</h3><div class="meta">Instagram Posts</div></div>
            </div>
            <div class="card">
                <img src="https://static.prod-images.emergentagent.com/jobs/f81b4164-3f7c-418b-9e38-85342e9419f0/images/af0b4e2fe3e045458d158fb0a670dad23536de7c26732176d00043cbda500171.png" alt="IG Story">
                <div class="card-info"><h3>SCAN. SCORE. KNOW.</h3><div class="meta">Instagram / TikTok Stories</div></div>
            </div>
        </div>
    </div>
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
