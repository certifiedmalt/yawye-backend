from fastapi import FastAPI, HTTPException, Depends, status
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
import openai
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
DB_NAME = os.getenv("DB_NAME") or "yawye_db"
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
scans_collection = db["scans"]
favorites_collection = db["favorites"]
product_cache_collection = db["product_cache"]  # New: Cache for faster lookups
scan_analytics_collection = db["scan_analytics"]  # New: Analytics tracking

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
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
EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY")

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
    return pwd_context.hash(password[:72])

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
    """Analyze ingredients using AI with focus on ultra-processed foods (UPFs)"""
    try:
        client = openai.AsyncOpenAI(api_key=EMERGENT_LLM_KEY)
        
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a food science expert specializing in ultra-processed foods (UPFs) and the NOVA classification system. Your expertise is in identifying harmful industrial ingredients, additives, and processing markers. Provide clear, consumer-friendly explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        response = completion.choices[0].message.content
        
        # Parse JSON from response
        import json
        # Clean response to extract JSON
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        analysis = json.loads(response_text)
        return analysis
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
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if not os.path.exists(icon_path):
        raise HTTPException(status_code=404, detail="Icon not found")
    return FileResponse(icon_path, media_type="image/png", filename="you-are-what-you-eat-icon.png")

@app.get("/api/health")
async def health_check():
    try:
        # Test MongoDB connection
        await users_collection.find_one({})
        return {"status": "healthy", "db": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "db_error": str(e)}

@app.get("/api/debug/test-register")
async def debug_test_register():
    try:
        hashed = get_password_hash("test123")
        result = await users_collection.insert_one({
            "email": f"debug-{datetime.utcnow().timestamp()}@test.com",
            "password": hashed,
            "name": "Debug Test",
            "subscription_tier": "free",
            "daily_scans": 0,
        })
        return {"status": "ok", "id": str(result.inserted_id)}
    except Exception as e:
        return {"status": "error", "error": str(e), "type": type(e).__name__}

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
        "daily_scans": 0,
        "last_scan_reset": datetime.utcnow(),
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
    # Reset daily scans if needed
    last_reset = current_user.get("last_scan_reset", datetime.utcnow())
    if datetime.utcnow() - last_reset > timedelta(days=1):
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$set": {"daily_scans": 0, "last_scan_reset": datetime.utcnow()}}
        )
        current_user["daily_scans"] = 0
    
    return {
        "id": str(current_user["_id"]),
        "email": current_user["email"],
        "name": current_user["name"],
        "subscription_tier": current_user.get("subscription_tier", "free"),
        "daily_scans": current_user.get("daily_scans", 0)
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
    
    # Check subscription limits
    subscription_tier = current_user.get("subscription_tier", "free")
    daily_scans = current_user.get("daily_scans", 0)
    
    if subscription_tier == "free" and daily_scans >= 5:
        raise HTTPException(
            status_code=403,
            detail="Daily scan limit reached. Upgrade to premium for unlimited scans."
        )
    
    # STEP 1: Check cache first (instant results!)
    cached_product = await get_cached_product(barcode)
    if cached_product:
        response_time = time.time() - start_time
        await log_scan_analytics(barcode, True, "cache", response_time)
        
        # Save to user's scan history
        scan_doc = {
            "user_id": str(current_user["_id"]),
            "barcode": barcode,
            "product_name": cached_product.get("product_name"),
            "brands": cached_product.get("brands"),
            "ingredients_text": cached_product.get("ingredients_text"),
            "image_url": cached_product.get("image_url"),
            "analysis": cached_product.get("analysis"),
            "scanned_at": datetime.utcnow(),
            "from_cache": True
        }
        await scans_collection.insert_one(scan_doc)
        
        # Update user's daily scan count
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"daily_scans": 1}}
        )
        
        return {
            "product_name": cached_product.get("product_name"),
            "brands": cached_product.get("brands"),
            "ingredients_text": cached_product.get("ingredients_text"),
            "image_url": cached_product.get("image_url"),
            "analysis": cached_product.get("analysis"),
            "from_cache": True,
            "response_time_ms": int(response_time * 1000)
        }
    
    # STEP 2: Parallel API calls with smart routing based on barcode prefix
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
        
        # Return the FIRST successful result
        for future in as_completed(futures, timeout=20):
            src = futures[future]
            try:
                result = future.result()
                if result:
                    product_data = result
                    source = src
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
            except Exception as e:
                logger.warning(f"{src} error: {e}")
    
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
    
    # STEP 7: Cache the result for future fast lookups
    product_data["analysis"] = analysis
    await cache_product(barcode, product_data)
    
    # STEP 8: Save to user's scan history
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
    
    # Update user's daily scan count
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$inc": {"daily_scans": 1}}
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
    # Simple upgrade (in production, integrate with payment processor)
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"subscription_tier": "premium"}}
    )
    
    return {"message": "Upgraded to premium", "subscription_tier": "premium"}

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
        # Create system message with strong guardrails
        system_message = """You are a health education assistant for "You Are What You Eat" app. 

CRITICAL RULES:
1. You provide EDUCATIONAL information only, NOT medical advice
2. ALWAYS remind users to consult healthcare professionals for personal health decisions
3. Focus on: general nutrition, food ingredients, UPFs, healthy eating principles
4. NEVER: diagnose, prescribe, recommend treatments, or give personalized medical advice
5. If asked medical questions, redirect to healthcare professionals
6. Keep responses concise (2-3 paragraphs max)
7. Be friendly and helpful but maintain boundaries

SAFE TOPICS:
- General nutrition education
- Understanding food labels and ingredients
- What are UPFs and why they matter
- General healthy eating tips
- How to use the app
- Food science basics

FORBIDDEN TOPICS:
- Medical diagnosis or treatment
- Personalized diet plans for medical conditions
- Medication interactions
- Specific health conditions
- Weight loss advice beyond general principles

If user asks forbidden topics, politely say: "I can't provide medical advice. Please consult a healthcare professional for personalized guidance. I can help with general nutrition education instead - what would you like to know?"
"""

        # Limit conversation history to last 10 messages
        recent_history = chat_req.conversation_history[-10:] if chat_req.conversation_history else []
        
        # Build messages for OpenAI
        messages = [{"role": "system", "content": system_message}]
        for msg in recent_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": chat_req.message})
        
        client = openai.AsyncOpenAI(api_key=EMERGENT_LLM_KEY)
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        response = completion.choices[0].message.content
        
        return {"response": response}
        
    except Exception as e:
        print(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from assistant")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
