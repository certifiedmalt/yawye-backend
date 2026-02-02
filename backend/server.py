from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import requests
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import jwt
from passlib.context import CryptContext
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio

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

# MongoDB
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")
client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
scans_collection = db["scans"]
favorites_collection = db["favorites"]

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"

# Open Food Facts API
OFF_API_URL = "https://world.openfoodfacts.org/api/v2/product"

# LLM Setup
EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY")

# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ScanRequest(BaseModel):
    barcode: str

class FavoriteRequest(BaseModel):
    product_id: str

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
    """Analyze ingredients using AI with focus on ultra-processed foods (UPFs)"""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"ingredient-analysis-{datetime.utcnow().timestamp()}",
            system_message="You are a food science expert specializing in ultra-processed foods (UPFs) and the NOVA classification system. Your expertise is in identifying harmful industrial ingredients, additives, and processing markers."
        ).with_model("openai", "gpt-5.2")
        
        prompt = f"""Analyze these ingredients from {product_name}:

{ingredients}

FOCUS: Identify ultra-processed food (UPF) ingredients and markers of industrial processing.

PRIORITIZE AS HIGH SEVERITY:
- Emulsifiers (E471, mono/diglycerides, lecithins, polysorbates)
- Artificial sweeteners (aspartame, sucralose, acesulfame K)
- Preservatives (sodium benzoate, potassium sorbate, BHA, BHT)
- Artificial colors (tartrazine, sunset yellow, etc.)
- Modified starches, maltodextrin, dextrose
- Hydrogenated oils, palm oil, interesterified fats
- Flavor enhancers (MSG, hydrolyzed proteins, yeast extract)
- Added sugars (high fructose corn syrup, glucose syrup, invert sugar)

PROCESSING SCORE SYSTEM:
- Score 8-10: Whole/minimally processed ingredients (real fruits, nuts, whole grains, simple ingredients)
- Score 5-7: Moderately processed (refined flour, sugar, salt, some processing)
- Score 1-4: Ultra-processed (multiple additives, emulsifiers, artificial ingredients, heavily processed)

Provide a JSON response with this exact structure:
{{
  "harmful_ingredients": [
    {{
      "name": "ingredient name",
      "health_risk": "explain why this UPF ingredient is harmful (focus on processing, not natural content)",
      "severity": "high/medium/low",
      "processing_level": "NOVA 4 - ultra-processed" or "NOVA 3 - processed" or "NOVA 2 - processed culinary ingredient",
      "study_reference": "Brief mention of research on ultra-processed foods and health outcomes (e.g., 'Studies link emulsifiers to gut inflammation and metabolic syndrome' or 'Research shows artificial sweeteners disrupt gut microbiome')"
    }}
  ],
  "beneficial_ingredients": [
    {{
      "name": "ingredient name",
      "health_benefit": "explain health benefit",
      "processing_level": "NOVA 1 - whole/minimally processed" or "natural ingredient",
      "study_reference": "Research on health benefits (e.g., 'Studies show whole grains reduce cardiovascular disease risk by 30%')"
    }}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage of ingredients that are ultra-processed (0-100%)",
  "processing_category": "Whole Food / Minimally Processed / Processed / Ultra-Processed",
  "recommendation": "Clear recommendation based on processing level and UPF content"
}}

IMPORTANT: 
- The more processed and artificial the ingredients, the LOWER the score
- Products with mostly UPF ingredients should score 1-3
- Focus on ADDITIVES and PROCESSING, not natural sugars or fats from whole foods
- Be specific about why industrial processing is harmful
- Cite research on ultra-processed foods, not general nutrition advice"""
        
        user_message = UserMessage(text=prompt)
        response = await chat.send_message(user_message)
        
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
    # Check subscription limits
    subscription_tier = current_user.get("subscription_tier", "free")
    daily_scans = current_user.get("daily_scans", 0)
    
    if subscription_tier == "free" and daily_scans >= 5:
        raise HTTPException(
            status_code=403,
            detail="Daily scan limit reached. Upgrade to premium for unlimited scans."
        )
    
    # Fetch product from Open Food Facts
    try:
        response = requests.get(f"{OFF_API_URL}/{scan_req.barcode}.json", timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Product not found")
        
        data = response.json()
        if data.get("status") != 1:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product = data.get("product", {})
        
        # Extract product info
        product_name = product.get("product_name", "Unknown Product")
        brands = product.get("brands", "Unknown Brand")
        ingredients_text = product.get("ingredients_text", "")
        image_url = product.get("image_url", "")
        
        if not ingredients_text:
            raise HTTPException(status_code=404, detail="No ingredients found for this product")
        
        # Analyze ingredients with AI
        analysis = await analyze_ingredients_with_ai(product_name, ingredients_text)
        
        # Save scan to database
        scan_doc = {
            "user_id": str(current_user["_id"]),
            "barcode": scan_req.barcode,
            "product_name": product_name,
            "brands": brands,
            "ingredients_text": ingredients_text,
            "image_url": image_url,
            "analysis": analysis,
            "scanned_at": datetime.utcnow()
        }
        await scans_collection.insert_one(scan_doc)
        
        # Update user's daily scan count
        await users_collection.update_one(
            {"_id": current_user["_id"]},
            {"$inc": {"daily_scans": 1}}
        )
        
        return {
            "product_name": product_name,
            "brands": brands,
            "ingredients_text": ingredients_text,
            "image_url": image_url,
            "analysis": analysis
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch product data: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
