from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
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
from emergentintegrations.llm.chat import LlmChat, UserMessage
import asyncio
import time
import logging

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
    """Analyze ingredients using AI with focus on ultra-processed foods (UPFs)"""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"ingredient-analysis-{datetime.utcnow().timestamp()}",
            system_message="You are a food science expert specializing in ultra-processed foods (UPFs) and the NOVA classification system. Your expertise is in identifying harmful industrial ingredients, additives, and processing markers. Provide clear, consumer-friendly explanations."
        ).with_model("openai", "gpt-5.2")
        
        prompt = f"""Analyze these ingredients from {product_name}:

{ingredients}

FOCUS: Identify ultra-processed food (UPF) ingredients and their direct health impacts.

PRIORITIZE AS HIGH SEVERITY:
- Emulsifiers (E471, mono/diglycerides, lecithins, polysorbates)
- Artificial sweeteners (aspartame, sucralose, acesulfame K)
- Preservatives (sodium benzoate, potassium sorbate, BHA, BHT)
- Artificial colors (tartrazine, sunset yellow, etc.)
- Modified starches, maltodextrin, dextrose
- Hydrogenated oils, palm oil, interesterified fats
- Flavor enhancers (MSG, hydrolyzed proteins, yeast extract)
- Added sugars (high fructose corn syrup, glucose syrup, invert sugar)

RESPONSE FORMAT - Provide JSON with this EXACT structure:
{{
  "harmful_ingredients": [
    {{
      "name": "ingredient name",
      "health_impact": "Clear, direct health impact in 1-2 sentences. Focus on WHAT it does to your body, not technical details. Consumer-friendly language.",
      "severity": "high/medium/low",
      "processing_level": "NOVA 4 - ultra-processed" or "NOVA 3 - processed",
      "research_summary": "Brief 2-3 sentence summary of key research findings. Mention specific health outcomes studied (e.g., 'Studies show increased inflammation markers and gut dysbiosis in regular consumers')"
    }}
  ],
  "beneficial_ingredients": [
    {{
      "name": "ingredient name",
      "health_benefit": "Clear, positive health impact in 1-2 sentences. Focus on benefits to the body.",
      "processing_level": "NOVA 1 - whole/minimally processed",
      "research_summary": "Brief 2-3 sentence summary of research. Focus on proven benefits (e.g., 'Multiple studies link regular consumption to 30% reduced cardiovascular disease risk')"
    }}
  ],
  "overall_score": 1-10,
  "upf_score": "percentage like 75%",
  "processing_category": "Whole Food / Minimally Processed / Processed / Ultra-Processed",
  "recommendation": "One clear sentence recommendation based on the analysis"
}}

WRITING STYLE:
- Health impacts: Clear, direct, consumer-friendly (8th grade reading level)
- Research summaries: Factual, specific, cite real findings
- No jargon in health impacts
- Research can be more technical but still accessible

SCORING:
- 8-10: Whole/minimally processed, mostly beneficial ingredients
- 5-7: Some processing, mix of good and concerning ingredients  
- 1-4: Ultra-processed, multiple harmful additives

Be honest and specific. If it's bad, say so clearly."""
        
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
        response = requests.get(f"{OFF_API_URL}/{scan_req.barcode}.json", timeout=30)
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
        
        # Create chat with content filtering
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"assistant-{str(current_user['_id'])}-{datetime.utcnow().timestamp()}",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        # Build conversation
        for msg in recent_history:
            if msg["role"] == "user":
                chat.send_message(UserMessage(text=msg["content"]))
        
        # Send current message
        user_message = UserMessage(text=chat_req.message)
        response = await chat.send_message(user_message)
        
        return {"response": response}
        
    except Exception as e:
        print(f"Assistant error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from assistant")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
