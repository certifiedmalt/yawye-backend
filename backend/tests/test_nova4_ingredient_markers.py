"""Code-level NOVA 4 / refined-oil detection: the score must not depend on whether the AI
happened to notice glucose syrup or sunflower oil. Repro: Mars Duo (5000159605144) came back
from the AI as 'Processed, 0% UPF, score 4' and the low-UPF rules boosted it to 7/10."""
import asyncio
import json
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

os.environ.setdefault("MONGO_URL", "mongodb://localhost:27017")
os.environ.setdefault("DB_NAME", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import server  # noqa: E402

MARS_INGREDIENTS = ("Sugar, Glucose Syrup, Skimmed Milk Powder, Cocoa Butter, Sunflower Oil, Cocoa Mass, "
                    "Lactose and Protein from Whey (from Milk), Palm Fat, Milk Fat, Barley Malt Extract, "
                    "Salt, Egg White Powder, Milk Protein, Natural Vanilla Extract")


def run(ingredients, ai_result, name="Product", off_nova=None):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(ai_result)))]
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=resp)
    with patch.object(server, "AsyncOpenAI", return_value=fake):
        return asyncio.run(server.analyze_ingredients_with_ai(name, ingredients, off_nova_group=off_nova))


def ai(score, category, upf, harmful=None):
    return {"harmful_ingredients": harmful or [], "beneficial_ingredients": [], "carcinogens_found": [],
            "healthier_alternatives": [], "overall_score": score, "upf_score": upf,
            "processing_category": category, "recommendation": "x"}


def test_mars_duo_is_ultra_processed_even_when_ai_misses_it():
    r = run(MARS_INGREDIENTS, ai(4, "Processed", "0%"), "Mars Duo")
    assert r["overall_score"] <= 3
    assert r["processing_category"] == "Ultra-Processed"
    names = " ".join(h["name"].lower() for h in r["harmful_ingredients"])
    assert "glucose syrup" in names and "sunflower oil" in names


def test_detectors():
    assert "glucose syrup" in server.detect_nova4_markers(MARS_INGREDIENTS)
    assert server.detect_refined_oils(MARS_INGREDIENTS) == ["sunflower oil", "palm fat"]
    assert server.detect_nova4_markers("Potatoes, Sunflower Oil, Salt") == []
    assert server.detect_refined_oils("Extra Virgin Olive Oil, Cold Pressed Rapeseed Oil") == []
    assert server.detect_nova4_markers("Emulsifier: E471, Colour (E150d)")
    assert server.detect_nova4_markers("Wholemeal flour, water, natural vanilla extract, salt") == []


def test_seed_oil_blocks_low_upf_boost():
    r = run("Mackerel, Tomato, Sunflower Oil, Salt", ai(5, "Processed", "0%"), "Mackerel in tomato")
    assert r["overall_score"] <= 5


def test_sugar_led_blocks_low_upf_boost():
    r = run("Sugar, Cocoa Butter, Cocoa Mass, Milk Powder", ai(4, "Processed", "0%"), "Chocolate")
    assert r["overall_score"] <= 5


def test_clean_products_unchanged():
    assert run("Oats", ai(10, "Whole Food", "0%"), "Oats")["overall_score"] == 10
    r = run("Wheat flour, Water, Salt, Yeast", ai(5, "Processed", "0%"), "Bread")
    assert r["overall_score"] == 7  # genuine 0% UPF boost still applies
    assert run("Sparkling Natural Mineral Water", ai(10, "Whole Food", "0%"), "Water")["overall_score"] == 10
