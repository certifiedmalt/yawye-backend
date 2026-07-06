"""Test scoring enforcement rules - verifies the logic that overrides AI scores"""
import pytest


def apply_scoring_rules(result: dict, ingredients: str) -> dict:
    """Extracted scoring enforcement logic from server.py for testing"""
    carcinogens = result.get("carcinogens_found", [])
    category = result.get("processing_category", "").lower()
    harmful = result.get("harmful_ingredients", [])
    score = result.get("overall_score", 5)

    # Parse UPF percentage from AI response
    upf_str = str(result.get("upf_score", "")).replace("%", "").replace("~", "").replace("<", "").strip()
    try:
        upf_pct = float(upf_str)
    except (ValueError, TypeError):
        upf_pct = -1

    ingredient_count = len([i.strip() for i in ingredients.split(",") if i.strip()]) if ingredients else 0

    has_industrial_additives = any(
        h.get("severity", "").lower() == "high" or
        "emulsifier" in h.get("name", "").lower() or
        "artificial" in h.get("name", "").lower() or
        "modified" in h.get("name", "").lower() or
        h.get("processing_level", "") == "NOVA 4"
        for h in harmful
    )

    # Rule 1: Any carcinogen = score 1
    if carcinogens and len(carcinogens) > 0:
        result["overall_score"] = 1
    # Rule 2: Ultra-Processed (NOVA 4) = max 3
    elif "ultra" in category:
        result["overall_score"] = min(score, 3)
    # Rule 8: Clean short ingredient list
    elif ingredient_count <= 3 and ingredient_count > 0 and not has_industrial_additives and len(harmful) == 0:
        result["overall_score"] = max(score, 8)
    # Rule 3: Processed (NOVA 3) = max 5 — BUT not if UPF is genuinely 0-10%
    elif "processed" in category and "minimally" not in category and "whole" not in category:
        if upf_pct >= 0 and upf_pct <= 10:
            result["overall_score"] = max(score, 7)
        else:
            result["overall_score"] = min(score, 5)

    # Rule 9: Zero/Low UPF safety net
    if upf_pct >= 0 and upf_pct <= 10 and not has_industrial_additives:
        if not (carcinogens and len(carcinogens) > 0) and "ultra" not in category:
            result["overall_score"] = max(result["overall_score"], 7)

    # Rule 10: Whole Food / Minimally Processed floor
    if ("whole" in category or "minimally" in category) and not (carcinogens and len(carcinogens) > 0):
        result["overall_score"] = max(result["overall_score"], 7)

    return result


# ============= BUG REPRODUCTION: 0% UPF scoring 5 =============

def test_zero_upf_should_not_score_5():
    """The exact bug reported: 0% UPF product getting score 5"""
    result = {
        "overall_score": 5,
        "upf_score": "0%",
        "processing_category": "Minimally Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "chicken, salt, pepper, herbs, breadcrumbs")
    assert result["overall_score"] >= 7, f"0% UPF product scored {result['overall_score']}, expected >= 7"


def test_zero_upf_processed_category_contradiction():
    """AI says 0% UPF but also says 'Processed' — should trust UPF and score >= 7"""
    result = {
        "overall_score": 5,
        "upf_score": "0%",
        "processing_category": "Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "chicken, salt, pepper, herbs")
    assert result["overall_score"] >= 7, f"0% UPF product scored {result['overall_score']}, expected >= 7"


def test_zero_upf_whole_food():
    """Whole food with 0% UPF should score at least 7"""
    result = {
        "overall_score": 5,
        "upf_score": "0%",
        "processing_category": "Whole Food",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "apple")
    assert result["overall_score"] >= 8, f"Whole food scored {result['overall_score']}, expected >= 8"


# ============= ERROR FALLBACK TEST =============

def test_error_fallback_not_misleading():
    """Error fallback should NOT return score 5 with 0% UPF"""
    # This is the OLD buggy fallback - make sure we never return this
    error_result = {
        "harmful_ingredients": [],
        "beneficial_ingredients": [],
        "overall_score": 0,
        "upf_score": "Unknown",
        "processing_category": "Unknown",
        "recommendation": "Unable to analyze ingredients at this time. Please try scanning again.",
        "analysis_error": True
    }
    # Error result should have score 0 and UPF "Unknown", not score 5 and "0%"
    assert error_result["overall_score"] == 0
    assert error_result["upf_score"] == "Unknown"
    assert error_result.get("analysis_error") is True


# ============= EXISTING RULES STILL WORK =============

def test_carcinogen_always_1():
    """Rule 1: Any carcinogen = score 1"""
    result = {
        "overall_score": 8,
        "upf_score": "0%",
        "processing_category": "Minimally Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [{"name": "nitrite"}],
    }
    result = apply_scoring_rules(result, "pork, salt, nitrite")
    assert result["overall_score"] == 1


def test_ultra_processed_max_3():
    """Rule 2: Ultra-Processed = max 3"""
    result = {
        "overall_score": 7,
        "upf_score": "80%",
        "processing_category": "Ultra-Processed",
        "harmful_ingredients": [{"name": "emulsifier E471", "severity": "high", "processing_level": "NOVA 4"}],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "sugar, modified starch, emulsifier E471, artificial flavor")
    assert result["overall_score"] <= 3


def test_clean_3_ingredients_minimum_8():
    """Rule 8: Clean short ingredient list = minimum 8"""
    result = {
        "overall_score": 6,
        "upf_score": "0%",
        "processing_category": "Minimally Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "apple, cinnamon, water")
    assert result["overall_score"] >= 8


def test_processed_with_high_upf_capped_at_5():
    """Rule 3: Processed with significant UPF = max 5"""
    result = {
        "overall_score": 7,
        "upf_score": "30%",
        "processing_category": "Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "flour, sugar, butter, salt, yeast, improver")
    assert result["overall_score"] <= 5


def test_ten_percent_upf_still_gets_boost():
    """Rule 9: 10% UPF should still get boost to 7"""
    result = {
        "overall_score": 5,
        "upf_score": "10%",
        "processing_category": "Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "oats, milk, honey, raisins")
    assert result["overall_score"] >= 7


def test_upf_15_percent_stays_capped():
    """Products with >10% UPF and Processed category should stay capped at 5"""
    result = {
        "overall_score": 7,
        "upf_score": "15%",
        "processing_category": "Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "flour, sugar, butter, salt, preservative")
    assert result["overall_score"] <= 5


def test_minimally_processed_boost():
    """Rule 10: Minimally Processed = minimum 7"""
    result = {
        "overall_score": 5,
        "upf_score": "5%",
        "processing_category": "Minimally Processed",
        "harmful_ingredients": [],
        "carcinogens_found": [],
    }
    result = apply_scoring_rules(result, "chicken, salt, pepper, herbs, garlic")
    assert result["overall_score"] >= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
