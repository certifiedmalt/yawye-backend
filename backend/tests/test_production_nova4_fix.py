"""Production Railway tests for NOVA 4 Rule 0 fix and regressions.
Tests are READ-ONLY (status endpoints) except one rescan call.
"""
import os
import pytest
import requests

BASE = "https://web-production-66c05.up.railway.app"
EMAIL = "jpsaila1986@gmail.com"
PASSWORD = "hello123"


@pytest.fixture(scope="module")
def token():
    r = requests.post(f"{BASE}/api/auth/login",
                      json={"email": EMAIL, "password": PASSWORD}, timeout=30)
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    tok = r.json().get("token") or r.json().get("access_token")
    assert tok, f"no token in {r.json()}"
    return tok


@pytest.fixture(scope="module")
def headers(token):
    return {"Authorization": f"Bearer {token}"}


def _get_status(headers, barcode):
    r = requests.get(f"{BASE}/api/scan/status/{barcode}", headers=headers, timeout=30)
    assert r.status_code == 200, f"{barcode} status={r.status_code} body={r.text[:300]}"
    return r.json()


def _extract(data):
    """Normalize product fields from status response."""
    a = data.get("analysis") or data.get("product") or data
    score = a.get("overall_score", data.get("overall_score"))
    cat = (a.get("processing_category") or data.get("processing_category") or "").lower()
    upf = a.get("upf_score", data.get("upf_score"))
    return score, cat, upf, a


# --- Philadelphia: primary bug fix verification ---
def test_philadelphia_is_ultra_processed_and_low_score(headers):
    data = _get_status(headers, "7622201693916")
    score, cat, upf, a = _extract(data)
    print(f"Philadelphia: score={score}, category={cat}, upf={upf}")
    assert score is not None and score <= 3, f"Philadelphia score must be <=3, got {score}"
    assert "ultra" in cat, f"Philadelphia category must be Ultra-Processed, got '{cat}'"


# --- Babybel Light: must NOT regress ---
def test_babybel_light_high_score_not_ultra(headers):
    data = _get_status(headers, "3073781081909")
    score, cat, upf, a = _extract(data)
    print(f"Babybel Light: score={score}, category={cat}, upf={upf}")
    assert score is not None and score >= 7, f"Babybel score must be >=7, got {score}"
    assert "ultra" not in cat, f"Babybel must NOT be ultra-processed, got '{cat}'"


# --- Peter's Yard crackers: earlier 0% UPF fix regression ---
def test_peters_yard_crackers_high_score(headers):
    data = _get_status(headers, "5060198820052")
    score, cat, upf, a = _extract(data)
    print(f"Peter's Yard: score={score}, category={cat}, upf={upf}")
    assert score is not None and score >= 7, f"Peter's Yard score must be >=7, got {score}"


# --- Welch's: must remain ultra-processed low score ---
def test_welchs_fruit_snacks_ultra_processed_low(headers):
    data = _get_status(headers, "0034856005995")
    score, cat, upf, a = _extract(data)
    print(f"Welch's: score={score}, category={cat}, upf={upf}")
    assert score is not None and score <= 3, f"Welch's score must be <=3, got {score}"
    assert "ultra" in cat, f"Welch's must be Ultra-Processed, got '{cat}'"


# --- One rescan test for determinism (costs $) ---
def test_philadelphia_rescan_deterministic(headers):
    r = requests.post(f"{BASE}/api/scan/rescan",
                      headers=headers,
                      json={"barcode": "7622201693916"},
                      timeout=120)
    assert r.status_code == 200, f"rescan failed: {r.status_code} {r.text[:500]}"
    data = r.json()
    score, cat, upf, a = _extract(data)
    print(f"Philadelphia RESCAN: score={score}, category={cat}, upf={upf}")
    assert score is not None and score <= 3, f"Rescan score must be <=3, got {score}"
    assert "ultra" in cat, f"Rescan category must be Ultra-Processed, got '{cat}'"
