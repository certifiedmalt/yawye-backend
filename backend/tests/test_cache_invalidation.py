"""Cache invalidation behavior tests for /api/scan/full and /api/scan/quick.

Verifies:
- Cached error results (score=5, upf=0%, category=Unknown) are invalidated on read
- Failed AI analyses are NOT persisted to product_cache.analysis
- Error fallback returns overall_score=0, upf_score='Unknown', analysis_error=True
"""
import os
import time
import uuid
import pytest
import requests
from pymongo import MongoClient

BASE_URL = "http://localhost:8001"  # internal; scan endpoints not exposed on prefix on preview when unauth
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "yawye_app")


@pytest.fixture(scope="module")
def mongo_db():
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    yield db
    # cleanup any TEST_ barcodes
    db.product_cache.delete_many({"barcode": {"$regex": "^TEST_"}})
    client.close()


@pytest.fixture(scope="module")
def auth_token():
    # register a fresh test user (avoids scan limit issues on reruns)
    email = f"TEST_cache_{uuid.uuid4().hex[:8]}@yawye.app"
    r = requests.post(f"{BASE_URL}/api/auth/register", json={
        "email": email, "password": "test1234", "name": "Cache Tester"
    })
    assert r.status_code == 200, f"register failed: {r.status_code} {r.text}"
    return r.json()["token"]


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}


def _seed_bad_cache(db, barcode):
    """Insert a stale error result matching the pre-fix cache bug."""
    db.product_cache.delete_many({"barcode": barcode})
    db.product_cache.insert_one({
        "barcode": barcode,
        "product_name": "Stale Bad Product",
        "brands": "",
        "ingredients_text": "chicken, salt",
        "image_url": "",
        "source": "test",
        "cached_at": time.time(),
        "analysis": {
            "overall_score": 5,
            "upf_score": "0%",
            "processing_category": "Unknown",
            "harmful_ingredients": [],
            "beneficial_ingredients": [],
            "recommendation": "old error",
        },
    })


# ============= /api/scan/full =============

def test_full_scan_invalidates_stale_error_cache(mongo_db, auth_headers):
    """Cached error result must be invalidated; analysis should be re-attempted."""
    barcode = f"TEST_{uuid.uuid4().hex[:10]}"
    _seed_bad_cache(mongo_db, barcode)

    r = requests.post(f"{BASE_URL}/api/scan/full",
                      json={"barcode": barcode}, headers=auth_headers, timeout=60)
    # Endpoint may return 404 (not found in any DB after re-fetch) OR 200 with re-analysis
    assert r.status_code in (200, 404), f"unexpected {r.status_code} {r.text}"

    # Verify the stale analysis was removed from cache
    cached = mongo_db.product_cache.find_one({"barcode": barcode})
    if cached is not None:
        stale_analysis = cached.get("analysis")
        if stale_analysis is not None:
            # If analysis was re-cached, must NOT be the stale error version
            assert not (
                stale_analysis.get("overall_score") == 5
                and stale_analysis.get("upf_score") == "0%"
                and stale_analysis.get("processing_category", "").lower() == "unknown"
            ), "Stale error cache was NOT invalidated"

    # Response should not be the stale error either (200 case)
    if r.status_code == 200:
        analysis = (r.json() or {}).get("analysis") or {}
        if analysis:
            assert not (
                analysis.get("overall_score") == 5
                and analysis.get("upf_score") == "0%"
                and analysis.get("processing_category", "").lower() == "unknown"
            ), "Stale error result was served from cache"


# ============= /api/scan/quick =============

def test_quick_scan_invalidates_stale_error_cache(mongo_db, auth_headers):
    barcode = f"TEST_{uuid.uuid4().hex[:10]}"
    _seed_bad_cache(mongo_db, barcode)

    r = requests.post(f"{BASE_URL}/api/scan/quick",
                      json={"barcode": barcode}, headers=auth_headers, timeout=60)
    assert r.status_code in (200, 404), f"unexpected {r.status_code} {r.text}"

    cached = mongo_db.product_cache.find_one({"barcode": barcode})
    if cached is not None:
        stale = cached.get("analysis")
        if stale is not None:
            assert not (
                stale.get("overall_score") == 5
                and stale.get("upf_score") == "0%"
                and stale.get("processing_category", "").lower() == "unknown"
            ), "Stale error cache was NOT invalidated in quick scan"

    if r.status_code == 200:
        body = r.json() or {}
        analysis = body.get("analysis") or {}
        if analysis:
            assert not (
                analysis.get("overall_score") == 5
                and analysis.get("upf_score") == "0%"
                and analysis.get("processing_category", "").lower() == "unknown"
            ), "Quick scan served stale error result"


# ============= Don't-cache-errors =============

def test_failed_ai_analysis_is_not_cached(mongo_db, auth_headers):
    """After scan/full with no OPENAI key, AI will fail. product_cache should
    contain product data but no analysis (or an analysis flagged analysis_error=True
    that will be invalidated on next read)."""
    # Use a real (non-TEST_) barcode that OpenFoodFacts likely resolves so
    # we actually hit AI analysis path. Fallback: skip if the OFF lookup fails.
    barcode = "3017620422003"  # Nutella — known OFF entry
    mongo_db.product_cache.delete_one({"barcode": barcode})

    r = requests.post(f"{BASE_URL}/api/scan/full",
                      json={"barcode": barcode}, headers=auth_headers, timeout=90)
    if r.status_code == 403:
        pytest.skip("scan quota exhausted for this test user")
    assert r.status_code in (200, 404), f"unexpected {r.status_code} {r.text}"

    cached = mongo_db.product_cache.find_one({"barcode": barcode})
    if r.status_code == 404 or cached is None:
        pytest.skip("Product not resolvable from OFF/USDA — cannot test AI cache path")

    analysis = cached.get("analysis")
    # Either: (a) no analysis key persisted, or (b) analysis with analysis_error=True
    if analysis is not None:
        assert analysis.get("analysis_error") is True, \
            f"Failed AI analysis was cached without error flag: {analysis}"


def test_error_fallback_shape_in_response(mongo_db, auth_headers):
    """When AI fails, the response should carry overall_score=0, upf_score='Unknown',
    analysis_error=True — not the misleading 5/0%."""
    barcode = "3017620422003"
    mongo_db.product_cache.delete_one({"barcode": barcode})

    r = requests.post(f"{BASE_URL}/api/scan/full",
                      json={"barcode": barcode}, headers=auth_headers, timeout=90)
    if r.status_code == 403:
        pytest.skip("scan quota exhausted")
    if r.status_code != 200:
        pytest.skip(f"scan returned {r.status_code} — cannot inspect fallback")

    analysis = (r.json() or {}).get("analysis") or {}
    if not analysis.get("analysis_error"):
        pytest.skip("AI analysis actually succeeded — cannot test error fallback")

    assert analysis.get("overall_score") == 0, f"error score {analysis.get('overall_score')}"
    assert analysis.get("upf_score") == "Unknown", f"error upf {analysis.get('upf_score')}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
