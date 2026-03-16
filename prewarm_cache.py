#!/usr/bin/env python3
"""
Pre-warm product cache by scanning common UK grocery products.
Runs against Railway production with proper error handling and retry logic.
"""
import requests
import time
import sys

PROD_URL = "https://web-production-66c05.up.railway.app"
SCAN_TIMEOUT = 45  # seconds per scan
DELAY_BETWEEN_SCANS = 1  # seconds between scans to avoid overwhelming the server

# Login
print("Logging in...")
r = requests.post(f"{PROD_URL}/api/auth/login", json={"email": "jpsaila1986@gmail.com", "password": "hello123"}, timeout=10)
if r.status_code != 200:
    print(f"Login failed: {r.status_code}")
    sys.exit(1)
TOKEN = r.json().get("token", "")
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}

# Check current cache size
try:
    cache_r = requests.get(f"{PROD_URL}/api/admin/cache_count?key=yawye2024clear", timeout=10)
    print(f"Current cache size: {cache_r.json().get('cached_products', '?')} products")
except:
    print("Could not check cache size")

# Top UK grocery barcodes - organized by likelihood of being in Open Food Facts
barcodes = [
    ("5449000000996", "Coca-Cola"),
    ("5449000131805", "Coca-Cola Zero"),
    ("5449000006004", "Diet Coke"),
    ("5449000131812", "Fanta Orange"),
    ("5449000054227", "Sprite"),
    ("5449000006448", "Fanta Lemon"),
    ("5449000131829", "Coca-Cola Cherry"),
    ("5000112637922", "Heinz Baked Beans"),
    ("5000112547795", "Heinz Tomato Soup"),
    ("5000157024671", "Heinz Beans"),
    ("5000157149824", "HP Sauce"),
    ("50184385", "Marmite"),
    ("5000347052286", "Lurpak"),
    ("5000108041828", "Lurpak Butter"),
    ("50201533", "Kelloggs Corn Flakes"),
    ("50105860", "Weetabix"),
    ("5000169372838", "Shreddies Original"),
    ("5000169411902", "Cheerios"),
    ("5000169422526", "Shreddies"),
    ("5000169186886", "Coco Pops"),
    ("5000169295601", "Special K"),
    ("5000169002520", "Cheerios Honey"),
    ("5000169173428", "Shredded Wheat"),
    ("5000128986724", "Cadbury Dairy Milk"),
    ("5000128076784", "Cadbury Buttons"),
    ("5000128587655", "Cadbury Wispa"),
    ("7622300489434", "Cadbury Bournville"),
    ("5000128953139", "Cadbury Creme Egg"),
    ("5000128046367", "Roses"),
    ("5000159459228", "Mars Bar"),
    ("5000159461177", "Snickers"),
    ("5000159484350", "Twix"),
    ("5000159407236", "Galaxy Chocolate"),
    ("5000159499132", "Maltesers"),
    ("5000159485876", "M&Ms Peanut"),
    ("5000159540728", "Celebrations"),
    ("5000159418621", "Bounty"),
    ("5000159449656", "Milky Way"),
    ("5000169195000", "Kit Kat"),
    ("5000169521489", "KitKat Chunky"),
    ("7622210449283", "Oreo"),
    ("5000295142015", "McVities Digestive"),
    ("5000295152625", "Jaffa Cakes"),
    ("5010029215960", "Walkers Ready Salted"),
    ("5010029211498", "Walkers Cheese Onion"),
    ("5010029220780", "Walkers Salt Vinegar"),
    ("5010029216004", "Walkers Prawn Cocktail"),
    ("5010029214116", "Walkers Sensations"),
    ("5010029208658", "Quavers"),
    ("5010029200980", "Walkers Max"),
    ("5010029216394", "Monster Munch"),
    ("5010029208061", "Wotsits"),
    ("5010029221077", "Doritos"),
    ("5010029012545", "Pringles Original"),
    ("5010029012552", "Pringles Sour Cream"),
    ("5010029212440", "Sensations Thai"),
    ("5010477348654", "PG Tips"),
    ("5010477336477", "PG Tips Original"),
    ("5000168001784", "Tetley Tea"),
    ("5000168002859", "Yorkshire Tea"),
    ("5000168178936", "Twinings Earl Grey"),
    ("5000168139777", "Twinings English Breakfast"),
    ("5000169169315", "Nescafe Gold"),
    ("5000169185315", "Nescafe Original"),
    ("5000232813350", "Tropicana OJ"),
    ("5000328520766", "Lucozade"),
    ("5000127599543", "Ribena Blackcurrant"),
    ("5000328527680", "Ribena"),
    ("5011546499253", "Innocent Smoothie"),
    ("5053990101603", "Naked Smoothie"),
    ("5060337502955", "Oatly Oat Milk"),
    ("5060337500357", "Oatly Barista"),
    ("5060166694258", "Monster Energy"),
    ("5060166693947", "Monster Ultra"),
    ("5000189508286", "Red Bull"),
    ("5060466510562", "Prime Hydration"),
    ("5000436588340", "Cathedral City"),
    ("5010081038297", "Warburtons"),
    ("5000184302360", "Branston Pickle"),
    ("5010044000350", "Hellmanns Mayo"),
    ("8711327370708", "Ben & Jerrys"),
    ("7613035087811", "Nescafe Dolce Gusto"),
    ("7613287356093", "SanPellegrino"),
    ("8710398526892", "Knorr Stock"),
    ("5000169445303", "Lucky Charms"),
    ("5000169511688", "Fitnesse"),
    ("5000159407243", "Galaxy Smooth Milk"),
    ("5000159418638", "Topic"),
    ("5000159484343", "Twix White"),
    ("5000171060204", "PG Tips Pyramid"),
    ("5000159409711", "Galaxy Ripple"),
]

success = 0
cached = 0
fail = 0
not_found = 0
total = len(barcodes)

print(f"\nStarting cache pre-warm for {total} products...")
print(f"Scan timeout: {SCAN_TIMEOUT}s, delay between scans: {DELAY_BETWEEN_SCANS}s")
print("-" * 60)

for i, (bc, name) in enumerate(barcodes):
    try:
        start = time.time()
        r = requests.post(
            f"{PROD_URL}/api/scan",
            json={"barcode": bc},
            headers=headers,
            timeout=SCAN_TIMEOUT
        )
        elapsed = time.time() - start

        if r.status_code == 200:
            d = r.json()
            src = d.get("source", "?")
            score = d.get("analysis", {}).get("overall_score", "?")
            if "cache" in str(src):
                cached += 1
                print(f"[{i+1}/{total}] CACHED: {name} score:{score} ({elapsed:.1f}s)")
            else:
                success += 1
                print(f"[{i+1}/{total}] NEW: {name} score:{score} via {src} ({elapsed:.1f}s)")
        elif r.status_code == 404:
            not_found += 1
            print(f"[{i+1}/{total}] NOT FOUND: {name} ({elapsed:.1f}s)")
        elif r.status_code == 403:
            print(f"[{i+1}/{total}] SCAN LIMIT: Need premium account")
            break
        else:
            fail += 1
            detail = ""
            try:
                detail = r.json().get("detail", "")[:80]
            except:
                detail = r.text[:80]
            print(f"[{i+1}/{total}] FAIL: {name} HTTP {r.status_code} {detail} ({elapsed:.1f}s)")

    except requests.exceptions.Timeout:
        fail += 1
        print(f"[{i+1}/{total}] TIMEOUT: {name} (>{SCAN_TIMEOUT}s)")
    except Exception as e:
        fail += 1
        print(f"[{i+1}/{total}] ERROR: {name} - {str(e)[:60]}")

    # Delay between scans
    if i < total - 1:
        time.sleep(DELAY_BETWEEN_SCANS)

print("-" * 60)
print(f"DONE: {success} new cached, {cached} already cached, {not_found} not found, {fail} failed out of {total}")

# Check final cache size
try:
    cache_r = requests.get(f"{PROD_URL}/api/admin/cache_count?key=yawye2024clear", timeout=10)
    print(f"Final cache size: {cache_r.json().get('cached_products', '?')} products")
except:
    pass
