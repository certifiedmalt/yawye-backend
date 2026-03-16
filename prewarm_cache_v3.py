#!/usr/bin/env python3
"""
Smart pre-warm v3: check cache first, then use AI prewarm for uncached products.
Uses separate HTTP sessions for scan vs prewarm to avoid connection pool issues.
"""
import requests, time, sys

PROD_URL = "https://web-production-66c05.up.railway.app"
KEY = "yawye2024clear"

# Create separate sessions
scan_session = requests.Session()
admin_session = requests.Session()

# Login
print("Logging in...")
r = requests.post(f"{PROD_URL}/api/auth/login", json={"email": "jpsaila1986@gmail.com", "password": "hello123"}, timeout=10)
TOKEN = r.json().get("token", "")
scan_session.headers.update({"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
admin_session.headers.update({"Content-Type": "application/json"})

# Check cache
try:
    cc = admin_session.get(f"{PROD_URL}/api/admin/cache_count?key={KEY}", timeout=10).json()
    print(f"Cache: {cc.get('cached_products', '?')} products")
except:
    pass

barcodes = [
    ("5449000000996", "Coca-Cola Original"),
    ("5449000131805", "Coca-Cola Zero Sugar"),
    ("5449000006004", "Diet Coke"),
    ("5449000131812", "Fanta Orange"),
    ("5449000054227", "Sprite"),
    ("5449000006448", "Fanta Lemon"),
    ("5449000131829", "Coca-Cola Cherry"),
    ("5000112637922", "Heinz Baked Beans"),
    ("5000112547795", "Heinz Cream of Tomato Soup"),
    ("5000157024671", "Heinz Beanz"),
    ("5000157149824", "HP Brown Sauce"),
    ("50184385", "Marmite Yeast Extract"),
    ("5000347052286", "Lurpak Spreadable Butter"),
    ("5000108041828", "Lurpak Danish Butter"),
    ("50201533", "Kelloggs Corn Flakes Cereal"),
    ("50105860", "Weetabix Wholegrain Cereal"),
    ("5000169372838", "Nestle Shreddies Original"),
    ("5000169411902", "Nestle Cheerios"),
    ("5000169422526", "Nestle Shreddies"),
    ("5000169186886", "Kelloggs Coco Pops"),
    ("5000169295601", "Kelloggs Special K"),
    ("5000169002520", "Nestle Cheerios Honey"),
    ("5000169173428", "Nestle Shredded Wheat"),
    ("5000128986724", "Cadbury Dairy Milk Chocolate"),
    ("5000128076784", "Cadbury Dairy Milk Buttons"),
    ("5000128587655", "Cadbury Wispa Chocolate Bar"),
    ("7622300489434", "Cadbury Bournville Dark Chocolate"),
    ("5000128953139", "Cadbury Creme Egg"),
    ("5000128046367", "Cadbury Roses Chocolates"),
    ("5000159459228", "Mars Bar Chocolate"),
    ("5000159461177", "Snickers Chocolate Bar"),
    ("5000159484350", "Twix Chocolate Bar"),
    ("5000159407236", "Galaxy Milk Chocolate"),
    ("5000159499132", "Maltesers Chocolate"),
    ("5000159485876", "M&Ms Peanut Chocolate"),
    ("5000159540728", "Celebrations Chocolates"),
    ("5000159418621", "Bounty Coconut Chocolate Bar"),
    ("5000159449656", "Milky Way Chocolate Bar"),
    ("5000169195000", "KitKat 4 Finger Chocolate Bar"),
    ("5000169521489", "KitKat Chunky Chocolate Bar"),
    ("7622210449283", "Oreo Original Cookies"),
    ("5000295142015", "McVities Digestive Biscuits"),
    ("5000295152625", "McVities Jaffa Cakes"),
    ("5010029215960", "Walkers Ready Salted Crisps"),
    ("5010029211498", "Walkers Cheese and Onion Crisps"),
    ("5010029220780", "Walkers Salt and Vinegar Crisps"),
    ("5010029216004", "Walkers Prawn Cocktail Crisps"),
    ("5010029214116", "Walkers Sensations Thai Sweet Chilli"),
    ("5010029208658", "Walkers Quavers Cheese Crisps"),
    ("5010029200980", "Walkers Max Strong Crisps"),
    ("5010029216394", "Walkers Monster Munch Pickled Onion"),
    ("5010029208061", "Walkers Wotsits Really Cheesy"),
    ("5010029221077", "Doritos Chilli Heatwave Tortilla Chips"),
    ("5010029012545", "Pringles Original Crisps"),
    ("5010029012552", "Pringles Sour Cream and Onion"),
    ("5010477348654", "PG Tips Original Tea Bags"),
    ("5000168001784", "Tetley Original Tea Bags"),
    ("5000168002859", "Yorkshire Tea Bags"),
    ("5000168178936", "Twinings Earl Grey Tea"),
    ("5000168139777", "Twinings English Breakfast Tea"),
    ("5000169169315", "Nescafe Gold Blend Coffee"),
    ("5000169185315", "Nescafe Original Instant Coffee"),
    ("5000232813350", "Tropicana Pure Orange Juice"),
    ("5000328520766", "Lucozade Energy Orange Drink"),
    ("5000127599543", "Ribena Blackcurrant Juice Drink"),
    ("5000328527680", "Ribena Original"),
    ("5011546499253", "Innocent Smoothie"),
    ("5053990101603", "Naked Green Machine Smoothie"),
    ("5060337502955", "Oatly Oat Milk Original"),
    ("5060337500357", "Oatly Barista Edition Oat Milk"),
    ("5060166694258", "Monster Energy Drink Original"),
    ("5060166693947", "Monster Energy Ultra Zero Sugar"),
    ("5000189508286", "Red Bull Energy Drink"),
    ("5060466510562", "Prime Hydration Drink"),
    ("5000436588340", "Cathedral City Mature Cheddar Cheese"),
    ("5010081038297", "Warburtons Toastie White Bread"),
    ("5000184302360", "Branston Original Pickle"),
    ("5010044000350", "Hellmanns Real Mayonnaise"),
    ("8711327370708", "Ben and Jerrys Cookie Dough Ice Cream"),
    ("7613035087811", "Nescafe Dolce Gusto Coffee Pods"),
    ("7613287356093", "SanPellegrino Sparkling Natural Mineral Water"),
    ("8710398526892", "Knorr Chicken Stock Pot"),
    ("5000169445303", "Lucky Charms Cereal"),
    ("5000169511688", "Nestle Fitness Cereal"),
    ("5000159407243", "Galaxy Smooth Milk Chocolate Bar"),
    ("5000159418638", "Mars Topic Chocolate Bar"),
    ("5000159484343", "Twix White Chocolate Bar"),
    ("5000159409711", "Galaxy Ripple Chocolate Bar"),
]

success = 0
cached = 0
ai_cached = 0
fail = 0
total = len(barcodes)

print(f"\nPre-warming {total} products...")
print("-" * 60)

for i, (bc, name) in enumerate(barcodes):
    try:
        # Step 1: Quick check if cached (3s timeout)
        try:
            r = scan_session.post(f"{PROD_URL}/api/scan", json={"barcode": bc}, timeout=3)
            if r.status_code == 200:
                d = r.json()
                src = d.get("source", "?")
                score = d.get("analysis", {}).get("overall_score", "?")
                cached += 1
                print(f"[{i+1}/{total}] CACHED: {name} score:{score}")
                continue
        except requests.exceptions.Timeout:
            pass  # Not cached, proceed to prewarm
        except Exception:
            pass

        # Step 2: Use AI prewarm endpoint
        time.sleep(1)  # Brief pause before AI call
        r2 = admin_session.post(
            f"{PROD_URL}/api/admin/prewarm?key={KEY}",
            json={"barcode": bc, "product_name": name},
            timeout=45
        )

        if r2.status_code == 200:
            d2 = r2.json()
            if d2.get("status") == "already_cached":
                cached += 1
                print(f"[{i+1}/{total}] CACHED: {name}")
            else:
                ai_cached += 1
                score = d2.get("score", "?")
                print(f"[{i+1}/{total}] AI: {name} score:{score}")
        else:
            fail += 1
            detail = ""
            try:
                detail = r2.json().get("detail", "")[:60]
            except:
                detail = r2.text[:60]
            print(f"[{i+1}/{total}] FAIL: {name} HTTP {r2.status_code} {detail}")

    except Exception as e:
        fail += 1
        print(f"[{i+1}/{total}] ERROR: {name} - {str(e)[:60]}")

    time.sleep(0.5)

print("-" * 60)
print(f"DONE: {ai_cached} AI-cached, {cached} already cached, {fail} failed / {total} total")

try:
    cc = admin_session.get(f"{PROD_URL}/api/admin/cache_count?key={KEY}", timeout=10).json()
    print(f"Final cache: {cc.get('cached_products', '?')} products")
except:
    pass
