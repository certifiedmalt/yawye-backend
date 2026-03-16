import requests, time, json

PROD_URL = "https://web-production-66c05.up.railway.app"

# Login
r = requests.post(f"{PROD_URL}/api/auth/login", json={"email":"jpsaila1986@gmail.com","password":"hello123"})
TOKEN = r.json().get("token","")
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}

# Top UK grocery barcodes
barcodes = [
    ("5449000000996", "Coca-Cola"),
    ("5449000131805", "Coca-Cola Zero"),
    ("5449000006004", "Diet Coke"),
    ("5000112546415", "Heinz Ketchup"),
    ("50184385", "Marmite"),
    ("5449000131812", "Fanta Orange"),
    ("5000112637922", "Heinz Baked Beans"),
    ("8711327370708", "Ben & Jerrys"),
    ("5000328520766", "Lucozade"),
    ("5000112547795", "Heinz Tomato Soup"),
    ("5000157024671", "Heinz Beans"),
    ("5010029215960", "Walkers Ready Salted"),
    ("5000128986724", "Cadbury Dairy Milk"),
    ("5000295142015", "McVities Digestive"),
    ("5449000054227", "Sprite"),
    ("5000328527680", "Ribena"),
    ("5010477348654", "PG Tips"),
    ("5000168001784", "Tetley Tea"),
    ("5000347052286", "Lurpak"),
    ("50201533", "Kelloggs Corn Flakes"),
    ("5010029211498", "Walkers Cheese Onion"),
    ("5000169150078", "Nescafe Gold"),
    ("8710398526892", "Knorr Stock"),
    ("5010029220780", "Walkers Salt Vinegar"),
    ("5000157149824", "HP Sauce"),
    ("5000108041828", "Lurpak Butter"),
    ("5060337502955", "Oatly Oat Milk"),
    ("5000436588340", "Cathedral City"),
    ("5010029208658", "Quavers"),
    ("5000168002859", "Yorkshire Tea"),
    ("5010081038297", "Warburtons"),
    ("5011546499253", "Innocent Smoothie"),
    ("5000169185315", "Nescafe Original"),
    ("5000169195000", "Kit Kat"),
    ("5000159459228", "Mars Bar"),
    ("5000159461177", "Snickers"),
    ("5000159484350", "Twix"),
    ("7622210449283", "Oreo"),
    ("5060166694258", "Monster Energy"),
    ("5060466510562", "Prime Hydration"),
    ("5000169411902", "Cheerios"),
    ("5000169422526", "Shreddies"),
    ("5000169186886", "Coco Pops"),
    ("5000169295601", "Special K"),
    ("5000232813350", "Tropicana OJ"),
    ("5000128076784", "Cadbury Buttons"),
    ("5000128953139", "Cadbury Creme Egg"),
    ("5000171060204", "PG Tips Pyramid"),
    ("5010029216004", "Walkers Prawn Cocktail"),
    ("5010029214116", "Walkers Sensations"),
    ("5000127599543", "Ribena Blackcurrant"),
    ("5000184302360", "Branston Pickle"),
    ("5010029221077", "Doritos"),
    ("5010044000350", "Hellmanns Mayo"),
    ("5000295152625", "Jaffa Cakes"),
    ("7613035087811", "Nescafe Dolce Gusto"),
    ("5000128587655", "Cadbury Wispa"),
    ("5000159407236", "Galaxy Chocolate"),
    ("5000159499132", "Maltesers"),
    ("5000159407243", "Galaxy Smooth Milk"),
    ("5010029216394", "Monster Munch"),
    ("50105860", "Weetabix"),
    ("5000169372838", "Shreddies Original"),
    ("7622300489434", "Cadbury Bournville"),
    ("5000159485876", "M&Ms Peanut"),
    ("5010029212440", "Sensations Thai"),
    ("5000168178936", "Twinings Earl Grey"),
    ("5000168139777", "Twinings English Breakfast"),
    ("5000169521489", "KitKat Chunky"),
    ("5000159418621", "Bounty"),
    ("5000159449656", "Milky Way"),
    ("5010029208061", "Wotsits"),
    ("5000159418638", "Topic"),
    ("5000159484343", "Twix White"),
    ("5010477336477", "PG Tips Original"),
    ("7613287356093", "SanPellegrino"),
    ("5449000006448", "Fanta Lemon"),
    ("5449000131829", "Coca-Cola Cherry"),
    ("5060166693947", "Monster Ultra"),
    ("5000189508286", "Red Bull"),
    ("5000169173428", "Shredded Wheat"),
    ("5010029200980", "Walkers Max"),
    ("5000169002520", "Cheerios Honey"),
    ("5000169445303", "Lucky Charms"),
    ("5010029012545", "Pringles Original"),
    ("5010029012552", "Pringles Sour Cream"),
    ("5053990101603", "Naked Smoothie"),
    ("5060337500357", "Oatly Barista"),
    ("5000169511688", "Fitnesse"),
    ("5000128046367", "Roses"),
    ("5000159540728", "Celebrations"),
    ("5000159409711", "Galaxy Ripple"),
]

success = 0
fail = 0
cached = 0

for i, (bc, name) in enumerate(barcodes):
    try:
        start = time.time()
        r = requests.post(f"{PROD_URL}/api/scan", json={"barcode": bc}, headers=headers, timeout=60)
        elapsed = time.time() - start
        
        if r.status_code == 200:
            d = r.json()
            src = d.get("source", "?")
            score = d.get("analysis", {}).get("overall_score", "?")
            if "cache" in str(src):
                cached += 1
                print(f"[{i+1}/{len(barcodes)}] CACHED: {name} - {elapsed:.1f}s")
            else:
                success += 1
                print(f"[{i+1}/{len(barcodes)}] OK: {name} Score:{score} - {elapsed:.1f}s via {src}")
        else:
            fail += 1
            print(f"[{i+1}/{len(barcodes)}] FAIL: {name} HTTP {r.status_code} - {elapsed:.1f}s")
    except Exception as e:
        fail += 1
        print(f"[{i+1}/{len(barcodes)}] ERROR: {name} - {str(e)[:50]}")
    
    # Small delay to avoid overwhelming the server
    time.sleep(0.5)

print(f"\n=== DONE: {success} new, {cached} already cached, {fail} failed out of {len(barcodes)} ===")
