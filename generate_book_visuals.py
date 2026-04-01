import asyncio
import os
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration

api_key = os.environ.get("EMERGENT_LLM_KEY")
image_gen = OpenAIImageGeneration(api_key=api_key)

VISUALS = [
    {
        "slug": "refinery_comparison",
        "prompt": "A clean, modern infographic on a dark charcoal background comparing two parallel industrial refining processes side by side. LEFT COLUMN titled 'VEGETABLE OIL REFINING' in amber text, showing 7 numbered steps flowing downward in amber/orange boxes: 1. Crush Seeds 2. Hexane Solvent Wash 3. Evaporate Solvent 4. Degum with Acid 5. Neutralise with Caustic Soda 6. Bleach with Clay 7. Deodorise at 260°C — ending with an arrow pointing to a cooking oil bottle. RIGHT COLUMN titled 'FUEL REFINING' in grey text, showing 7 matching steps in grey boxes: 1. Distil Crude Oil 2. Solvent Separation 3. Remove Impurities 4. Chemical Treatment 5. Neutralise Acids 6. Bleach/Filter 7. High-Heat Stabilise — ending with an arrow pointing to a fuel canister. Both columns use the SAME visual styling to emphasize their similarity. At the bottom in large white bold text: 'One is sold as food. The other is sold as fuel.' Minimalist, flat design, no photographs, clean sans-serif typography."
    },
    {
        "slug": "nova_classification",
        "prompt": "A modern infographic poster on a dark charcoal background showing the NOVA Food Classification System as 4 horizontal color-coded tiers stacked vertically. TOP TIER in bright green: 'GROUP 1 — UNPROCESSED' with small flat icons of an apple, egg, fish, and bag of rice. SECOND TIER in amber/yellow: 'GROUP 2 — CULINARY INGREDIENTS' with flat icons of olive oil bottle, butter, and salt shaker. THIRD TIER in orange: 'GROUP 3 — PROCESSED FOODS' with flat icons of cheese wheel, bread loaf, and tinned sardines. BOTTOM TIER in deep red, noticeably larger and bolder: 'GROUP 4 — ULTRA-PROCESSED' with flat icons of a soda can, crisp packet, and candy bar. A prominent white dashed line between Group 3 and Group 4 labeled 'THE LINE THAT MATTERS' in white bold text. Clean, authoritative, minimalist flat design for a book illustration."
    },
    {
        "slug": "fizzy_drink_anatomy",
        "prompt": "An exploded scientific diagram of a silver soda can on a dark charcoal background. The can is centered, slightly transparent. Seven labeled callout lines radiate outward from different parts of the can to colored info boxes arranged around it: 1. Red box: 'PHOSPHORIC ACID — also used in rust removers and industrial descalers' 2. Orange box: 'CITRIC ACID — also used in dishwasher tablets and limescale cleaners' 3. Amber box: 'SWEETNESS — 9+ teaspoons per can, triggers dopamine reward pathways' 4. Yellow box: 'CARBONATION — stimulates pain receptors, disguised as refreshment' 5. Brown box: 'CARAMEL COLOUR — industrial pigment, not kitchen caramel' 6. Purple box: 'FLAVOUR SYSTEM — dozens of synthetic aromatic compounds' 7. Grey box: 'PRESERVATIVES — sodium benzoate, potassium sorbate'. At the bottom in large white text: 'This is not a beverage. It is a chemical system.' Clinical, scientific diagram style, minimalist, print quality."
    },
    {
        "slug": "identity_pyramid",
        "prompt": "A clean geometric pyramid diagram on a dark charcoal background. Five horizontal layers stacked to form a pyramid shape, widest at bottom, narrowest at top. BOTTOM LAYER (largest, bright green): 'IDENTITY' with subtitle 'I am someone who eats real food'. SECOND LAYER (amber): 'ENVIRONMENT' with subtitle 'Make the good choice the easy choice'. THIRD LAYER (orange): 'ROUTINE' with subtitle 'Default meals, anchors, safety nets'. FOURTH LAYER (coral/salmon): 'BIOLOGY' with subtitle 'Stable energy, mood, hunger'. TOP LAYER (smallest, gold): 'MOMENTUM' with subtitle 'Small wins become big shifts'. Title above the pyramid: 'THE REAL-FOOD LIFE PYRAMID' in white bold text. Minimalist flat geometric design, clean sans-serif typography, book illustration print quality."
    }
]

async def generate_all():
    for v in VISUALS:
        print(f"Generating {v['slug']}...")
        try:
            images = await image_gen.generate_images(
                prompt=v["prompt"],
                model="gpt-image-1",
                number_of_images=1
            )
            if images and len(images) > 0:
                path = f"/app/marketing/book_visual_{v['slug']}.png"
                with open(path, "wb") as f:
                    f.write(images[0])
                size_kb = round(os.path.getsize(path) / 1024)
                print(f"  SUCCESS: {path} ({size_kb}KB)")
            else:
                print(f"  FAIL: No image returned for {v['slug']}")
        except Exception as e:
            print(f"  ERROR: {v['slug']}: {e}")

asyncio.run(generate_all())
print("Done!")
