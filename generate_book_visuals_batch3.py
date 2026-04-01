import asyncio
import os
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration

api_key = os.environ.get("EMERGENT_LLM_KEY")
image_gen = OpenAIImageGeneration(api_key=api_key)

VISUALS = [
    {
        "slug": "30_day_calendar",
        "prompt": "A 30-day visual calendar planner infographic on dark charcoal background. Designed as a monthly grid (Monday to Sunday across the top, 5 rows of days). Four color-coded weekly bands: WEEK 1 (Days 1-7) in GREEN band labeled 'ONE REAL MEAL A DAY - Pick your anchor meal. Cook it every day.' WEEK 2 (Days 8-14) in AMBER band labeled 'SWAP THE BIGGEST TRIGGER - One swap. Big impact.' WEEK 3 (Days 15-21) in ORANGE band labeled 'FIX THE ENVIRONMENT - The Saturday Reset. Redesign your kitchen.' WEEK 4 (Days 22-30) in GOLD band labeled 'BUILD THE RHYTHM - Add a second real meal. Stabilise.' Each day cell has a small empty checkbox. Title at top in white bold: 'YOUR FIRST 30 DAYS'. Clean minimalist calendar design for a book, printable."
    },
    {
        "slug": "10_swaps_table",
        "prompt": "A bold two-column swap table infographic on dark charcoal background, designed to be stuck on a fridge. Title at top in large white bold: '10 SWAPS THAT CHANGE EVERYTHING'. Left column in red/orange with header 'INSTEAD OF' lists: Fizzy drink, Crisps, Chocolate bar, UPF cereal, Protein bar, Ready meal, UPF lunch, Energy drink, UPF dessert, Packaged snack. Right column in green with header 'TRY' lists matching alternatives: Sparkling water plus citrus, Nuts or seeds, Fruit plus dark chocolate, Eggs yoghurt or oats, Boiled eggs or cheese, Leftovers, Simple real-food plate, Water plus salt plus citrus, Dark chocolate or fruit, Anything with one ingredient. Large green arrows between columns. Bottom text: Not restriction. Upgrades. Clean bold typography, high contrast, practical poster design."
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
