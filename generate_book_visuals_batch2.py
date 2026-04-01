import asyncio
import os
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration

api_key = os.environ.get("EMERGENT_LLM_KEY")
image_gen = OpenAIImageGeneration(api_key=api_key)

VISUALS = [
    {
        "slug": "upf_disease_timeline",
        "prompt": "A dual-axis line graph infographic on dark charcoal background. X-axis shows decades from 1960s to 2020s. Two rising lines track almost identically: Line 1 in amber/orange labeled 'UPF Market Share (% of household calories)' rising from about 10% to 55%. Line 2 in red labeled 'Chronic Disease Prevalence (obesity, T2 diabetes)' rising in near-perfect parallel. Key decade markers on the timeline with small labels. Title at top in white bold: 'THE CURVES THAT TELL THE TRUTH'. Subtitle: 'When the food changed, the health changed.' At bottom small text: 'Correlation is not causation. But when curves are this close, it demands explanation.' Clean minimalist data visualization style, no photographs."
    },
    {
        "slug": "childrens_calories_pie",
        "prompt": "A stark, impactful pie chart infographic on dark charcoal background. Large circle divided into two segments: 70% in deep alarming red labeled 'ULTRA-PROCESSED FOODS' and 30% in bright green labeled 'REAL FOOD'. A small silhouette of a child stands next to the pie chart for emotional scale. Title at top in white bold: 'WHAT YOUR CHILDREN EAT'. Below the chart in white text: 'Up to 70% of children daily calories now come from ultra-processed products. Their brains, hormones, and microbiomes are still developing.' Clean minimalist design, data visualization style for a book."
    },
    {
        "slug": "see_the_matrix_checklist",
        "prompt": "A clean, bold checklist poster infographic on dark charcoal background, designed to look like something you would stick on a fridge. Title at top in large white bold: 'SEE THE MATRIX — 10 RULES'. Ten numbered items with empty checkbox squares, each in clean white sans-serif text: 1. If it has a health claim, be suspicious. 2. If it has a long ingredient list, it is a formulation. 3. If it melts in your mouth, it is engineered. 4. If it is everywhere, it is engineered. 5. If it is cheap fast and hyper-palatable, it is designed for repeat purchase. 6. If it is marketed to children, it is a formulation. 7. If it is a drink with flavour, it is a chemical system. 8. If it is healthy but in a packet, be careful. 9. If it is hard to stop eating, it was designed that way. 10. Once you see it, you cannot unsee it. Clean typography, high contrast, fridge-magnet aesthetic."
    },
    {
        "slug": "label_comparison",
        "prompt": "A side-by-side comparison infographic on dark charcoal background. LEFT SIDE with green border shows 'REAL BREAD' with a simplified ingredient label showing: Wheat flour, water, salt, yeast. Below it annotation in green: '4 ingredients. All recognisable. Group 3.' RIGHT SIDE with red border shows 'SUPERMARKET BREAD' with a longer label: Wheat flour, water, yeast, salt, soya flour, vegetable oil rapeseed, emulsifiers E472e E481, flour treatment agent, preservative calcium propionate, dextrose. Red arrows pointing to specific ingredients labeled EMULSIFIER, INDUSTRIAL OIL, PRESERVATIVE. Below annotation in red: '12 ingredients. Group 4.' Title at top in white: 'THE LABEL THAT TELLS THE TRUTH'. Clean flat design, print quality."
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
