import asyncio
import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai import OpenAITextToSpeech

tts = OpenAITextToSpeech(api_key=os.environ['EMERGENT_LLM_KEY'])

scripts = {
    "01_upf_shock_test": "Thought this was healthy? Watch this. This bar looks healthy... but it's ultra-processed. Hidden sugars, additives, emulsifiers. But this one? Minimally processed. Score: nine out of ten. Scan your snacks before you buy. Download You Are What You Eat now.",
    
    "02_kids_snacks_audit": "Parents, you need to see this. I scanned my kids' favourite snacks. Two of them were ultra-processed. This one scored nine out of a hundred. But these swaps? So much better. Scan before you pack their lunch.",
    
    "03_upf_supermarket_challenge": "Find a cereal that isn't ultra-processed. Let's try this one... ultra-processed. This one? Ultra-processed. And this one? Finally! A decent score. Most cereals are ultra-processed. Scan your next shop.",
    
    "04_upf_vs_wholefood_swap": "This swap changed everything. Same price. Same aisle. But this yoghurt drink? Ultra-processed. Score: fourteen out of a hundred. And this whole yoghurt? Minimally processed. Score: seventy-eight. Make smarter swaps instantly.",
    
    "05_fridge_upf_score": "What's your fridge score? Let's find out. Ultra-processed. Ultra-processed. This one's good. Ultra-processed again. I scanned my whole fridge. Half of it was ultra-processed. Check your fridge.",
    
    "06_upf_breakfast_breakdown": "Your breakfast might be ultra-processed. Cereal? Ultra-processed. Toast spread? Ultra-processed. But this yoghurt? Good score. Small swaps make a huge difference. Scan your breakfast tomorrow.",
    
    "07_upf_on_a_budget": "You don't need to spend more to avoid ultra-processed food. Same price. Same shelf. One is ultra-processed. One isn't. It's not about cost. It's about ingredients. Scan before you buy.",
    
    "08_upf_label_lies": "High protein doesn't mean healthy. Watch this. This high-protein snack? Ultra-processed. Full of additives. Marketing tricks you. Scanning doesn't. See through the labels.",
    
    "09_upf_lunch_challenge": "Make a U-P-F free lunch in sixty seconds. Scan as you go. This? Ultra-processed. Swap it. This? Perfect. And this? Even better. Most lunchbox fillers are ultra-processed. But these swaps? Easy. Scan your lunch.",
    
    "10_upf_hidden_healthy": "This healthy smoothie shocked me. Looks clean, right? Scan it. Ultra-processed. But blend your own with real fruit? Score: ninety-two out of a hundred. U-P-F hides in foods you'd never expect. Scan before you sip."
}

async def generate_all():
    os.makedirs('/app/marketing/voiceovers', exist_ok=True)
    
    for name, text in scripts.items():
        print(f"Generating voiceover: {name}...")
        try:
            audio_bytes = await tts.generate_speech(
                text=text,
                model="tts-1-hd",
                voice="nova",
                speed=1.05,
                response_format="mp3"
            )
            filepath = f'/app/marketing/voiceovers/{name}.mp3'
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            print(f"  DONE: {filepath} ({len(audio_bytes)} bytes)")
        except Exception as e:
            print(f"  FAILED: {e}")

asyncio.run(generate_all())
print("\nAll voiceovers complete!")
