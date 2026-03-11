import asyncio, os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'
from emergentintegrations.llm.openai import OpenAITextToSpeech

async def gen():
    tts = OpenAITextToSpeech(api_key=os.environ['EMERGENT_LLM_KEY'])
    out = '/app/marketing/voiceovers/split'
    os.makedirs(out, exist_ok=True)

    # Each script has 3 lines timed to clips A (0-8s), B (8-16s), C (16-24s)
    lines = {
        "01a": "What's your fridge score? Let's find out.",
        "01b": "Ultra-processed. Ultra-processed. This one's good. Ultra-processed again.",
        "01c": "Half my fridge was ultra-processed. Check yours.",

        "02a": "High protein. Clean label. Let's scan it. Ultra-processed.",
        "02b": "Try another. Ultra-processed. And this one? Ultra-processed. All of them.",
        "02c": "But this one? Eight out of ten. Minimally processed. Scan before you shake.",

        "03a": "Date night dinner shop. Let's scan this pasta sauce. Ultra-processed.",
        "03b": "Try another. Still ultra-processed. Everything is failing.",
        "03c": "Fresh ingredients? Nine out of ten. Now we're cooking. Scan your dinner tonight.",

        "07a": "Your breakfast might be ultra-processed. Cereal? Three out of ten. Ultra-processed.",
        "07b": "Toast spread? Ultra-processed. Orange juice? Ultra-processed. Your entire breakfast failed.",
        "07c": "Small swaps make a huge difference. Scan your breakfast tomorrow.",

        "09a": "Hey guys, I'm obsessed with this clean smoothie. So good for you.",
        "09b": "Scan it. Ultra-processed. Two out of ten. Awkward.",
        "09c": "Blend real fruit instead. Nine out of ten. Stop trusting labels. Start scanning.",

        "12a": "Parents, you need to see this. Scanning my kid's favourite snack. Ultra-processed.",
        "12b": "They grabbed another one. Scan it. Ultra-processed again.",
        "12c": "Finally, this one scores eight out of ten. Minimally processed. Scan before you pack their lunch.",

        "13a": "Meal deal for lunch. Let's scan the sandwich. Two out of ten. Ultra-processed.",
        "13b": "Crisps? Ultra-processed. Drink? Ultra-processed. Every single item failed.",
        "13c": "Swapped for a salad. Seven out of ten. Much better. Scan before you buy.",

        "14a": "My PT recommended this protein bar. Let's scan it. Ultra-processed.",
        "14b": "He tried another. Ultra-processed. And his shake? Ultra-processed. Every one.",
        "14c": "I pulled out mine. Eight out of ten. Minimally processed. See through the labels.",
    }

    for name, text in lines.items():
        print(f'Generating {name}...')
        audio = await tts.generate_speech(text=text, model='tts-1-hd', voice='onyx', speed=1.0, response_format='mp3')
        with open(f'{out}/{name}.mp3', 'wb') as f:
            f.write(audio)
        print(f'  DONE ({len(audio)} bytes)')

asyncio.run(gen())
print("\nAll split voiceovers generated!")
