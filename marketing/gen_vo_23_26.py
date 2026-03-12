import asyncio, os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'
from emergentintegrations.llm.openai import OpenAITextToSpeech

async def gen():
    tts = OpenAITextToSpeech(api_key=os.environ['EMERGENT_LLM_KEY'])
    out = '/app/marketing/voiceovers/split'
    os.makedirs(out, exist_ok=True)

    lines = {
        "23a": "That fizzy drink looks refreshing right?",
        "23b": "Sodium benzoate plus vitamin C. When they mix they create benzene. A known carcinogen. Inside your bottle.",
        "23c": "You Are What You Eat shows you every chemical. Every risk. Every fact they dont want you to know.",
        "23d": "Scan yours free. You Are What You Eat.",

        "24a": "These white sweets look harmless enough.",
        "24b": "That white coating is titanium dioxide. Banned in food across the entire EU since twenty twenty two. Still in yours.",
        "24c": "You Are What You Eat tells you which chemicals are banned in other countries but still legal in your food.",
        "24d": "Scan yours. You Are What You Eat. Free download.",

        "25a": "Your kids morning cereal. Looks wholesome right?",
        "25b": "It contains BHT. The same chemical used in jet fuel and embalming fluid. In your breakfast.",
        "25c": "You Are What You Eat scans it and shows you everything. The score. The carcinogens. Healthier alternatives.",
        "25d": "Scan your breakfast. You Are What You Eat.",

        "26a": "Fresh bread. Nothing wrong with that surely.",
        "26b": "Potassium bromate. Banned in over thirty countries. Linked to cancer. Still in your bread.",
        "26c": "You Are What You Eat flags every banned chemical. Shows you which countries have outlawed whats in your food.",
        "26d": "Know what you eat. Download You Are What You Eat. Free.",
    }

    for name, text in lines.items():
        print(f'Generating {name}...')
        audio = await tts.generate_speech(text=text, model='tts-1-hd', voice='onyx', speed=1.0, response_format='mp3')
        with open(f'{out}/{name}.mp3', 'wb') as f:
            f.write(audio)
        print(f'  DONE ({len(audio)} bytes)')

asyncio.run(gen())
print("\nAll voiceovers for scripts 23-26 generated!")
