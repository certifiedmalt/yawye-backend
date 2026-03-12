import asyncio, os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'
from emergentintegrations.llm.openai import OpenAITextToSpeech

async def gen():
    tts = OpenAITextToSpeech(api_key=os.environ['EMERGENT_LLM_KEY'])
    out = '/app/marketing/voiceovers/split'
    os.makedirs(out, exist_ok=True)

    lines = {
        "21a": "This red dye is in your sweets, your cereal, your kids snacks.",
        "21b": "It's banned in cosmetics. Too dangerous for your skin. But it's still in your food.",
        "21c": "You Are What You Eat scans any product and tells you the truth. Every chemical. Every risk.",
        "21d": "Scan yours free. You Are What You Eat. Link in bio.",

        "22a": "That bacon smells incredible right?",
        "22b": "The World Health Organisation classifies processed meat as a Group One carcinogen. The same category as tobacco and asbestos.",
        "22c": "You Are What You Eat shows you the carcinogens in your food. The score. The facts. The alternatives.",
        "22d": "Know what you eat. Download You Are What You Eat. Free.",
    }

    for name, text in lines.items():
        print(f'Generating {name}...')
        audio = await tts.generate_speech(text=text, model='tts-1-hd', voice='onyx', speed=1.0, response_format='mp3')
        with open(f'{out}/{name}.mp3', 'wb') as f:
            f.write(audio)
        print(f'  DONE ({len(audio)} bytes)')

asyncio.run(gen())
print("\nAll voiceovers for scripts 21-22 generated!")
