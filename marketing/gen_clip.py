import os, sys
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'
from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])
script = sys.argv[1]
out_dir = '/app/marketing/production'
os.makedirs(out_dir, exist_ok=True)

prompts = {
    "16a": "Cinematic first-person POV shot. A hand picks up a smartphone from a table. The phone screen shows a dark near-black health app with bright emerald green accents and a green barcode leaf icon. The app dashboard shows bold white text Hello and a prominent bright green button labeled Scan a Product. The hand taps the green scan button. Modern apartment setting, natural light, 4K cinematic.",
    "16b": "Cinematic first-person POV shot. A hand holds a smartphone up to scan a protein bar barcode on a kitchen counter. The phone screen shows a dark near-black background health app with bright emerald green corner bracket guides framing the barcode. The scan completes and the screen shows a large circular progress ring in red with the number 3 in bold red, /10 in gray, and a blue badge reading Ultra-Processed. Dark gray cards with red warning badges appear below. Kitchen counter, natural lighting, 4K cinematic.",
    "16c": "Cinematic close-up of a smartphone screen on a dark near-black background showing a food analysis result. Dark gray rounded cards with red HIGH severity pill badges listing Artificial Sweeteners, Emulsifiers, and Modified Starch. A finger scrolls down through the ingredient warnings. Camera slowly pulls back to reveal a concerned face. Kitchen setting, warm lighting, 4K cinematic.",
    "16d": "Cinematic shot of a person at a kitchen counter looking directly at camera seriously. They hold up their phone showing a dark near-black health app with the emerald green You Are What You Eat branding visible and a bright green Scan a Product button. They point at the phone emphatically. Warm kitchen lighting, direct to camera, professional advertisement ending, 4K cinematic.",
    "17a": "Cinematic shot of a young man sitting next to his middle-aged mother on a couch at home. He shows her his smartphone screen displaying a dark near-black health app with bright emerald green accents and a green Scan a Product button. She looks curious and takes the phone. Warm cozy living room, soft lighting, heartwarming family moment, 4K cinematic.",
    "17b": "Cinematic kitchen shot. A middle-aged woman holds a phone up to scan a packet of biscuits from her cupboard. The phone screen shows a dark black background with a red circular progress ring, number 2, /10, Ultra-Processed badge. Her expression changes from curious to shocked. She puts a hand over her mouth. Her son nods knowingly. Warm kitchen lighting, 4K cinematic.",
    "17c": "Cinematic kitchen shot. A middle-aged woman pulls out a jar of cooking sauce and scans it. Phone shows dark background, red ring, number 3, /10, Ultra-Processed. She puts it down and stares at it. She scans a tin of soup. Red ring, number 4, /10. She looks at her son with wide eyes. Warm kitchen, family atmosphere, 4K cinematic.",
    "17d": "Cinematic warm shot in a supermarket. A mother and son shop together. She holds the phone and scans a product confidently. The screen shows a green ring, number 8, /10, Minimally Processed. She smiles proudly and puts it in the trolley. They walk together, both smiling. Warm supermarket lighting, heartwarming ending, 4K cinematic.",
    "18a": "Cinematic morning kitchen shot. A person stands at their kitchen counter with a cup of coffee in morning sunlight. They pick up their smartphone showing a dark-themed health scoring app with bright emerald green accents. They hold the phone up to a cereal box to scan its barcode. The phone screen displays a low score with a red warning indicator. They look at their breakfast with a surprised expression. Warm morning light, 4K cinematic.",
    "18b": "Cinematic supermarket shot. Text overlay reads Day 3. A person walks through aisles scanning multiple products rapidly. Phone flashes between red scores and Ultra-Processed badges. They look overwhelmed. Shopping basket is nearly empty. Bright supermarket lighting, 4K cinematic.",
    "18c": "Cinematic supermarket shot. Text overlay reads Day 5. A person confidently scans products before placing them in a full trolley. Phone shows green rings with scores of 7, 8, 9. They nod approvingly. Trolley fills with fresh produce and clean items. Bright lighting, confident energy, 4K cinematic.",
    "18d": "Cinematic kitchen shot. Text overlay reads Day 7. A person opens their fridge full of fresh colourful foods. They hold up their phone showing the dark app with a bright green score and health streak visible. They smile at camera, arms spread showing off the new fridge. Warm golden kitchen lighting, triumphant ending, 4K cinematic.",
    "19a": "Cinematic casual outdoor shot. Two male friends standing outside a supermarket. One pulls out his phone showing a dark near-black health app with bright emerald green accents. He challenges his friend who holds a shopping bag confidently. They walk into the store together. Natural daylight, friendly rivalry, 4K cinematic.",
    "19b": "Cinematic supermarket shot. One friend scans items from the others shopping bag. Phone shows dark background, red ring, number 2, /10, Ultra-Processed for a protein shake. Red ring, number 3 for rice cakes. The bag owners expression changes from confident to embarrassed. Bright supermarket lighting, comedic, 4K cinematic.",
    "19c": "Cinematic close-up of a phone screen showing a dark near-black background with a bright green circular ring, number 6, /10. Camera pulls back to show both friends staring at the single acceptable item among red-scored products. They look at each other. One item out of many. Supermarket lighting, 4K cinematic.",
    "19d": "Cinematic shot of both friends walking out of a supermarket. The embarrassed friend is looking at his own phone, downloading the app. The screen shows a dark themed app store page for You Are What You Eat with emerald green branding. He shows the download to his friend. They bump fists. Natural daylight, 4K cinematic.",
    "20a": "Cinematic supermarket shot. A person walks in holding their phone up determinedly. The phone shows a dark near-black health app. They scan the first product on the end cap. Red ring, Ultra-Processed. They shake their head and move on purposefully. Comedic energy, determined expression, bright lighting, 4K cinematic.",
    "20b": "Cinematic supermarket shot. A persons partner stands by the trolley looking impatient, checking their watch. The scanner person is three aisles away, scanning every item. Phone flashes between red scores. They put items back frantically. The partner calls out. Comedic timing, bright lighting, 4K cinematic.",
    "20c": "Cinematic supermarket checkout queue shot. A person is scanning items IN the queue. People behind look annoyed. Phone shows red ring, number 2, /10. They pull the item out of the trolley and hand it to a confused cashier. They scan the next item. Red again. More items come out. Comedic chaos, bright lighting, 4K cinematic.",
    "20d": "Cinematic home kitchen shot. A person opens their fridge perfectly organized with fresh colourful whole foods. They smile proudly. Their partner walks in, sees the fridge, looks impressed. The person holds up their phone showing the dark app with a bright green score. They both smile. Warm golden kitchen lighting, satisfying ending, 4K cinematic.",
}

prompt = prompts[script]
outfile = f"{out_dir}/{script}.mp4"
print(f"Generating clip {script}...")
try:
    video_bytes = video_gen.text_to_video(prompt=prompt, model="sora-2", size="1280x720", duration=8, max_wait_time=600)
    if video_bytes:
        video_gen.save_video(video_bytes, outfile)
        print(f"DONE: {outfile} ({len(video_bytes)} bytes)")
    else:
        print("FAILED: no bytes")
except Exception as e:
    print(f"FAILED: {e}")
