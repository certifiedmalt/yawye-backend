import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

# Video 4: White man comparing two products using the app
prompt4 = """Cinematic shot of a middle-aged white man with short brown hair and a beard in a supermarket aisle. He holds two packaged food products, one in each hand. He picks up his smartphone from the shopping cart and scans the first product. The phone screen clearly shows a dark black background app with a scan result: a large circular ring indicator glowing red with the number "3" and "/10" in white text, text reading "Ultra-Processed" in a blue badge, and dark cards with red "HIGH" severity badges. He shakes his head and puts that product down. He then scans the second product and the phone screen changes to show a bright green circular ring score with "8" and "/10", a blue badge reading "Minimally Processed", and dark cards with green checkmark icons. He smiles and places the second product in his cart. Bright supermarket lighting, eye-level camera angle, professional advertisement quality, 4K cinematic."""

print("Generating Video 4/4: White man - comparing two products...")
video_bytes = video_gen.text_to_video(
    prompt=prompt4,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/ad_v2_clip4_man_compare.mp4')
    print("DONE: Video 4 saved!")
else:
    print("FAILED: Video 4")
