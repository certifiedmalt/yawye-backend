import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

prompt = """Close-up cinematic shot of a person in a modern supermarket, holding up their smartphone to scan a barcode on a healthy organic granola bar product. The phone screen shows a dark-themed health scanning app with bright green corner brackets framing the barcode. The scan completes and transitions to show a large circular green health score "9/10" with text "Minimally Processed" on screen. The person smiles and confidently places the product in their shopping cart. Clean bright lighting, shallow depth of field on the phone. Professional advertisement quality, 4K cinematic, vertical format."""

print("Generating Video 2: App Score Reveal (Healthy)...")

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/app_score_reveal_healthy.mp4')
    print("DONE: Video 2 saved to /app/marketing/app_score_reveal_healthy.mp4")
else:
    print("FAILED: Video 2 generation failed")
