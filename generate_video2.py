import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

prompt = """A confident young adult in a modern supermarket, holding up their smartphone to scan a food barcode. The phone screen shows a clear health app with green checkmarks and a health score. The person smiles with relief and understanding. Clean bright lighting, healthy fresh produce visible in the background. Cinematic style, 4K quality."""

print("🎬 Generating Clip 2: The Solution...")

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/yawye_ad_clip2_solution.mp4')
    print("✅ Clip 2 saved: /app/yawye_ad_clip2_solution.mp4")
else:
    print("❌ Clip 2 generation failed")
