import os
import sys
sys.path.insert(0, '/app/backend')

# Set the key directly
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

prompt = """A concerned young adult in a supermarket aisle, holding a packaged food product and reading the ingredients label with a confused, worried expression. The person squints at tiny text on the label. Surrounding shelves filled with colorful ultra-processed junk food packages. Soft dramatic lighting emphasizing their concern. Cinematic style, 4K quality."""

print("🎬 Generating Clip 1: The Problem...")
print("This may take 2-5 minutes...")

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/yawye_ad_clip1_problem.mp4')
    print("✅ Clip 1 saved: /app/yawye_ad_clip1_problem.mp4")
else:
    print("❌ Clip 1 generation failed")
