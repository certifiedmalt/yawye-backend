import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

print("Testing image-to-video with app screenshot as starting frame...")

video_bytes = video_gen.text_to_video(
    prompt="Camera slowly pulls back from this phone screen to reveal a parent in a bright supermarket holding the phone, with a toddler sitting in the shopping trolley next to them. The parent looks at the red score on the phone with a concerned frown and shakes their head at the toddler. Warm supermarket lighting, shallow depth of field, cinematic advertisement quality.",
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600,
    image_path="/app/marketing/app_screens/result_bad_4.png",
    mime_type="image/png"
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/test_image_to_video.mp4')
    print(f"SUCCESS! Saved ({len(video_bytes)} bytes)")
else:
    print("FAILED")
