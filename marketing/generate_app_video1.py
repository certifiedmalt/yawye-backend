import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

prompt = """Close-up cinematic shot of a person's hand holding a modern smartphone in a bright grocery store. The phone screen displays a sleek dark-themed health app with a large circular health score showing "3/10" in red, with text "Ultra-Processed" visible on screen. The person looks at the phone with a concerned expression, then slowly puts the food product back on the shelf. Shallow depth of field focuses on the phone screen. Warm supermarket lighting. Professional advertisement quality, 4K cinematic."""

print("Generating Video 1: App Score Reveal (Unhealthy)...")

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/app_score_reveal_unhealthy.mp4')
    print("DONE: Video 1 saved to /app/marketing/app_score_reveal_unhealthy.mp4")
else:
    print("FAILED: Video 1 generation failed")
