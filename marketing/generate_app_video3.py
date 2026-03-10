import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

prompt = """Dramatic close-up overhead shot of a kitchen counter with a variety of packaged food products scattered around. A hand reaches in and places a smartphone in the center of frame. The phone screen illuminates showing a sleek dark-themed food analysis app with the title "You Are What You Eat" in white text and a green scan icon. The camera slowly pushes in toward the phone screen as the app dashboard becomes visible, showing health streak stats, a prominent green "Scan a Product" button, and dark card UI elements. Moody kitchen lighting with warm tones, shallow depth of field, professional advertisement quality, 4K cinematic, vertical format."""

print("Generating Video 3: App Dashboard Hero Shot...")

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/app_dashboard_hero.mp4')
    print("DONE: Video 3 saved to /app/marketing/app_dashboard_hero.mp4")
else:
    print("FAILED: Video 3 generation failed")
