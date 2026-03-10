import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

# Video 3: Latina woman browsing the app dashboard at home
prompt3 = """Cinematic medium shot of a young Latina woman with long dark hair sitting at a modern kitchen table with grocery bags around her. She looks at her smartphone which shows a dark near-black background app dashboard. The phone screen clearly displays: bold white text "Hello, Maria!" at the top, below that a dark gray rounded card with a green scan icon showing "5 Scans Remaining" and a gold star icon showing "Free" subscription status. Below that another dark gray card titled "Your Health Streak" with a small orange flame icon and "3 days" in white text. A large bright green rounded button labeled "Scan a Product" with a white barcode icon is prominently visible. She taps the green scan button and the screen transitions to the camera scanning view with green corner brackets. She picks up a product from the bag and holds it to the camera. Soft warm kitchen lighting, modern apartment setting, professional advertisement, 4K cinematic."""

print("Generating Video 3/4: Latina woman - app dashboard to scan...")
video_bytes = video_gen.text_to_video(
    prompt=prompt3,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/ad_v2_clip3_woman_dashboard.mp4')
    print("DONE: Video 3 saved!")
else:
    print("FAILED: Video 3")
