import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

# Video 1: Young Black woman scanning a product - sees unhealthy result
prompt1 = """Cinematic close-up of a young Black woman with natural curly hair in a bright modern supermarket. She holds up her iPhone to scan a cereal box barcode. The phone screen is clearly visible showing a dark near-black background app with bright green corner bracket guides framing the barcode area, white text below reads "Align barcode within the frame". The scan completes and the screen transitions to show a result page: dark black background, a large circular ring score indicator glowing red showing the number "2" with "/10" below it in white text, a blue pill-shaped badge reading "Ultra-Processed", and below that dark gray cards with red warning badges. She looks concerned, furrows her brow, and puts the cereal box back on the shelf. Shallow depth of field, warm supermarket lighting, professional advertisement quality, 4K cinematic."""

print("Generating Video 1/4: Young Black woman - unhealthy scan result...")
video_bytes = video_gen.text_to_video(
    prompt=prompt1,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/ad_v2_clip1_woman_unhealthy.mp4')
    print("DONE: Video 1 saved!")
else:
    print("FAILED: Video 1")
