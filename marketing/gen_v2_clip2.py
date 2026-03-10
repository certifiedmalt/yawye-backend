import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

# Video 2: Asian man scanning a healthy product - sees great result
prompt2 = """Cinematic shot of a young Asian man with short black hair wearing a casual t-shirt in a well-lit grocery store produce section. He holds his smartphone up to a granola bar package. The phone screen is clearly visible showing a dark black background health app. The screen shows a scan result: a large bright green circular ring score indicator with the number "9" in bold green text and "/10" below it in gray, a blue badge reading "Minimally Processed", and underneath dark gray rounded cards with green checkmark icons listing beneficial ingredients in white text like "Oats" and "Almonds". He smiles broadly, nods approvingly, and places the product into his shopping basket. Fresh vegetables and fruits visible in the background. Warm natural lighting, shallow depth of field on the phone screen, professional advertisement, 4K cinematic."""

print("Generating Video 2/4: Asian man - healthy scan result...")
video_bytes = video_gen.text_to_video(
    prompt=prompt2,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/ad_v2_clip2_man_healthy.mp4')
    print("DONE: Video 2 saved!")
else:
    print("FAILED: Video 2")
