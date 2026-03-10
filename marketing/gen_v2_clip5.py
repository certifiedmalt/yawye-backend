import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])

prompt = """Cinematic shot of a gorgeous athletic young woman with long sun-kissed hair, glowing tan skin, wearing a trendy fitted crop top and high-waisted leggings after a gym workout. She walks into a premium health food store looking radiant and confident. She picks up a sleek protein bar package, holds up her iPhone and scans the barcode. The phone screen shows a dark black background app with a bright green circular score ring displaying "9/10" in bold green, a blue badge reading "Minimally Processed", and dark cards with green checkmarks. She gives a satisfied smile, flips her hair, drops the bar into her basket and walks toward the exit with an effortless confident stride. The camera captures her from a flattering angle. Beautiful golden hour sunlight floods through large windows, shallow depth of field, warm color grading. High-end fitness lifestyle brand advertisement, aspirational, 4K cinematic."""

print("Generating: Attractive fit woman using app...")
video_bytes = video_gen.text_to_video(
    prompt=prompt,
    model="sora-2",
    size="1280x720",
    duration=8,
    max_wait_time=600
)

if video_bytes:
    video_gen.save_video(video_bytes, '/app/marketing/ad_v2_clip5_fit_woman.mp4')
    print("DONE: Video saved!")
else:
    print("FAILED")
