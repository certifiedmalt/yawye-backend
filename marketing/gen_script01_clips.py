import os
os.environ['EMERGENT_LLM_KEY'] = 'sk-emergent-7562a399754063b92B'

from emergentintegrations.llm.openai.video_generation import OpenAIVideoGeneration

video_gen = OpenAIVideoGeneration(api_key=os.environ['EMERGENT_LLM_KEY'])
out_dir = '/app/marketing/script01_parts'
os.makedirs(out_dir, exist_ok=True)

clips = [
    {
        "file": f"{out_dir}/clip_a.mp4",
        "prompt": "Close-up cinematic shot in a bright modern supermarket. A young woman's hand reaches for a granola bar on the shelf. She picks it up and holds her iPhone up to the barcode. The phone screen shows a dark black health app with green corner bracket guides framing the barcode area. The scan completes and the screen flashes to show a large red circular score ring with the number 3 in bold red, a blue badge reading Ultra Processed, and dark cards with red warning icons. She looks at the screen with a shocked concerned expression. Warm supermarket lighting, shallow depth of field on the phone, professional advertisement quality, 4K cinematic."
    },
    {
        "file": f"{out_dir}/clip_b.mp4",
        "prompt": "Cinematic continuation shot in a bright supermarket. A young woman shakes her head and puts a granola bar back on the shelf. She picks up a different natural oat bar next to it and scans the barcode with her phone. The phone screen shows a dark black app with a bright green circular score ring displaying the number 9 in bold green text, a blue badge reading Minimally Processed, and dark cards with green checkmark icons listing healthy ingredients. She smiles broadly and nods approvingly. Warm lighting, shallow depth of field focused on phone screen, professional advertisement, 4K cinematic."
    },
    {
        "file": f"{out_dir}/clip_c.mp4",
        "prompt": "Cinematic shot of a confident young woman in a supermarket placing a natural oat bar into her shopping basket with a satisfied smile. Quick cut to a close-up of a smartphone screen showing a dark black health app with a bright green score ring and the text You Are What You Eat in white. The app logo glows. Camera slowly pulls back. Clean bright supermarket lighting, professional brand advertisement ending shot, 4K cinematic quality."
    }
]

for i, clip in enumerate(clips):
    label = chr(65 + i)
    print(f"Generating Clip {label}...")
    try:
        video_bytes = video_gen.text_to_video(
            prompt=clip["prompt"],
            model="sora-2",
            size="1280x720",
            duration=8,
            max_wait_time=600
        )
        if video_bytes:
            video_gen.save_video(video_bytes, clip["file"])
            print(f"  DONE: Clip {label} saved ({len(video_bytes)} bytes)")
        else:
            print(f"  FAILED: Clip {label} - no bytes returned")
    except Exception as e:
        print(f"  FAILED: Clip {label} - {e}")

print("\nAll clips for Script 1 generated!")
