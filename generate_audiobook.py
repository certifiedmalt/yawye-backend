import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

# Selected sections for the audiobook preview
SECTIONS = [
    {
        "title": "OPENING QUOTES",
        "text": """You Are What You Eat.

"Coca-Cola dissolves rust, cleans toilets better than some detergents, and strips corrosion off car batteries. We call it a drink. We give it to children."

"The World Health Organisation classifies processed meats containing nitrates in the same carcinogenic risk group as smoking tobacco. One comes with tumour warnings. The other comes with a cartoon pig."

"Many ultra-processed foods contain additives also used in shampoo, floor polish, and industrial lubricants. The same chemicals that keep machinery running smoothly are used to keep your food soft, creamy, and shelf-stable."

"We are the first generation in history whose children are growing up on products engineered to bypass their biology. The consequences will last a lifetime."

"Children now get up to 70 percent of their calories from ultra-processed foods. Their brains, hormones, and microbiomes are still developing, and we're feeding them products engineered to hijack all three." """
    },
    {
        "title": "PROLOGUE",
        "text": """Prologue. The Experiment.

The story of modern food doesn't begin in a kitchen. It begins in a laboratory. Not the kind with chefs and chopping boards, the kind with centrifuges, solvents, industrial mixers, and white coats. The kind where food is not grown, cooked, or prepared, but engineered.

For most of human history, eating was simple. Food came from soil, sunlight, water, and time. Then, in the space of a single generation, everything changed.

Today, more than half of what we eat is not food in any traditional sense. It is ultra-processed, a category defined by the World Health Organisation's NOVA scale, the same system used to classify industrial edible substances designed for stability, shelf life, and profit.

And the most disturbing part? This shift didn't happen slowly. It didn't happen naturally. It didn't happen with your consent.

It happened quietly. It happened quickly."""
    },
    {
        "title": "INTRODUCTION",
        "text": """Introduction. The Cost of Convenience.

The modern world is getting sicker. Not in small ways, not in subtle ways, but in ways so widespread and so rapid that entire branches of medicine have had to rewrite themselves just to keep up.

In a single generation, rates of obesity, type 2 diabetes, fatty liver disease, and metabolic disorders have surged. Conditions once seen only in adults now appear in children. Diseases that used to take decades to develop are showing up earlier, faster, and more aggressively.

And all of this has happened in lockstep with one change: the rise of ultra-processed food.

This isn't coincidence. It's correlation stacked on correlation until it becomes impossible to ignore.

As real food disappeared from our plates, something else took its place. Products engineered in laboratories, optimised for profit, and designed to override the biological systems that kept humans healthy for thousands of years.

We didn't evolve to handle this. Our children certainly didn't."""
    },
    {
        "title": "INTRODUCTION PART 2",
        "text": """Human biology is ancient. Ultra-processed food is brand new. And the collision between the two is reshaping our health in real time.

Look at the timeline. As ultra-processed foods became cheaper, more available, more aggressively marketed, and more deeply embedded in daily life, chronic disease rose alongside them. Not gradually. Exponentially. The curve of UPF consumption and the curve of metabolic disease are almost identical.

We were told this was convenience. We were told this was progress. We were told this was harmless.

But convenience has a cost. And we are living inside the bill.

Ultra-processed foods are not just unhealthy. They are biologically disruptive. They alter hunger signals. They distort cravings. They change the microbiome, the control centre of immunity, metabolism, and even mood. They push blood sugar higher, faster, and more often. They create inflammation that quietly damages tissues over years. They encourage fat storage. They interfere with hormones. They reshape the reward pathways in the brain."""
    },
    {
        "title": "INTRODUCTION PART 3",
        "text": """And when these effects accumulate, day after day, year after year, the result is the world we see now. A population struggling with conditions that were once rare, now common. Once exceptional, now expected.

This isn't a failure of discipline. It's a failure of the food environment.

A food environment that rewards companies for engineering products people can't stop eating. A food environment where the cheapest calories are the most harmful. A food environment where children are targeted before they can read. A food environment where profit grows faster than public health can collapse.

You didn't choose this system. You were born into it. And the truth is simple: you cannot win a biological battle against a product designed to bypass your biology.

But you can understand it. You can see it clearly. And once you do, you can finally step outside it.

This book is not about blame. It's about exposure. It's about showing you the forces that shaped your cravings, your habits, your weight, your energy, your mood, and your children's development. Forces you were never meant to notice.

Because once you understand the problem, the solution becomes obvious. And once you see the system, you can no longer be controlled by it.

This is the beginning of taking back your biology. This is the beginning of taking back your health. This is the beginning of taking back your life."""
    },
    {
        "title": "CHAPTER 3 OPENING",
        "text": """Chapter Three. The Refinery. The Industrial Birth of Seed Oils.

If you could watch your food being made, you would never eat the same way again.

Most people imagine oil the way they imagine olive oil. Sunlight, olives, a stone press, a slow trickle of golden liquid. Seed oils are nothing like that. They don't drip from seeds. They don't squeeze out naturally. They don't exist in any meaningful quantity without heavy machinery, chemical solvents, and industrial heat. Seed oils are not pressed. They are extracted. And the process looks far more like a fuel refinery than a kitchen.

Corn. Soy. Rapeseed. Cottonseed. Sunflower. Safflower. These plants were never part of the human diet in oil form. For most of human history, they couldn't be, because you cannot get meaningful oil out of them without industrial intervention.

A sunflower seed contains a trace of oil. A corn kernel contains even less. To turn these into litres of oil, you need the same tools used in chemical manufacturing. High pressure rollers, solvent tanks, steam injectors, centrifuges, bleaching towers, deodorising columns. This is not cooking. This is industry."""
    },
    {
        "title": "CHAPTER 3 PROCESS",
        "text": """The process begins. Step one: Crushing. The seeds are not gently pressed. They are crushed, ground, and pulverised into a fine meal. The goal is simple: break the seed so aggressively that every microscopic droplet of oil becomes accessible. This is the first moment where the process stops resembling food.

Step two: Solvent Extraction. The ground seed meal is washed in hexane, a petroleum-derived solvent also used in industrial degreasing, glue production, and chemical extraction. Hexane dissolves the oil out of the seed. The mixture becomes a slurry, part food, part chemical bath. This is the moment most people would stop calling it food.

Step three: Desolventising. The hexane-oil mixture is heated so the solvent evaporates. The solvent is captured and reused. The oil remains, but it is dark, bitter, and full of impurities. It smells nothing like food. So the industry keeps going.

Step four: Degumming. The crude oil contains phospholipids, natural components of seeds. Industry calls them gums. They are removed using water, steam, or acid.

Step five: Neutralisation. Free fatty acids are removed using sodium hydroxide, the same chemical used in soap making. This reduces bitterness. It also removes nutrients. The oil is now chemically altered."""
    },
    {
        "title": "CHAPTER 3 BLEACHING AND COMPARISON",
        "text": """Step six: Bleaching. This is the moment the oil stops looking like food. The oil is heated and mixed with acid-activated bleaching clays, the same materials used in petroleum refining, wastewater treatment, and chemical purification. These clays pull out pigments, metals, breakdown products, and oxidation compounds. What was once brown becomes pale. What was once visibly industrial becomes visually clean. Not because it has become healthier, but because it has become more marketable.

Step seven: Deodorisation. The oil is heated again, this time to temperatures between 200 and 260 degrees Celsius. This removes the harsh, industrial smell created by the refining process. But high heat does something else. It breaks fatty acids. It destroys nutrients. It creates new compounds. It alters the structure of the oil itself. This is the moment the oil becomes shelf-stable, and biologically unfamiliar.

At this point, the process looks so industrial that it's worth stepping back and asking a simple question: What other products are made this way?

The clear, odourless oil on supermarket shelves is the result of crushing, solvent extraction, boiling, degumming, neutralisation, bleaching, and high-heat deodorisation. It is not the oil of a seed. It is the oil of a refinery. And it is now one of the most consumed substances in the modern diet. Not because it is healthy. Not because it is natural. But because it is profitable."""
    },
    {
        "title": "CHAPTER 3 MYTH VS TRUTH",
        "text": """The Myth versus the Truth, and why the truth is even more revealing.

There's a story that circulates online, dramatic, viral, and easy to repeat. Seed oils were originally made for livestock, but the animals kept dying, so they banned it for animals and fed it to humans instead.

It's a powerful story. It feels like something that should be true. But it isn't what happened. There is no historical record of livestock dying from seed oils, or governments banning seed oils in animal feed. It's a myth, compelling, but inaccurate. And the real story is actually more shocking.

The truth. Seed oils began as industrial waste. In the late 1800s to early 1900s, cottonseed oil was used for machinery lubrication, candles, soap, industrial greases, and lamp fuel. It was cheap, abundant, and considered unfit for human consumption.

Then chemists discovered they could harden it. Hydrogenation, a German industrial process, turned liquid cottonseed oil into a white, lard-like fat. This was the birth of Crisco in 1911. Not a food innovation. A chemical one."""
    },
    {
        "title": "CHAPTER 3 CLOSING",
        "text": """Procter and Gamble launched one of the most aggressive food marketing campaigns in history. Pure. Modern. Clean. Scientific. They reframed industrial oil as a health product.

And here is the livestock connection that is true. Livestock didn't die from seed oils. They did something else. They got fatter. Faster. Cheaper. Cottonseed meal, soybean meal, and corn by-products, all high in industrial polyunsaturated fatty acids, were used because they increased weight gain, reduced feed costs, and boosted profitability. This is historically documented. Animals didn't collapse. They grew.

The twist? The same industrial oils used to fatten livestock cheaply became the backbone of the modern human diet. Not because they were healthy. Not because they were traditional. Not because they were demanded. But because they were cheap, abundant, profitable, chemically malleable, and easy to market.

The myth says seed oils were banned for animals. The truth says they were perfect for fattening them. The myth says seed oils were fed to humans by accident. The truth says they were fed to humans by design. The myth says seed oils were a mistake. The truth says they were a business model.

And that business model still shapes the modern diet today."""
    }
]

async def generate_audiobook():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    
    for i, section in enumerate(SECTIONS):
        print(f"[{i+1}/{len(SECTIONS)}] Generating: {section['title']}...")
        text = section["text"].strip()
        
        # Split into chunks if needed (4096 char limit)
        chunks = []
        if len(text) <= 4000:
            chunks = [text]
        else:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > 4000:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        for j, chunk in enumerate(chunks):
            try:
                audio_bytes = await tts.generate_speech(
                    text=chunk,
                    model="tts-1-hd",
                    voice="onyx",
                    response_format="mp3",
                    speed=0.95
                )
                all_audio.extend(audio_bytes)
                print(f"  Chunk {j+1}/{len(chunks)} done ({len(audio_bytes)} bytes)")
            except Exception as e:
                print(f"  ERROR on chunk {j+1}: {e}")
    
    # Save the combined audio
    output_path = os.path.join(output_dir, "audiobook_preview.mp3")
    with open(output_path, "wb") as f:
        f.write(all_audio)
    
    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 1)
    print(f"\nAudiobook preview saved: {output_path} ({size_mb} MB)")

asyncio.run(generate_audiobook())
