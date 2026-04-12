import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

COMBINED_BOOK = [

    # ===== OPENING CREDITS =====
    """You Are What You Eat.

... ... ...

Written by Jason Psaila.

... ... ...

This audiobook is a companion to the You Are What You Eat app. ... Available on Google Play and the App Store.

... ... ...""",

    # ===== PERSONAL NOTE FROM AUTHOR =====
    """A note from the author.

... ... ...

My name is Jason Psaila. I wrote this book because I got tired of being confused.

... Confused by contradictory headlines. Confused by food labels that made no sense. Confused by an industry that seemed to be hiding in plain sight.

... So I spent years reading the research. ... Talking to the experts. ... And building an app that turns that research into something you can use every single day.

... This book is what I wish someone had given me years ago. ... Not rules. Not guilt. ... Just clarity.

... I hope it does the same for you.

... ... ...""",

    # ===== V1: OPENING QUOTES (dramatic hook) =====
    """... ... ...

"Coca-Cola dissolves rust... cleans toilets better than some detergents... and strips corrosion off car batteries."

... We call it a drink. ... We give it to children.

... ... ...

"The World Health Organisation classifies processed meats containing nitrates... in the same carcinogenic risk group... as smoking tobacco."

... One comes with tumour warnings. ... The other comes with a cartoon pig.

... ... ...

"Many ultra-processed foods contain additives also used in shampoo, floor polish, and industrial lubricants. The same chemicals that keep machinery running smoothly... are used to keep your food soft, creamy, and shelf-stable."

... ... ...

"We are the first generation in history... whose children are growing up on products engineered to bypass their biology."

... The consequences will last a lifetime.

... ... ...

"Children now get up to seventy percent of their calories from ultra-processed foods. Their brains, hormones, and microbiomes are still developing..."

... and we're feeding them products engineered to hijack all three.""",

    # ===== V1: PROLOGUE =====
    """... ... ...

Prologue. ... The Experiment.

... ... ...

The story of modern food doesn't begin in a kitchen.

... It begins in a laboratory.

Not the kind with chefs and chopping boards... the kind with centrifuges, solvents, industrial mixers, and white coats. The kind where food is not grown, cooked, or prepared...

... but engineered.

... ... ...

For most of human history, eating was simple. Food came from soil, sunlight, water, and time.

... Then, in the space of a single generation... everything changed.

... ... ...

Today, more than half of what we eat is not food in any traditional sense. It is ultra-processed... a category defined by the World Health Organisation's NOVA scale... the same system used to classify industrial edible substances designed for stability, shelf life, and profit.

And the most disturbing part?

... This shift didn't happen slowly. It didn't happen naturally. It didn't happen with your consent.

... It happened quietly. ... It happened quickly.""",

    # ===== V2: INTRODUCTION =====
    """... ... ...

Introduction.

... This book is about clarity. For decades, nutrition has been confusing, contradictory, and overwhelming. But over the last six years, something remarkable has happened... the science has converged.

Across hundreds of studies... dozens of countries... and millions of participants... one message is unmistakable:

... Ultra-processed food... is the single most important dietary factor... driving modern disease.

... ... ...

This book is not about guilt, restriction, or perfection. ... It is about understanding how UPFs affect the body and mind... and learning simple, sustainable ways to reduce them.

You will not find calorie charts, complicated rules, or moral judgments here. ... You will find clarity... structure... and a system that works for real families with real lives.

And you will find a companion... the app that turns this book into daily action. Together, they form a blueprint for a healthier, calmer, more energised life.

... This is not a diet. ... It is a return to food that looks like food.

... ... ...""",

    # ===== V2: CHAPTER 1 - THE 10 KEY STUDIES =====
    """Chapter One. ... The New Science of Ultra-Processed Food. ... What the Last Six Years Finally Made Clear.

... ... ...

Ultra-processed food isn't a trend, a buzzword, or a moral panic. Over the last six years, the science has hardened into something much more powerful... a global, cross-disciplinary consensus.

Nutrition researchers... metabolic scientists... epidemiologists... mental health specialists... and policy bodies... have all converged on the same conclusion:

... Ultra-processed food is not just food with additives. ... It is a category of products that reliably predicts worse health outcomes... across almost every system of the body.

... ... ...

This chapter distils the ten most important studies from 2020 to 2026.

... Study one... The Lancet UPF Series, 2025... The Turning Point.
Study two... The 2024 Umbrella Review... UPF and Metabolic Disease.
Study three... Nature Medicine, 2025... A Framework for UPF Harm.
Study four... Cardiometabolic Meta-Analysis, 2023... Quantifying the Damage.
Study five... Clinical Nutrition Umbrella Review, 2024... 49 Health Outcomes.
Study six... Global Burden of Disease, 2022... UPF as a Population-Level Threat.
Study seven... NOVA Validation Studies, 2020 to 2025... The Category Is Real.
Study eight... WHO Sugar-Sweetened Beverage Tax Evidence, 2022... Policy That Works.
Study nine... Mental Health Meta-Analyses, 2023 to 2024... The Brain to Food Connection.
Study ten... Children's Health Evidence, 2024... A Wake-Up Call.

... ... ...

Across all ten studies... three themes emerge:

... One. ... UPF is a category with independent harm.
... Two. ... The mechanisms are now understood.
... Three. ... The evidence is strong enough for policy.

... Key takeaway... This is no longer a debate. The science is settled. Ultra-processed food is harmful... and the evidence is now strong enough to act on.

... ... ...""",

    # ===== V2: CHAPTER 2 - FIVE MECHANISMS =====
    """Chapter Two. ... The Five Mechanisms of UPF Harm.

... ... ...

If Chapter One established that UPF is harmful... this chapter explains why. The last six years of research have clarified five core mechanisms that make UPFs fundamentally different from whole foods... not just nutritionally... but biologically and behaviourally.

... ... ...

Mechanism One... Food Matrix Destruction.
... Whole foods have structure. UPFs do not. When the natural matrix is destroyed, digestion becomes faster, satiety signals weaken, and glucose and lipids hit the bloodstream harder and faster. This drives overeating... and metabolic stress.

... ... ...

Mechanism Two... Hyper-Palatability Engineering.
... UPFs are engineered to be irresistible. Manufacturers combine sugar, fat, salt, flavour enhancers, and emulsifiers in ratios rarely found in nature. This creates bliss point foods... that override normal appetite regulation.

... ... ...

Mechanism Three... Additives and Emulsifiers.
... Emulsifiers, stabilisers, artificial sweeteners, and colourants... their chronic, combined, long-term effects are now linked to inflammation... microbiome disruption... and impaired gut barrier function.

... ... ...

Mechanism Four... Packaging and Endocrine Disruptors.
... Plastics, liners, and heat-sealed packaging can leach chemicals like phthalates and bisphenols into food. These compounds interfere with hormones... metabolism... and appetite regulation.

... ... ...

Mechanism Five... Rapid Glycaemic Load.
... UPFs often deliver carbohydrates in their most rapidly absorbable form. This causes glucose spikes... insulin surges... and subsequent crashes that drive hunger, cravings, and fat storage.

... ... ...

Key takeaway... The harm from UPFs is not just about what's in them. It's about what they do to your body. The damage is structural... behavioural... and biological.

... ... ...

Now that you understand the mechanisms... let's see them in action. Starting with one of the most disturbing examples in the modern food system.

... ... ...""",

    # ===== V1: CHAPTER 3 - THE REFINERY (seed oil deep-dive) =====
    """Chapter Three. ... The Refinery. ... The Industrial Birth of Seed Oils.

... ... ...

If you could watch your food being made... you would never eat the same way again.

... ... ...

Most people imagine oil the way they imagine olive oil. Sunlight. Olives. A stone press. A slow trickle of golden liquid.

... Seed oils are nothing like that.

They don't drip from seeds. They don't squeeze out naturally. They don't exist in any meaningful quantity... without heavy machinery, chemical solvents, and industrial heat.

... Seed oils are not pressed. ... They are extracted.

... And the process looks far more like a fuel refinery... than a kitchen.

... ... ...

The process begins.

... Step one. ... Crushing. The seeds are crushed... ground... and pulverised into a fine meal.

... Step two. ... Solvent Extraction. The ground seed meal is washed... in hexane. A petroleum-derived solvent. The same chemical used in industrial degreasing.

... Step three. ... Desolventising. The hexane-oil mixture is heated so the solvent evaporates.

... Step four. ... Degumming. Phospholipids are removed using water, steam, or acid.

... Step five. ... Neutralisation. Free fatty acids are removed using sodium hydroxide. The same chemical used in soap making.

... Step six. ... Bleaching. The oil is heated and mixed with acid-activated bleaching clays. The same materials used in petroleum refining and wastewater treatment.

... Step seven. ... Deodorisation. The oil is heated to two hundred and sixty degrees Celsius. This removes the industrial smell... but breaks fatty acids and destroys nutrients.

... ... ...

Vegetable oil refining. Crush. Extract with hexane. Evaporate solvent. Degum. Neutralise with caustic soda. Bleach. Deodorise at two hundred and sixty degrees. Bottle and sell as food.

... Fuel refining. Distil crude oil. Use solvents to separate fractions. Remove impurities. Neutralise acids. Bleach or filter. Heat to high temperatures. Blend and store as fuel.

... The clear, odourless oil on supermarket shelves... is the oil of a refinery.

... ... ...

Key takeaway... The oil in your kitchen went through the same industrial process as petroleum. Not because it's healthy. Because it's profitable.

... ... ...""",

    # ===== V1: CHAPTER 4 - THE ADDITIVE EXPLOSION =====
    """Chapter Four. ... The Additive Explosion.

... ... ...

If Chapter Three showed you how modern oils are born in refineries... this chapter shows you what happens next. When those oils become the foundation for a new kind of product.

... Not food. ... Not ingredients. ... But formulations.

... ... ...

Ultra-processed foods are not built the way meals are built. They are assembled... the way products are assembled. From components, stabilisers, emulsifiers, colours, and chemicals.

... ... ...

Emulsifiers are the glue of the ultra-processed food world. They make oil and water mix. They make sauces smooth. They make ice cream soft. They make bread stay soft... for weeks.

... Common emulsifiers include lecithins... mono and diglycerides... polysorbates... carboxymethylcellulose... carrageenan.

... These are not kitchen ingredients. ... They are industrial tools.

... ... ...

Sweetness used to be rare. ... Now it is engineered. A strawberry yoghurt may contain no strawberries. A vanilla drink may contain no vanilla. A cheese puff may contain no cheese.

... Flavours are designed to mimic nature... and then surpass it.

... ... ...

Preservatives are the reason ultra-processed foods last weeks on shelves... months in warehouses... years in storage. They stop mould. They stop bacteria. They stop decay.

... Real food spoils. ... Ultra-processed food doesn't. ... That alone should tell you something.

... ... ...

Key takeaway... Every additive solves a business problem. How do we make this cheaper? How do we make this last longer? How do we make this more addictive? None of them solve a health problem.

... ... ...""",

    # ===== V1: CHAPTER 5 - THE FIZZY DRINK =====
    """Chapter Five. ... The Fizzy Drink. ... The Most Engineered Product in the Modern Diet.

... ... ...

The UK now spends more than twenty-one billion pounds a year on soft drinks. These aren't beverages. They're chemical formulations.

... A fizzy drink is not made. ... It is assembled.

... ... ...

Number one. ... The acid. Phosphoric acid in colas. Citric acid in fruit sodas. These same acids appear in rust removers, dishwasher tablets, and industrial descalers.

Number two. ... The sweetness. A single can contains more sugar than most people would ever add to anything they make at home. Sweetness is one of the most powerful reward triggers in the human brain. ... Drink. Dopamine. Repeat.

Number three. ... The colour. Caramel colour in colas is not caramel from a pan. It is industrial caramel. Without colour, the drink would look like chemical water.

Number four. ... The flavour. Flavour is a formula, often containing dozens of compounds. ... It is not taste. It is engineering.

Number five. ... The carbonation. CO2 triggers pain receptors in the mouth. Without carbonation, the drink is syrup. With carbonation... it is addictive.

Number six. ... The preservatives. Under certain conditions, sodium benzoate plus vitamin C can form benzene in trace amounts. Fizzy drinks behave like chemical systems... because they are chemical systems.

... ... ...

Fizzy drinks do not nourish. They stimulate. They do not hydrate. They override. They do not satisfy. They provoke.

... They are the perfect example of a product designed not for health... but for consumption.

... Key takeaway... A fizzy drink is not a beverage. It is a behavioural product... engineered to be consumed quickly, repeatedly, and in large quantities. Especially by children.

... ... ...

Now that you've seen what's inside these products... let's talk about what they do to your brain.

... ... ...""",

    # ===== V2: CHAPTER 3 - HOW UPF HIJACKS BEHAVIOUR =====
    """Chapter Six. ... How Ultra-Processed Food Hijacks Behaviour.

... ... ...

UPFs don't just affect the body. ... They affect the brain. The last six years of research have shown that UPFs exploit the same neural pathways involved in reward... habit formation... and compulsive behaviour.

... ... ...

The Dopamine Loop. ... Hyper-palatable foods trigger rapid dopamine spikes. Over time, the brain adapts, requiring more stimulation to achieve the same reward.

Satiety Signal Disruption. ... UPFs bypass the mechanical and chemical signals that whole foods trigger. Without fibre, structure, or slow digestion... the body struggles to register fullness.

Craving Amplification. ... Rapid glucose spikes followed by crashes create a physiological craving cycle. The body seeks fast energy... and UPFs deliver it... temporarily.

Habit Formation. ... UPFs are cheap, convenient, and everywhere. This environmental saturation makes them easy to default to.

Emotional Eating. ... UPFs provide immediate sensory reward. In moments of stress, boredom, or low mood, they offer a quick dopamine hit... reinforcing the loop.

... ... ...

Key takeaway... UPF is not a willpower issue. It is a design issue. These foods are engineered to be eaten quickly, repeatedly, and in large quantities. Understanding this frees you from blame.

... ... ...""",

    # ===== V2: CHAPTER 8 - THE SCIENCE OF CRAVINGS =====
    """Chapter Seven. ... The Science of Cravings. ... Why Your Body Wants What It Doesn't Need.

... ... ...

Cravings are not a character flaw. ... They are chemistry. And ultra-processed foods are engineered to exploit that chemistry... with precision.

... The Dopamine Spike. UPFs deliver rapid sensory reward... triggering dopamine release. Over time, the brain adapts. This is why cravings intensify... not fade.

... The Blood Sugar Whiplash. UPFs spike glucose... spike insulin... then cause a crash. The crash triggers hunger and a drive for fast energy... usually more UPF.

... The Gut Brain Loop. Diets high in UPF promote bacteria that thrive on sugar. These microbes send signals that influence cravings... mood... and appetite.

... The Stress Connection. Cortisol increases appetite for high-reward foods. UPFs provide immediate relief... reinforcing the stress-craving-UPF loop.

... The Habit Layer. Cravings are not just biological. They are contextual. Time of day, location, mood, and routine all trigger learned associations.

... ... ...

Key takeaway... Cravings are predictable, understandable, and manageable. You can break the loop... not through willpower... but through strategy.

... ... ...""",

    # ===== V1: CHAPTERS 6+7 - MODERN ILLNESS + CHILDREN'S CRISIS =====
    """Chapter Eight. ... The Human Cost.

... ... ...

The modern food economy exploded. ... And something else exploded with it.

Over the same decades that ultra-processed foods became dominant, the world saw dramatic increases in... obesity. Type 2 diabetes. Fatty liver disease. Chronic inflammation. Cardiovascular issues.

... No single food caused these conditions. But the system changed. And the body responded.

... ... ...

Not every consequence shows up in a diagnosis. Most show up in everyday life.

Fatigue. Cravings. Mood swings. Low energy. Poor sleep. Difficulty concentrating. Constant hunger. Afternoon crashes. Brain fog.

... These are not character flaws. They are biological signals. Signals that the body is trying to cope with a diet it was never designed for.

... ... ...

And nowhere is this more urgent... than with our children.

... ... ...

Children are not miniature adults. They are developing systems. Fragile. Sensitive. Rapidly changing.

For the first time in human history, children are growing up in a food system where ultra-processed foods dominate their calories. Marketing is relentless. Real food is the exception.

... Their brains are more sensitive to sweetness. Their taste preferences are still forming. Their microbiome is still developing. Their bodies are smaller... meaning the same dose hits harder.

... ... ...

Children are marketed to more aggressively than any demographic on Earth. Cartoons. Mascots. YouTube ads. TikTok trends. Fun size snacks. Kid-friendly flavours.

... Children do not stand a chance. ... And parents are outnumbered.

... Parents are not failing. ... Parents are overwhelmed.

... ... ...

Key takeaway... You are not the problem. The system is. And our children are paying the highest price.

... ... ...""",

    # ===== V2: CHAPTER 4 - THE UPF AUDIT =====
    """Chapter Nine. ... The UPF Audit. ... How to Identify and Replace Ultra-Processed Foods.

... ... ...

Knowledge is useless without action. This chapter gives you a simple, practical system to identify UPFs... and replace them.

... ... ...

The Four Question UPF Test.

... One. Does it contain ingredients you wouldn't cook with at home?
... Two. Does it have more than five to seven ingredients?
... Three. Does it contain emulsifiers, stabilisers, colourants, or artificial sweeteners?
... Four. Does it come from a factory... not a kitchen?

... If the answer is yes to two or more... it's almost certainly UPF.

... ... ...

Swap, Don't Stop.

... Flavoured yoghurt... becomes plain yoghurt with fruit.
... Breakfast cereal... becomes oats with nuts and berries.
... Crisps... become nuts, seeds, or popcorn.
... Ready meals... become batch-cooked meals.

... ... ...

Build a Whole Food Default.

... People don't rise to the level of their goals. ... They fall to the level of their defaults.

... ... ...

The Eighty Percent Rule.

... You don't need to eliminate UPFs entirely. Reducing them to twenty percent or less of total intake... delivers most of the benefit.

... Key takeaway... Swap, don't stop. The goal isn't perfection. It's replacement.

... ... ...""",

    # ===== V2: CHAPTER 5 - FAMILY STRATEGY =====
    """Chapter Ten. ... The Family Strategy. ... Reducing UPF With Kids Without Battles.

... ... ...

Kids don't need perfection, pressure, or policing. ... They need structure... modelling... and environment.

... The Parent First Principle. Children copy what they see, not what they're told. No lectures required.

... The Home Environment Rule. If it's in the house... it gets eaten. If it's not... it doesn't. Parents don't need to restrict... they need to curate.

... The Three Category Food System. Everyday foods... Sometimes foods... Treat foods. This removes shame and gives kids clarity.

... The Snack Swap Strategy. Kids eat what's easy. Crisps become popcorn. Cereal bars become fruit and nuts. Fizzy drinks become flavoured water. Small swaps compound.

... The One Treat No Drama Rule. Allow one treat per day. Predictable, calm, consistent.

... The Family Meal Advantage. Shared meals reduce UPF intake by up to forty percent.

... Key takeaway... UPF reduction with kids is a leadership challenge, not a discipline challenge. Change the environment... behaviour follows.

... ... ...""",

    # ===== V2: CHAPTER 6 - 30-DAY RESET =====
    """Chapter Eleven. ... The Metabolic Reset. ... A Thirty Day Plan to Rebuild Your Health.

... ... ...

This is not a diet. ... It is a pattern.

... ... ...

Week One... Stabilise. Swap breakfast for a whole food option. Replace sugary drinks with water. Add protein to every meal. ... Outcome... Fewer crashes. Fewer cravings.

... ... ...

Week Two... Replace. Swap the highest impact UPFs. Batch cook two to three meals. Build a whole food snack list. ... Outcome... Energy stabilises. Hunger normalises.

... ... ...

Week Three... Rebuild. Three meals, one to two snacks. Twelve hour overnight fast. One big salad or veg-heavy meal daily. ... Outcome... Appetite regulation improves.

... ... ...

Week Four... Sustain. Eighty twenty UPF rule. Weekly batch cook. One treat window per day. ... Outcome... A sustainable lifestyle... not a temporary fix.

... Key takeaway... You don't need a new diet. You need one real meal. And then another. And then another.

... ... ...""",

    # ===== V2: CHAPTER 9 - METABOLISM EXPLAINED =====
    """Chapter Twelve. ... The UPF Metabolism Link Explained Simply.

... ... ...

Metabolism is not just calories in, calories out. Ultra-processed foods disrupt this system in five major ways.

... Insulin Overload. Frequent glucose spikes force the pancreas to release large amounts of insulin. Over time, cells become less responsive.

... Chronic Inflammation. Additives and emulsifiers irritate the gut lining. Low-grade inflammation interferes with metabolic regulation.

... Mitochondrial Stress. The body's energy factories function best with steady, nutrient-dense fuel. UPFs provide erratic, low-quality energy.

... Hormonal Disruption. Chemicals from packaging interfere with appetite hormones like leptin and ghrelin.

... Microbiome Imbalance. UPFs starve beneficial bacteria and feed opportunistic species.

... ... ...

Key takeaway... Metabolism is not broken. It is responding to the environment. Change the environment... and the body recalibrates.

... ... ...""",

    # ===== V2: CHAPTER 10 - LONG-TERM PLAN =====
    """Chapter Thirteen. ... The Long Term Plan. ... How to Live UPF Light for Life.

... ... ...

Short-term changes are easy. Long-term change requires identity... environment... and rhythm.

... Identity Shift. People who succeed don't say... I'm trying to eat better. They say... I'm someone who eats real food. Identity drives behaviour.

... The Eighty Twenty Lifestyle. Eighty percent whole or minimally processed. Twenty percent flexible. Almost all the benefit... none of the stress.

... The Weekly Rhythm. One batch cook session. One big shop. One treat window per day. One family meal per day. Rhythm beats motivation.

... The Environment Reset. Keep whole foods visible. Keep UPFs out of the house. Environment is destiny.

... The Social Strategy. UPFs are everywhere... parties, work, travel. Enjoy the moment, then return to your rhythm. No guilt. No spirals.

... Key takeaway... You don't need perfection. You need a rhythm you can repeat.

... ... ...""",

    # ===== V2: CHAPTER 7 - APP INTEGRATION =====
    """Chapter Fourteen. ... Turning Knowledge Into Daily Action.

... ... ...

Books change understanding. ... Apps change behaviour.

... The Ingredient Scanner. Scan any food and instantly see... UPF score... additives... processing level... and healthier alternatives. This removes guesswork and builds awareness.

... The Daily Food Log. Not calorie counting... pattern tracking. The app highlights whole food percentage... UPF exposure... and weekly trends.

... The Streak System. Streaks work. The app rewards whole food days, batch cook sessions, and UPF-free breakfasts. This builds momentum.

... The Family Mode. Parents can track household patterns, plan meals, and create shared goals.

... The Habit Engine. Shopping suggestions. Swap recommendations. Meal ideas. Reminders to prep or batch cook.

... The Thirty Day Reset Integration. The app guides you through the reset with daily tasks, tips, and progress tracking.

... ... ...

Key takeaway... This app is the practical extension of this book. The tool that makes change stick.

... ... ...""",

    # ===== V2: FINAL CHAPTER =====
    """The Final Chapter. ... The Movement Starts With You.

... ... ...

You've reached the end of the book... but the journey is just beginning. Reducing UPF is not a trend. ... It is a movement... one that starts in homes, kitchens, schools, workplaces, and communities.

... ... ...

Every whole food meal you choose...
... Every swap you make...
... Every habit you build...
... Every child you influence...
... Every person you inspire...
... shifts the world a little further away... from UPF dependence.

... ... ...

You don't need perfection. ... You need momentum.
... You don't need discipline. ... You need structure.
... You don't need willpower. ... You need environment.

... ... ...

The science is clear. ... The tools are in your hands. ... The app is your companion. ... The path is simple.

... Now you get to lead... not by preaching... but by living.

... ... ...

This is how change spreads.
... This is how health returns.
... This is how a movement begins.

... ... ...""",

    # ===== CLOSING CREDITS + END CARD =====
    """... ... ...

You Are What You Eat.

... Written by Jason Psaila.

... ... ...

Thank you for listening.

... If this book has changed how you see food... share it with someone you care about. Leave a review. Start a conversation.

... Visit yawye dot app... to download the companion app. Scan your next meal. See the score for yourself.

... ... ...

For more information, follow You Are What You Eat on social media.

... ... ...

Together... we eat better.""",
]

async def generate_combined_audiobook():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    total = len(COMBINED_BOOK)
    
    for i, text in enumerate(COMBINED_BOOK):
        text = text.strip()
        print(f"\n[{i+1}/{total}] Generating section ({len(text)} chars)...", flush=True)
        
        # Split into chunks if over 4000 chars
        chunks = []
        if len(text) <= 4000:
            chunks = [text]
        else:
            sentences = re.split(r'(?<=[.!?"])\s+', text)
            current = ""
            for s in sentences:
                if len(current) + len(s) + 1 > 3800:
                    if current:
                        chunks.append(current.strip())
                    current = s
                else:
                    current += " " + s
            if current:
                chunks.append(current.strip())
        
        for j, chunk in enumerate(chunks):
            try:
                audio = await tts.generate_speech(
                    text=chunk,
                    model="tts-1-hd",
                    voice="onyx",
                    response_format="mp3",
                    speed=0.92
                )
                all_audio.extend(audio)
                print(f"  Chunk {j+1}/{len(chunks)} done ({len(audio)} bytes)", flush=True)
            except Exception as e:
                print(f"  ERROR chunk {j+1}: {e}", flush=True)
    
    path = os.path.join(output_dir, "audiobook_combined.mp3")
    with open(path, "wb") as f:
        f.write(all_audio)
    
    size_mb = round(os.path.getsize(path) / (1024*1024), 1)
    
    # Compare all versions
    print(f"\n{'='*50}")
    print(f"COMBINED audiobook saved: {path}")
    print(f"  Size: {size_mb} MB")
    for name in ["audiobook_full.mp3", "audiobook_v2.mp3"]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            s = round(os.path.getsize(p) / (1024*1024), 1)
            print(f"  vs {name}: {s} MB")
    print(f"{'='*50}")

asyncio.run(generate_combined_audiobook())
