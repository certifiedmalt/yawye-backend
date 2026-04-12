import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

BOOK_V2 = [
    # INTRODUCTION
    """You Are What You Eat.

... ... ...

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

    # CHAPTER 1 PART 1
    """Chapter One. ... The New Science of Ultra-Processed Food. ... What the Last Six Years Finally Made Clear.

... ... ...

Ultra-processed food isn't a trend, a buzzword, or a moral panic. Over the last six years, the science has hardened into something much more powerful... a global, cross-disciplinary consensus.

Nutrition researchers... metabolic scientists... epidemiologists... mental health specialists... and policy bodies... have all converged on the same conclusion:

... Ultra-processed food is not just food with additives. ... It is a category of products that reliably predicts worse health outcomes... across almost every system of the body.

... ... ...

This chapter distils the ten most important studies from 2020 to 2026... the ones that changed the conversation from... is UPF harmful?... to... how fast can we reduce it?

... ... ...""",

    # CHAPTER 1 PART 2 - THE STUDIES
    """Study one... The Lancet UPF Series, 2025... The Turning Point.

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

... This chapter establishes the scientific foundation that makes the practical guidance in this book... credible... urgent... and unavoidable.

... ... ...""",

    # CHAPTER 2
    """Chapter Two. ... The Five Mechanisms of UPF Harm.

... ... ...

If Chapter One established that UPF is harmful... this chapter explains why. The last six years of research have clarified five core mechanisms that make UPFs fundamentally different from whole foods... not just nutritionally... but biologically and behaviourally.

... ... ...

Mechanism One... Food Matrix Destruction.

... Whole foods have structure. ... UPFs do not. When the natural matrix is destroyed, digestion becomes faster, satiety signals weaken, and glucose and lipids hit the bloodstream harder and faster. ... This drives overeating... and metabolic stress.

... ... ...

Mechanism Two... Hyper-Palatability Engineering.

... UPFs are engineered to be irresistible. Manufacturers combine sugar, fat, salt, flavour enhancers, and emulsifiers in ratios rarely found in nature. ... This creates bliss point foods... that override normal appetite regulation.

... ... ...

Mechanism Three... Additives and Emulsifiers.

... Emulsifiers, stabilisers, artificial sweeteners, and colourants are not inherently evil... but their chronic, combined, long-term effects are now linked to inflammation... microbiome disruption... and impaired gut barrier function.

... ... ...

Mechanism Four... Packaging and Endocrine Disruptors.

... Plastics, liners, and heat-sealed packaging can leach chemicals like phthalates and bisphenols into food. These compounds interfere with hormones... metabolism... and appetite regulation.

... ... ...

Mechanism Five... Rapid Glycaemic Load.

... UPFs often deliver carbohydrates in their most rapidly absorbable form. This causes glucose spikes... insulin surges... and subsequent crashes that drive hunger, cravings, and fat storage.

... ... ...

Together... these five mechanisms explain why UPFs consistently predict worse health outcomes... even when calories, sugar, fat, and fibre are matched. ... The harm is structural... behavioural... and biological.

... ... ...""",

    # CHAPTER 3
    """Chapter Three. ... How Ultra-Processed Food Hijacks Behaviour.

... ... ...

UPFs don't just affect the body. ... They affect the brain. The last six years of research have shown that UPFs exploit the same neural pathways involved in reward... habit formation... and compulsive behaviour.

... ... ...

The Dopamine Loop.

... Hyper-palatable foods trigger rapid dopamine spikes. Over time, the brain adapts, requiring more stimulation to achieve the same reward. ... This mirrors the pattern seen in other compulsive behaviours.

... ... ...

Satiety Signal Disruption.

... UPFs bypass the mechanical and chemical signals that whole foods trigger. Without fibre, structure, or slow digestion... the body struggles to register fullness.

... ... ...

Craving Amplification.

... Rapid glucose spikes followed by crashes create a physiological craving cycle. The body seeks fast energy... and UPFs deliver it... temporarily.

... ... ...

Habit Formation.

... UPFs are cheap, convenient, and everywhere. This environmental saturation makes them easy to default to... especially under stress, fatigue, or time pressure.

... ... ...

Emotional Eating.

... UPFs provide immediate sensory reward. In moments of stress, boredom, or low mood, they offer a quick dopamine hit... reinforcing the loop.

... ... ...

This chapter reframes UPF not as a willpower issue... but as a design issue. These foods are engineered to be eaten quickly, repeatedly, and in large quantities. ... Understanding this... frees people from blame... and gives them tools to break the cycle.

... ... ...""",

    # CHAPTER 4
    """Chapter Four. ... The UPF Audit. ... How to Identify and Replace Ultra-Processed Foods.

... ... ...

Knowledge is useless without action. This chapter gives readers a simple, practical system to identify UPFs in the real world... and replace them without spending more money or more time.

... ... ...

Step One... The Four Question UPF Test.

... One. Does it contain ingredients you wouldn't cook with at home?
... Two. Does it have more than five to seven ingredients?
... Three. Does it contain emulsifiers, stabilisers, colourants, or artificial sweeteners?
... Four. Does it come from a factory... not a kitchen?

... If the answer is yes to two or more... it's almost certainly UPF.

... ... ...

Step Two... Swap, Don't Stop.

... The goal isn't perfection. It's replacement.

... Flavoured yoghurt... becomes plain yoghurt with fruit.
... Breakfast cereal... becomes oats with nuts and berries.
... Crisps... become nuts, seeds, or popcorn.
... Ready meals... become batch-cooked meals.

... ... ...

Step Three... Build a Whole Food Default.

... People don't rise to the level of their goals. ... They fall to the level of their defaults. This section teaches readers how to build a home environment where whole foods are the easiest option.

... ... ...

Step Four... The Eighty Percent Rule.

... You don't need to eliminate UPFs entirely. Reducing them to twenty percent or less of total intake... delivers most of the benefit.

... ... ...

This chapter turns the science into a system... one that is simple, sustainable, and family-friendly.

... ... ...""",

    # CHAPTER 5
    """Chapter Five. ... The Family Strategy. ... Reducing UPF With Kids Without Battles.

... ... ...

If UPF reduction is hard for adults... it can feel impossible with children. But the last decade of behavioural science shows something powerful... kids don't need perfection, pressure, or policing. ... They need structure... modelling... and environment.

... ... ...

The Parent First Principle.

... Children copy what they see, not what they're told. When adults shift their own defaults... breakfast, snacks, drinks... kids follow naturally. No lectures required.

... ... ...

The Home Environment Rule.

... If it's in the house... it gets eaten. If it's not... it doesn't. This is the single strongest predictor of a child's diet quality. Parents don't need to restrict... they need to curate.

... ... ...

The Three Category Food System.

... Instead of good versus bad, use...
... Everyday foods... whole foods.
... Sometimes foods... minimally processed.
... Treat foods... UPFs.
... This removes shame and gives kids clarity.

... ... ...

The Eighty Twenty Family Pattern.

... Children don't need to eliminate UPFs. They need a pattern where whole foods dominate. Eighty percent whole or minimally processed... twenty percent flexible.

... ... ...

The Snack Swap Strategy.

... Kids eat what's easy. Replace...
... Crisps with popcorn or nuts.
... Flavoured yoghurt with plain yoghurt and honey.
... Cereal bars with fruit and nut mixes.
... Fizzy drinks with flavoured water.
... Small swaps compound.

... ... ...

The One Treat No Drama Rule.

... Allow one treat per day or per occasion. Predictable, calm, and consistent. This prevents binge-restrict cycles.

... ... ...

The Family Meal Advantage.

... Shared meals reduce UPF intake by up to forty percent. Not because of the food... but because of the structure, routine, and modelling.

... ... ...

This chapter reframes UPF reduction with kids as a leadership challenge... not a discipline challenge. When the environment changes... behaviour follows.

... ... ...""",

    # CHAPTER 6
    """Chapter Six. ... The Metabolic Reset. ... A Thirty Day Plan to Rebuild Your Health.

... ... ...

This chapter gives readers a simple, structured thirty day reset designed to lower UPF intake, stabilise blood sugar, reduce cravings, and rebuild metabolic resilience. ... It is not a diet. ... It is a pattern.

... ... ...

Week One... Stabilise.

... Goal... Remove the biggest blood sugar disruptors.

... Swap breakfast for a whole food option... oats, eggs, yoghurt, fruit.
... Replace sugary drinks with water, tea, or coffee.
... Add protein to every meal.

... Outcome... Fewer crashes. Fewer cravings.

... ... ...

Week Two... Replace.

... Goal... Swap the highest impact UPFs.

... Replace cereal, crisps, biscuits, ready meals.
... Batch cook two to three meals for the week.
... Build a whole food snack list.

... Outcome... Energy stabilises. Hunger normalises.

... ... ...

Week Three... Rebuild.

... Goal... Add structure and routine.

... Three meals, one to two snacks.
... Twelve hour overnight fast.
... One big salad or veg-heavy meal daily.

... Outcome... Appetite regulation improves.

... ... ...

Week Four... Sustain.

... Goal... Build long-term habits.

... Eighty twenty UPF rule.
... Weekly batch cook session.
... One treat window per day.

... Outcome... A sustainable lifestyle... not a temporary fix.

... ... ...

The thirty day reset is designed to be family-friendly, affordable, and repeatable. It gives readers a metabolic foundation they can maintain for life.

... ... ...""",

    # CHAPTER 7
    """Chapter Seven. ... The App Integrated Journey. ... Turning Knowledge Into Daily Action.

... ... ...

Books change understanding. ... Apps change behaviour. This chapter shows readers how the companion app turns the science of UPF reduction into a daily, personalised system.

... ... ...

The Ingredient Scanner.

... Readers can scan any food and instantly see... UPF score... additives... processing level... and healthier alternatives. ... This removes guesswork and builds awareness.

... ... ...

The Daily Food Log.

... Not calorie counting... pattern tracking. The app highlights... whole food percentage... UPF exposure... and weekly trends. ... This creates accountability without obsession.

... ... ...

The Streak System.

... Behavioural science is clear... streaks work. The app rewards... whole food days... batch cook sessions... and UPF-free breakfasts. ... This builds momentum.

... ... ...

The Family Mode.

... Parents can track household patterns, plan meals, and create shared goals. ... This turns UPF reduction into a team effort.

... ... ...

The Habit Engine.

... The app nudges users with... shopping suggestions... swap recommendations... meal ideas... and reminders to prep or batch cook. ... This bridges the gap between intention and action.

... ... ...

The Thirty Day Reset Integration.

... The app guides users through the reset from Chapter Six with daily tasks, tips, and progress tracking.

... This chapter positions the app as the practical extension of the book... the tool that makes change stick.

... ... ...""",

    # CHAPTER 8
    """Chapter Eight. ... The Science of Cravings. ... Why Your Body Wants What It Doesn't Need.

... ... ...

Cravings are not a character flaw. ... They are chemistry. And ultra-processed foods are engineered to exploit that chemistry... with precision.

... ... ...

The Dopamine Spike.

... UPFs deliver rapid sensory reward... sweetness, crunch, salt, fat... triggering dopamine release. Over time, the brain adapts, requiring more stimulation for the same effect. ... This is why cravings intensify... not fade.

... ... ...

The Blood Sugar Whiplash.

... UPFs often contain rapidly absorbable carbohydrates. They spike glucose... spike insulin... then cause a crash. The crash triggers hunger, irritability, and a drive for fast energy... usually more UPF.

... ... ...

The Gut Brain Loop.

... The microbiome shifts in response to diet. Diets high in UPF promote bacteria that thrive on sugar and refined carbs. These microbes send signals that influence cravings... mood... and appetite.

... ... ...

The Stress Connection.

... Cortisol increases appetite for high-energy, high-reward foods. UPFs provide immediate relief... reinforcing the stress, craving, UPF loop.

... ... ...

The Habit Layer.

... Cravings are not just biological. They are contextual. Time of day, location, mood, and routine all trigger learned associations.

... ... ...

This chapter reframes cravings as predictable... understandable... and manageable. When readers understand the mechanisms... they can break the loop... not through willpower... but through strategy.

... ... ...""",

    # CHAPTER 9
    """Chapter Nine. ... The UPF Metabolism Link Explained Simply.

... ... ...

Metabolism is not just calories in, calories out. It is a complex system involving hormones, enzymes, gut bacteria, and cellular signalling. Ultra-processed foods disrupt this system... in five major ways.

... ... ...

Insulin Overload.

... Frequent glucose spikes from UPFs force the pancreas to release large amounts of insulin. Over time, cells become less responsive... the first step toward insulin resistance.

... ... ...

Chronic Inflammation.

... Additives, emulsifiers, and certain fats can irritate the gut lining and immune system. Low-grade inflammation interferes with metabolic regulation and fat storage.

... ... ...

Mitochondrial Stress.

... Mitochondria... the body's energy factories... function best with steady, nutrient-dense fuel. UPFs provide erratic, low-quality energy that impairs efficiency.

... ... ...

Hormonal Disruption.

... Endocrine disrupting chemicals from packaging can interfere with appetite hormones like leptin and ghrelin... making hunger harder to regulate.

... ... ...

Microbiome Imbalance.

... UPFs starve beneficial bacteria and feed opportunistic species. A disrupted microbiome affects digestion... cravings... immunity... and metabolic flexibility.

... ... ...

This chapter gives readers a simple, empowering message... metabolism is not broken. ... It is responding to the environment. Change the environment... and the body recalibrates.

... ... ...""",

    # CHAPTER 10
    """Chapter Ten. ... The Long Term Plan. ... How to Live UPF Light for Life.

... ... ...

Short-term changes are easy. Long-term change requires identity... environment... and rhythm. This chapter gives readers a sustainable blueprint for living UPF-light... without rigidity or obsession.

... ... ...

Identity Shift.

... People who succeed don't say... I'm trying to eat better. ... They say... I'm someone who eats real food. ... Identity drives behaviour.

... ... ...

The Eighty Twenty Lifestyle.

... Perfection is unnecessary and unsustainable. Eighty percent whole or minimally processed. Twenty percent flexible. ... This delivers almost all the health benefit... with none of the stress.

... ... ...

The Weekly Rhythm.

... A sustainable routine includes... one batch cook session... one big shop... one treat window per day... one family meal per day. ... Rhythm beats motivation.

... ... ...

The Environment Reset.

... Keep whole foods visible and accessible. Keep UPFs out of the house or in inconvenient places. ... Environment is destiny.

... ... ...

The Social Strategy.

... UPFs are everywhere... parties, work, travel. The strategy... enjoy the moment, then return to your rhythm. No guilt. No spirals.

... ... ...

The App as a Long Term Companion.

... The app tracks patterns, celebrates streaks, and nudges users back on track. It turns long-term health into a game... one you can win.

... ... ...

This chapter closes the loop. Readers now have the science, the strategy, and the system... to live UPF-light for life.

... ... ...""",

    # FINAL CHAPTER
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

... ... ...

You Are What You Eat.

... ... ...""",
]

async def generate_audiobook_v2():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    total = len(BOOK_V2)
    
    for i, text in enumerate(BOOK_V2):
        text = text.strip()
        print(f"\n[{i+1}/{total}] Generating section ({len(text)} chars)...")
        
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
                print(f"  Chunk {j+1}/{len(chunks)} OK ({len(audio)} bytes)")
            except Exception as e:
                print(f"  ERROR chunk {j+1}: {e}")
    
    path = os.path.join(output_dir, "audiobook_v2.mp3")
    with open(path, "wb") as f:
        f.write(all_audio)
    
    size_mb = round(os.path.getsize(path) / (1024*1024), 1)
    print(f"\nV2 audiobook saved: {path} ({size_mb} MB)")
    
    # Compare with original
    orig_path = os.path.join(output_dir, "audiobook_full.mp3")
    if os.path.exists(orig_path):
        orig_size = round(os.path.getsize(orig_path) / (1024*1024), 1)
        print(f"Original audiobook: {orig_size} MB")
        print(f"V2 audiobook: {size_mb} MB")
        print(f"Difference: {round(size_mb - orig_size, 1)} MB")

asyncio.run(generate_audiobook_v2())
