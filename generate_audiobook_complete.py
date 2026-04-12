import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

COMPLETE_BOOK = [

    # ===== 1. OPENING CREDITS =====
    """You Are What You Eat.

... ... ...

Written by Jason Psaila.

... ... ...

This audiobook is a companion to the You Are What You Eat app. ... Available on Google Play and the App Store.

... ... ...""",

    # ===== 2. PERSONAL NOTE FROM AUTHOR =====
    """A note from the author.

... ... ...

My name is Jason Psaila. I wrote this book because I got tired of being confused.

... Confused by contradictory headlines. Confused by food labels that made no sense. Confused by an industry that seemed to be hiding in plain sight.

... So I spent years reading the research. ... Talking to the experts. ... And building an app that turns that research into something you can use every single day.

... This book is what I wish someone had given me years ago. ... Not rules. Not guilt. ... Just clarity.

... I hope it does the same for you.

... ... ...""",

    # ===== 3. V1: OPENING QUOTES =====
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

    # ===== 4. V1: PROLOGUE =====
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

... It happened quietly. ... It happened quickly.

This is the beginning of taking back your biology. ... This is the beginning of taking back your health. ... This is the beginning of taking back your life.""",

    # ===== 5. V2: INTRODUCTION =====
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

    # ===== 6. V1: CH1 - THE FOOD REVOLUTION PART 1 =====
    """... ... ...

Chapter One. ... The Food Revolution That Happened Behind Your Back.

... ... ...

You didn't notice it happening. ... Nobody did.

There was no announcement, no warning, no moment where the world paused and said: "From today, your food will be different."

... It happened quietly. ... Gradually. ... Invisibly.

... The way all revolutions do... when the people in charge don't want you to see them coming.

... ... ...

For most of human history, food was simple. It came from farms, fields, oceans, orchards, and gardens. It was grown, raised, caught, cooked, shared.

... Then, in the space of a single generation... that world disappeared.

Real food didn't vanish overnight. It was pushed aside... replaced by products that looked like food, smelled like food, tasted like food... but were something else entirely.

... Something engineered. ... Something manufactured. ... Something designed to behave like food without being bound by any of its limitations.

... ... ...

Shelf life replaced freshness. ... Profit replaced nourishment. ... Formulas replaced recipes.

... And the consequences have been catastrophic.""",

    # ===== 7. V1: CH1 - PART 2 =====
    """Walk into any supermarket today and you'll see abundance. Bright colours, friendly mascots, health claims, convenience.

... But beneath the packaging lies a different story. One of industrial processes... chemical engineering... and corporate strategy.

... ... ...

Ultra-processed foods didn't take over because they were better. ... They took over because they were more profitable.

They're cheaper to produce than real food. ... They last longer on shelves than real food. ... They can be flavoured, coloured, thickened, softened, sweetened, and stabilised into anything a marketing team can imagine.

... And most importantly... they can be engineered to make you want more.

... This is not a conspiracy. ... It's a business model.

... A business model that has reshaped the global diet... and with it... global health.

... ... ...

Look at the timeline.

As ultra-processed foods became dominant, something else rose alongside them. Obesity. Type 2 diabetes. Fatty liver disease. Metabolic syndrome. Chronic inflammation. And shifts in cancer patterns.

These conditions didn't rise slowly. ... They rose sharply. ... They rose predictably. ... They rose in parallel with the spread of ultra-processed foods.

... ... ...

This isn't about individual choices. It's about an environment that changed faster than human biology could adapt.

... Your body is ancient. ... Ultra-processed food is brand new. ... And the collision between the two has created a health crisis unlike anything the world has seen.""",

    # ===== 8. V1: CH1 - PART 3 =====
    """The food industry didn't just change what we eat. ... It changed how we eat.

It changed the pace of eating... softer textures, faster chewing, quicker absorption. It changed the psychology of eating... engineered flavours, bliss points, addictive combinations. It changed the biology of eating... disrupting hunger signals, altering the microbiome, hijacking reward pathways.

... And it changed the culture of eating... replacing meals with snacks, cooking with convenience, nourishment with novelty.

... ... ...

We didn't evolve for this. ... We weren't prepared for this. ... We weren't warned about this.

... But the companies who built this system... knew exactly what they were doing.

... ... ...

They knew that the more processed a product is, the cheaper it is to make. They knew that the more engineered a product is, the easier it is to manipulate. They knew that the more addictive a product is... the more profitable it becomes.

... And they knew... that once ultra-processed food became normal... nobody would question it.

... Until now.

... ... ...

You are not imagining it. You are not overreacting. ... You are not the problem.

The system changed. The food changed. ... And your body has been trying to cope ever since.

The rest of this book will show you how deep the transformation goes... and how you can finally break free from it.""",

    # ===== 9. V2: CH1 - THE 10 KEY STUDIES =====
    """Chapter Two. ... The New Science of Ultra-Processed Food. ... What the Last Six Years Finally Made Clear.

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

... Key takeaway... This is no longer a debate. The science is settled.

... ... ...""",

    # ===== 10. V2: CH2 - FIVE MECHANISMS =====
    """Chapter Three. ... The Five Mechanisms of UPF Harm.

... ... ...

If the previous chapter established that UPF is harmful... this chapter explains why. The last six years of research have clarified five core mechanisms that make UPFs fundamentally different from whole foods.

... Mechanism One... Food Matrix Destruction. Whole foods have structure. UPFs do not. When the natural matrix is destroyed, digestion becomes faster, satiety signals weaken, and glucose and lipids hit the bloodstream harder and faster.

... Mechanism Two... Hyper-Palatability Engineering. UPFs are engineered to be irresistible. Manufacturers combine sugar, fat, salt, flavour enhancers, and emulsifiers in ratios rarely found in nature.

... Mechanism Three... Additives and Emulsifiers. Their chronic, combined, long-term effects are now linked to inflammation... microbiome disruption... and impaired gut barrier function.

... Mechanism Four... Packaging and Endocrine Disruptors. Plastics, liners, and heat-sealed packaging can leach chemicals like phthalates and bisphenols into food.

... Mechanism Five... Rapid Glycaemic Load. UPFs often deliver carbohydrates in their most rapidly absorbable form. This causes glucose spikes... insulin surges... and subsequent crashes.

... ... ...

Together... these five mechanisms explain why UPFs consistently predict worse health outcomes... even when calories, sugar, fat, and fibre are matched.

... Now that you understand the mechanisms... let's see them in action.

... ... ...""",

    # ===== 11-16. V1: CH3 - SEED OILS (6 PARTS - ALREADY IN FINAL) =====
    # Part 1
    """Chapter Four. ... The Refinery. ... The Industrial Birth of Seed Oils.

... ... ...

If you could watch your food being made... you would never eat the same way again.

... ... ...

Most people imagine oil the way they imagine olive oil. Sunlight. Olives. A stone press. A slow trickle of golden liquid.

... Seed oils are nothing like that.

They don't drip from seeds. ... They don't squeeze out naturally. ... They don't exist in any meaningful quantity... without heavy machinery, chemical solvents, and industrial heat.

... Seed oils are not pressed. ... They are extracted.

... And the process looks far more like a fuel refinery... than a kitchen.

... ... ...

The raw material. Corn. Soy. Rapeseed. Cottonseed. Sunflower. Safflower.

These plants were never part of the human diet in oil form. For most of human history, they couldn't be... because you cannot get meaningful oil out of them... without industrial intervention.

... A sunflower seed contains a trace of oil. ... A corn kernel contains even less.

To turn these into litres of oil, you need the same tools used in chemical manufacturing. High-pressure rollers. Solvent tanks. Steam injectors. Centrifuges. Bleaching towers. Deodorising columns.

... ... ...

This is not cooking. ... This is industry.""",

    # Part 2
    """... ... ...

The process begins.

... ... ...

Step one. ... Crushing. The seeds are not gently pressed. They are crushed... ground... and pulverised into a fine meal.

... This is the first moment where the process stops resembling food.

... ... ...

Step two. ... Solvent Extraction. The ground seed meal is washed... in hexane. ... Hexane. ... A petroleum-derived solvent. The same chemical used in industrial degreasing... in glue production... and in chemical extraction.

... Hexane dissolves the oil out of the seed. The mixture becomes a slurry... part food... part chemical bath.

... ... ...

Let that sit for a moment. The oil in your kitchen cupboard... was washed out of crushed seeds... using a petroleum solvent.

... This is the moment most people would stop calling it food.

... ... ...

Step three. ... Desolventising. The hexane-oil mixture is heated so the solvent evaporates. The oil remains... but it is dark, bitter, and full of impurities.

... It smells nothing like food. ... So the industry keeps going.""",

    # Part 3
    """... ... ...

Step four. ... Degumming. The crude oil contains phospholipids... natural components of seeds. Industry calls them "gums." They are removed using water, steam, or acid.

... ... ...

Step five. ... Neutralisation. Free fatty acids are removed... using sodium hydroxide. ... Sodium hydroxide. ... The same chemical used in soap making.

... The oil is now chemically altered.

... ... ...

Step six. ... Bleaching. ... This is the moment the oil stops looking like food.

The oil is heated and mixed with acid-activated bleaching clays. ... The same materials used in... petroleum refining. ... Wastewater treatment. ... Chemical purification.

... ... ...

These clays pull out pigments... metals... breakdown products... and oxidation compounds.

What was once brown... becomes pale. What was once visibly industrial... becomes visually "clean."

... Not because it has become healthier. ... But because it has become more marketable.

... Think about that. The bleaching step doesn't make the oil better for you. It makes the oil easier to sell to you.""",

    # Part 4
    """... ... ...

Step seven. ... The final transformation. ... Deodorisation.

The oil is heated again... this time to temperatures between two hundred... and two hundred and sixty degrees Celsius.

... This removes the harsh, industrial smell created by the refining process.

But high heat does something else. ... It breaks fatty acids. ... It destroys nutrients. ... It creates new compounds. ... It alters the structure of the oil itself.

... ... ...

Vegetable oil refining. Crush seeds. Extract with hexane solvent. Evaporate solvent. Degum. Neutralise with caustic soda. Bleach with activated clays. Deodorise at two hundred and sixty degrees. Bottle and sell as food.

... ... ...

Fuel refining. Distil crude oil. Use solvents to separate fractions. Remove impurities. Neutralise acids. Bleach or filter. Heat to high temperatures. Blend and store as fuel.

... ... ...

The clear, odourless oil on supermarket shelves... is not the oil of a seed. ... It is the oil of a refinery.

... And it is now one of the most consumed substances in the modern diet. Not because it is healthy. Not because it is natural. ... But because it is profitable.""",

    # Part 5 - Myth vs Truth
    """... ... ...

The Myth... versus the Truth.

... ... ...

There's a story that circulates online... dramatic, viral, and easy to repeat. "Seed oils were originally made for livestock, but the animals kept dying, so they banned it for animals... and fed it to humans instead."

... It's a powerful story. ... But it isn't what happened. There is no historical record of livestock dying from seed oils.

... And the real story... is actually more shocking.

... ... ...

The truth. Seed oils began... as industrial waste.

In the late eighteen hundreds to early nineteen hundreds, cottonseed oil was used for machinery lubrication... candles... soap... industrial greases... and lamp fuel.

... It was cheap. It was abundant. ... And it was considered unfit for human consumption.

... ... ...

Then, chemists discovered they could harden it. Hydrogenation... a German industrial process... turned liquid cottonseed oil into a white, lard-like fat.

... This was the birth of Crisco... in nineteen eleven.

... Not a food innovation. ... A chemical one.""",

    # Part 6 - Closing
    """Procter and Gamble launched one of the most aggressive food marketing campaigns in history. ... "Pure." ... "Modern." ... "Clean." ... "Scientific."

... They reframed industrial oil... as a health product.

... ... ...

And here is the livestock connection that is true. ... Livestock didn't die from seed oils. ... They did something else. ... They got fatter. ... Faster. ... Cheaper.

The same industrial oils used to fatten livestock cheaply... became the backbone of the modern human diet.

Not because they were healthy. ... Not because they were traditional. ... But because they were cheap... abundant... profitable... and easy to market.

... ... ...

The myth says seed oils were a mistake. ... The truth says they were a business model.

... And that business model... still shapes the modern diet today.

... ... ...""",

    # ===== 17-20. V1: CH4 - ADDITIVES (4 PARTS) =====
    """Chapter Five. ... The Additive Explosion.

... ... ...

If the previous chapter showed you how modern oils are born in refineries... this chapter shows you what happens next. When those oils become the foundation for a new kind of product.

... Not food. ... Not ingredients. ... But formulations.

... ... ...

Ultra-processed foods are not built the way meals are built. They are assembled... the way products are assembled. From components, stabilisers, emulsifiers, colours, and chemicals... that each play a role in creating the illusion of food.

... A recipe is simple. Ingredients. Heat. Time. Skill.

A formulation is different. Stabilisers. Emulsifiers. Humectants. Anti-foaming agents. Acidity regulators. Colourants. Preservatives. Artificial flavours. Sweeteners. Gums. Texturisers.

... Ultra-processed foods are not built to nourish. ... They are built to perform.""",

    """... ... ...

The emulsifier problem.

Emulsifiers are the glue of the ultra-processed food world. They make oil and water mix. They make sauces smooth. They make ice cream soft. They make bread stay soft... for weeks.

... Common emulsifiers include lecithins... mono and diglycerides... polysorbates... carboxymethylcellulose... carrageenan.

... These are not kitchen ingredients. ... They are industrial tools.

... ... ...

Stabilisers and gums.

Stabilisers keep ultra-processed foods from separating, collapsing, or spoiling.

Gums... guar, xanthan, locust bean... give ultra-processed foods their signature textures. The chewiness of a protein bar. The thickness of a milkshake. The stretch of a plant-based cheese.

... These gums absorb water and swell, creating volume without calories.

This is why ultra-processed foods feel filling... but leave you hungry again soon after.

... Your stomach feels full. ... Your biology does not.""",

    """... ... ...

Sweeteners and flavours.

Sweetness used to be rare. ... Now it is engineered.

A strawberry yoghurt may contain no strawberries. A vanilla drink may contain no vanilla. A cheese puff may contain no cheese.

... Flavours are designed to mimic nature... and then surpass it. This is why ultra-processed foods taste "better" than real food. ... They are engineered to.

... ... ...

Colours. Colour is psychology. If a product looks vibrant, fresh, or fruity... the brain believes it tastes better. ... These colours are not added for nutrition. ... They are added for persuasion. They make products look alive... even when nothing inside them is.

... ... ...

Preservatives. Preservatives are the reason ultra-processed foods last weeks on shelves... months in warehouses... years in storage. They stop mould. They stop bacteria. They stop decay.

... Real food spoils. ... Ultra-processed food doesn't. ... That alone should tell you something.""",

    """... ... ...

The engineering of texture.

Crispiness. Crunchiness. Chewiness. Creaminess. Melt-in-the-mouth softness.

These sensations are engineered using starches, gums, emulsifiers, aeration, extrusion, and high-pressure processing.

... Texture is not an accident. ... It is a design choice. And it is one of the reasons ultra-processed foods are so easy to overeat.

... ... ...

Every additive solves a problem. How do we make this cheaper? How do we make this last longer? How do we make this more addictive? How do we make this feel like real food?

... The result is a product that behaves like food... but is built like a chemical system.

... Additives are not "poisons." But they are industrial tools, designed to make products cheaper, more stable, more profitable... and more addictive.

... And they are now everywhere. In your snacks. In your drinks. In your sauces. In your bread. In your cereal. ... In your children's lunchboxes.

... ... ...""",

    # ===== 21-24. V1: CH5 - FIZZY DRINKS (4 PARTS) =====
    """Chapter Six. ... The Fizzy Drink. ... The Most Engineered Product in the Modern Diet.

... ... ...

The UK now spends more than twenty-one billion pounds a year on soft drinks. These aren't beverages. ... They're chemical formulations.

Engineered for stimulation. Engineered for repeat consumption. Engineered to dominate.

... A fizzy drink is not made. ... It is assembled.

The components: acid... sweetness... colour... flavour... carbonation... preservatives... stabilisers... treated water.

Each one is engineered. Each one has a job. Each one exists to create the illusion of refreshment.

... Let's take them one by one.""",

    """... ... ...

Number one. ... The acid.

Phosphoric acid in colas. Citric acid in fruit sodas. Malic acid in sour drinks.

... These same acids appear in... rust removers... dishwasher tablets... metal cleaners... and industrial descalers.

Phosphoric acid gives cola its signature "bite." Citric acid enhances fruitiness... but the citric acid in drinks is not squeezed from lemons. It is industrial citric acid... produced by fermenting sugar with mould.

... They're not added to nourish you. ... They're added to engineer the sensation of "refreshment."

... ... ...

Number two. ... The sweetness.

A single can contains more sugar than most people would ever add to anything they make at home.

... Sweetness is one of the most powerful reward triggers in the human brain. Fizzy drinks deliver it in a form that is fast... intense... unbuffered by fibre.

... Your brain learns the pattern quickly. Drink. Dopamine. Repeat.""",

    """... ... ...

Number three. ... The colour.

Caramel colour in colas is not caramel from a pan. It is industrial caramel. ... Without colour, the drink would look like chemical water. ... With colour, it looks like a treat.

... ... ...

Number four. ... The flavour.

Flavour is a formula... often containing dozens of compounds, blended to create a sensory illusion. ... The recipe is proprietary. ... The effect is universal. ... Flavour is not taste. ... It is engineering.

... ... ...

Number five. ... The carbonation.

CO2 triggers pain receptors in the mouth... the same receptors activated by mustard and chilli. ... Without carbonation, the drink is syrup. ... With carbonation... it is addictive.

... ... ...

Number six. ... The preservatives.

Under certain conditions... heat plus light plus benzoate plus vitamin C... benzene can form in trace amounts. ... Fizzy drinks behave like chemical systems... because they are chemical systems.""",

    """... ... ...

Fizzy drinks are engineered to be consumed quickly and repeatedly.

They are sweet enough to trigger dopamine. Acidic enough to feel refreshing. Carbonated enough to stimulate the mouth. Flavoured enough to override satiety. Coloured enough to create expectation. Preserved enough to last forever.

... This is not a beverage. ... This is a behavioural product.

... Fizzy drinks do not nourish. They stimulate. They do not hydrate. They override. They do not satisfy. ... They provoke.

... And they are consumed everywhere. ... Especially by children.

... ... ...

Now that you've seen what's inside these products... let's talk about what they do to your brain.

... ... ...""",

    # ===== 25. V2: HOW UPF HIJACKS BEHAVIOUR =====
    """Chapter Seven. ... How Ultra-Processed Food Hijacks Behaviour.

... ... ...

UPFs don't just affect the body. ... They affect the brain.

... The Dopamine Loop. Hyper-palatable foods trigger rapid dopamine spikes. Over time, the brain adapts, requiring more stimulation to achieve the same reward.

... Satiety Signal Disruption. UPFs bypass the mechanical and chemical signals that whole foods trigger. Without fibre, structure, or slow digestion... the body struggles to register fullness.

... Craving Amplification. Rapid glucose spikes followed by crashes create a physiological craving cycle.

... Habit Formation. UPFs are cheap, convenient, and everywhere. This environmental saturation makes them easy to default to.

... Emotional Eating. UPFs provide immediate sensory reward. In moments of stress, boredom, or low mood, they offer a quick dopamine hit... reinforcing the loop.

... ... ...

UPF is not a willpower issue. It is a design issue. Understanding this frees you from blame.

... ... ...""",

    # ===== 26. V2: THE SCIENCE OF CRAVINGS =====
    """Chapter Eight. ... The Science of Cravings. ... Why Your Body Wants What It Doesn't Need.

... ... ...

Cravings are not a character flaw. ... They are chemistry.

... The Dopamine Spike. UPFs deliver rapid sensory reward... triggering dopamine release. Over time, the brain adapts. This is why cravings intensify... not fade.

... The Blood Sugar Whiplash. UPFs spike glucose... spike insulin... then cause a crash. The crash triggers hunger and a drive for fast energy.

... The Gut Brain Loop. Diets high in UPF promote bacteria that thrive on sugar. These microbes send signals that influence cravings... mood... and appetite.

... The Stress Connection. Cortisol increases appetite for high-reward foods. UPFs provide immediate relief... reinforcing the loop.

... The Habit Layer. Cravings are not just biological. They are contextual. Time of day, location, mood, and routine all trigger learned associations.

... ... ...

Cravings are predictable, understandable, and manageable. You can break the loop... not through willpower... but through strategy.

... ... ...""",

    # ===== 27. V1: CH6 - RISE OF MODERN ILLNESS =====
    """Chapter Nine. ... The Rise of Modern Illness.

... ... ...

The modern food economy exploded. ... And something else exploded with it.

Not overnight. Not dramatically. But steadily... quietly... relentlessly.

A rise in conditions that were once rare. A rise in symptoms that were once unusual. A rise in illnesses that were once confined to adulthood... now appearing in children.

... ... ...

For most of human history, chronic metabolic diseases were rare. Food was perishable. Fibrous. Slow-digesting. Nutrient-dense.

... Then came refined oils. Industrial sweeteners. Engineered flavours. Additives. Ready meals. Hyper-palatable formulations.

... And the timeline shifted.

... ... ...

Over the same decades that ultra-processed foods became dominant, the world saw dramatic increases in... obesity. Type 2 diabetes. Fatty liver disease. Metabolic dysfunction. Chronic inflammation. Cardiovascular issues.

... No single food "caused" these conditions. ... But the system changed. ... And the body responded.""",

    # ===== 28. V1: INVISIBLE SYMPTOMS =====
    """... ... ...

The human body evolved for whole foods... slow energy... fibre... protein... natural fats.

Instead, it received refined oils... engineered sweetness... industrial textures... additives... emulsifiers... liquid sugar... chemical acidity.

... This is not a small mismatch. ... It is a biological collision.

... Your body is ancient. Your food is modern. And the gap between the two... is where modern illness lives.

... ... ...

Not every consequence shows up in a diagnosis. Most show up in everyday life.

Fatigue... cravings... mood swings... low energy... poor sleep... difficulty concentrating... constant hunger... afternoon crashes... brain fog.

... These are not character flaws. ... They are biological signals.

... ... ...

Ultra-processed foods create cycles. Eat. Crash. Crave. Repeat.

People blame themselves. They think they lack willpower. ... But the truth is simple.

... The system is designed this way.

... You are not the problem. ... The system is.

... And once you see the system... you can step outside it.

... ... ...""",

    # ===== 29-30. V1: CH7 - CHILDREN'S CRISIS =====
    """Chapter Ten. ... The Children's Crisis.

... ... ...

Children are not miniature adults. ... They are developing systems. Fragile. Sensitive. Rapidly changing.

For the first time in human history, children are growing up in a food system where ultra-processed foods dominate their calories... marketing is relentless... and real food is the exception.

... This chapter is not about blame. ... It is about exposure.

Because children are not choosing this. ... It is being chosen for them.

... ... ...

Every generation before this one grew up eating real meals, real ingredients, real food environments.

This generation is growing up eating formulations... additives... refined oils... engineered sweetness... cartoon-branded snacks... liquid sugar.

Children are not just eating ultra-processed foods. ... They are immersed in them.

... In their lunchboxes. In their schools. In their sports clubs. In their birthday parties. In their vending machines.

... This is not a dietary shift. ... It is a cultural one.""",

    """... ... ...

Children are more vulnerable because... their biology is still being built.

Their brains are more sensitive to sweetness. Their taste preferences are still forming. Their microbiome is still developing. Their bodies are smaller... meaning the same dose hits harder.

... Children are not resilient to ultra-processed foods. ... They are exposed to them.

... ... ...

And children are marketed to more aggressively than any demographic on Earth.

Cartoons... mascots... bright colours... YouTube ads... TikTok trends... "fun size" snacks... "kid-friendly" flavours.

... Children do not stand a chance. ... And parents are outnumbered.

... ... ...

Parents are not failing. ... Parents are overwhelmed.

They are fighting convenience... marketing... peer pressure... school environments... time scarcity... engineered cravings.

... Parents are not choosing ultra-processed foods for their children. ... Ultra-processed foods are choosing their children.

... ... ...""",

    # ===== 31. V1: CH8 - THE HIDDEN COSTS =====
    """Chapter Eleven. ... The Hidden Costs.

... ... ...

Ultra-processed foods don't just change the body. They change the day. They change the mood. They change the energy. They change the wallet. They change the rhythm of life.

... This chapter is about the costs that don't show up on labels.

... ... ...

The cost of energy. Ultra-processed foods create energy instability. Fast energy. Fast crashes. Fast hunger. Fast cravings. ... A rollercoaster disguised as convenience.

... ... ...

The cost of cravings. Cravings feel personal. They feel like weakness. ... But cravings are not moral failures. They are biological responses to engineered foods. ... Cravings are not a bug. ... They are the business model.

... ... ...

The cost of mood. Ultra-processed foods create irritability, mood swings, emotional volatility, low resilience, afternoon crashes, brain fog. Not because people are unstable. ... But because their blood sugar is.

... ... ...

The cost of money. A one-pound snack becomes a thirty-pound weekly habit. A two-pound drink becomes a sixty-pound monthly cycle. ... Ultra-processed foods are cheap at the checkout. But expensive in the long run.

... ... ...

The cost of self-perception. Perhaps the most painful cost of all. Ultra-processed foods make people feel weak... undisciplined... guilty... ashamed... "broken."

... But they're not broken. ... They're responding exactly as biology responds to engineered foods.

... Ultra-processed foods don't just change the body. ... They change the story people tell about themselves.

... ... ...""",

    # ===== 32. V1: CH9 - SEEING THE MATRIX =====
    """Chapter Twelve. ... Seeing the Matrix.

... ... ...

Most people don't choose ultra-processed foods. ... They drift into them. They pick up what's convenient. What's familiar. What's marketed.

... This chapter changes that.

... ... ...

Rule one. If it has a health claim... be suspicious. Real food doesn't need to advertise itself.

Rule two. If it has a long ingredient list... it's not food. It's a formulation.

Rule three. If it melts in your mouth... it's engineered to override satiety.

Rule four. If it's everywhere... it's engineered. Real food is found in a few places.

Rule five. If it's cheap, fast, and hyper-palatable... it's designed for repeat purchase.

Rule six. If it's marketed to children... it's a formulation. Real food doesn't need a mascot.

Rule seven. If it's a drink with flavour... it's a chemical system.

Rule eight. If it's "healthy" but still comes in a packet... be careful.

Rule nine. If it's hard to stop eating... it was designed that way.

... ... ...

Rule ten. ... Once you see it... you can't unsee it.

... ... ...

You start seeing the matrix. In your cupboard. In your fridge. In your child's lunchbox. In your office. In your supermarket.

... And once you see it... you can step outside it.

... ... ...""",

    # ===== 33. V1: CH10 - RECLAIMING YOUR BIOLOGY =====
    """Chapter Thirteen. ... Reclaiming Your Biology.

... ... ...

You've seen the system. You've seen the tricks. You've seen the patterns. You've seen the consequences.

... Now it's time to take back control.

Not through willpower. Not through discipline. Not through guilt.

... But through environment... awareness... and strategy.

Because biology doesn't respond to motivation. ... Biology responds to inputs.

... Change the inputs... and everything changes.

... ... ...

The first truth. It's not you. It's the environment.

... You don't need more willpower. ... You need a different environment.

... ... ...

The second truth. Small shifts change everything.

Biology doesn't need dramatic change. It needs consistent nudges. Hunger stabilises. Cravings reduce. Energy improves. Mood evens out. Sleep deepens. Clarity returns.

... ... ...

The method has three pillars.

Pillar one. Stabilise your biology. Add protein. Add fibre. Add healthy fats. Add real meals.

Pillar two. Change your environment. Make real food visible. Make ultra-processed foods invisible and inconvenient.

Pillar three. Shift your identity. When you become "the kind of person who eats real food"... the decisions make themselves.

... ... ...

... One real meal a day changes everything.

... You don't need a new diet. ... You need one real meal.

... And then another. ... And then another.

... Not perfection. ... Progress.

... ... ...""",

    # ===== 34. V2: CH4 - THE UPF AUDIT =====
    """Chapter Fourteen. ... The UPF Audit. ... How to Identify and Replace Ultra-Processed Foods.

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

... You don't need to eliminate UPFs entirely. Reducing them to twenty percent or less... delivers most of the benefit.

... ... ...""",

    # ===== 35. V1: CH11 - THE METHOD AND 10 SWAPS =====
    """Chapter Fifteen. ... The You Are What You Eat Method.

... ... ...

Most people don't need more information. ... They need a method.

... ... ...

Ten real-food swaps that change everything.

Swap one. Fizzy drink... becomes sparkling water with citrus. Same sensation. None of the engineering.

Swap two. Crisps... become nuts or seeds. Real fat. Real crunch. Real satiety.

Swap three. Chocolate bar... becomes fruit with protein.

Swap four. Ultra-processed breakfast cereal... becomes eggs, yoghurt, or oats. Start the day with anchors, not spikes.

Swap five. Protein bar... becomes a real food snack. Protein bars are ultra-processed foods in gym clothes.

Swap six. Ready meal... becomes leftovers. Real food without extra effort.

Swap seven. Ultra-processed lunch... becomes a simple real-food plate.

Swap eight. Energy drink... becomes water with a pinch of salt and citrus.

Swap nine. Ultra-processed dessert... becomes dark chocolate or fruit.

Swap ten. "Healthy" packaged snack... becomes anything with one ingredient. If it has a mascot, it's not food.

... ... ...""",

    # ===== 36. V2: CH5 - FAMILY STRATEGY =====
    """Chapter Sixteen. ... The Family Strategy. ... Reducing UPF With Kids Without Battles.

... ... ...

Kids don't need perfection, pressure, or policing. ... They need structure... modelling... and environment.

... The Parent First Principle. Children copy what they see, not what they're told.

... The Home Environment Rule. If it's in the house... it gets eaten. If it's not... it doesn't.

... The Three Category Food System. Everyday foods... Sometimes foods... Treat foods.

... The Snack Swap Strategy. Crisps become popcorn. Cereal bars become fruit and nuts. Fizzy drinks become flavoured water. Small swaps compound.

... The One Treat No Drama Rule. Allow one treat per day. Predictable, calm, consistent.

... The Family Meal Advantage. Shared meals reduce UPF intake by up to forty percent.

... ... ...""",

    # ===== 37. V1: CH12 - YOUR FIRST 30 DAYS =====
    """Chapter Seventeen. ... Your First Thirty Days.

... ... ...

Transformation doesn't happen in a year. It doesn't happen in a month. ... It happens in moments. Small decisions, repeated consistently.

... ... ...

Week one. One real meal a day.

This is the anchor. One meal per day built entirely from real ingredients. Protein. Vegetables. Healthy fats. You don't need to change the other meals yet.

... By day seven, you'll notice fewer crashes, more predictable hunger, and slightly better sleep.

... ... ...

Week two. Swap the biggest trigger.

Everyone has one. The product that pulls hardest. Identify yours. Swap it. One swap. Big impact.

... By day fourteen, the craving weakens. Not disappears. Weakens.

... ... ...

Week three. Fix the environment.

Go through your kitchen. Move ultra-processed foods to the highest cupboard. Put real food at eye level. Prep a safety net: boiled eggs, washed vegetables, nuts, cheese, tinned fish.

... By day twenty-one, you reach for real food automatically.

... ... ...

Week four. Build the rhythm.

Add a second real meal. Stabilise eating times. Reduce mindless snacking.

... By day thirty... energy is noticeably more stable. Cravings are quieter. Mood is steadier. You start to feel like a different person... not because you've suffered... but because your biology has settled.

... ... ...""",

    # ===== 38. V2: CH9 - METABOLISM EXPLAINED =====
    """Chapter Eighteen. ... The UPF Metabolism Link Explained Simply.

... ... ...

Metabolism is not just calories in, calories out.

... Insulin Overload. Frequent glucose spikes force the pancreas to release large amounts of insulin. Over time, cells become less responsive.

... Chronic Inflammation. Additives and emulsifiers irritate the gut lining. Low-grade inflammation interferes with metabolic regulation.

... Mitochondrial Stress. The body's energy factories function best with steady, nutrient-dense fuel.

... Hormonal Disruption. Chemicals from packaging interfere with appetite hormones like leptin and ghrelin.

... Microbiome Imbalance. UPFs starve beneficial bacteria and feed opportunistic species.

... ... ...

Metabolism is not broken. It is responding to the environment. Change the environment... and the body recalibrates.

... ... ...""",

    # ===== 39. V1: CH13-15 - IDENTITY, ENVIRONMENT, TOOLS =====
    """Chapter Nineteen. ... Identity Shift.

... ... ...

Most people try to change their eating habits by changing their behaviour. And every time, the same thing happens. They run out of motivation.

... Not because they're weak. But because they're using the wrong tool.

... Behaviour doesn't create identity. ... Identity creates behaviour.

... ... ...

There is a moment when you stop thinking "I'm trying to eat better"... and start thinking... "This is just what I do."

... That moment is the identity shift. It's the moment cravings lose their power. Real food feels natural. Confidence returns.

... It's the moment the system loses its grip.

... ... ...

Chapter Twenty. ... Designing Your Real-Food Life.

... ... ...

Identity is the engine. But environment is the road.

Design your home. Make real food visible. Make ultra-processed food less accessible. Have a default meal.

Design your work environment. Bring anchors. Avoid the vending machine. Protect the three PM window.

Design your social life. Eat real before you go out. Don't moralise food.

... Make the good choice the easy choice.

... ... ...

Chapter Twenty-One. ... The Tools That Help.

The ten-second label check. Look at the ingredient list on the back, not the front. Scan for words you wouldn't find in a kitchen: emulsifiers, stabilisers, modified starch, maltodextrin, flavourings, hydrogenated oils.

... If you see them... it's Group Four. Put it back.

The "could I make this?" test. Ask yourself: "Could I make this at home, with ingredients I recognise?" If the answer is no... it's ultra-processed.

The food scanner. Your phone becomes your x-ray. Point it at a barcode and see what you're really holding.

... The tools don't replace your judgement. ... They accelerate it.

... ... ...""",

    # ===== 40. V2: CH7 - APP INTEGRATION =====
    """Chapter Twenty-Two. ... Turning Knowledge Into Daily Action.

... ... ...

Books change understanding. ... Apps change behaviour.

... The Ingredient Scanner. Scan any food and instantly see... UPF score... additives... processing level... and healthier alternatives.

... The Daily Food Log. Not calorie counting... pattern tracking. Whole food percentage... UPF exposure... weekly trends.

... The Streak System. Streaks work. The app rewards whole food days, batch cook sessions, and UPF-free breakfasts.

... The Family Mode. Parents can track household patterns, plan meals, and create shared goals.

... The Habit Engine. Shopping suggestions. Swap recommendations. Meal ideas. Reminders to prep.

... The Thirty Day Reset Integration. The app guides you through the reset with daily tasks, tips, and progress tracking.

... This app is the practical extension of this book. The tool that makes change stick.

... ... ...""",

    # ===== 41. V1: CH16-18 - THE FUTURE AND CLOSING =====
    """Chapter Twenty-Three. ... The Real-Food Future.

... ... ...

There is a moment in every transformation where the personal becomes universal.

You start by changing your meals. Then your energy changes. Then your mood changes. Then your identity changes. Then your environment changes.

... And then something unexpected happens. Your life changes... and the people around you feel it.

... Energy is contagious. Clarity is contagious. ... Real food is contagious.

... ... ...

Imagine a home where real food is visible. Meals are simple. Energy is stable. Children are calmer. Mornings are smoother. Evenings are easier.

A home where biology is respected. Where the environment is designed for humans... not for corporations.

... ... ...

Chapter Twenty-Four. ... Living Outside the Matrix.

... ... ...

There is a moment when the world looks different. Not because the world changed... but because you did.

You walk into a supermarket and see the layout for what it is. You walk past a vending machine and feel nothing. You feel hunger as a signal, not a crisis. You feel energy as a baseline, not a rare event.

... This is what it feels like to live outside the matrix.

... Not perfect. Not pure. Not restrictive.

... Just free.

... ... ...

Chapter Twenty-Five. ... The Beginning.

... ... ...

Every journey has a moment where you realise you're not going back.

... Because you've changed.

... ... ...

There comes a day when you notice something subtle.

You walk past an ultra-processed food... and feel nothing.

No pull. No craving. No negotiation. No internal battle.

... Just neutrality.

... That's the moment you realise. ... You're free.

... ... ...

You are not broken. ... You were living in a system designed to overwhelm you.

You are not weak. ... You were eating foods engineered to hijack your biology.

You are not failing. ... You were fighting a battle you were never meant to fight alone.

But now you see the system. Now you have the method. Now you have the tools. Now you have the identity.

... And now you have the power.

... ... ...

This is not the end.

... This is the beginning.

... The beginning of your real-food life. ... The beginning of your real-energy life. ... The beginning of your real-clarity life. ... The beginning of your real-you life.

... A life lived outside the matrix. ... With freedom, stability, and confidence.

... A life built on real food.

... A life built on... real you.""",

    # ===== 42. V2: FINAL CHAPTER =====
    """The Final Chapter. ... The Movement Starts With You.

... ... ...

You've reached the end of the book... but the journey is just beginning. Reducing UPF is not a trend. ... It is a movement.

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

    # ===== 43. CLOSING CREDITS =====
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

async def generate_complete_audiobook():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    total = len(COMPLETE_BOOK)
    
    for i, text in enumerate(COMPLETE_BOOK):
        text = text.strip()
        print(f"\n[{i+1}/{total}] Generating section ({len(text)} chars)...", flush=True)
        
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
    
    print(f"\n{'='*50}", flush=True)
    print(f"COMPLETE audiobook saved: {path}", flush=True)
    print(f"  Size: {size_mb} MB", flush=True)
    print(f"  Sections: {total}", flush=True)
    print(f"  Chapters: 25 + credits", flush=True)
    for name in ["audiobook_full.mp3", "audiobook_v2.mp3"]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            s = round(os.path.getsize(p) / (1024*1024), 1)
            print(f"  vs {name}: {s} MB", flush=True)
    print(f"{'='*50}", flush=True)

asyncio.run(generate_complete_audiobook())
