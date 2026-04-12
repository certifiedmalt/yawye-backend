import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

FINAL_BOOK = [

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

    # ===== 3. V1: OPENING QUOTES (FULL) =====
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

    # ===== 4. V1: PROLOGUE (FULL) =====
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

    # ===== 6. V2: CHAPTER 1 - THE 10 KEY STUDIES =====
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

    # ===== 7. V2: CHAPTER 2 - FIVE MECHANISMS =====
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

    # ===== 8. V1: CHAPTER 3 - THE REFINERY PART 1 (FULL) =====
    """Chapter Three. ... The Refinery. ... The Industrial Birth of Seed Oils.

... ... ...

If you could watch your food being made...

... you would never eat the same way again.

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

    # ===== 9. V1: CHAPTER 3 - THE PROCESS PART 2 (FULL) =====
    """... ... ...

The process begins.

... ... ...

Step one. ... Crushing.

The seeds are not gently pressed. They are crushed... ground... and pulverised into a fine meal. The goal is simple: break the seed so aggressively that every microscopic droplet of oil becomes accessible.

... This is the first moment where the process stops resembling food.

... ... ...

Step two. ... Solvent Extraction.

The ground seed meal is washed... in hexane.

... Hexane. ... A petroleum-derived solvent.

The same chemical used in industrial degreasing... in glue production... and in chemical extraction.

... Hexane dissolves the oil out of the seed. The mixture becomes a slurry... part food... part chemical bath.

... ... ...

Let that sit for a moment.

The oil in your kitchen cupboard... was washed out of crushed seeds... using a petroleum solvent.

... This is the moment most people would stop calling it food.

... ... ...

Step three. ... Desolventising.

The hexane-oil mixture is heated so the solvent evaporates. The solvent is captured and reused. The oil remains... but it is dark, bitter, and full of impurities.

... It smells nothing like food.

... So the industry keeps going.""",

    # ===== 10. V1: CHAPTER 3 - DEGUMMING THROUGH BLEACHING (FULL) =====
    """... ... ...

Step four. ... Degumming.

The crude oil contains phospholipids... natural components of seeds. Industry calls them "gums." They are removed using water, steam, or acid. The oil becomes clearer... but still unappealing.

... ... ...

Step five. ... Neutralisation.

Free fatty acids are removed... using sodium hydroxide.

... Sodium hydroxide. ... The same chemical used in soap making.

... This reduces bitterness. It also removes nutrients.

... The oil is now chemically altered.

... ... ...

Step six. ... Bleaching.

... This is the moment the oil stops looking like food.

The oil is heated and mixed with acid-activated bleaching clays.

... The same materials used in... petroleum refining. ... Wastewater treatment. ... Chemical purification.

... ... ...

These clays pull out pigments... metals... breakdown products... and oxidation compounds.

What was once brown... becomes pale. What was once visibly industrial... becomes visually "clean."

... Not because it has become healthier.

... But because it has become more marketable.

... ... ...

Think about that. The bleaching step doesn't make the oil better for you. It makes the oil easier to sell to you.""",

    # ===== 11. V1: CHAPTER 3 - DEODORISATION AND COMPARISON (FULL) =====
    """... ... ...

Step seven. ... The final transformation. ... Deodorisation.

The oil is heated again... this time to temperatures between two hundred... and two hundred and sixty degrees Celsius.

... This removes the harsh, industrial smell created by the refining process.

But high heat does something else.

... It breaks fatty acids. ... It destroys nutrients. ... It creates new compounds. ... It alters the structure of the oil itself.

... This is the moment the oil becomes shelf-stable... and biologically unfamiliar.

... ... ...

At this point, the process looks so industrial... that it's worth stepping back... and asking a simple question.

... What other products are made this way?

... ... ...

Vegetable oil refining. Step one: crush seeds. Step two: extract with hexane solvent. Step three: evaporate solvent. Step four: degum. Step five: neutralise with caustic soda. Step six: bleach with activated clays. Step seven: deodorise at two hundred and sixty degrees. Step eight: bottle and sell as food.

... ... ...

Fuel refining. Step one: distil crude oil. Step two: use solvents and catalysts to separate fractions. Step three: remove impurities. Step four: treat with chemicals to neutralise acids. Step five: bleach or filter to remove colour. Step six: heat to high temperatures to stabilise. Step seven: blend and store as fuel.

... ... ...

The clear, odourless oil on supermarket shelves is the result of crushing... solvent extraction... boiling... degumming... neutralisation... bleaching... and high-heat deodorisation.

... It is not the oil of a seed.

... It is the oil of a refinery.

... ... ...

And it is now one of the most consumed substances in the modern diet. Not because it is healthy. Not because it is natural.

... But because it is profitable.""",

    # ===== 12. V1: CHAPTER 3 - MYTH VS TRUTH (FULL) =====
    """... ... ...

The Myth... versus the Truth. And why the truth is even more revealing.

... ... ...

There's a story that circulates online... dramatic, viral, and easy to repeat.

"Seed oils were originally made for livestock, but the animals kept dying, so they banned it for animals... and fed it to humans instead."

... It's a powerful story. It feels like something that should be true. It fits the tone of industrial betrayal.

... But it isn't what happened.

There is no historical record of livestock dying from seed oils. No governments banning seed oils in animal feed. No manufacturers redirecting toxic oil into the human food supply.

... It's a myth. Compelling... but inaccurate.

... And the real story... is actually more shocking.

... ... ...

The truth.

Seed oils began... as industrial waste.

In the late eighteen hundreds to early nineteen hundreds, cottonseed oil was used for machinery lubrication... candles... soap... industrial greases... and lamp fuel.

... It was cheap. It was abundant. ... And it was considered unfit for human consumption.

... ... ...

Then, chemists discovered they could harden it.

Hydrogenation... a German industrial process... turned liquid cottonseed oil into a white, lard-like fat.

... This was the birth of Crisco... in nineteen eleven.

... Not a food innovation. ... A chemical one.""",

    # ===== 13. V1: CHAPTER 3 - CLOSING (FULL) =====
    """Procter and Gamble launched one of the most aggressive food marketing campaigns in history. ... "Pure." ... "Modern." ... "Clean." ... "Scientific."

... They reframed industrial oil... as a health product.

... ... ...

And here is the livestock connection that is true.

... Livestock didn't die from seed oils. ... They did something else.

... They got fatter. ... Faster. ... Cheaper.

Cottonseed meal, soybean meal, and corn by-products... all high in industrial polyunsaturated fatty acids... were used because they increased weight gain... reduced feed costs... and boosted profitability.

... This is historically documented. Animals didn't collapse. ... They grew.

... ... ...

And here is the twist.

... The same industrial oils used to fatten livestock cheaply... became the backbone of the modern human diet.

Not because they were healthy. ... Not because they were traditional. ... Not because they were demanded.

... But because they were cheap... abundant... profitable... chemically malleable... and easy to market.

... ... ...

The myth says seed oils were banned for animals. ... The truth says they were perfect for fattening them.

The myth says seed oils were fed to humans by accident. ... The truth says they were fed to humans... by design.

The myth says seed oils were a mistake. ... The truth says they were a business model.

... ... ...

And that business model... still shapes the modern diet today.

... Key takeaway... The oil in your kitchen went through the same industrial process as petroleum. Not because it's healthy. Because it's profitable.

... ... ...""",

    # ===== 14. V1: CHAPTER 4 - THE ADDITIVE EXPLOSION PART 1 (FULL) =====
    """Chapter Four. ... The Additive Explosion.

... ... ...

If Chapter Three showed you how modern oils are born in refineries... this chapter shows you what happens next. When those oils become the foundation for a new kind of product.

... Not food. ... Not ingredients. ... But formulations.

... ... ...

Ultra-processed foods are not built the way meals are built. They are assembled... the way products are assembled. From components, stabilisers, emulsifiers, colours, and chemicals... that each play a role in creating the illusion of food.

This is the part of the story most people never see.

Because additives don't live in kitchens. ... They live in laboratories.

... ... ...

A recipe is simple. Ingredients. Heat. Time. Skill.

A formulation is different. Stabilisers. Emulsifiers. Humectants. Anti-foaming agents. Acidity regulators. Colourants. Preservatives. Artificial flavours. Sweeteners. Gums. Texturisers.

... Each one has a job. Each one solves a problem. Each one makes the product behave... in a way nature never intended.

... ... ...

Ultra-processed foods are not built to nourish. ... They are built to perform.""",

    # ===== 15. V1: CHAPTER 4 - EMULSIFIERS AND STABILISERS (FULL) =====
    """... ... ...

The emulsifier problem.

Emulsifiers are the glue of the ultra-processed food world.

They make oil and water mix. They make sauces smooth. They make ice cream soft. They make chocolate glossy. They make bread stay soft... for weeks.

... Common emulsifiers include lecithins... mono and diglycerides... polysorbates... carboxymethylcellulose... carrageenan.

... These are not kitchen ingredients. ... They are industrial tools.

... ... ...

Emulsifiers change the texture of food... and they also change how food interacts with the gut. They make food easier to eat quickly. They make food easier to over-consume. They make food behave like a product... not a meal.

... ... ...

Stabilisers and gums.

Stabilisers keep ultra-processed foods from separating, collapsing, or spoiling.

Gums... guar, xanthan, locust bean... give ultra-processed foods their signature textures. The chewiness of a protein bar. The thickness of a milkshake. The stretch of a plant-based cheese. The "creaminess" of a low-fat yoghurt.

... These gums absorb water and swell, creating volume without calories.

This is why ultra-processed foods feel filling... but leave you hungry again soon after.

... Your stomach feels full. ... Your biology does not.""",

    # ===== 16. V1: CHAPTER 4 - SWEETENERS, COLOURS, PRESERVATIVES (FULL) =====
    """... ... ...

Sweeteners and flavours.

Sweetness used to be rare. ... Now it is engineered.

Ultra-processed foods use high-intensity sweeteners... sugar alcohols... syrups... flavour enhancers... artificial flavours... and "natural flavours"... a legal category that can include dozens of compounds.

... Flavours are not fruits. They are formulas.

A strawberry yoghurt may contain no strawberries. A vanilla drink may contain no vanilla. A cheese puff may contain no cheese.

... Flavours are designed to mimic nature... and then surpass it.

This is why ultra-processed foods taste "better" than real food. ... They are engineered to.

... ... ...

Colours.

Colour is psychology.

If a product looks vibrant, fresh, or fruity... the brain believes it tastes better. So ultra-processed foods use artificial colours, natural colourants, caramel colour, beetroot red, turmeric yellow, paprika extract.

... These colours are not added for nutrition. ... They are added for persuasion.

They make products look alive... even when nothing inside them is.

... ... ...

Preservatives.

Preservatives are the reason ultra-processed foods last weeks on shelves... months in warehouses... years in storage. They stop mould. They stop bacteria. They stop oxidation. They stop decay.

... In other words... they stop food from behaving like food.

... Real food spoils. ... Ultra-processed food doesn't.

... That alone should tell you something.""",

    # ===== 17. V1: CHAPTER 4 - TEXTURE AND CLOSING (FULL) =====
    """... ... ...

The engineering of texture.

Texture is one of the most powerful tools in the ultra-processed food arsenal.

Crispiness. Crunchiness. Chewiness. Creaminess. Melt-in-the-mouth softness.

These sensations are engineered using starches, gums, emulsifiers, aeration, extrusion, and high-pressure processing.

... Texture is not an accident. ... It is a design choice.

And it is one of the reasons ultra-processed foods are so easy to overeat. Your brain is wired to respond to texture. ... Ultra-processed foods exploit that wiring.

... ... ...

Here is the truth.

Ultra-processed foods are not competing on nutrition. ... They are competing on engineering.

Every additive solves a problem. How do we make this cheaper? How do we make this last longer? How do we make this more addictive? How do we make this feel like real food? How do we make this survive shipping? How do we make this irresistible to children?

... The result is a product that behaves like food... but is built like a chemical system.

... This is not a conspiracy. ... It is a business model.

... ... ...

Additives are not "poisons." They are not "toxins." They are not "illegal."

But they are industrial tools, designed to make products cheaper, more stable, more profitable, more palatable... and more addictive.

... And they are now everywhere.

In your snacks. In your drinks. In your sauces. In your bread. In your cereal. ... In your children's lunchboxes.

... Key takeaway... Every additive solves a business problem. None of them solve a health problem.

... ... ...""",

    # ===== 18. V1: CHAPTER 5 - FIZZY DRINKS PART 1 (FULL) =====
    """Chapter Five. ... The Fizzy Drink. ... The Most Engineered Product in the Modern Diet.

... ... ...

Before you understand the biology, the chemistry, or the psychology of fizzy drinks... you need to understand the economics.

The UK now spends more than twenty-one billion pounds a year on soft drinks. And over eight to ten billion of that is on fizzy and energy drinks alone.

... These aren't beverages. ... They're chemical formulations.

And they have become one of the most profitable product categories in the entire food system. Engineered for stimulation. Engineered for repeat consumption. Engineered to dominate.

... Fizzy drinks are not drinks. ... They are products.

And once you see how they're built... you'll never see them the same way again.

... ... ...

A fizzy drink is not made. ... It is assembled.

The components: acid... sweetness... colour... flavour... carbonation... preservatives... stabilisers... treated water.

Each one is engineered. Each one has a job. Each one exists to create the illusion of refreshment.

... Let's take them one by one.""",

    # ===== 19. V1: CHAPTER 5 - ACIDS (FULL) =====
    """... ... ...

Number one. ... The acid. The bite behind the bubbles.

Fizzy drinks are acidic. Far more acidic than most people realise.

Phosphoric acid in colas. Citric acid in fruit sodas. Malic acid in sour drinks.

... These acids are chosen not for nutrition, but for performance. Sharpness. Stability. Shelf life. And the engineered sensation of "refreshment."

... ... ...

And here's the part that stops people mid-sentence.

These same acids appear in... rust removers... dishwasher tablets... metal cleaners... cosmetics... and industrial descalers.

... Not because fizzy drinks are toxic. But because these acids are powerful, predictable, and effective.

... ... ...

Phosphoric acid gives cola its signature "bite." It drops the pH so low that bacteria can't grow... and flavour hits harder. ... It's not a flavour. It's a tool.

Citric acid enhances fruitiness and boosts sweetness perception. But the citric acid in drinks is not squeezed from lemons. It is almost always industrial citric acid... produced by fermenting sugar with mould. ... It tastes like fruit. It behaves like chemistry.

Malic acid gives a deeper, longer-lasting sourness... the kind that makes your mouth water and your brain light up. ... It's not natural sourness. It's engineered sourness.

... ... ...

These acids aren't rare. They're the industry's favourites. Phosphoric acid for bite. Citric acid for brightness. Malic acid for depth. The same acids used in rust removers, limescale cleaners, cosmetics, and industrial descalers.

... They're not added to nourish you. ... They're added to engineer the sensation of "refreshment"... a sensation designed to override biology.""",

    # ===== 20. V1: CHAPTER 5 - SWEETNESS, COLOUR, FLAVOUR (FULL) =====
    """... ... ...

Number two. ... The sweetness. The dopamine trigger.

A single can of regular fizzy drink contains more sugar than most people would ever add to anything they make at home.

... But the sweetness isn't just about taste. ... It's about dopamine.

Sweetness is one of the most powerful reward triggers in the human brain. Fizzy drinks deliver it in a form that is fast... intense... unbuffered by fibre... unbalanced by nutrients.

... It hits the bloodstream like a signal flare.

And when sugar is removed, the industry doesn't reduce sweetness... it increases it... using high-intensity sweeteners hundreds of times sweeter than sugar.

... Your brain learns the pattern quickly. Drink. Dopamine. Repeat.

... ... ...

Number three. ... The colour. The psychology of appearance.

Colour is persuasion. A cola must be brown. An orange soda must be orange. A lemon-lime drink must be clear.

These colours are not natural. They are engineered. Caramel colour... used in colas... is created by heating sugars with acids or alkalis. It is not caramel from a pan. It is industrial caramel.

... Without colour, the drink would look like chemical water. ... With colour, it looks like a treat.

... ... ...

Number four. ... The flavour. The illusion of fruit.

Flavour is the most mysterious part of a fizzy drink. It is not fruit. It is not natural. It is not simple.

Flavour is a formula... often containing dozens of compounds, blended to create a sensory illusion.

... The recipe is proprietary. ... The effect is universal.

... Flavour is not taste. ... It is engineering.""",

    # ===== 21. V1: CHAPTER 5 - CARBONATION, PRESERVATIVES, CLOSING (FULL) =====
    """... ... ...

Number five. ... The carbonation. ... The pain that feels like pleasure.

Carbonation is not just bubbles. ... It is a sensory amplifier.

CO2 triggers pain receptors in the mouth... the same receptors activated by mustard and chilli. This creates a tingling sensation that makes the drink feel "alive."

... Without carbonation, the drink is syrup. ... With carbonation... it is addictive.

... ... ...

Number six. ... The preservatives.

Sodium benzoate. Potassium sorbate. These prevent microbial growth.

But here's the part most people don't know. Under certain conditions... heat plus light plus benzoate plus vitamin C... benzene can form in trace amounts.

... Fizzy drinks behave like chemical systems... because they are chemical systems.

... ... ...

Number seven. ... The water.

The water in fizzy drinks is not tap water. It is treated, filtered, softened, and adjusted to create the perfect base for the formula.

... It is not water. ... It is a solvent.

... ... ...

Fizzy drinks are engineered to be consumed quickly and repeatedly.

They are sweet enough to trigger dopamine. Acidic enough to feel refreshing. Carbonated enough to stimulate the mouth. Flavoured enough to override satiety. Coloured enough to create expectation. Preserved enough to last forever.

... This is not a beverage. ... This is a behavioural product.

... And it is one of the most profitable inventions in human history.

... ... ...

Fizzy drinks do not nourish. They stimulate. They do not hydrate. They override. They do not satisfy. ... They provoke.

They are the perfect example of a product designed not for health... but for consumption.

... And they are consumed everywhere. ... Especially by children.

... Key takeaway... A fizzy drink is not a beverage. It is a behavioural product... engineered to be consumed quickly, repeatedly, and in large quantities.

... ... ...

Now that you've seen what's inside these products... let's talk about what they do to your brain.

... ... ...""",

    # ===== 22. V2: HOW UPF HIJACKS BEHAVIOUR =====
    """Chapter Six. ... How Ultra-Processed Food Hijacks Behaviour.

... ... ...

UPFs don't just affect the body. ... They affect the brain. The last six years of research have shown that UPFs exploit the same neural pathways involved in reward... habit formation... and compulsive behaviour.

... ... ...

The Dopamine Loop. ... Hyper-palatable foods trigger rapid dopamine spikes. Over time, the brain adapts, requiring more stimulation to achieve the same reward. ... This mirrors the pattern seen in other compulsive behaviours.

... ... ...

Satiety Signal Disruption. ... UPFs bypass the mechanical and chemical signals that whole foods trigger. Without fibre, structure, or slow digestion... the body struggles to register fullness.

... ... ...

Craving Amplification. ... Rapid glucose spikes followed by crashes create a physiological craving cycle. The body seeks fast energy... and UPFs deliver it... temporarily.

... ... ...

Habit Formation. ... UPFs are cheap, convenient, and everywhere. This environmental saturation makes them easy to default to... especially under stress, fatigue, or time pressure.

... ... ...

Emotional Eating. ... UPFs provide immediate sensory reward. In moments of stress, boredom, or low mood, they offer a quick dopamine hit... reinforcing the loop.

... ... ...

Key takeaway... UPF is not a willpower issue. It is a design issue. These foods are engineered to be eaten quickly, repeatedly, and in large quantities. Understanding this frees you from blame.

... ... ...""",

    # ===== 23. V2: THE SCIENCE OF CRAVINGS =====
    """Chapter Seven. ... The Science of Cravings. ... Why Your Body Wants What It Doesn't Need.

... ... ...

Cravings are not a character flaw. ... They are chemistry. And ultra-processed foods are engineered to exploit that chemistry... with precision.

... The Dopamine Spike. UPFs deliver rapid sensory reward... sweetness, crunch, salt, fat... triggering dopamine release. Over time, the brain adapts, requiring more stimulation for the same effect. ... This is why cravings intensify... not fade.

... ... ...

The Blood Sugar Whiplash. UPFs often contain rapidly absorbable carbohydrates. They spike glucose... spike insulin... then cause a crash. The crash triggers hunger, irritability, and a drive for fast energy... usually more UPF.

... ... ...

The Gut Brain Loop. The microbiome shifts in response to diet. Diets high in UPF promote bacteria that thrive on sugar and refined carbs. These microbes send signals that influence cravings... mood... and appetite.

... ... ...

The Stress Connection. Cortisol increases appetite for high-energy, high-reward foods. UPFs provide immediate relief... reinforcing the stress, craving, UPF loop.

... ... ...

The Habit Layer. Cravings are not just biological. They are contextual. Time of day, location, mood, and routine all trigger learned associations.

... ... ...

Key takeaway... Cravings are predictable, understandable, and manageable. You can break the loop... not through willpower... but through strategy.

... ... ...""",

    # ===== 24. V1: CHAPTER 6+7 - MODERN ILLNESS + CHILDREN'S CRISIS (FULL) =====
    """Chapter Eight. ... The Human Cost. ... The Rise of Modern Illness.

... ... ...

The modern food economy exploded. ... And something else exploded with it.

Not overnight. Not dramatically. But steadily... quietly... relentlessly... year after year, decade after decade... until it became the background noise of modern life.

A rise in conditions that were once rare. A rise in symptoms that were once unusual. A rise in illnesses that were once confined to adulthood... now appearing in children.

... ... ...

For most of human history, chronic metabolic diseases were rare. Not because people were healthier... but because the environment was different.

Food was perishable. Fibrous. Slow-digesting. Nutrient-dense. Limited by nature.

... Then came refined oils. Industrial sweeteners. Engineered flavours. Additives. Shelf-stable snacks. Fizzy drinks. Ready meals. Hyper-palatable formulations.

... And the timeline shifted.

... ... ...

Over the same decades that ultra-processed foods became dominant, the world saw dramatic increases in... obesity. Type 2 diabetes. Fatty liver disease. Metabolic dysfunction. Chronic inflammation. Cardiovascular issues.

... No single food "caused" these conditions. No single ingredient is to blame.

... But the system changed. ... And the body responded.""",

    # ===== 25. V1: INVISIBLE SYMPTOMS AND EMOTIONAL COST (FULL) =====
    """... ... ...

The biological mismatch.

The human body evolved for whole foods... slow energy... fibre... protein... natural fats... seasonal eating.

Instead, it received refined oils... engineered sweetness... industrial textures... additives... emulsifiers... hyper-palatable formulations... liquid sugar... chemical acidity.

... This is not a small mismatch. ... It is a biological collision.

... Your body is ancient. Your food is modern. And the gap between the two... is where modern illness lives.

... ... ...

Not every consequence of ultra-processed food shows up in a diagnosis. Most show up in everyday life.

The symptoms people feel but rarely connect to food... fatigue... cravings... mood swings... low energy... poor sleep... difficulty concentrating... constant hunger... afternoon crashes... irritability... brain fog.

... These are not character flaws. ... They are biological signals.

Signals that the system is overwhelmed. Signals that the environment is mismatched. Signals that the body is trying to cope... with a diet it was never designed for.

... ... ...

And ultra-processed foods create cycles. Eat. Crash. Crave. Repeat.

People blame themselves. They think they lack willpower. They think they're weak. They think they're the problem.

... But the truth is simple.

... The system is designed this way.

... ... ...

Once you see the pattern... once you see the timeline... once you see the mismatch... you understand something powerful.

... You are not the problem. ... The system is.

And once you see the system... you can step outside it.""",

    # ===== 26. V1: THE CHILDREN'S CRISIS PART 1 (FULL) =====
    """... ... ...

And nowhere is this more urgent... than with our children.

... ... ...

Chapter Nine. ... The Children's Crisis.

... ... ...

Children are not miniature adults. ... They are developing systems. Fragile. Sensitive. Rapidly changing. And profoundly shaped by their environment.

And today's environment... is unlike anything any generation of children has ever faced.

... ... ...

For the first time in human history, children are growing up in a food system where ultra-processed foods dominate their calories... fizzy drinks are normalised... snacks are constant... sweetness is everywhere... marketing is relentless... and real food is the exception, not the rule.

... This chapter is not about blame. ... It is about exposure.

Because children are not choosing this. ... It is being chosen for them.

... ... ...

Every generation before this one grew up eating real meals, real ingredients, real food environments.

This generation is growing up eating formulations... additives... refined oils... engineered sweetness... industrial textures... cartoon-branded snacks... liquid sugar... hyper-palatable products.

Children are not just eating ultra-processed foods. ... They are immersed in them.

... In their lunchboxes. In their schools. In their sports clubs. In their birthday parties. In their vending machines. In their after-school snacks. In their drinks. In their celebrations. In their routines.

... This is not a dietary shift. ... It is a cultural one.""",

    # ===== 27. V1: CHILDREN'S CRISIS - BIOLOGY AND MARKETING (FULL) =====
    """... ... ...

Children are more vulnerable than adults for one simple reason.

... Their biology is still being built.

Their brains. Their hormones. Their microbiomes. Their taste preferences. Their reward pathways. Their appetite regulation. Their metabolic systems.

... All under construction.

... And ultra-processed foods hit every one of those systems harder.

... ... ...

Their brains are more sensitive to sweetness. High sweetness rewires reward pathways faster in children.

Their taste preferences are still forming. What they eat now... becomes what they crave later.

Their microbiome is still developing. Additives, emulsifiers, and engineered foods shape it in ways we're only beginning to understand.

Their appetite signals are immature. Hyper-palatable foods override satiety more easily.

Their bodies are smaller. Meaning the same dose of sugar, additives, or sweetness... hits harder.

... Children are not resilient to ultra-processed foods. ... They are exposed to them.

... ... ...

And children are marketed to more aggressively than any demographic on Earth.

Ultra-processed food companies use cartoons... mascots... bright colours... collectible packaging... influencers... YouTube ads... TikTok trends... sports sponsorships... school partnerships... "fun size" snacks... "kid-friendly" flavours.

... Children do not stand a chance. ... And parents are outnumbered.

... This is not a fair fight. ... It is a billion-pound industry... targeting developing brains.

... ... ...

Parents are not failing. ... Parents are overwhelmed.

They are fighting convenience... marketing... peer pressure... school environments... time scarcity... rising food prices... engineered cravings... constant exposure.

... Parents are not choosing ultra-processed foods for their children. ... Ultra-processed foods are choosing their children.

... Key takeaway... You are not the problem. The system is. And our children are paying the highest price.

... ... ...""",

    # ===== 28. V2: THE UPF AUDIT (FULL) =====
    """Chapter Ten. ... The UPF Audit. ... How to Identify and Replace Ultra-Processed Foods.

... ... ...

Knowledge is useless without action. This chapter gives you a simple, practical system to identify UPFs in the real world... and replace them without spending more money or more time.

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

... People don't rise to the level of their goals. ... They fall to the level of their defaults. Build a home environment where whole foods are the easiest option.

... ... ...

Step Four... The Eighty Percent Rule.

... You don't need to eliminate UPFs entirely. Reducing them to twenty percent or less of total intake... delivers most of the benefit.

... Key takeaway... Swap, don't stop. The goal isn't perfection. It's replacement.

... ... ...""",

    # ===== 29. V2: FAMILY STRATEGY (FULL) =====
    """Chapter Eleven. ... The Family Strategy. ... Reducing UPF With Kids Without Battles.

... ... ...

If UPF reduction is hard for adults... it can feel impossible with children. But the last decade of behavioural science shows something powerful... kids don't need perfection, pressure, or policing. ... They need structure... modelling... and environment.

... ... ...

The Parent First Principle. ... Children copy what they see, not what they're told. When adults shift their own defaults... breakfast, snacks, drinks... kids follow naturally. No lectures required.

... ... ...

The Home Environment Rule. ... If it's in the house... it gets eaten. If it's not... it doesn't. This is the single strongest predictor of a child's diet quality. Parents don't need to restrict... they need to curate.

... ... ...

The Three Category Food System. ... Instead of good versus bad, use...
... Everyday foods... whole foods.
... Sometimes foods... minimally processed.
... Treat foods... UPFs.
... This removes shame and gives kids clarity.

... ... ...

The Eighty Twenty Family Pattern. ... Children don't need to eliminate UPFs. They need a pattern where whole foods dominate. Eighty percent whole or minimally processed... twenty percent flexible.

... ... ...

The Snack Swap Strategy. ... Kids eat what's easy. Replace...
... Crisps with popcorn or nuts.
... Flavoured yoghurt with plain yoghurt and honey.
... Cereal bars with fruit and nut mixes.
... Fizzy drinks with flavoured water.
... Small swaps compound.

... ... ...

The One Treat No Drama Rule. ... Allow one treat per day or per occasion. Predictable, calm, and consistent. This prevents binge-restrict cycles.

... ... ...

The Family Meal Advantage. ... Shared meals reduce UPF intake by up to forty percent. Not because of the food... but because of the structure, routine, and modelling.

... Key takeaway... UPF reduction with kids is a leadership challenge, not a discipline challenge. When the environment changes... behaviour follows.

... ... ...""",

    # ===== 30. V2: 30-DAY METABOLIC RESET (FULL) =====
    """Chapter Twelve. ... The Metabolic Reset. ... A Thirty Day Plan to Rebuild Your Health.

... ... ...

This chapter gives you a simple, structured thirty day reset designed to lower UPF intake, stabilise blood sugar, reduce cravings, and rebuild metabolic resilience. ... It is not a diet. ... It is a pattern.

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

Key takeaway... You don't need a new diet. You need one real meal. And then another. And then another.

... ... ...""",

    # ===== 31. V2: METABOLISM EXPLAINED =====
    """Chapter Thirteen. ... The UPF Metabolism Link Explained Simply.

... ... ...

Metabolism is not just calories in, calories out. It is a complex system involving hormones, enzymes, gut bacteria, and cellular signalling. Ultra-processed foods disrupt this system in five major ways.

... Insulin Overload. Frequent glucose spikes from UPFs force the pancreas to release large amounts of insulin. Over time, cells become less responsive... the first step toward insulin resistance.

... ... ...

Chronic Inflammation. Additives, emulsifiers, and certain fats can irritate the gut lining and immune system. Low-grade inflammation interferes with metabolic regulation and fat storage.

... ... ...

Mitochondrial Stress. Mitochondria... the body's energy factories... function best with steady, nutrient-dense fuel. UPFs provide erratic, low-quality energy that impairs efficiency.

... ... ...

Hormonal Disruption. Endocrine disrupting chemicals from packaging can interfere with appetite hormones like leptin and ghrelin... making hunger harder to regulate.

... ... ...

Microbiome Imbalance. UPFs starve beneficial bacteria and feed opportunistic species. A disrupted microbiome affects digestion... cravings... immunity... and metabolic flexibility.

... ... ...

Key takeaway... Metabolism is not broken. It is responding to the environment. Change the environment... and the body recalibrates.

... ... ...""",

    # ===== 32. V2: LONG-TERM PLAN (FULL) =====
    """Chapter Fourteen. ... The Long Term Plan. ... How to Live UPF Light for Life.

... ... ...

Short-term changes are easy. Long-term change requires identity... environment... and rhythm. This chapter gives you a sustainable blueprint for living UPF-light... without rigidity or obsession.

... ... ...

Identity Shift. ... People who succeed don't say... I'm trying to eat better. ... They say... I'm someone who eats real food. ... Identity drives behaviour.

... ... ...

The Eighty Twenty Lifestyle. ... Perfection is unnecessary and unsustainable. Eighty percent whole or minimally processed. Twenty percent flexible. ... This delivers almost all the health benefit... with none of the stress.

... ... ...

The Weekly Rhythm. ... A sustainable routine includes... one batch cook session... one big shop... one treat window per day... one family meal per day. ... Rhythm beats motivation.

... ... ...

The Environment Reset. ... Keep whole foods visible and accessible. Keep UPFs out of the house or in inconvenient places. ... Environment is destiny.

... ... ...

The Social Strategy. ... UPFs are everywhere... parties, work, travel. The strategy... enjoy the moment, then return to your rhythm. No guilt. No spirals.

... Key takeaway... You don't need perfection. You need a rhythm you can repeat.

... ... ...""",

    # ===== 33. V2: APP INTEGRATION =====
    """Chapter Fifteen. ... Turning Knowledge Into Daily Action.

... ... ...

Books change understanding. ... Apps change behaviour. This chapter shows you how the companion app turns the science of UPF reduction into a daily, personalised system.

... The Ingredient Scanner. Scan any food and instantly see... UPF score... additives... processing level... and healthier alternatives. This removes guesswork and builds awareness.

... ... ...

The Daily Food Log. Not calorie counting... pattern tracking. The app highlights... whole food percentage... UPF exposure... and weekly trends. This creates accountability without obsession.

... ... ...

The Streak System. Behavioural science is clear... streaks work. The app rewards... whole food days... batch cook sessions... and UPF-free breakfasts. This builds momentum.

... ... ...

The Family Mode. Parents can track household patterns, plan meals, and create shared goals. This turns UPF reduction into a team effort.

... ... ...

The Habit Engine. The app nudges you with... shopping suggestions... swap recommendations... meal ideas... and reminders to prep or batch cook. This bridges the gap between intention and action.

... ... ...

The Thirty Day Reset Integration. The app guides you through the reset from Chapter Twelve with daily tasks, tips, and progress tracking.

... Key takeaway... This app is the practical extension of this book. The tool that makes change stick.

... ... ...""",

    # ===== 34. V2: FINAL CHAPTER =====
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

    # ===== 35. CLOSING CREDITS + END CARD =====
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

async def generate_final_audiobook():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    total = len(FINAL_BOOK)
    
    for i, text in enumerate(FINAL_BOOK):
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
    
    print(f"\n{'='*50}")
    print(f"FINAL audiobook saved: {path}")
    print(f"  Size: {size_mb} MB")
    print(f"  Sections: {total}")
    for name in ["audiobook_full.mp3", "audiobook_v2.mp3"]:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            s = round(os.path.getsize(p) / (1024*1024), 1)
            print(f"  vs {name}: {s} MB")
    print(f"{'='*50}")

asyncio.run(generate_final_audiobook())
