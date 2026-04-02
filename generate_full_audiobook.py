import asyncio
import os
import re
from dotenv import load_dotenv
load_dotenv("/app/backend/.env")

from emergentintegrations.llm.openai import OpenAITextToSpeech

api_key = os.environ.get("EMERGENT_LLM_KEY")
tts = OpenAITextToSpeech(api_key=api_key)

# Dramatic pause markers:
# "..." = short pause (~0.5s)
# "... ... ..." = medium pause (~1.5s)  
# Section breaks use extra periods and line breaks

FULL_BOOK = [
    # ===== OPENING QUOTES =====
    """You Are What You Eat.

... ... ...

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

    # ===== PROLOGUE =====
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

    # ===== INTRODUCTION PART 1 =====
    """... ... ...

Introduction. ... The Cost of Convenience.

... ... ...

The modern world is getting sicker.

Not in small ways. ... Not in subtle ways. ... But in ways so widespread and so rapid... that entire branches of medicine have had to rewrite themselves just to keep up.

... ... ...

In a single generation... rates of obesity, type 2 diabetes, fatty liver disease, and metabolic disorders have surged. Conditions once seen only in adults... now appear in children. Diseases that used to take decades to develop... are showing up earlier, faster, and more aggressively.

... And all of this has happened in lockstep with one change.

... The rise of ultra-processed food.

... ... ...

This isn't coincidence. It's correlation... stacked on correlation... until it becomes impossible to ignore.

As real food disappeared from our plates, something else took its place. Products engineered in laboratories... optimised for profit... and designed to override the biological systems that kept humans healthy for thousands of years.

... We didn't evolve to handle this. ... Our children certainly didn't.""",

    # ===== INTRODUCTION PART 2 =====
    """Human biology is ancient. ... Ultra-processed food is brand new. ... And the collision between the two... is reshaping our health in real time.

... ... ...

Look at the timeline. As ultra-processed foods became cheaper, more available, more aggressively marketed, and more deeply embedded in daily life... chronic disease rose alongside them.

... Not gradually. ... Exponentially.

The curve of UPF consumption... and the curve of metabolic disease... are almost identical.

... ... ...

We were told this was convenience. ... We were told this was progress. ... We were told this was harmless.

... But convenience has a cost.

... And we are living inside the bill.

... ... ...

Ultra-processed foods are not just "unhealthy." ... They are biologically disruptive.

They alter hunger signals. ... They distort cravings. ... They change the microbiome... the control centre of immunity, metabolism, and even mood. They push blood sugar higher, faster, and more often. They create inflammation that quietly damages tissues over years. They encourage fat storage. They interfere with hormones. ... They reshape the reward pathways in the brain.""",

    # ===== INTRODUCTION PART 3 =====
    """And when these effects accumulate... day after day... year after year... the result is the world we see now.

... A population struggling with conditions that were once rare... now common. ... Once exceptional... now expected.

... ... ...

This isn't a failure of discipline. ... It's a failure of the food environment.

A food environment that rewards companies for engineering products people can't stop eating. ... A food environment where the cheapest calories are the most harmful. ... A food environment where children are targeted before they can read. ... A food environment where profit grows faster... than public health can collapse.

... ... ...

You didn't choose this system. ... You were born into it.

And the truth is simple... you cannot win a biological battle... against a product designed to bypass your biology.

... But you can understand it. You can see it clearly. And once you do... you can finally step outside it.

... ... ...

This book is not about blame. ... It's about exposure.

It's about showing you the forces that shaped your cravings, your habits, your weight, your energy, your mood, and your children's development... forces you were never meant to notice.

Because once you understand the problem... the solution becomes obvious. And once you see the system... you can no longer be controlled by it.

... ... ...

This is the beginning of taking back your biology. ... This is the beginning of taking back your health. ... This is the beginning of taking back your life.""",

    # ===== CHAPTER 1 PART 1 =====
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

    # ===== CHAPTER 1 PART 2 =====
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

    # ===== CHAPTER 1 PART 3 =====
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

    # ===== CHAPTER 3 PART 1 - THE REFINERY OPENING =====
    """... ... ...

Chapter Three. ... The Refinery. ... The Industrial Birth of Seed Oils.

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

    # ===== CHAPTER 3 PART 2 - THE PROCESS (with pauses for each step) =====
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

    # ===== CHAPTER 3 PART 3 - DEGUMMING THROUGH BLEACHING =====
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

    # ===== CHAPTER 3 PART 4 - DEODORISATION AND THE PUNCHLINE =====
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

    # ===== CHAPTER 3 - MYTH VS TRUTH =====
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

    # ===== CHAPTER 3 - CLOSING =====
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

And that business model... still shapes the modern diet today.""",

    # ===== CHAPTER 4 - THE ADDITIVE EXPLOSION PART 1 =====
    """... ... ...

Chapter Four. ... The Additive Explosion.

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

    # ===== CHAPTER 4 - EMULSIFIERS AND STABILISERS =====
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

    # ===== CHAPTER 4 - SWEETENERS, COLOURS, PRESERVATIVES =====
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

    # ===== CHAPTER 4 - TEXTURE AND CLOSING =====
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

... You are not imagining it. You are not overreacting. ... You are not the problem.

The system changed. The food changed. ... And your biology has been trying to cope ever since.""",

    # ===== CHAPTER 5 - FIZZY DRINKS PART 1 =====
    """... ... ...

Chapter Five. ... The Fizzy Drink. ... The Most Engineered Product in the Modern Diet.

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

    # ===== CHAPTER 5 - ACIDS =====
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

    # ===== CHAPTER 5 - SWEETNESS, COLOUR, FLAVOUR =====
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

    # ===== CHAPTER 5 - CARBONATION, PRESERVATIVES, CLOSING =====
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

... And they are consumed everywhere. ... Especially by children.""",

    # ===== CHAPTER 6 - RISE OF MODERN ILLNESS =====
    """... ... ...

Chapter Six. ... The Rise of Modern Illness.

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

    # ===== CHAPTER 6 - INVISIBLE SYMPTOMS AND EMOTIONAL COST =====
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

    # ===== CHAPTER 7 - THE CHILDREN'S CRISIS PART 1 =====
    """... ... ...

Chapter Seven. ... The Children's Crisis.

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

    # ===== CHAPTER 7 - BIOLOGY AND MARKETING =====
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

... Parents are not choosing ultra-processed foods for their children. ... Ultra-processed foods are choosing their children.""",

    # ===== CHAPTER 8 - THE HIDDEN COSTS =====
    """... ... ...

Chapter Eight. ... The Hidden Costs.

... ... ...

Ultra-processed foods don't just change the body. They change the day. They change the mood. They change the energy. They change the wallet. They change the rhythm of life.

Most people don't notice it happening. ... They just feel like life is harder than it should be.

... This chapter is about the costs that don't show up on labels.

... ... ...

The cost of energy. Most people think they're tired because they're busy... they're stressed... they're getting older. But there's another layer. Ultra-processed foods create energy instability. Fast energy. Fast crashes. Fast hunger. Fast cravings. ... A rollercoaster disguised as convenience.

... ... ...

The cost of cravings. Cravings feel personal. They feel like weakness. They feel like a lack of discipline. ... But cravings are not moral failures. They are biological responses to engineered foods. Ultra-processed foods are designed to dissolve quickly, spike dopamine, override satiety, and encourage repeat consumption. ... Cravings are not a bug. ... They are the business model.

... ... ...

The cost of mood. Ultra-processed foods create irritability, mood swings, emotional volatility, low resilience, afternoon crashes, brain fog. Not because people are unstable. ... But because their blood sugar is.

... ... ...

The cost of money. Ultra-processed foods look cheap. But they're expensive in disguise. More frequent purchases. More snacks. More drinks. More cravings. More impulse buys. A one-pound snack becomes a thirty-pound weekly habit. A two-pound drink becomes a sixty-pound monthly cycle. ... Ultra-processed foods are cheap at the checkout. But expensive in the long run.

... ... ...

The cost of self-perception. Perhaps the most painful cost of all. Ultra-processed foods make people feel weak... undisciplined... guilty... ashamed... confused... frustrated... "broken."

... But they're not broken. ... They're responding exactly as biology responds to engineered foods.

... Ultra-processed foods don't just change the body. ... They change the story people tell about themselves.""",

    # ===== CHAPTER 9 - SEEING THE MATRIX =====
    """... ... ...

Chapter Nine. ... Seeing the Matrix.

... ... ...

Most people don't choose ultra-processed foods. ... They drift into them. They pick up what's convenient. What's familiar. What's marketed. What's colourful. What's on offer. What's at eye level.

Ultra-processed foods don't look like a system. ... They look like normal life.

... This chapter changes that.

... ... ...

Rule one. If it has a health claim... be suspicious. Real food doesn't need to advertise itself. Apples don't say "high fibre." Salmon doesn't say "omega-3 rich." Broccoli doesn't say "immune boosting." ... Only ultra-processed foods need to convince you they're healthy.

Rule two. If it has a long ingredient list... it's not food. It's a formulation.

Rule three. If it melts in your mouth... it's engineered to override satiety. Foods that dissolve quickly bypass chewing, bypass fullness signals, deliver calories fast, and encourage overeating. ... Real food fights back. Ultra-processed food surrenders instantly.

Rule four. If it's everywhere... it's engineered. Real food is found in a few places. Ultra-processed food is found in all places.

Rule five. If it's cheap, fast, and hyper-palatable... it's designed for repeat purchase.

Rule six. If it's marketed to children... it's a formulation. Real food doesn't need a mascot.

Rule seven. If it's a drink with flavour... it's a chemical system.

Rule eight. If it's "healthy" but still comes in a packet... be careful. Ultra-processed foods don't disappear when they put on gym clothes.

Rule nine. If it's hard to stop eating... it was designed that way. This is not cooking. This is neuroscience.

... ... ...

Rule ten. ... Once you see it... you can't unsee it.

... ... ...

This is the moment the reader wakes up. Because once you understand the rules, you start noticing them everywhere. In your cupboard. In your fridge. In your child's lunchbox. In your office. In your supermarket. In your daily routine.

... You start seeing the matrix.

... And once you see it... you can step outside it.""",

    # ===== CHAPTER 10 - RECLAIMING YOUR BIOLOGY =====
    """... ... ...

Chapter Ten. ... Reclaiming Your Biology.

... ... ...

You've seen the system. You've seen the tricks. You've seen the patterns. You've seen the consequences.

... Now it's time to take back control.

Not through willpower. Not through discipline. Not through guilt. Not through perfection.

... But through environment... awareness... and strategy.

Because biology doesn't respond to motivation. ... Biology responds to inputs.

... Change the inputs... and everything changes.

... ... ...

The first truth. It's not you. It's the environment.

You are living in a food environment designed to overwhelm your biology. Ultra-processed foods are engineered to dissolve fast, spike dopamine, override satiety, encourage overeating, and create cravings. ... Your biology is not broken. It's responding exactly as it should.

... You don't need more willpower. ... You need a different environment.

... ... ...

The second truth. Small shifts change everything.

Biology doesn't need dramatic change. It needs consistent nudges. Hunger stabilises. Cravings reduce. Energy improves. Mood evens out. Sleep deepens. Clarity returns.

... Your body wants to work with you. It just needs the right inputs.

... ... ...

The method has three pillars.

Pillar one. Stabilise your biology. Add protein. Add fibre. Add healthy fats. Add slow carbs. Add real meals. Not by removing things... but by adding anchors.

Pillar two. Change your environment. Make real food visible and accessible. Make ultra-processed foods invisible and inconvenient. Have a default meal you can make in ten minutes without thinking.

Pillar three. Shift your identity. When you become "the kind of person who eats real food"... the decisions make themselves.

... ... ...

If you take nothing else from this chapter, take this.

... One real meal a day changes everything.

It stabilises energy. It reduces cravings. It improves mood. It anchors your biology. It creates momentum. It shifts your identity.

... You don't need a new diet. ... You need one real meal.

... And then another. ... And then another.

... Not perfection. ... Progress.""",

    # ===== CHAPTER 11 - THE METHOD AND SWAPS =====
    """... ... ...

Chapter Eleven. ... The You Are What You Eat Method.

... ... ...

Most people don't need more information. ... They need a method.

A way to take everything they've learned... the system, the tricks, the consequences, the hidden costs... and turn it into something they can actually do.

Not a diet. Not a programme. Not a challenge.

... A method.

... ... ...

The method has three parts. See the system. Change the environment. Rebuild the biology.

You've already done the first part. ... Now we move into the second and third.

... ... ...

Ten real-food swaps that change everything.

Swap one. Fizzy drink... becomes sparkling water with citrus. Same sensation. None of the engineering.

Swap two. Crisps... become nuts or seeds. Real fat. Real crunch. Real satiety.

Swap three. Chocolate bar... becomes fruit with protein. Sweetness with stability.

Swap four. Ultra-processed breakfast cereal... becomes eggs, yoghurt, or oats. Start the day with anchors, not spikes.

Swap five. Protein bar... becomes a real food snack. Protein bars are ultra-processed foods in gym clothes.

Swap six. Ready meal... becomes leftovers. Real food without extra effort.

Swap seven. Ultra-processed lunch... becomes a simple real-food plate. Protein plus veg plus carbs equals stability.

Swap eight. Energy drink... becomes water with a pinch of salt and citrus. Hydration without the chemical system.

Swap nine. Ultra-processed dessert... becomes dark chocolate or fruit. Satisfaction without the crash.

Swap ten. "Healthy" packaged snack... becomes anything with one ingredient. If it has a mascot, it's not food.""",

    # ===== CHAPTER 12 - YOUR FIRST 30 DAYS =====
    """... ... ...

Chapter Twelve. ... Your First 30 Days.

... ... ...

Transformation doesn't happen in a year. It doesn't happen in a month. It doesn't happen in a week.

... It happens in moments. Small decisions, repeated consistently, that slowly shift your biology, your habits, and your identity.

... ... ...

Week one. One real meal a day.

This is the anchor. The single most powerful change you can make. One meal per day built entirely from real ingredients. Protein. Vegetables. Healthy fats. That's it. You don't need to change the other meals yet. Just build this one.

... By day seven, you'll notice fewer crashes, more predictable hunger, and slightly better sleep.

... ... ...

Week two. Swap the biggest trigger.

Everyone has one. The product that pulls hardest. The thing you reach for automatically. Identify yours. Swap it. One swap. Big impact.

... By day fourteen, the craving weakens. Not disappears. Weakens.

... ... ...

Week three. Fix the environment.

Go through your kitchen. Move ultra-processed foods to the highest cupboard. Put real food at eye level. Prep a safety net: boiled eggs, washed vegetables, nuts, cheese, tinned fish. Write down your default ten-minute meal.

... By day twenty-one, you reach for real food automatically because it's what's visible.

... ... ...

Week four. Build the rhythm.

Add a second real meal. Stabilise eating times. Reduce mindless snacking. Begin noticing how your body feels after real food versus ultra-processed food.

... By day thirty... energy is noticeably more stable. Cravings are quieter, not louder. Mood is steadier. You start to feel like a different person... not because you've suffered... but because your biology has settled.""",

    # ===== CHAPTERS 13-15 - IDENTITY, ENVIRONMENT, TOOLS =====
    """... ... ...

Chapter Thirteen. ... Identity Shift.

... ... ...

Most people try to change their eating habits by changing their behaviour. They try new diets, new rules, new routines, new restrictions.

And every time, the same thing happens. They run out of motivation. They run out of willpower. They run out of energy.

... Not because they're weak. But because they're using the wrong tool.

... Behaviour doesn't create identity. ... Identity creates behaviour.

... ... ...

There is a moment, and it happens quietly, when you stop thinking "I'm trying to eat better"... and start thinking... "This is just what I do."

... That moment is the identity shift.

It's the moment cravings lose their power. Ultra-processed foods lose their appeal. Real food feels natural. Energy feels stable. Confidence returns.

... It's the moment the system loses its grip.

... ... ...

Chapter Fourteen. ... Designing Your Real-Food Life.

... ... ...

Identity is the engine. But environment is the road. If your environment is still built for ultra-processed foods, your biology will always be fighting uphill.

Design your home. Make real food visible. Make ultra-processed food less accessible. Have a default meal. Keep a safety net stocked.

Design your work environment. Bring anchors. Avoid the vending machine. Create a real-food lunch routine. Protect the three PM window.

Design your social life. Eat real before you go out. Choose the anchor first. Don't drink your calories. Don't moralise food.

... Make the good choice the easy choice. ... That is the core principle.

... ... ...

Chapter Fifteen. ... The Tools That Help.

The ten-second label check. Look at the ingredient list on the back, not the front. Count the ingredients. Scan for words you wouldn't find in a kitchen: emulsifiers, stabilisers, modified starch, maltodextrin, flavourings, hydrogenated oils.

... If you see them... it's Group Four. Put it back.

The "could I make this?" test. Look at the product. Ask yourself: "Could I make this at home, with ingredients I recognise?" If the answer is no... it's ultra-processed.

The food scanner. Your phone becomes your x-ray. Point it at a barcode and see what you're really holding.

... The tools don't replace your judgement. ... They accelerate it.""",

    # ===== CHAPTER 16-18 - THE FUTURE AND CLOSING =====
    """... ... ...

Chapter Sixteen. ... The Real-Food Future.

... ... ...

There is a moment in every transformation where the personal becomes universal.

You start by changing your meals. Then your energy changes. Then your mood changes. Then your identity changes. Then your environment changes.

... And then something unexpected happens. Your life changes... and the people around you feel it.

... Energy is contagious. Clarity is contagious. Stability is contagious. ... Real food is contagious.

... ... ...

Imagine a home where real food is visible. Meals are simple. Energy is stable. Children are calmer. Mornings are smoother. Evenings are easier.

A home where biology is respected. Where identity is reinforced. Where the environment is designed for humans... not for corporations.

... ... ...

Chapter Seventeen. ... Living Outside the Matrix.

... ... ...

There is a moment when the world looks different. Not because the world changed... but because you did.

You walk into a supermarket and see the layout for what it is. You walk past a vending machine and feel nothing. You feel hunger as a signal, not a crisis. You feel energy as a baseline, not a rare event. You feel calm where there used to be noise.

... This is what it feels like to live outside the matrix.

... Not perfect. Not pure. Not restrictive.

... Just free.

... ... ...

Chapter Eighteen. ... The Beginning.

... ... ...

Every journey has a moment where you realise you're not going back.

Not because you're forcing yourself forward. Not because you're scared of slipping.

... But because you've changed.

... ... ...

There comes a day, and it always comes, when you notice something subtle.

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
]

async def generate_full_audiobook():
    output_dir = "/app/marketing"
    all_audio = bytearray()
    total = len(FULL_BOOK)
    
    for i, text in enumerate(FULL_BOOK):
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
    
    path = os.path.join(output_dir, "audiobook_full.mp3")
    with open(path, "wb") as f:
        f.write(all_audio)
    
    size_mb = round(os.path.getsize(path) / (1024*1024), 1)
    est_mins = round(size_mb * 1.0)  # rough estimate
    print(f"\nFull audiobook saved: {path} ({size_mb} MB, ~{est_mins} min)")

asyncio.run(generate_full_audiobook())
