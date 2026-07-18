# YAWYE Roadmap

## 📅 TOMORROW (user has full day free) — Console Session + Build 52
User will be in App Store Connect + Play Console. Agent guides step-by-step.

**Products to create in BOTH consoles (copy-paste ready):**
| Product ID | Type | Price | Notes |
|---|---|---|---|
| `yawye_premium_annual` | Auto-renew sub, 1 year | £14.99 / $19.99 | "SAVE 37%" vs monthly |
| `yawye_family_monthly` | Auto-renew sub, 1 month | £3.49 / $4.49 | Up to 5 profiles |
| `yawye_family_annual` | Auto-renew sub, 1 year | £24.99 / $29.99 | Best value tier |

- Apple: Monetization → Subscriptions → existing group → "+" (submit with next build)
- Google: Monetize → Subscriptions → add base plans (live within hours)
- Agent prep: paywall UI with annual + SAVE badge, expo-iap wiring, backend product IDs, Stripe annual price for web → **Build 52**

## P0 / Near-term
1. **Annual pricing tier** (Build 52) — annual plans cut churn 30-40% (Trash Panda data: $5.99/mo, $39.99/yr worked)
2. **Family profiles / Family Plan** (Build 53) — profile switcher, per-profile history/streaks, entitlement logic. Store products created tomorrow; feature ships later. Family accounts churn far less.
3. **CUSTOM RULES** (user-saved idea, market-validated — Fooducate died of generic scoring):
   - Users toggle personal scoring preferences: "flag refined seed oils", "less sugar", allergen filters, "no artificial sweeteners"
   - Default score stays evidence-based (RD-safe); strictness is opt-in → solves the seed-oil dilemma both ways
   - Start with ONE rule (seed-oil flag) as MVP, expand later
   - Premium feature candidate = upsell driver

## P1
- SendGrid password reset emails (BLOCKED: waiting for user's SG. API key; domain auth must be redone WITHOUT www)
- Email campaign centre in dashboard (once SendGrid live) — 59 zero-scan users reachable by email vs 4 by push
- Google Analytics on website
- "How We Score" methodology page on website (RD trust signal; every criticism of Yuka answered)
- iOS 1.0.51: in Apple review — user must press "Release This Version" if Pending Developer Release

## P2
- Pantry View (scan whole pantry → overall health score + "swap these first" list) — also the retention fix for post-first-week churn
- Push-permission prompt already moved to after first scan (shipped Build 51)
- Railway rootDirectory refactor (kill duplicate /app/server.py)
- Second paywall "success moment": after 10th scan result ("unlock Safe Swaps for your whole shelf")
- Multi-language support (i18n)
- Meta ads geo-targeted US test ($10/day, tracked via /go/metaads) — only real US geo-targeting
- US-products reel script (Trader Joe's/Goldfish/Gatorade) to shift IG algorithm toward US audience

## Marketing campaign state (Jun 2026)
- 1 YES: Shweta @myfoodspree (Nashville, 117K) — signed up, premium active, first US user, FILMING VIDEO. Watch /go/foodspree
- Sent: Kate @thatcrunchymomkate, @crunchycountrymom, @theketologist(?), dishingouthealth email
- Queued cities/scripts: Austin, SLC, Nashville, Denver (@foodles_denver 18% ER top priority), Atlanta (@lovinglylina scripted), San Diego (@diet.assassinista scripted, @playdatesandpints scripted)
- Big swings parked: Jen Smiley @wakeupandreadthelabels (affiliate offer), Courtney Swan @realfoodology (podcast guest pitch), @wellnesswithcharms, @dishingouthealth
- Tracked links: youarewhatyoueat.store/go/<code> — any code works, bot-filtered, dashboard panel
- Lessons: DM 12-2pm GMT (US morning scroll); UPF-native creators found via comment-mining Jen Smiley/Eddie Abbew + "Ultra-Processed People" book tags; avoid anti-diet/HAES creators (hostile to scoring apps) and Food Babe/FlavCity
