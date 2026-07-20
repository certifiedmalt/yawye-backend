# YAWYE (You Are What You Eat) - Product Requirements Document

## Original Problem Statement
A food product analysis app that scans barcodes, provides AI-powered health scores (penalizing ultra-processed foods), offers premium features through subscriptions, and acts as a companion to a book franchise. Live on Apple App Store.

## Architecture
- **Frontend**: React Native (Expo), TypeScript, EAS Build
- **Backend**: Python, FastAPI
- **Database**: MongoDB
- **Monetization**: expo-iap (StoreKit 2 / Google Play Billing), Stripe Checkout (Web)
- **AI**: OpenAI GPT-4o for ingredient analysis
- **Deployment**: Railway (Python 3.13)

## Core Features (Implemented)
- Barcode scanning with OpenFoodFacts + AI fallback
- AI health scoring with strict NOVA classification rules
- Safe Swaps (healthier alternatives for low-scoring products)
- Daily quests, streaks, XP, levels, badges (gamification)
- Health assistant chat
- Native IAP subscriptions (StoreKit 2 / Google Play Billing)
- Web Stripe checkout for subscriptions
- Google Ads conversion tracking on website
- Push notifications

## Scoring Rules (Updated Feb 2026)
- Rule 1: Carcinogens = score 1
- Rule 2: Ultra-Processed (NOVA 4) = max 3
- Rule 3: Processed (NOVA 3) = max 5 (only if UPF > 10%)
- Rule 4: NOVA 2 culinary ingredients = 5-7
- Rule 5: Only NOVA 1 can score 8+
- Rule 6: Alcohol = always 1
- Rule 7: Processed meat = always 1
- Rule 8: Clean 1-3 natural ingredients = minimum 8
- Rule 9: 0-10% UPF safety net = minimum 7 (unless carcinogens)
- Rule 10: Whole Food / Minimally Processed floor = 7

## What's Been Implemented
- (Jun 2026) FOUNDER DASHBOARD LIVE: https://youarewhatyoueat.store/dashboard (password = admin key yawye2024clear, noindex, sessionStorage). Shows: total users/premium/est revenue, conversion funnel, scan distribution, country chips, recent-signups table (email, scans, push token status, per-user recent scan scores, "never scanned" flags), 7/30/90-day selector. Backed by new `/api/admin/cohort_diagnostics` endpoint. KEY DIAGNOSTIC FINDING: zero scan errors in cohort — but only ~18% of low-scan users have push tokens (day-2 nudge can't reach them) and scan_analytics logger isn't wired into current scan endpoints (instrumentation gap).
- (Jun 2026) RELIABILITY FIX (testing agent verified 100%, iteration_5): Rule 0 NOVA 4 consistency — OpenFoodFacts nova_group now fetched/cached and used as authoritative ground truth; if OFF says NOVA 4 OR the AI flags any NOVA 4 marker ingredient, product is deterministically coerced to Ultra-Processed (score capped at 3) regardless of AI mood. Contradictory low UPF% masked as "unknown" when override fires. /api/scan/rescan refreshes nova_group+ingredients from OFF for old cache entries. Verified live: Philadelphia (7622201693916) 5/10 "Processed" -> 3/10 Ultra-Processed across repeated AI runs; Babybel Light stays 9/10 (OFF nova 3, clean); Peter's Yard + Welch's regressions clean. Regression suite: /app/backend/tests/test_production_nova4_fix.py.
- (Jun 2026) COUNTRY TRACKING LIVE: IP->country capture (ip-api.com, fire-and-forget) at register + login; stored once per user (`country` field). `funnel_stats` now returns `by_country` breakdown (users + premium per country). Existing users get tagged as they next log in. Also: `/api/admin/geo_estimate` infers regions from scanned barcode prefixes (~6 genuine US users found). Day-2 nudge upgraded: local on-device scheduling at 6pm user timezone (ships in build 49, cancels at 5 scans, `/api/auth/day2-local-scheduled` handoff), server fallback gated to 16-21 UTC, threshold <=4 scans.
- (Jun 2026) GROWTH TOOLING: (1) `/api/admin/funnel_stats?key=...&days=N` — activation/paywall/premium funnel + scan distribution. Key insight: paywall→premium converts at 26-35% (elite); real leak is 39% of users never scan once. (2) Day-2 re-engagement push: auto-scheduler (every 6h) nudges users 24-72h post-signup with <=2 scans via Expo push; idempotent (day2_nudge_sent flag); manual trigger `/api/admin/run_day2_nudge?key=...&dry_run=true`. Both LIVE on Railway, tested (local e2e + prod dry-run). (3) Android build 48 (v1.0.47) FINISHED on EAS — user uploading AAB to Play Console. (4) Fixed TDZ crash in result.tsx (testing agent verified). (5) Marketing pack: 3 TikTok scripts + 5 real scan screenshot pairs in /app/marketing-screenshots/.
- (Jun 2026) WEBSITE REDESIGN LIVE: dark "wake-up-call" scroll journey on youarewhatyoueat.store — hero (poor diet > smoking), Diary of a CEO video (dzUDhstqXbg) + stat cards, 2 more UPF videos (ZOE BAxkGg8nk3w, Spector zO7tyleoxwE), NOVA scale chart (4 levels + common foods, color-graded), app solution section, pricing (£0/£1.99), official app-icon logo swapped into nav (asset: /assets/yawye-logo.jpeg in both repos). Google Ads tracking (AW-18007960189) preserved. Tailwind CDN, static HTML. Testing agent iteration_3: 100% pass. Book excluded per user.
- (Jun 2026) WEBSITE MIGRATED TO GITHUB PAGES + CUSTOM DOMAIN LIVE: https://youarewhatyoueat.store (and www) now serves the marketing site from GitHub Pages repo `certifiedmalt/yawye-website` with valid SSL (https_enforced). Railway SSL issuance was stuck for hours, so site hosting was moved off Railway; API/mobile backend remains on Railway unchanged. DNS at Namecheap: 4 A records (@ -> 185.199.108-111.153), www CNAME -> certifiedmalt.github.io. subscribe.html calls Railway API absolutely (CORS OK). Dead yawye.app custom domain removed from Railway. Testing agent verified all pages/links 100%. KNOWN GAP: /api/stripe/create-checkout-session returns 500 "Stripe not configured" — STRIPE_SECRET_KEY not set in Railway env (pre-existing).
- (Jun 2026) FIXED & DEPLOYED: 0% UPF → 5/10 score bug. Root causes: (1) fix commits were never pushed to GitHub/Railway; (2) scan/quick cache-invalidation branch hung — after unsetting bad analysis it hit the "pending analysis" branch and returned "analyzing" forever without re-analyzing. Fixes: pushed all commits, set cached=None after invalidation to force re-fetch + background re-analysis, time-bounded pending branch (2 min freshness, stale = re-analyze). Verified LIVE on Railway: Peter's Yard crackers (5060198820052, 5060198821219) now score 8/10.
- Apple App Store Approval (LIVE)
- expo-iap replacing RevenueCat
- Scan race condition fix
- Safe Swaps feature
- Website redesign with Stripe
- AI Scoring logic (Rule 8 + Rule 9 + Rule 10)
- Gamification auto-create and dynamic quests
- Google Ads tracking
- Python 3.13 Railway deploy fixes
- **Feb 2026**: Fixed 0% UPF scoring bug (products with 0% UPF were getting score 5 instead of 7+)
- **Jun 2026**: Scan-limit paywall fix (Build 50). iOS Build 50 submitted to App Store Connect via EAS Submit. Android Build 50 AAB delivered to user for Play Console upload.
- **Jun 2026**: Added Premium Subscribers table to founder dashboard (backend user_stats enriched with created_at/country; deployed to Railway + GitHub Pages).
- **Jun 2026**: Push notification targeting added (`max_scans` filter + `dry_run` on /api/admin/send-notification). Sent re-engagement push to 4 zero-scan users with tokens.
- **Jun 2026**: Dashboard v2 — Push Centre (compose, audience targeting, reach preview, campaign history via push_campaigns collection), User Admin panel (search, grant/revoke premium, reset password, fix scan count, view scans), Failed Scans monitor (/api/admin/failed_scans). Also fixed /api/scan/quick to log scan_analytics (was never logging).
- **Jun 2026**: Deleted 20 dev/test accounts (+70 junk scan records) from production. Kept applereview@yawye.app, googlereviewer@yawye.app, test.screenshot@yawye.app. Added /api/admin/delete_user endpoint + Delete button in dashboard User Admin panel.
- **Jun 2026**: Influencer tracked links: youarewhatyoueat.store/go/<code> (GitHub Pages 404.html redirect trick) → logs click to POST /api/track/click (device, IP country) → redirects to App Store (?ct=code) or Play Store (utm referrer). Dashboard "Tracked Links" panel via /api/admin/link_stats. Verified e2e. Competitive positioning doc saved at /app/memory/COMPETITIVE_POSITIONING.md. Influencer strategy: Austin + SLC first; exclusive DM drafted for @thatcrunchymomkate (/go/kate).
- **Jun 2026**: Failed-scans fixes (live in production): backend rejects QR/URL/invalid barcodes; AI fallback no longer serves "Unknown product" with fake scores; cache guard invalidates unknown-product entries.
- **Jun 2026**: Build 51 features (code ready, BUILD NOT YET KICKED OFF): teach-the-app identify flow (POST /api/scan/identify + result.tsx card), push permission moved to after first scan (utils/notifications.ts), QR scanner fix, fixed rescan crash bugs (missing Alert import, undefined setAnalysis). Verified identify endpoint in production.
- **Jun 2026**: Pending premium auto-grant: /api/admin/grant_pending_premium — premium activates on signup. Granted to myfoodspree@gmail.com (first influencer yes: Shweta @myfoodspree, Nashville, link /go/foodspree). CLAIMED — Shweta signed up (first US user), premium active, filming video.
- **Jun 2026**: Scoring accuracy fixes (live, verified on 10 US products): acrylamide reclassified as process-formed (cap 4, never auto-1); NOVA-3 floor 3; fried-snack guard (chips/crisps/fries max 5, excluded from clean-list/whole-food boosts); temperature 0 for deterministic scores. Tostitos 1→4 consistent; Coke=1, Oats=10 regression passed.
- **Jun 2026**: Extended process-formed carcinogen guard to full family: furan, HCAs, PAHs/benzo(a)pyrene, 3-MCPD/glycidyl esters, ethyl carbamate (Rule 11 cap 4, whole/minimally-processed exempt). Verified: coffee 7/10, smoked salmon 6/10, cashews 9/10, raspberries 10/10, Tostitos 4/10. Nitrosamines/processed meat still auto-1 (added nitrites).
- **Jun 2026**: Refined seed oil positioning (option 3): evidence-based score (7-8) + full-context analysis (industrial refining in shocking_facts, cold-pressed alternatives, no bare "healthy" claims). Palm oil correctly penalized 3-4 (glycidyl esters). Added no-fake-citation rule to prompt. 10-oil test documented.
- **Jun 2026**: Yuka-criticism tests passed: natural vs added sugar (raisins/OJ 9, candy/sweetened yogurt 3-5); MSG handled per evidence (soy sauce 6, no "hazardous" labels). Bug fixed: process-formed compounds now relocated out of carcinogen panel in code (AI disobeyed prompt on non-English products); score 1 strictly reserved for added carcinogens (floor 2 otherwise). Verified: MAMA noodles 1→2, Coke=1, Tostitos=4.
- **Jun 2026**: Lazy cache refresh (live): cache entries >90 days old trigger silent background re-fetch on scan; re-analysis only if normalized ingredients changed (reformulation detection — Yuka's #1 complaint category). Both branches tested. Barcodes do NOT change on reformulation (GS1).
- **Jun 2026**: CRITICAL FIX — AI barcode identification was hallucinating product names (2 different barcodes both "identified" as Heinz Baked Beans). Fixed: no-guess prompt + confidence=="high" gate in caller + temp 0. Purged 4 hallucinated cache entries + 4 test fakes via new /api/admin/cache_delete. Verified: hallucinated barcode now honest Unknown (0), real products still resolve.

- **Jun 2026**: Dashboard v3 additions (live, verified): (1) INTERNAL TEST badges — scan analytics now log user_id; failed-scans panel tags founder/review/test accounts (INTERNAL_EMAILS) with amber "INTERNAL TEST" chip so agent/founder testing never looks like real user failures. (2) Comped/Influencer separation — `comped` flag on users (auto-set on pending_premium signup, toggleable via User Admin "Mark as comped/paying" + /api/admin/set_comped); user_stats returns paying vs comped splits; revenue card = PAYING × £1.99 only; new 🎁 Comped panel. Shweta (myfoodspree@gmail.com) marked comped — revenue now £26/mo (13 paying). (3) 📊 Member Usage panel via /api/admin/usage_stats: per-segment (paying/comped/free) avg lifetime scans, avg scans 7d/30d, active % — key insight: paying subs avg 24 scans/30d & 85% active vs free 1.5 scans & 31%. Also fixed: search_users was leaking bcrypt password hashes (excluded "password_hash" but field is "password").
- **Jun 2026**: IARC VERIFICATION GATE (live, verified in production): AI can no longer trigger score 1 with invented carcinogen ratings. Code-level whitelist of genuinely IARC-classified food agents (nitrites, alcohol, aflatoxins, E150d/4-MEI, aspartame, BHA, TiO2, etc.); unverified claims (Red 40, Yellow 6, TBHQ, "palm oil", "artificial colors") demoted to harmful panel with honest note. Prompt corrected — Red 40/Yellow 6/Red 3 were wrongly listed as Group 2B in our own prompt (root cause). Gate also applied to name-based analysis + AI barcode identification paths. New endpoint: POST /api/admin/carcinogen_audit?key=..&purge=true — found & purged 89 polluted cache entries (stale pre-fix fear-mongering: Wholemeal Bread 1/10 via glycidyl esters, McVities 1/10 via acrylamide, dye-based auto-1s). Verified live: Doritos 1→3, Wholemeal Bread 1→3, McVities 1→3, Coke stays 1 (genuine E150d), Peter's Yard stays 8, MAMA stays 2.
- **Jun 2026**: VERIFIED LIVE post-fork: GTIN checksum gate + GS1 prefix verification (commit 40fb2ed) deployed on Railway. Invalid-checksum barcodes rejected with friendly retry message; real products (Coke) unaffected. Root/backend server.py confirmed in sync, commit pushed to origin/main.

## Pending Issues
- P0: Password reset emails (needs SendGrid API key from user)
- P1: Google Play Store: user needs to upload AAB Build 50
- P2: Stripe webhook configuration pending

## Upcoming Tasks
- P1: Google Analytics tracking on website
- P2: Pantry View feature
- P2: Family Profiles
- P2: Railway rootDirectory refactor
- P3: Multi-language support (i18n)

## Key DB Schema
- `users`: `{email, password, role, subscription_tier, total_scans, push_token}`
- `scans`: `{barcode, user_id, product_data, analysis, scanned_at, source}`
- `gamification`: `{user_id, xp, level, current_streak, longest_streak, daily_quests}`
- `product_cache`: `{barcode, product_data, analysis, categories_tags, cached_at}`

## Test Credentials
- Admin: jpsaila1986@gmail.com / hello123
- Apple Review: applereview@yawye.app / Review2024!
- Test: test.screenshot@yawye.app / test1234
