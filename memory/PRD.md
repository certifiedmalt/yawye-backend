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
