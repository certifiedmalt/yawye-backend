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

## Pending Issues
- P0: Password reset emails (needs SendGrid API key from user)
- P1: Custom domain migration (youarewhatyoueat.store DNS propagation)
- P1: Google Play Store app outdated (user needs to upload AAB build 47)
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
