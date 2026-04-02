# You Are What You Eat - PRD

## Original Problem Statement
Food scanning app that analyzes products via barcode, provides health scores based on ingredients/processing level via AI (GPT-4o), and offers premium features through RevenueCat subscriptions. Live on Google Play, submitting to Apple App Store.

## Architecture
- **Frontend**: React Native (Expo), TypeScript, EAS Build/Submit
- **Backend**: Python, FastAPI (deployed on Railway)
- **Database**: MongoDB (on Railway)
- **AI**: OpenAI GPT-4o
- **Subscriptions**: RevenueCat (Google Play + Apple App Store)
- **APIs**: Open Food Facts, USDA, UPC Item DB, Brocade, FatSecret

## Key DB Schema
- `product_cache`: `{barcode, product_name, brands, ingredients_text, image_url, analysis, analysis_error, cached_at, source}`
- `users`: `{email, password_hash, role, subscription_tier, total_scans}`
- `scans`: `{user_id, barcode, product_name, analysis, scanned_at, source}`
- `gamification`: `{user_id, daily_quests}`

## Key API Endpoints
- `POST /api/scan/quick` — Two-stage scan: instant product lookup from 7 databases, AI fallback for unknown barcodes
- `GET /api/scan/status/{barcode}` — Poll for AI analysis completion
- `POST /api/scan` — Legacy full-scan fallback
- `POST /api/admin/prewarm` — Admin tool to cache items
- `POST /api/admin/set_premium` — Grant premium to user by email
- `POST /api/admin/fix_scan_count` — Fix inflated scan counts
- `GET /api/admin/user_scans` — View user's scan history
- `GET /api/admin/search_users` — Search users by email/name
- `POST /api/webhooks/revenuecat` — Subscription webhook (now handles anonymous IDs)

## What's Been Implemented
- Two-stage scan with 7 food DB sources + AI barcode identification fallback (0% 404s)
- User-Agent headers on ALL external API calls
- Background AI analysis with error reporting to status endpoint
- RevenueCat Apple + Google subscription integration
- Platform-specific subscription text (iOS vs Android)
- Pre-warmed cache with ~90 products
- Marketing assets library
- Strict scoring: Carcinogens = 1/10, NOVA 4 = max 3/10, enforced in Python code
- Scan count dedup (5-min window) to prevent count inflation

## Current Status (April 2026)
- iOS v1.0.28 (build 29): Pending Apple App Store review
- Android v1.0.29 (build 30): Build submitted to EAS — includes Build A fixes
- Backend: Webhook hardened for anonymous RC IDs, mirrored to root server.py

## Build A Fixes (v1.0.29) — Completed
1. **RevenueCat Anonymous User Bug (P0 FIXED)**: Root cause was `initialized` flag blocking `Purchases.logIn(userId)`. Separated SDK `configure` from user `logIn` — now uses `configured` + `loggedInUserId` states so login always fires when user ID is available.
2. **Backend Webhook Hardened (P0 FIXED)**: Webhook now handles non-ObjectId app_user_ids gracefully — falls back to aliases lookup, then email from subscriber_attributes, instead of silently ignoring.
3. **Back Button UX Fix (P1 FIXED)**: Custom `HeaderBackButton` with 44x44pt touch target + hitSlop in `_layout.tsx`, replacing default back buttons.
4. **NOVA Color Scheme (P1 FIXED)**: Processing badges in `result.tsx` now color-coded: Green (Whole Food), Yellow-Green (Minimally Processed), Amber (Processed), Red (Ultra-Processed).

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- RevenueCat iOS: appl_qDwlqIUvHJHGuewqEExfpAgaCpw
- RevenueCat Android: goog_sSuefaqGfyQKJvmIkNrWEyVElTx
- Apple IAP Product ID: yawye_premium_monthly
- Admin key: yawye2024clear

## Pending Issues
1. P1: Password reset emails (Resend domain `yawye.app` not verified)
2. P2: Custom domain yawye.app not configured
3. P2: Railway rootDirectory refactor (currently duplicating server.py to root)

## Upcoming Tasks
1. P0: Monitor EAS Android build completion for v1.0.29 and submit to Google Play
2. P0: Monitor Apple App Store review for v1.0.28
3. P1: Build B — Ship Library screen, Share button with store links, product confirmation step
4. P1: Marketing video creation
5. P2: Guide iOS App Store Connect setup

## Book Franchise ("You Are What You Eat")
- **Status**: Manuscript reviewed (Feb 2026). All deliverables complete and live on Railway.
- **Asset Library URL**: `https://web-production-66c05.up.railway.app/api/marketing`

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat
- Bulk fix_scan_count for all existing users

## Critical Notes for Next Agent
- Railway deploys from root — `server.py` and `requirements.txt` must be copied from `/backend/` to root
- Open Food Facts REQUIRES User-Agent header on ALL requests
- AI fallback for unknown barcodes ensures zero 404s
- Scoring rules enforced in Python code, NOT just AI prompt (carcinogens=1, NOVA4=max 3)
- Scan count now deduped with 5-min window to prevent inflation
- iOS subscription has NO free trial; Android has 7-day free trial
- Production URL: https://web-production-66c05.up.railway.app
- RevenueCat webhook now handles: valid ObjectId, aliases, email fallback, and anonymous IDs (logs warning, returns ok)
