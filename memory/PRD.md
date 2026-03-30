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
- `POST /api/webhooks/revenuecat` — Subscription webhook

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

## Current Status (March 30, 2026)
- iOS v1.0.28 (build 29): Submitted to App Store Connect, awaiting review
- Android v1.0.28: Live on Google Play
- Backend: Scan count bug fixed, all admin endpoints deployed to Railway
- rosannaleggett@gmail.com upgraded to premium, scan count corrected

## Bug Fixes This Session
- **Scan count inflation bug (CRITICAL)**: `total_scans` was incrementing on every API call including cache hits and retries. Free users could exhaust 5-scan limit from 1 actual scan. Fixed with 5-minute dedup window — count only increments on new unique scan records.

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- RevenueCat iOS: appl_qDwlqIUvHJHGuewqEExfpAgaCpw
- RevenueCat Android: goog_sSuefaqGfyQKJvmIkNrWEyVElTx
- Apple IAP Product ID: yawye_premium_monthly
- Admin key: yawye2024clear

## Pending Issues
1. P1: Password reset emails (Resend domain `yawye.app` not verified)
2. P1: NOVA-based color scheme on result page (deferred by user)
3. P2: Custom domain yawye.app not configured
4. P2: Railway rootDirectory refactor (currently duplicating server.py to root)

## Upcoming Tasks
1. P0: Monitor Apple App Store review for v1.0.28
2. P1: Marketing video creation
3. P2: Guide iOS App Store Connect setup

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
