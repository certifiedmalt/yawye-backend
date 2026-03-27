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
- `GET /api/scan/status/{barcode}` — Poll for AI analysis completion (returns complete/analyzing/error)
- `POST /api/scan` — Legacy full-scan fallback
- `POST /api/admin/prewarm` — Admin tool to cache items
- `POST /api/webhooks/revenuecat` — Subscription webhook

## What's Been Implemented
- Two-stage scan with 7 parallel food DB sources + AI barcode identification fallback
- User-Agent headers on ALL API calls (Open Food Facts requires this)
- Background AI analysis with error reporting to status endpoint
- RevenueCat Apple App Store integration (`appl_qDwlqIUvHJHGuewqEExfpAgaCpw`)
- Platform-specific subscription text (iOS: "Subscribe Now", Android: "Start 7-Day Free Trial")
- Subscription product `yawye_premium_monthly` configured in App Store Connect + RevenueCat
- Pre-warmed cache with ~90 products
- Marketing assets library (screenshots, icons, feature graphics)

## Current Status (March 27, 2026)
- iOS v1.0.28 (build 29): Building on EAS, awaiting submission to App Store Connect
- Android v1.0.28 (build 29): Building on EAS
- Backend: All scan fixes deployed to Railway
- Subscription: Fully configured for both platforms

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- RevenueCat iOS: appl_qDwlqIUvHJHGuewqEExfpAgaCpw
- RevenueCat Android: goog_sSuefaqGfyQKJvmIkNrWEyVElTx
- Apple IAP Product ID: yawye_premium_monthly
- Google Play Product ID: premium_monthly:monthly-base

## Pending Issues
1. P1: Password reset emails (Resend domain not verified)
2. P2: Custom domain yawye.app not configured
3. P2: Minor UI bugs (keyboard/splash screen) — verify on v1.0.28

## Upcoming Tasks
1. P0: Submit iOS build to App Store Connect once EAS finishes → resubmit for Apple review
2. P0: Upload Android .aab to Google Play Console
3. P1: Build admin dashboard (user stats, scan analytics)
4. P2: Refactor Railway deployment (rootDirectory setting)

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat
- Marketing video creation

## Critical Notes for Next Agent
- Railway deploys from root — `server.py` and `requirements.txt` must be copied from `/backend/` to root
- Open Food Facts REQUIRES User-Agent header on ALL requests (403 without it)
- When no food database has a barcode, AI (GPT-4o) identifies the product — zero 404s
- iOS subscription has NO free trial; Android has 7-day free trial
- EAS build commands: `npx eas-cli build --platform ios --non-interactive --no-wait`
- EAS submit: `npx eas-cli submit --platform ios --latest --non-interactive`
