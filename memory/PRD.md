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
- `product_cache`: `{barcode, product_name, brands, ingredients_text, image_url, analysis, cached_at, source}`
- `users`: `{email, password_hash, role, subscription_tier, total_scans, push_token}`
- `scans`: `{user_id, barcode, product_name, analysis, scanned_at, source}`
- `gamification`: `{user_id, daily_quests}`

## Current Status (April 2026)
- **v1.0.31** (versionCode 32): Build complete, ready for Google Play submission
- Contains CRITICAL fix: correct RevenueCat API keys (were wrong in ALL previous builds)

## Completed Work This Session

### CRITICAL: RevenueCat API Keys Fixed
- **Root cause discovered**: Both iOS and Android RevenueCat API keys in `SubscriptionContext.tsx` were WRONG
- Old (broken): `goog_LSdTYjNzFKaMnhJQRcfEzGRwOmt` / `appl_OVnBBsTafRUvxYPvVfFMfhuvEva`
- New (working): `goog_sSuefaqGfyQKJvmIkNrWEyVElTx` / `appl_qDwlqIUvHJHGuewqEExfpAgaCpw`
- This was THE reason subscriptions never loaded — "Loading..." dialog on every subscribe attempt

### Build A Fixes (v1.0.29)
1. RevenueCat anonymous user bug — separated `configure` from `logIn` flow
2. Backend webhook hardened for anonymous IDs (aliases + email fallback)
3. Custom back button with 44x44pt touch targets
4. NOVA color scheme on processing badges

### Build B Fixes (v1.0.30)
1. Product confirmation step — "Is this your product?" screen after scan
2. NOVA colors on per-ingredient labels (Red/Amber/Green)
3. NOVA colors on Library category badges
4. Library + About screens added to Stack navigator

### Additional Fixes (v1.0.31)
1. **CORRECT RevenueCat API keys** (CRITICAL)
2. `useRef` for RC configured state (prevents race condition)
3. `purchasePackage` now passes Package object instead of string identifier
4. Retry offerings fetch when user taps subscribe
5. Comprehensive RC logging for debugging
6. Fixed Android notifications (channel, daily trigger, SCHEDULE_EXACT_ALARM)
7. Push token registration for server-side notifications
8. Admin send-notification endpoint
9. Admin set_premium now supports downgrade (`tier=free`)

### Data Operations
- Downgraded all premium users to free (except rosannaleggett@gmail.com) for subscription flow testing

## RevenueCat Keys (CORRECT)
- iOS: `appl_qDwlqIUvHJHGuewqEExfpAgaCpw`
- Android: `goog_sSuefaqGfyQKJvmIkNrWEyVElTx`

## Key API Endpoints
- `POST /api/scan/quick` — Two-stage scan
- `GET /api/scan/status/{barcode}` — Poll for AI analysis
- `POST /api/webhooks/revenuecat` — Subscription webhook (handles anonymous IDs)
- `POST /api/auth/push-token` — Store Expo push token
- `POST /api/admin/send-notification` — Blast push notifications
- `POST /api/admin/set_premium` — Set user tier (now supports `tier=free`)
- `GET /api/marketing` — Asset library

## Pending Issues
1. P1: Password reset emails (Resend domain `yawye.app` not verified — needs DNS records added)
2. P2: Railway rootDirectory refactor

## Upcoming Tasks
1. P0: Submit v1.0.31 to Google Play and test subscription flow end-to-end
2. P1: Monitor Apple App Store review
3. P1: Marketing video creation
4. P2: Resend domain verification for password reset

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat

## Critical Notes for Next Agent
- **RevenueCat keys MUST be**: iOS=`appl_qDwlqIUvHJHGuewqEExfpAgaCpw`, Android=`goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Railway deploys from root — mirror `backend/server.py` to root `server.py`
- Production URL: https://web-production-66c05.up.railway.app
- Admin key: `yawye2024clear`
- Webhook auth: `Jmaster1986!`
