# You Are What You Eat - PRD

## Original Problem Statement
Food scanning app that analyzes products via barcode, provides health scores based on ingredients/processing level via AI (GPT-4o), and offers premium features through RevenueCat subscriptions. Live on Google Play, submitting to Apple App Store.

## Architecture
- **Frontend**: React Native (Expo), TypeScript, EAS Build/Submit
- **Backend**: Python, FastAPI (deployed on Railway)
- **Database**: MongoDB (on Railway)
- **AI**: OpenAI GPT-4o
- **Subscriptions**: RevenueCat (Google Play + Apple App Store)

## Current Status (April 2026)
- **v1.0.32** (versionCode 33): Built and ready for Google Play. "Subscribe Now" button, correct RC keys, all UI fixes.
- Backend deployed on Railway with hardened webhook

## RevenueCat Keys (CORRECT - verified against API)
- iOS: `appl_qDwlqIUvHJHGuewqEExfpAgaCpw`
- Android: `goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Secret: `sk_lkgtpXULPFdEblMPvMXeaRJPDcepH`

## Google Cloud Service Account
- Project: `lithe-augury-486501-i9`
- Email: `revenuecat@lithe-augury-486501-i9.iam.gserviceaccount.com`
- JSON key downloaded and available
- Google Play Android Developer API: ENABLED
- **PENDING**: Service account needs to be granted access in Google Play Console (Settings → API Access)
- **PENDING**: JSON needs to be uploaded to RevenueCat dashboard (requires Google Play Console linking first)

## Completed This Session
1. CRITICAL: Fixed wrong RevenueCat API keys (were invalid in ALL previous builds)
2. RevenueCat anonymous user bug — separated configure/logIn flow with useRef
3. Backend webhook hardened for anonymous IDs (aliases + email fallback)
4. Custom back button with 44x44pt touch targets
5. NOVA color scheme on processing badges, ingredient labels, and library badges
6. Product confirmation step ("Is this your product?") after barcode scan
7. Fixed Android notifications (channel, daily trigger, SCHEDULE_EXACT_ALARM permission)
8. Push token registration + admin send-notification endpoint
9. Admin set_premium now supports downgrade (tier=free)
10. Downgraded all premium users to free (except Rosanna) for testing
11. Removed "7-Day Free Trial" text, changed to "Subscribe Now"
12. Granted promotional premium entitlement via RevenueCat API
13. Google Cloud service account created for RevenueCat integration
14. Multiple EAS Android builds (v1.0.29 through v1.0.32)

## Pending Issues
1. P0: Google Play service account not linked in Play Console (needs desktop access)
2. P1: Password reset emails (Resend domain not verified - DNS records needed at Squarespace)
3. P2: Railway rootDirectory refactor

## Upcoming Tasks
1. P0: Link service account in Google Play Console (desktop only)
2. P0: Upload JSON to RevenueCat after linking
3. P1: Submit v1.0.32 to Google Play
4. P1: Marketing video creation
5. P2: Resend domain DNS verification

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat
- iOS App Store submission

## Critical Notes for Next Agent
- RevenueCat keys MUST be: iOS=`appl_qDwlqIUvHJHGuewqEExfpAgaCpw`, Android=`goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Railway deploys from root — mirror `backend/server.py` to root `server.py`
- Production URL: https://web-production-66c05.up.railway.app
- Admin key: `yawye2024clear`, Webhook auth: `Jmaster1986!`
- Google Play service account linking is BLOCKED until user has desktop access
- User's subscription was charged by Google Play but NOT tracked in RevenueCat (service account not connected)
- Promotional premium entitlement granted manually, expires May 3, 2026
