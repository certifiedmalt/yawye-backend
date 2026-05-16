# You Are What You Eat - PRD

## Original Problem Statement
Food scanning app that analyzes products via barcode, provides health scores based on ingredients/processing level via AI (GPT-4o), and offers premium features through subscriptions. Live on Google Play, submitting to Apple App Store. Companion app to a book/audiobook franchise.

## Architecture
- **Frontend**: React Native (Expo), TypeScript, EAS Build/Submit
- **Backend**: Python, FastAPI (deployed on Railway)
- **Database**: MongoDB (on Railway)
- **AI**: OpenAI GPT-4o
- **Subscriptions**: RevenueCat (Google Play + Apple App Store)

## Current Status (April 21, 2026)
- **v1.0.33** (build 34): Submitted to App Store Connect — awaiting Apple processing
- Backend deployed on Railway with hardened webhook
- First REAL subscription processed successfully
- iOS v1.0.32 rejected twice by Apple (Guidelines 1.4.1, 2.1(b), 3.1.2(c))
- v1.0.33 includes all Apple Review fixes (citations, subscription metadata, IAP flow)

## RevenueCat Keys
- iOS: `appl_qDwlqIUvHJHGuewqEExfpAgaCpw`
- Android: `goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Secret: `sk_lkgtpXULPFdEblMPvMXeaRJPDcepH`

## Google Cloud Service Account
- Email: `revenuecat@lithe-augury-486501-i9.iam.gserviceaccount.com`
- JSON uploaded to RevenueCat: YES (April 7, 2026)
- Google Play Android Developer API: NEEDS ENABLING (causes auto-refunds)

## App Store Connect
- Issuer ID: `15e998af-a427-4258-a434-7ed291c56f1b`
- API Key: `QS39X6QRC7` (EAS Submit)
- iOS v1.0.33 (build 34): Submitted April 21, 2026
- Subscription: `yawye_premium_monthly` — MUST be submitted for review alongside the build
- Shared Secret: PENDING (need to set up for RevenueCat iOS validation)

## Completed This Session (April 21, 2026)
1. Analyzed Apple App Store rejection video — identified 4 rejection reasons across 3 guidelines
2. Fixed IAP purchase flow (Guideline 2.1(b)): Added timeouts, better error handling
3. Fixed subscription metadata (Guideline 3.1.2(c)): Added title, price, duration, Terms/Privacy links
4. Fixed medical citations (Guideline 1.4.1): Static "Sources & References" section on result + about pages with 7+ clickable PubMed/WHO/IARC/BMJ links
5. Added Legal section to About page (Terms, Privacy, Support links)
6. Created .easignore to reduce build archive from 1.2GB to 678MB
7. Pushed code to GitHub, triggered EAS Build, submitted v1.0.33 to App Store Connect

## User Stats
- Total users: 68+
- Premium subscribers: 2 (Jason, Rosie Leggett)

## Pending Issues
1. P0: Apple App Store review — v1.0.33 submitted, waiting for Apple processing + user must submit IAP products
2. P0: Google Play subscriptions auto-refunding — User needs to enable Google Play Android Developer API
3. P1: Apple Shared Secret → RevenueCat (for iOS subscription validation)
4. P2: Password reset emails (Resend domain not verified — DNS records needed)
5. P2: Railway rootDirectory refactor

## User Action Items (App Store Connect)
1. Submit IAP product `yawye_premium_monthly` for review (with screenshot)
2. Add Terms of Use URL (`https://yawye.app/terms-of-service`) to App Description or EULA field
3. Confirm Privacy Policy URL is set to `https://yawye.app/privacy-policy`
4. Confirm Paid Apps Agreement is active
5. Once v1.0.33 finishes processing, create a new version and submit BOTH build + IAP for review

## Upcoming Tasks
1. P1: Complete audiobook generation (3 sections remaining — needs Emergent LLM Key top-up)
2. P2: Create marketing videos
3. P2: Railway deployment refactor

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat
- Sell audiobook directly through the app

## Critical Notes for Next Agent
- v1.0.33 build 34 IPA submitted to App Store Connect on April 21, 2026
- User MUST submit IAP products separately in App Store Connect before Apple review
- Production URL: https://web-production-66c05.up.railway.app
- Admin key: `yawye2024clear`, Webhook auth: `Jmaster1986!`
- GitHub PAT: stored in git remote config
- Emergent LLM Key budget exhausted — user needs to add balance for audiobook completion
- Google Play Android Developer API must be enabled by user in Google Cloud Console
