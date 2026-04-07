# You Are What You Eat - PRD

## Original Problem Statement
Food scanning app that analyzes products via barcode, provides health scores based on ingredients/processing level via AI (GPT-4o), and offers premium features through subscriptions. Live on Google Play, submitting to Apple App Store. Companion app to a book/audiobook franchise.

## Architecture
- **Frontend**: React Native (Expo), TypeScript, EAS Build/Submit
- **Backend**: Python, FastAPI (deployed on Railway)
- **Database**: MongoDB (on Railway)
- **AI**: OpenAI GPT-4o
- **Subscriptions**: RevenueCat (Google Play + Apple App Store)

## Current Status (April 7, 2026)
- **v1.0.32** (versionCode 33): Live on Google Play, submitted for Apple review
- Backend deployed on Railway with hardened webhook
- First REAL subscription processed successfully (Status: Processed, not Refunded!)

## RevenueCat Keys
- iOS: `appl_qDwlqIUvHJHGuewqEExfpAgaCpw`
- Android: `goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Secret: `sk_lkgtpXULPFdEblMPvMXeaRJPDcepH`

## Google Cloud Service Account
- Email: `revenuecat@lithe-augury-486501-i9.iam.gserviceaccount.com`
- JSON uploaded to RevenueCat: YES (April 7, 2026)
- Propagation: Up to 36 hours from upload

## App Store Connect
- Issuer ID: `15e998af-a427-4258-a434-7ed291c56f1b`
- API Key: `QS39X6QRC7` (EAS Submit)
- iOS v1.0.32 (build 33): Submitted for Apple review (April 7, 2026)
- Subscription: `yawye_premium_monthly` attached to submission
- Shared Secret: PENDING (need to set up for RevenueCat iOS validation)

## Completed This Session (April 7, 2026)
1. Uploaded Google Play Service Account JSON to RevenueCat (P0 BLOCKER resolved)
2. Resubmitted iOS v1.0.32 to Apple App Store review (with subscription attached)
3. Verified first REAL subscription: Order GPA.3392-2092-7721-00952, Status: Processed
4. Confirmed 68 total users, 2 premium subscribers
5. Set Jason's account back to free then resubscribed successfully
6. Helped Miha (zeta.mol18@gmail.com) re-register after password issue

## User Stats
- Total users: 68
- Premium subscribers: 2 (Jason, Rosie Leggett)
- Notable active free users: Miha (53 scans), Dan (7), Michelle (7), Lisa (5), Glory (5)

## Pending Issues
1. P1: Apple App Store review (waiting for approval, 24-48 hours)
2. P1: Apple Shared Secret → RevenueCat (for iOS subscription validation)
3. P2: Password reset emails (Resend domain not verified - DNS records needed)
4. P2: Railway rootDirectory refactor

## Upcoming Tasks (After Apple Approval)
1. P0: Publish audiobook (ACX/Audible + Findaway Voices) - split into chapters first
2. P0: Launch media campaign (screenshots + AI-generated videos)
3. P1: Set up Apple Shared Secret in RevenueCat for iOS purchases
4. P2: Resend domain DNS verification for password reset emails

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Amazon IAP via RevenueCat
- Sell audiobook directly through the app

## Audiobook Assets
- Full audiobook: 61.5 MB MP3 (18 chapters)
- Preview clip: 14.2 MB MP3
- Book visuals: 9 HD illustrations
- Needs: Split into individual chapter files, cover art (2400x2400), ISBN

## Critical Notes for Next Agent
- RevenueCat keys MUST be: iOS=`appl_qDwlqIUvHJHGuewqEExfpAgaCpw`, Android=`goog_sSuefaqGfyQKJvmIkNrWEyVElTx`
- Railway deploys from root — mirror `backend/server.py` to root `server.py`
- Production URL: https://web-production-66c05.up.railway.app
- Admin key: `yawye2024clear`, Webhook auth: `Jmaster1986!`
- Google Play Service Account JSON uploaded to RevenueCat on April 7, 2026
- First real subscription CONFIRMED working (Status: Processed)
- Apple review submitted April 7, 2026 - check status before doing anything
- After Apple approval: need to add Shared Secret to RevenueCat for iOS
- Audiobook ready but needs splitting into chapters before publishing
- User wants to launch audiobook + media campaign simultaneously after Apple approval
