# You Are What You Eat (YAWYE) - Product Requirements Document

## Original Problem Statement
Build a mobile food scanning app that analyzes food products via barcode, provides AI-powered health scores based on ingredients and processing level, and offers premium features through subscriptions.

## Core Principles
- Ultra-processed foods and certain harmful ingredients should be scored negatively
- AI explanations must be detailed and science-backed
- App must work independently of the development agent

## Tech Stack
- **Frontend**: React Native (Expo), TypeScript
- **Backend**: Python, FastAPI
- **Database**: MongoDB (on Railway)
- **Deployment**: Railway (backend), Google Play Store (frontend)
- **AI**: OpenAI GPT-4o (via openai SDK, user API key)
- **Monetization**: RevenueCat SDK, Google Play Billing, RevenueCat Webhooks
- **Email**: Resend (domain verification pending)

## Architecture
```
/
├── backend/
│   ├── server.py       # FastAPI backend
│   └── requirements.txt
├── frontend/
│   ├── app/            # Expo Router pages
│   ├── context/        # Auth + Subscription contexts
│   └── utils/          # Subscription utilities
├── marketing/          # Generated marketing assets
├── server.py           # Railway deployment copy
└── requirements.txt    # Railway deployment copy
```

## Key Screens
1. Login/Register (dark theme, green accents)
2. Main Dashboard (greeting, stats, scan button, health streak, assistant)
3. Barcode Scanner (camera with green corner brackets)
4. Result Screen (circular health score ring, harmful/beneficial ingredients)
5. Health Assistant (AI chat with disclaimer)
6. Daily Quiz & Achievements (gamification)

## Key API Endpoints
- `POST /api/scan` - Scan barcode, analyze with GPT-4o
- `POST /api/webhooks/revenuecat` - Subscription webhook
- `POST /api/subscription/upgrade` - Manual subscription upgrade fallback
- `POST /api/assistant/chat` - AI health assistant
- `GET /api/gamification/stats` - Streak & XP data
- `POST /api/gamification/update-streak` - Update streak after scan

## What's Been Implemented
- Full barcode scanning with OpenFood Facts API + GPT-4o analysis
- User auth (register, login, forgot password)
- Subscription system (RevenueCat + Google Play Billing + webhook)
- Gamification (streaks, XP, daily quests, achievements)
- Health Assistant AI chat
- Push notifications
- App version v1.0.26 live on Google Play

## Pending Issues
1. **P0**: End-to-end subscription flow needs real user validation
2. **P1**: Password reset emails blocked (Resend domain not verified)
3. **P2**: Health Assistant keyboard may hide text input
4. **P2**: Splash screen may show generic logo

## Completed Tasks (This Session - March 10, 2026)
- Generated 8 AI marketing images (phone mockups, banners, lifestyle, comparison, IG story)
- Generated 2 new Sora 2 video clips (app score reveal, dashboard hero)
- Created marketing assets catalog (assets-catalog.html)

## Upcoming Tasks
1. **P0**: Submit latest build to Google Play
2. **P1**: iOS App Store Connect setup
3. **P1**: RevenueCat for Amazon IAP
4. **P2**: Deploy static website to GitHub Pages

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Fix Railway deployment duplication (root directory config)

## Marketing Assets Generated
### Images (8 total)
- Dashboard mockup, Scan screen mockup, Unhealthy result (3/10), Healthy result (9/10)
- Wide promo banner, Before/After comparison, Lifestyle shopping photo, Instagram story ad

### Videos (5 total)
- app_score_reveal_unhealthy.mp4 (NEW - phone showing 3/10 score)
- app_dashboard_hero.mp4 (NEW - phone on kitchen counter)
- yawye_ad_clip1_problem.mp4 (person confused at labels)
- yawye_ad_clip2_solution.mp4 (person scanning with app)
- yawye_ad_clip2_couple.mp4 (couple shopping together)

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- Premium: jason.psaila@kwik-fit.com
- Test: jpsaila@live.com
