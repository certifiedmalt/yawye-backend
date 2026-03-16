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
- **Deployment**: Railway (backend), Google Play Store + Apple App Store (frontend)
- **AI**: OpenAI GPT-4o (via openai SDK, user API key)
- **Monetization**: RevenueCat SDK, Google Play Billing, Apple IAP, RevenueCat Webhooks
- **Email**: Resend (domain verification pending)

## Architecture
```
/
├── backend/
│   ├── server.py       # FastAPI backend (main application)
│   └── requirements.txt
├── frontend/
│   ├── app/            # Expo Router pages
│   ├── context/        # Auth + Subscription contexts
│   └── utils/          # Subscription utilities
├── server.py           # Railway deployment copy (MUST stay in sync with backend/server.py)
├── requirements.txt    # Railway deployment copy
├── prewarm_cache_v3.py # Smart cache pre-warming script
└── marketing/          # Marketing assets, videos, screenshots
```

## Key API Endpoints
- `POST /api/scan` - Scan barcode, analyze with GPT-4o (full result)
- `POST /api/admin/prewarm?key=yawye2024clear` - Pre-warm cache by product name (AI analysis)
- `POST /api/admin/cache_insert?key=yawye2024clear` - Direct cache insertion
- `GET /api/admin/cache_count?key=yawye2024clear` - Get number of cached products
- `DELETE /api/cache/clear?key=yawye2024clear` - Clear product cache
- `POST /api/webhooks/revenuecat` - Subscription webhook
- `POST /api/subscription/upgrade` - Manual subscription upgrade fallback
- `POST /api/assistant/chat` - AI health assistant
- `GET /api/gamification/stats` - Streak & XP data

## What's Been Implemented
- Full barcode scanning with 7 food databases + GPT-4o analysis
- User auth (register, login, forgot password)
- Subscription system (RevenueCat + Google Play + Apple IAP + webhook)
- Gamification (streaks, XP, daily quests, achievements)
- Health Assistant AI chat
- Push notifications
- App version v1.0.26 live on Google Play
- iOS build submitted to App Store Connect (pending Apple review)
- Static website served from backend (privacy, terms, support)
- AI analysis with carcinogen detection, chemical breakdown, shocking facts, alternatives
- **Product cache pre-warmed with 90 common UK grocery products** (Mar 2026)
- **Global exception handler for unhandled errors** (Mar 2026)
- **TimeoutError handling in ThreadPoolExecutor** (Mar 2026)

## Bug Fixes Applied (Mar 2026)
- Fixed SyntaxError from orphaned except blocks in scan_product function
- Fixed unhandled `TimeoutError` from `concurrent.futures.as_completed()` causing 500 errors on cold scans
- Added global FastAPI exception handler for any unhandled exceptions
- Moved `concurrent.futures` import to module level
- Created admin endpoints for cache management (prewarm, insert, count)
- Pre-warmed cache with 90 products — all returning in 2-5ms

## Pending Issues
1. **P1**: Password reset emails blocked (Resend domain `yawye.app` not verified by user)
2. **P1**: Cold scans for products NOT in cache still take 20-30s (mobile timeout issue)
3. **P2**: Custom domain `yawye.app` not pointed to Railway (user action needed)
4. **P2**: Health Assistant keyboard may hide text input (user verification pending)
5. **P2**: Splash screen may show generic logo (user verification pending)

## Upcoming Tasks
1. **P0**: Wait for Apple App Store review approval
2. **P1**: After approval, build new app version with longer network timeout + two-stage scan UI
3. **P1**: Continue expanding cache with more products
4. **P2**: Refactor Railway deployment (rootDirectory config to remove file duplication)

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- RevenueCat for Amazon IAP
- Two-stage scan UI (frontend shows product name instantly, AI analysis loads after)

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- Premium: jason.psaila@kwik-fit.com
- Test: jpsaila@live.com
- Cache admin key: yawye2024clear
