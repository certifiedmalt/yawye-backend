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
- **AI**: OpenAI GPT-4o-mini (via openai SDK, user API key) — switched from GPT-4o for speed
- **Monetization**: RevenueCat SDK, Google Play Billing, Apple IAP, RevenueCat Webhooks
- **Email**: Resend (domain verification pending)
- **Video Gen**: Sora 2 (via Emergent LLM Key)
- **Voiceover**: OpenAI TTS (via Emergent LLM Key)
- **Video Assembly**: FFmpeg

## Architecture
```
/
├── backend/
│   ├── server.py       # FastAPI backend (serves marketing library too)
│   └── requirements.txt
├── frontend/
│   ├── app/            # Expo Router pages
│   ├── context/        # Auth + Subscription contexts
│   └── utils/          # Subscription utilities
├── marketing/
│   ├── gen_clip.py             # Sora 2 clip generation script
│   ├── gen_split_voiceovers.py # OpenAI TTS voiceover generation
│   ├── assemble_v4.sh          # FFmpeg 4-part video assembly (logo + text + VO)
│   ├── production/             # Generated clips and voiceovers
│   ├── voiceovers/split/       # Per-clip voiceover audio files
│   ├── real_icon_watermark.png # App logo for video watermarking
│   └── FINAL_*.mp4             # Finished assembled ad videos
├── server.py           # Railway deployment copy
└── requirements.txt    # Railway deployment copy
```

## Key API Endpoints
- `POST /api/scan` - Scan barcode, analyze with GPT-4o-mini (full result)
- `POST /api/scan/quick` - Two-stage: returns product info instantly, analysis pending
- `GET /api/scan/status/{barcode}` - Poll for analysis completion (two-stage flow)
- `DELETE /api/cache/clear?key=yawye2024clear` - Clear product cache (no auth)
- `DELETE /api/admin/cache` - Clear product cache (admin auth)
- `POST /api/webhooks/revenuecat` - Subscription webhook
- `POST /api/subscription/upgrade` - Manual subscription upgrade fallback
- `POST /api/assistant/chat` - AI health assistant
- `GET /api/gamification/stats` - Streak & XP data
- `GET /api/marketing` - Marketing asset library page (multi-select enabled)
- `GET /api/marketing/video/{filename}` - Serve individual video
- `DELETE /api/marketing/video/{filename}` - Delete marketing asset

## What's Been Implemented
- Full barcode scanning with 7 food databases + GPT-4o-mini analysis
- User auth (register, login, forgot password)
- Subscription system (RevenueCat + Google Play Billing + Apple IAP + webhook) - VERIFIED WORKING
- Gamification (streaks, XP, daily quests, achievements)
- Health Assistant AI chat (GPT-4o-mini) - FIXED and deployed
- Push notifications
- App version v1.0.26 live on Google Play
- iOS build submitted to App Store Connect (pending Apple review)
- Full automated video ad pipeline (Sora 2 + TTS + FFmpeg)
- Marketing asset library with multi-select Save/Share/Delete
- iPad App Store screenshots (2048x2732)
- Static website served from backend (privacy, terms, support)
- AI analysis with carcinogen detection, chemical breakdown, shocking facts, alternatives

## Speed Optimizations (Implemented)
- **GPT-4o-mini**: Switched from GPT-4o for faster AI responses
- **In-memory LRU cache**: Top 500 products cached in RAM (~230ms repeat scans)
- **MongoDB cache**: 30-day persistent cache with auto-promotion to memory
- **AI temperature 0.3**: Consistent, repeatable scores
- **Early termination**: Stops querying databases once ingredients found
- **9s database timeout**: Not-found products fail faster
- **Trimmed AI prompts**: ~60% smaller input, removed study_link output
- **7 food databases**: OFF Global, OFF UK, USDA, UPC Item DB, Brocade.io, OFF Search, FatSecret
- **Two-stage scan API**: `/api/scan/quick` returns product info in ~6s, analysis loads after

## Speed Optimizations (Noted for Next Build - Frontend Required)
- **Two-stage UI**: Frontend calls `/api/scan/quick` first → shows product name/image instantly → polls `/api/scan/status/{barcode}` → loads AI analysis when ready
- **Pre-warm cache**: Batch scan top 500-1000 UK product barcodes to pre-populate cache
- **Connection pooling**: Use requests.Session() for reused HTTP connections
- **Gzip compression**: Enable response compression middleware
- **Request deduplication**: Prevent double-scans within 5 seconds

## Pending Issues
1. **P1**: Password reset emails blocked (Resend domain `yawye.app` not verified by user)
2. **P2**: Custom domain `yawye.app` not pointed to Railway (user action needed)
3. **P2**: Health Assistant keyboard may hide text input (user verification pending)
4. **P2**: Splash screen may show generic logo (user verification pending)

## Upcoming Tasks
1. **P0**: Wait for Apple App Store review approval
2. **P1**: Set up In-App Purchase subscription on App Store Connect (DONE - `premium_monthly` at £1.99)
3. **P1**: Update frontend to use two-stage scan (`/api/scan/quick` → poll → full result) — requires new app build
4. **P1**: Pre-warm cache with top UK product barcodes
5. **P2**: Update website & marketing with App Store link once iOS is live
6. **P2**: Refactor Railway deployment (rootDirectory config to remove file duplication)

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Fix Railway deployment duplication (rootDirectory config)
- RevenueCat for Amazon IAP
- Clean up /app/marketing/ directory (remove superseded scripts)

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- Premium: jason.psaila@kwik-fit.com
- Test: jpsaila@live.com
