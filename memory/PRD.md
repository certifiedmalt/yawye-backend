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
- `POST /api/scan` - Scan barcode, analyze with GPT-4o
- `POST /api/webhooks/revenuecat` - Subscription webhook
- `POST /api/subscription/upgrade` - Manual subscription upgrade fallback
- `POST /api/assistant/chat` - AI health assistant
- `GET /api/gamification/stats` - Streak & XP data
- `GET /api/marketing` - Marketing asset library page
- `GET /api/marketing/video/{filename}` - Serve individual video
- `DELETE /api/marketing/video/{filename}` - Delete marketing asset

## What's Been Implemented
- Full barcode scanning with OpenFood Facts API + GPT-4o analysis
- User auth (register, login, forgot password)
- Subscription system (RevenueCat + Google Play Billing + webhook) - VERIFIED WORKING
- Gamification (streaks, XP, daily quests, achievements)
- Health Assistant AI chat (GPT-4o) - FIXED and deployed
- Push notifications
- App version v1.0.26 live on Google Play
- Full automated video ad pipeline (Sora 2 + TTS + FFmpeg)
- Marketing asset library with Save/Delete

## Pending Issues
1. **P1**: Password reset emails blocked (Resend domain not verified by user)
2. **P2**: Health Assistant keyboard may hide text input (user verification pending)
3. **P2**: Splash screen may show generic logo (user verification pending)

## Completed Marketing Videos (13 total)

### Original 8 Videos (Scripts 01-14, 3-clip format)
- FINAL_01.mp4 - Fridge Score Expose
- FINAL_01_upf_shock_test.mp4 - UPF Shock Test
- FINAL_02.mp4 - Gym Bro Protein Shake
- FINAL_03.mp4 - Date Night Dinner
- FINAL_07.mp4 - Breakfast Betrayal
- FINAL_09.mp4 - Influencer Fraud
- FINAL_12.mp4 - Parents Snack Check
- FINAL_13.mp4 - Meal Deal Lunch
- FINAL_14.mp4 - PT Protein Bar

### New 5 Videos (Scripts 16-20, 4-clip format, ~18-19s each)
- FINAL_16.mp4 (2.7MB, 19s) - First Scan Discovery
- FINAL_17.mp4 (3.5MB, 17s) - Teaching Mum
- FINAL_18.mp4 (4.8MB, 19s) - 7-Day Challenge
- FINAL_19.mp4 (4.2MB, 18s) - The Mate Challenge
- FINAL_20.mp4 (4.8MB, 19s) - Supermarket Warrior

All videos feature: real app logo watermark, synced male voiceover, text overlays, brand green (#00E676) CTA.

## Upcoming Tasks
1. **P1**: Submit latest build to Google Play
2. **P1**: iOS App Store Connect setup
3. **P2**: UPF-focused landing page for ad campaigns
4. **P2**: Deploy static website (privacy/terms) to GitHub Pages

## Future/Backlog
- Multi-language support (i18n)
- Samsung Galaxy Store publishing
- Fix Railway deployment duplication (rootDirectory config)
- RevenueCat for Amazon IAP

## Credentials
- Admin: jpsaila1986@gmail.com / hello123
- Premium: jason.psaila@kwik-fit.com
- Test: jpsaila@live.com
