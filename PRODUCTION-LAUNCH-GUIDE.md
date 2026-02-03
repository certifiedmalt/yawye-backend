# 🚀 Production Launch Guide - You Are What You Eat

## ✅ What's Already Done

- [x] Complete app built (barcode scanning, AI analysis, auth)
- [x] RevenueCat integrated with your API keys
- [x] Multi-language support (6 languages)
- [x] Multi-currency pricing (25+ regions)
- [x] Privacy Policy & Terms of Service created
- [x] Bundle IDs configured

---

## 📋 Complete Checklist to Launch

### **Phase 1: Legal Documents (30 minutes)**

#### 1.1 Host Privacy Policy & Terms

**Option A: Use Carrd.co (Free & Easy)**
1. Go to https://carrd.co
2. Sign up for free account
3. Create new site
4. Choose "Simple Landing Page" template
5. Add two sections:
   - Privacy Policy (copy from `/app/legal-documents/privacy-policy.md`)
   - Terms of Service (copy from `/app/legal-documents/terms-of-service.md`)
6. Publish site
7. Get URL: `https://yoursite.carrd.co`

**Option B: Buy Domain (Better for production)**
1. Go to https://namecheap.com
2. Buy domain: `youarewhatyoueat.com` (~$12/year)
3. Use Carrd.co or GitHub Pages to host
4. Point domain to hosting

**Important URLs you'll need:**
- Privacy Policy: `https://yourdomain.com/privacy-policy`
- Terms of Service: `https://yourdomain.com/terms`

---

### **Phase 2: RevenueCat Products Setup (15 minutes)**

#### 2.1 Create Entitlements
1. Go to RevenueCat Dashboard: https://app.revenuecat.com
2. Click **"Entitlements"** in left sidebar
3. Click **"+ New"**
4. Name it: **"premium"**
5. Identifier: **"premium"** (important!)
6. Click **"Save"**

#### 2.2 Create Products
1. Still in RevenueCat, click **"Products"** tab
2. Click **"+ New"** 

**Monthly Product:**
- Name: **"Premium Monthly"**
- Identifier: **"premium_monthly"**
- Duration: **Monthly**
- Free Trial: **7 days** (optional)
- Link to entitlement: **"premium"**
- Save

**Yearly Product:**
- Name: **"Premium Yearly"**
- Identifier: **"premium_yearly"**
- Duration: **Yearly**
- Free Trial: **7 days** (optional)
- Link to entitlement: **"premium"**
- Save

#### 2.3 Create Offering
1. Click **"Offerings"** tab
2. Click **"+ New"**
3. Name: **"default"**
4. Identifier: **"default"** (important!)
5. Add packages:
   - Add **"premium_monthly"** product
   - Add **"premium_yearly"** product
6. Mark as **"Current Offering"**
7. Save

---

### **Phase 3: Apple App Store Setup**

#### 3.1 Register Apple Developer Account
1. Go to https://developer.apple.com
2. Click "Account"
3. Enroll ($99/year)
4. Complete identity verification (may take 24-48 hours)

#### 3.2 Create App in App Store Connect
1. Go to https://appstoreconnect.apple.com
2. Click **"My Apps"** → **"+"** → **"New App"**
3. Fill in:
   - **Platform**: iOS
   - **Name**: You Are What You Eat
   - **Primary Language**: English (US)
   - **Bundle ID**: com.youarewhatyoueat.app (must match your app)
   - **SKU**: youarewhatyoueat (unique identifier)
4. Click **"Create"**

#### 3.3 Create In-App Purchases (Subscriptions)
1. In your app, go to **"Features"** → **"In-App Purchases"**
2. Click **"+"** → **"Auto-Renewable Subscription"**

**Create Subscription Group:**
- Name: Premium Subscriptions
- Click **"Create"**

**Monthly Subscription:**
- Reference Name: Premium Monthly
- Product ID: **premium_monthly** (must match RevenueCat!)
- Subscription Group: Premium Subscriptions
- Duration: 1 Month
- Price: $4.99 (select for each region)
- Free Trial: 7 Days
- Localization:
  - Display Name: Premium Monthly
  - Description: Unlimited scans, full ingredient analysis
- Screenshot: (you'll upload later)
- Review Information: test@youarewhatyoueat.com
- Submit for review

**Yearly Subscription:**
- Reference Name: Premium Yearly
- Product ID: **premium_yearly**
- Subscription Group: Premium Subscriptions
- Duration: 1 Year
- Price: $39.99 (select for each region)
- Free Trial: 7 Days
- Localization: Same as above
- Submit for review

#### 3.4 Connect RevenueCat to App Store
1. Go to RevenueCat Dashboard
2. Click your iOS app
3. Scroll to **"App Store Connect Integration"**
4. Generate Shared Secret:
   - Go to App Store Connect
   - Users & Access → Shared Secret → Generate
   - Copy and paste into RevenueCat
5. Generate In-App Purchase Key:
   - App Store Connect → Keys → In-App Purchase
   - Create API Key
   - Download .p8 file
   - Upload to RevenueCat
6. Save

---

### **Phase 4: Google Play Store Setup**

#### 4.1 Register Google Play Developer Account
1. Go to https://play.google.com/console
2. Create account ($25 one-time)
3. Complete identity verification
4. Accept agreements

#### 4.2 Create App in Google Play Console
1. Click **"Create app"**
2. Fill in:
   - **App name**: You Are What You Eat
   - **Default language**: English (United States)
   - **App or game**: App
   - **Free or paid**: Free
3. Complete declarations
4. Click **"Create app"**

#### 4.3 Set Up Store Listing
1. Go to **"Main store listing"**
2. Fill in:
   - **Short description** (80 chars):
     "Scan barcodes, analyze ingredients, avoid ultra-processed foods"
   - **Full description** (4000 chars):
     ```
     Discover what's really in your food with You Are What You Eat!
     
     🔍 SCAN & ANALYZE
     Simply scan any product barcode to instantly reveal:
     • Ultra-processed food (UPF) content
     • Harmful additives and preservatives
     • Health risks backed by scientific studies
     • Beneficial ingredients
     • Overall health score (1-10)
     
     🧪 AI-POWERED ANALYSIS
     Our advanced AI analyzes ingredients using the NOVA classification system, focusing on:
     • Emulsifiers
     • Artificial sweeteners
     • Preservatives
     • Modified starches
     • Hydrogenated oils
     • And more...
     
     📚 SCIENCE-BACKED
     Every analysis includes references to actual research studies and clinical trials.
     
     ✨ FEATURES
     FREE:
     • 5 scans per day
     • Basic ingredient analysis
     • Health scores
     
     PREMIUM:
     • Unlimited daily scans
     • Detailed UPF analysis
     • Scan history
     • Save favorites
     • Multi-language support
     
     🌍 GLOBAL DATABASE
     Access to millions of products worldwide through Open Food Facts.
     
     💡 MAKE INFORMED CHOICES
     Knowledge is power. Understand what you're eating and make healthier choices for you and your family.
     
     Download now and start your journey to better eating!
     ```
   - **App icon**: 512x512px (create one or use Canva)
   - **Feature graphic**: 1024x500px
   - **Phone screenshots**: At least 2 (take from your app)
   - **Category**: Health & Fitness
   - **Contact email**: support@youarewhatyoueat.com
   - **Privacy policy URL**: https://yourdomain.com/privacy-policy
3. Save

#### 4.4 Create In-App Products (Subscriptions)
1. Go to **"Monetization"** → **"Subscriptions"**
2. Click **"Create subscription"**

**Monthly Subscription:**
- Product ID: **premium_monthly** (must match RevenueCat!)
- Name: Premium Monthly
- Description: Unlimited scans and full analysis
- Billing period: 1 Month
- Price: $4.99 USD
- Free trial: 7 days
- Grace period: 3 days
- Save

**Yearly Subscription:**
- Product ID: **premium_yearly**
- Name: Premium Yearly
- Description: Unlimited scans and full analysis - Save 33%!
- Billing period: 1 Year
- Price: $39.99 USD
- Free trial: 7 days
- Grace period: 3 days
- Save

#### 4.5 Connect RevenueCat to Google Play
1. Go to Google Play Console
2. **"Setup"** → **"API access"**
3. Link Google Cloud project
4. Create service account
5. Grant permissions:
   - View financial data
   - Manage subscriptions
   - View app information
6. Download JSON key file
7. Go to RevenueCat Dashboard
8. Upload JSON key to your Android app
9. Save

---

### **Phase 5: Build Production App**

#### 5.1 Install EAS CLI
```bash
npm install -g eas-cli
```

#### 5.2 Login to Expo
```bash
cd /app/frontend
eas login
```

#### 5.3 Configure EAS Build
Create `/app/frontend/eas.json`:
```json
{
  "build": {
    "production": {
      "node": "18.18.0",
      "ios": {
        "bundleIdentifier": "com.youarewhatyoueat.app",
        "buildType": "release"
      },
      "android": {
        "buildType": "apk"
      }
    }
  }
}
```

#### 5.4 Build iOS
```bash
eas build --platform ios --profile production
```

This will:
- Ask you to create Apple credentials
- Upload code to Expo servers
- Build IPA file
- Takes ~15-20 minutes

#### 5.5 Build Android
```bash
eas build --platform android --profile production
```

This will:
- Ask you to create Android keystore
- Upload code to Expo servers
- Build AAB file
- Takes ~15-20 minutes

---

### **Phase 6: Test Before Submitting**

#### 6.1 iOS Testing (TestFlight)
1. After iOS build completes, go to App Store Connect
2. **"TestFlight"** tab
3. Upload your IPA file (or use EAS Submit)
4. Add internal testers (your email)
5. Install TestFlight app on iPhone
6. Test:
   - Login/Register
   - Barcode scanning
   - AI analysis
   - Subscriptions (use sandbox account)
   - All major features

**Create Sandbox Tester:**
- App Store Connect → Users & Access → Sandbox Testers
- Add tester with fake email
- Use this to test in-app purchases

#### 6.2 Android Testing (Internal Testing)
1. Go to Google Play Console
2. **"Testing"** → **"Internal testing"**
3. Create new release
4. Upload AAB file
5. Add testers (your email)
6. Install app via Play Store link
7. Test same as iOS

---

### **Phase 7: Submit to App Stores**

#### 7.1 Submit to Apple App Store
1. App Store Connect → Your App
2. Go to **"App Store"** tab
3. Create new version (e.g., 1.0)
4. Fill in:
   - **Screenshots**: Upload 6.5" and 5.5" iPhone screenshots
   - **Description**: Copy from Google Play
   - **Keywords**: barcode, food, ingredients, health, nutrition, upf
   - **Support URL**: https://yourdomain.com
   - **Marketing URL**: https://yourdomain.com
   - **Privacy Policy URL**: https://yourdomain.com/privacy-policy
   - **Age Rating**: 4+
   - **Content Rights**: Check that you own rights
5. Select build from TestFlight
6. Answer review questions
7. Click **"Submit for Review"**

**Review time**: 1-3 days typically

#### 7.2 Submit to Google Play Store
1. Google Play Console → Your App
2. **"Release"** → **"Production"**
3. Click **"Create new release"**
4. Upload AAB from EAS build
5. Release name: 1.0
6. Release notes:
   ```
   🎉 Initial Release
   - Scan product barcodes
   - AI-powered ingredient analysis
   - Identify ultra-processed foods
   - Health scores and recommendations
   - Multi-language support
   ```
7. Review and rollout: 100% of users
8. **"Review release"** → **"Start rollout to Production"**

**Review time**: Usually within 24-48 hours

---

### **Phase 8: Post-Launch**

#### 8.1 Monitor Analytics
- RevenueCat: Track subscriptions, MRR, churn
- App Store Connect: Downloads, crashes
- Google Play Console: Installs, ratings

#### 8.2 Respond to Reviews
- Reply to user reviews (good and bad)
- Fix bugs quickly
- Add requested features

#### 8.3 Marketing
- Create social media accounts
- Post about launch
- Reach out to health/food bloggers
- Submit to Product Hunt
- Create TikTok/Instagram Reels showing app in action

---

## 🎯 Quick Reference Checklist

### Before You Can Submit:
- [ ] Privacy policy hosted online
- [ ] Terms of service hosted online
- [ ] RevenueCat entitlements created
- [ ] RevenueCat products created
- [ ] Apple Developer account ($99)
- [ ] Google Play Developer account ($25)
- [ ] App icon designed (512x512px)
- [ ] Screenshots taken (multiple sizes)
- [ ] App description written
- [ ] Support email set up

### For Apple:
- [ ] In-app purchases created in App Store Connect
- [ ] RevenueCat connected to App Store
- [ ] iOS build completed with EAS
- [ ] TestFlight testing done
- [ ] All App Store fields filled
- [ ] Submitted for review

### For Google:
- [ ] Subscriptions created in Google Play Console
- [ ] RevenueCat connected to Google Play
- [ ] Android build completed with EAS
- [ ] Internal testing done
- [ ] All Play Store fields filled
- [ ] Submitted for review

---

## 💰 Total Costs to Launch

| Item | Cost |
|------|------|
| Apple Developer | $99/year |
| Google Play Developer | $25 one-time |
| Domain name | ~$12/year |
| App icon design (optional) | $0-50 |
| **Total First Year** | **~$136-186** |

## 📞 Need Help?

- **RevenueCat Support**: https://community.revenuecat.com
- **Apple Support**: https://developer.apple.com/contact
- **Google Support**: https://support.google.com/googleplay/android-developer

---

## 🎉 Estimated Timeline

| Phase | Time |
|-------|------|
| Legal documents | 30 mins |
| RevenueCat setup | 15 mins |
| Apple account | 2-3 days (verification) |
| Google account | 1 day (verification) |
| App Store setup | 1 hour |
| Build apps | 1 hour |
| Testing | 2-3 hours |
| Submit | 1 hour |
| **Total active time** | **~6-8 hours** |
| **Total calendar time** | **3-5 days** |

---

**You've got this! 🚀 Follow these steps and you'll be live on both app stores in less than a week!**
