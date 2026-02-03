# Git Setup Instructions - You Are What You Eat

## Option A: Using Emergent's GitHub Button (Recommended)

1. Look for "Save to GitHub" button in Emergent interface
2. Click it and follow the prompts
3. Choose repository name: `you-are-what-you-eat-app`
4. Set to Private
5. Click Push/Save

## Option B: Manual Git Setup (If needed)

If you need to set up Git manually, follow these steps:

### Step 1: Create Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon → **"New repository"**
3. Repository name: `you-are-what-you-eat-app`
4. Description: "Mobile app for scanning barcodes and analyzing food ingredients"
5. Select **Private**
6. **DO NOT** initialize with README
7. Click **"Create repository"**

### Step 2: Initialize Git Locally

```bash
cd /app

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: You Are What You Eat app with barcode scanning, AI analysis, and subscriptions"
```

### Step 3: Connect to GitHub

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/you-are-what-you-eat-app.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Enter Credentials

When prompted:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your regular password)

**How to create a Personal Access Token:**
1. GitHub → Settings → Developer settings
2. Personal access tokens → Tokens (classic)
3. Generate new token
4. Select scopes: `repo` (all)
5. Generate token
6. Copy and use as password

---

## What Gets Saved to GitHub

Your entire project:
```
/app
├── backend/
│   ├── server.py          # FastAPI backend with AI analysis
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables (RevenueCat keys)
├── frontend/
│   ├── app/               # All React Native screens
│   ├── context/           # Auth & Subscription contexts
│   ├── i18n/              # 6 languages translations
│   ├── utils/             # Currency & subscription utils
│   ├── package.json       # Node dependencies
│   └── app.json           # Expo configuration
├── legal-documents/
│   ├── privacy-policy.md
│   └── terms-of-service.md
└── PRODUCTION-LAUNCH-GUIDE.md
```

---

## After Pushing to GitHub

### Clone to Another Computer:
```bash
git clone https://github.com/YOUR_USERNAME/you-are-what-you-eat-app.git
cd you-are-what-you-eat-app
```

### Install Dependencies:
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
yarn install
```

### Run App:
```bash
# Backend
cd backend
python server.py

# Frontend (new terminal)
cd frontend
expo start
```

---

## Future Updates

After making changes:
```bash
git add .
git commit -m "Description of changes"
git push origin main
```

---

## Branches (For feature development)

```bash
# Create new feature branch
git checkout -b feature/new-feature

# Make changes, then push
git push origin feature/new-feature

# Merge via GitHub Pull Request when ready
```

---

## Important Notes

✅ **Your code is now backed up**
✅ **You can access it from anywhere**
✅ **You can share with developers**
✅ **You have version history**

⚠️ **Keep your .env file secure** - Never commit API keys publicly
⚠️ **Use Private repository** - Protects your intellectual property

---

## Need Help?

- GitHub Docs: https://docs.github.com
- Git Tutorial: https://git-scm.com/docs/gittutorial
- Emergent Support: support@emergent.sh
