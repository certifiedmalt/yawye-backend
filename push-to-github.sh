#!/bin/bash

echo "=========================================="
echo "Push to GitHub - You Are What You Eat App"
echo "=========================================="
echo ""
echo "Your repository: https://github.com/certifiedmalt/you-are-what-you-eat-app"
echo ""
echo "IMPORTANT: You'll need a GitHub Personal Access Token (not your password)"
echo ""
echo "How to get a token:"
echo "1. Go to: https://github.com/settings/tokens"
echo "2. Click 'Generate new token (classic)'"
echo "3. Give it a name: 'Emergent Push'"
echo "4. Select scope: 'repo' (full control)"
echo "5. Click 'Generate token'"
echo "6. COPY the token (you won't see it again!)"
echo ""
echo "=========================================="
echo ""

# Push to GitHub
cd /app
git push -u origin main

echo ""
echo "When prompted:"
echo "  Username: certifiedmalt"
echo "  Password: [paste your GitHub token]"
echo ""
