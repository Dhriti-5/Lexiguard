# LexiGuard Dashboard - Quick Setup Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Node.js 16+ installed
- Python 3.8+ installed
- Firebase account (free tier is sufficient)
- Existing LexiGuard project setup

---

## Step 1: Install Dependencies

### Frontend
```powershell
cd lexiguard-frontend
npm install firebase
```

### Backend (Optional - for future token verification)
```powershell
cd ..\lexiguard-backend
pip install firebase-admin
```

---

## Step 2: Firebase Setup

### A. Create Firebase Project
1. Go to https://console.firebase.google.com/
2. Click "Add project" or select existing "LexiGuard" project
3. Follow the wizard (Analytics optional)

### B. Enable Authentication
1. In Firebase Console → Authentication
2. Click "Get started"
3. Enable "Google" sign-in method
4. Save

### C. Enable Firestore
1. In Firebase Console → Firestore Database
2. Click "Create database"
3. Select "Start in production mode"
4. Choose location (e.g., us-central1)
5. Click "Enable"

### D. Get Firebase Config
1. Project Settings (gear icon) → General
2. Scroll to "Your apps" section
3. Click "</>" (Web) to add a web app
4. Register app nickname: "LexiGuard Web"
5. Copy the `firebaseConfig` object

---

## Step 3: Configure Environment Variables

### Create/Update: `lexiguard-frontend/.env`

```env
# Firebase Configuration
REACT_APP_FIREBASE_API_KEY=AIzaSy...
REACT_APP_FIREBASE_AUTH_DOMAIN=lexiguard-xxxxx.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=lexiguard-xxxxx
REACT_APP_FIREBASE_STORAGE_BUCKET=lexiguard-xxxxx.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=123456789
REACT_APP_FIREBASE_APP_ID=1:123456789:web:abcdef123456

# Backend API (existing)
REACT_APP_API_URL=http://localhost:8000
```

**⚠️ Important:** Replace the values above with YOUR Firebase config values!

---

## Step 4: Deploy Firestore Security Rules

### Option A: Using Firebase Console (Easiest)
1. Go to Firestore → Rules tab
2. Copy the contents of `firestore.rules` file
3. Paste into the editor
4. Click "Publish"

### Option B: Using Firebase CLI (Recommended for production)
```powershell
# Install Firebase CLI
npm install -g firebase-tools

# Login
firebase login

# Initialize (select Firestore)
firebase init firestore

# Deploy rules
firebase deploy --only firestore:rules

# Deploy indexes
firebase deploy --only firestore:indexes
```

---

## Step 5: Create Firestore Indexes

### Option A: Automatic (Recommended)
1. Run the app and navigate to `/dashboard`
2. Firestore will show an error with a link to create the index
3. Click the link and create the index

### Option B: Manual
1. Firestore Console → Indexes tab
2. Click "Add index"
3. Collection ID: `userAnalyses`
4. Add fields:
   - `userID` (Ascending)
   - `uploadTimestamp` (Descending)
5. Query scope: Collection
6. Click "Create"

Or use the `firestore.indexes.json` file with Firebase CLI.

---

## Step 6: Test the Setup

### Start Backend
```powershell
cd lexiguard-backend
python -m uvicorn main:app --reload --port 8000
```

### Start Frontend (new terminal)
```powershell
cd lexiguard-frontend
npm start
```

### Test Flow
1. Navigate to http://localhost:3000
2. Click "Login" and sign in with Google
3. Go to "Upload" page
4. Upload a test document (or use sample_contract.txt)
5. Choose "Detailed Analysis"
6. Click "Analyze Document"
7. Wait for analysis to complete
8. You should be redirected to Results page
9. Click "Dashboard" in nav menu
10. You should see your analysis card!

### Verify Firestore
1. Open Firebase Console → Firestore
2. You should see a new collection: `userAnalyses`
3. Inside, your analysis document with redacted text

---

## Step 7: Verify Privacy Features

### Check PII Redaction
1. In Dashboard, look for green badge: "✓ PII Redacted"
2. Click "View Analysis" on an analysis card
3. In the analysis text, you should see placeholders like:
   - `[PERSON_NAME]`
   - `[EMAIL_ADDRESS]`
   - `[STREET_ADDRESS]`
   - `[PHONE_NUMBER]`

### Verify Security
1. Open browser DevTools → Application → IndexedDB
2. Find Firestore local cache
3. Open `userAnalyses` collection
4. Verify only YOUR user analyses are visible
5. Try accessing `/results/:analysisId` with an ID you don't own
   - Should redirect to Dashboard or show error

---

## ✅ Success Checklist

- [ ] Firebase project created
- [ ] Authentication enabled (Google)
- [ ] Firestore enabled
- [ ] Security rules deployed
- [ ] Environment variables configured
- [ ] Frontend dependencies installed
- [ ] Can login with Google
- [ ] Can upload and analyze document
- [ ] Analysis auto-saves to Firestore
- [ ] Dashboard displays analyses
- [ ] Can view analysis from Dashboard
- [ ] PII redaction badge visible
- [ ] Can rename, star, delete analyses
- [ ] Search, filter, sort work correctly
- [ ] Stats cards show accurate counts

---

## 🐛 Troubleshooting

### "Firebase not configured" error
- Check `.env` file exists in `lexiguard-frontend`
- Verify all `REACT_APP_FIREBASE_*` variables are set
- Restart development server: `Ctrl+C` then `npm start`

### "Permission denied" in Firestore
- Ensure Security Rules are deployed
- Check user is logged in: See profile photo in nav bar
- Verify `userID` in Firestore matches your Firebase UID

### Analyses not appearing in Dashboard
- Open browser console (F12), check for errors
- Go to Firebase Console → Firestore → userAnalyses
- Check if documents exist with your `userID`
- Create index if Firestore prompts you

### Analysis not saving
- Check browser console for errors
- Verify Firestore Security Rules allow `create`
- Ensure user is authenticated before analyzing
- Check network tab for 403/401 errors

---

## 🎯 Next Steps

### Recommended Enhancements
1. Add pagination to Dashboard (for 50+ analyses)
2. Implement bulk operations (delete multiple)
3. Add tags and notes features
4. Create shareable analysis links
5. Export analyses to PDF/CSV

### Backend Integration (Optional)
If you want backend token verification:

1. Download Firebase Service Account Key
   - Firebase Console → Project Settings → Service Accounts
   - Click "Generate new private key"
   - Save as `serviceAccountKey.json` in backend folder

2. Update backend `main.py`:
```python
from firebase_admin import auth, credentials, initialize_app

cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred)

# Add middleware to verify tokens
@app.middleware("http")
async def verify_token(request: Request, call_next):
    if "/dashboard" in request.url.path or "/results" in request.url.path:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        try:
            decoded = auth.verify_id_token(token)
            request.state.user_id = decoded['uid']
        except:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
```

---

## 📚 Additional Resources

- **Firebase Docs**: https://firebase.google.com/docs
- **Firestore Queries**: https://firebase.google.com/docs/firestore/query-data/queries
- **Security Rules Guide**: https://firebase.google.com/docs/firestore/security/get-started
- **React Firebase**: https://firebase.google.com/docs/web/setup

---

## 🆘 Need Help?

If you encounter issues:

1. Check the detailed guide: `DASHBOARD-FEATURE-GUIDE.md`
2. Review browser console for errors (F12)
3. Check Firebase Console for Firestore errors
4. Verify all environment variables are set correctly
5. Ensure backend is running on port 8000
6. Try clearing browser cache and local storage

---

**Dashboard Setup Complete! 🎉**

Your LexiGuard now has a powerful, privacy-first dashboard for managing legal document analyses.

**Remember:** Only redacted analyses are stored. Original documents are NEVER saved to Firestore!
