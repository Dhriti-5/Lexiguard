# 🎯 LexiGuard Dashboard Feature - Complete Implementation Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Setup Instructions](#setup-instructions)
5. [Firestore Configuration](#firestore-configuration)
6. [Security Rules](#security-rules)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

---

## 🌟 Overview

The LexiGuard Dashboard transforms your one-time analysis tool into a **persistent, intelligent legal companion** with:

- ✅ **Privacy-First Architecture**: Only redacted analyses are stored (NO original documents)
- 📊 **Comprehensive Analytics**: Track risk patterns across all your documents
- 🔐 **User Isolation**: Firebase Authentication + Firestore Security Rules
- 🎨 **Beautiful UI**: Consistent with existing LexiGuard design language
- 🚀 **Scalable**: Built on Firebase/Firestore for auto-scaling

---

## 🏗️ Architecture

### Data Flow

```
User Upload → FastAPI Backend → Analysis (Gemini + DLP) → Firestore Storage
                                      ↓
                              Redacted Text Only
                              (PII Replaced)
                                      ↓
                              Dashboard Display
```

### Tech Stack

**Frontend:**
- React 18+ with Hooks
- React Router v6 (with URL parameters)
- Firebase JS SDK (Auth + Firestore)
- Framer Motion (animations)
- Tailwind CSS

**Backend:**
- FastAPI (Python)
- Firebase Admin SDK (optional for token verification)
- Google Cloud DLP (PII redaction)
- Google Gemini AI (analysis)

**Database:**
- Cloud Firestore (NoSQL, real-time)

---

## ✨ Features

### 1. Dashboard Home (`/dashboard`)

**Stats Overview:**
- Total Analyses
- High/Medium/Low Risk Counts
- Total Clauses Analyzed

**Analysis Management:**
- Search by title/filename
- Filter by type (Standard/Detailed/Starred)
- Sort by date/title/risk level
- Inline title editing
- Star/favorite analyses
- Delete with confirmation

**Privacy Badge:**
- Prominently displays "PII Redacted" status
- Explains that only redacted data is stored

### 2. Results Page Enhancement (`/results/:analysisId`)

**Supports Two Access Patterns:**
1. **Direct from Upload**: Immediate results with optional save
2. **From Dashboard**: Load analysis from Firestore by ID

**Features:**
- Full analysis display (Standard or Detailed)
- Chat with document (using redacted text)
- Generate negotiation emails
- Export to PDF (if implemented)

### 3. Upload Page Integration

**Automatic Saving:**
- After successful analysis, automatically saves to Firestore
- User is authenticated via Firebase
- Redirects to `/results/:analysisId` with saved ID

---

## 🛠️ Setup Instructions

### Step 1: Install Frontend Dependencies

```bash
cd lexiguard-frontend
npm install firebase framer-motion
```

### Step 2: Install Backend Dependencies

```bash
cd lexiguard-backend
pip install firebase-admin
```

### Step 3: Configure Firebase

**A. Create Firebase Project:**
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or select existing "LexiGuard"
3. Enable **Authentication** → Google Sign-In
4. Enable **Cloud Firestore** → Start in production mode

**B. Get Firebase Config:**

1. Go to Project Settings → General
2. Scroll to "Your apps" → Web app
3. Copy the config object

**C. Update Frontend `.env`:**

Create/update `lexiguard-frontend/.env`:

```env
REACT_APP_FIREBASE_API_KEY=your_api_key
REACT_APP_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=your_project_id
REACT_APP_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
REACT_APP_FIREBASE_APP_ID=your_app_id
```

### Step 4: Deploy Firestore Security Rules

**Go to Firestore Console → Rules tab:**

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // User Analyses Collection
    match /userAnalyses/{analysisId} {
      // Allow read/update/delete only if authenticated and it's their document
      allow read, update, delete: if request.auth != null 
                                   && request.auth.uid == resource.data.userID;
      
      // Allow create only if authenticated and they are setting their own userID
      allow create: if request.auth != null 
                    && request.auth.uid == request.resource.data.userID;
    }
  }
}
```

**Click "Publish" to deploy.**

### Step 5: Create Firestore Indexes (If Needed)

Firestore will prompt you to create indexes when you first query with sorting. 

**Expected Index:**
- Collection: `userAnalyses`
- Fields: `userID` (Ascending), `uploadTimestamp` (Descending)

You can also create this manually in Firestore Console → Indexes.

---

## 🔐 Firestore Configuration

### Collection: `userAnalyses`

**Document Structure:**

```javascript
{
  // Security & Identity
  "userID": "firebase_auth_uid_string",          // CRITICAL for security rules
  "analysisId": "auto_generated_by_firestore",   // Document ID
  
  // Document Info
  "documentTitle": "Rental Agreement for ...",   // User-editable
  "originalFilename": "lease_agreement.pdf",     // Original file name
  "uploadTimestamp": Timestamp,                  // Server timestamp
  "fileType": "PDF",                             // PDF, DOCX, TXT
  
  // Privacy & Content
  "piiRedacted": true,                           // Always true for stored docs
  "redactedDocumentText": "Full redacted text...", // Core privacy-safe content
  
  // Analysis Results
  "summary": "AI-generated summary...",
  "risks": [                                     // For standard analysis
    {
      "clause_text": "[REDACTED] clause...",
      "severity": "High",
      "risk_explanation": "Explanation..."
    }
  ],
  "clauses": [                                   // For detailed analysis
    {
      "clause": "Original clause text...",
      "risk_level": "High",
      "impact": "Impact description...",
      "recommendation": "Action to take...",
      "explanation": "Detailed explanation..."
    }
  ],
  "analysisType": "detailed",                    // "standard" or "detailed"
  "total_risky_clauses": 5,
  
  // Additional Features
  "suggestions": [],
  "fairness_analysis": [],
  "tags": ["lease", "rental"],                   // User-defined tags
  "starred": false,                              // User favorite flag
  "notes": ""                                    // User notes
}
```

---

## 🔒 Security Rules

### Frontend Security

**AuthContext Protection:**
- All dashboard/results routes wrapped in `<ProtectedRoute>`
- Firebase `currentUser` checked before any Firestore operation
- Automatic redirect to `/login` if not authenticated

**Firestore Service Layer:**
- Every function accepts `userId` parameter
- Double-checks ownership before read/write
- Returns user-friendly error messages

### Backend Security (Optional Enhancement)

**For future backend integration:**

```python
from firebase_admin import auth, credentials, initialize_app

# Initialize Firebase Admin
cred = credentials.Certificate("path/to/serviceAccountKey.json")
initialize_app(cred)

# Middleware to verify Firebase ID Token
async def verify_firebase_token(authorization: str):
    try:
        token = authorization.split("Bearer ")[-1]
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
```

---

## 📖 Usage Guide

### For End Users

**1. Analyze a Document:**
   - Go to `/upload`
   - Choose Standard or Detailed analysis
   - Upload file or paste text
   - Click "Analyze Document"
   - Results auto-save to dashboard if logged in

**2. View Dashboard:**
   - Navigate to `/dashboard`
   - See all your past analyses
   - View stats and privacy notice

**3. Manage Analyses:**
   - **Rename**: Click edit icon next to title
   - **Star**: Click star icon to favorite
   - **Search**: Type in search bar
   - **Filter**: Use dropdowns for type/starred
   - **Sort**: By date, title, or risk level
   - **View**: Click "View Analysis" button
   - **Delete**: Click trash icon, confirm deletion

**4. Revisit Results:**
   - Click "View Analysis" on any dashboard card
   - Opens full results page at `/results/:analysisId`
   - All features available (chat, negotiation, email)

### For Developers

**Firestore Service API:**

```javascript
import { 
  saveAnalysis,           // Save new analysis
  getUserAnalyses,        // Get all user analyses
  getAnalysisById,        // Get single analysis
  updateAnalysisTitle,    // Rename analysis
  toggleStarredAnalysis,  // Star/unstar
  deleteAnalysis,         // Delete analysis
  getAnalysisStats        // Get dashboard stats
} from '../services/firestoreService';

// Example: Save analysis after upload
const analysisId = await saveAnalysis(currentUser.uid, analysisData);

// Example: Load all analyses
const analyses = await getUserAnalyses(currentUser.uid);

// Example: Get single analysis for Results page
const analysis = await getAnalysisById(analysisId, currentUser.uid);
```

---

## 🔌 API Reference

### Frontend Routes

```
GET  /dashboard              → Dashboard home page
GET  /results/:analysisId    → Results page (from Firestore)
POST /results                → Results page (from fresh analysis)
GET  /upload                 → Upload & analyze page
```

### Firestore Service Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `saveAnalysis` | `userId`, `analysisData` | `analysisId` | Save new analysis to Firestore |
| `getUserAnalyses` | `userId` | `Array<Analysis>` | Get all user's analyses |
| `getAnalysisById` | `analysisId`, `userId` | `Analysis` | Get single analysis |
| `updateAnalysisTitle` | `analysisId`, `userId`, `newTitle` | `void` | Rename analysis |
| `toggleStarredAnalysis` | `analysisId`, `userId`, `starred` | `void` | Toggle star |
| `updateAnalysisTags` | `analysisId`, `userId`, `tags` | `void` | Update tags |
| `updateAnalysisNotes` | `analysisId`, `userId`, `notes` | `void` | Update notes |
| `deleteAnalysis` | `analysisId`, `userId` | `void` | Delete analysis |
| `getAnalysisStats` | `userId` | `StatsObject` | Get dashboard stats |

---

## 🐛 Troubleshooting

### Issue: "Firebase not initialized" error

**Solution:**
1. Check `.env` file has all Firebase config values
2. Restart development server: `npm start`
3. Clear browser cache

### Issue: "Permission denied" in Firestore

**Solution:**
1. Verify Firestore Security Rules are deployed
2. Check user is authenticated: `console.log(currentUser)`
3. Ensure `userID` field matches `currentUser.uid`

### Issue: Analyses not appearing in Dashboard

**Solution:**
1. Open browser console, check for errors
2. Verify Firestore data: Firebase Console → Firestore
3. Check if `uploadTimestamp` field exists (required for sorting)
4. Create composite index if prompted by Firestore

### Issue: Can't load analysis from URL

**Solution:**
1. Check URL format: `/results/:analysisId` (not `/results?id=...`)
2. Verify `analysisId` exists in Firestore
3. Ensure user owns the analysis (security rule)

### Issue: Slow Dashboard Loading

**Solution:**
1. Check Firestore indexes are created
2. Limit number of analyses loaded (add pagination)
3. Use Firestore caching: `enableIndexedDbPersistence(db)`

---

## 🎨 Design Consistency

All new components follow the existing LexiGuard theme:

**Colors:**
- Primary: `#064E3B` (dark teal)
- Secondary: `#0FC6B2` (cyan)
- Accent: `#0F2A40` (dark blue)
- Background: Gradient from black → `#0F2A40` → `#064E3B`

**Components:**
- All use existing `Card`, `Badge`, `Button` from `ui/` folder
- Framer Motion for smooth animations
- Consistent spacing and typography

---

## 📊 Performance Considerations

**Frontend:**
- Use React.memo() for analysis cards
- Implement virtual scrolling for 100+ analyses
- Lazy load Firestore queries (offset/limit)

**Firestore:**
- Composite indexes for filtered queries
- Limit query size (e.g., 50 analyses per load)
- Use Firestore local cache

**Future Enhancements:**
- Server-side pagination
- Real-time listeners for multi-device sync
- Firestore data migration scripts

---

## 🚀 Deployment Checklist

- [ ] Firebase project created
- [ ] Firestore enabled (production mode)
- [ ] Security rules deployed
- [ ] Composite index created
- [ ] Frontend `.env` configured
- [ ] `npm install` dependencies
- [ ] Backend `firebase-admin` installed (if using)
- [ ] Test authentication flow
- [ ] Test save/load analysis
- [ ] Test dashboard filters/search
- [ ] Test delete with confirmation
- [ ] Verify PII redaction badge displays

---

## 📝 Future Enhancements

**Short-term:**
- [ ] Bulk delete analyses
- [ ] Export dashboard to CSV
- [ ] Tags autocomplete
- [ ] Analysis comparison view

**Long-term:**
- [ ] Real-time collaboration
- [ ] AI-powered insights (trends across analyses)
- [ ] Mobile app (React Native + Firestore)
- [ ] API for third-party integrations

---

## 📞 Support

**Questions?**
- Check existing issues in GitHub
- Review Firebase documentation: https://firebase.google.com/docs
- Firestore queries: https://firebase.google.com/docs/firestore/query-data/queries

**Contributing:**
- Follow existing code style
- Test thoroughly before PR
- Update this guide if adding features

---

## ✅ Success Metrics

Your dashboard is working correctly if:

1. ✅ User can see all their past analyses
2. ✅ Privacy badge shows "PII Redacted"
3. ✅ Search/filter/sort work smoothly
4. ✅ Can rename and delete analyses
5. ✅ Clicking "View Analysis" loads full results
6. ✅ Stats cards show accurate counts
7. ✅ No console errors in browser
8. ✅ Firestore security rules block unauthorized access

---

**Built with ❤️ for LexiGuard by the Development Team**

*Last Updated: October 2025*
