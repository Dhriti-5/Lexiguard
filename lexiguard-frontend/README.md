# ⚖️ LexiGuard Frontend

The **LexiGuard Frontend** is a modern, intuitive web application that empowers users to understand complex legal documents through AI-powered analysis. Built with React and TailwindCSS, it provides a seamless experience for document upload, analysis, and management.

---

## 🌟 Features

### Core Functionality
- **📄 Document Analysis** - Upload PDF/DOCX files or paste text for instant AI analysis
- **📊 Two Analysis Modes**:
  - **Standard**: Quick overview with risk flags and summary
  - **Detailed**: Comprehensive clause-by-clause breakdown with recommendations
- **🎯 Risk Detection** - Automatic flagging of unfair clauses with severity levels (High/Medium/Low)
- **⚖️ Fairness Score** - Benchmark contracts against fair-practice standards
- **💬 Chat with Document** - Ask questions and get context-aware answers
- **✉️ Negotiation Assistant** - Generate professional emails to contest risky clauses
- **📧 PDF Email Reports** - Send comprehensive analysis reports via email

### Dashboard & Management
- **📊 Analytics Dashboard** - Track all your analyses in one place
- **🔍 Advanced Search & Filter** - Find analyses by title, type, or risk level
- **⭐ Favorites System** - Star important analyses for quick access
- **✏️ Inline Editing** - Rename analyses directly from the dashboard
- **🔐 Privacy-First Design** - Only redacted content stored (PII removed)
- **📱 Responsive Design** - Works seamlessly on desktop, tablet, and mobile

### Security & Privacy
- **🔒 Firebase Authentication** - Secure Google Sign-In
- **🛡️ PII Redaction** - Automatic removal of sensitive information (DLP API)
- **👤 User Isolation** - Each user can only access their own data
- **🔐 Firestore Security Rules** - Server-side access control

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Framework** | React 18+ with Hooks |
| **Routing** | React Router DOM v6 |
| **Styling** | TailwindCSS, shadcn/ui components |
| **Animations** | Framer Motion |
| **Authentication** | Firebase Auth (Google Sign-In) |
| **Database** | Cloud Firestore |
| **Icons** | Lucide React |
| **Content Rendering** | React Markdown |
| **Backend Integration** | FastAPI + Google Cloud Run |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ (check with `node -v`)
- **npm** or **yarn** (check with `npm -v`)
- **Google Cloud Account** (for Firebase & API keys)
- **Backend Running** (see Backend README)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishagandhi0711/Lexiguard.git
   cd Lexiguard/lexiguard-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the `lexiguard-frontend` root directory:
   
   ```env
   # Backend API URL
   REACT_APP_BACKEND_URL=http://localhost:8000
   
   # Firebase Configuration
   REACT_APP_FIREBASE_API_KEY=your_firebase_api_key
   REACT_APP_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
   REACT_APP_FIREBASE_PROJECT_ID=your_project_id
   REACT_APP_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
   REACT_APP_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
   REACT_APP_FIREBASE_APP_ID=your_app_id
   ```

4. **Start the development server**
   ```bash
   npm start
   ```

5. **Access the application**
   
   Open your browser and navigate to: **http://localhost:3000**

---

## 🔥 Firebase Setup

### Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" or select existing project
3. Enter project name (e.g., "LexiGuard")
4. Follow the setup wizard

### Step 2: Enable Authentication

1. In Firebase Console, go to **Build → Authentication**
2. Click on **Sign-in method** tab
3. Enable **Google** provider:
   - Toggle "Enable" switch
   - Add support email (your email)
   - Click "Save"
4. Add authorized domains:
   - `localhost` (default)
   - Your production domain (when deploying)

### Step 3: Enable Firestore Database

1. In Firebase Console, go to **Build → Firestore Database**
2. Click "Create database"
3. Choose **Production mode**
4. Select location closest to your users

### Step 4: Deploy Security Rules

Go to **Firestore → Rules** tab and paste:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // User Analyses Collection
    match /userAnalyses/{analysisId} {
      // Users can only access their own documents
      allow read, update, delete: if request.auth != null 
                                   && request.auth.uid == resource.data.userID;
      
      // Users can only create documents with their own userID
      allow create: if request.auth != null 
                    && request.auth.uid == request.resource.data.userID;
    }
  }
}
```

Click **Publish** to deploy.

### Step 5: Create Firestore Index

Firestore will prompt you to create an index when you first query. Or create manually:

1. Go to **Firestore → Indexes**
2. Click "Create Index"
3. Configure:
   - **Collection ID**: `userAnalyses`
   - **Field 1**: `userID` (Ascending)
   - **Field 2**: `uploadTimestamp` (Descending)
4. Click "Create Index"

### Step 6: Get Firebase Configuration

1. Go to **Project Settings** (gear icon)
2. Scroll to **Your apps** → Web app
3. Copy the configuration object
4. Paste values into your `.env` file

---

## 📂 Project Structure

```
lexiguard-frontend/
├── public/
│   ├── favicon.ico
│   ├── index.html
│   └── manifest.json
│
├── src/
│   ├── components/              # Reusable UI components
│   │   ├── ui/                 # shadcn/ui components
│   │   ├── home/               # Homepage sections
│   │   ├── BackToTop.jsx
│   │   ├── FairnessScore.js
│   │   ├── LanguageSelector.jsx
│   │   ├── Layout.jsx
│   │   ├── ProtectedRoute.jsx
│   │   └── ...
│   │
│   ├── contexts/
│   │   └── AuthContext.jsx     # Firebase authentication context
│   │
│   ├── firebase/
│   │   └── config.js           # Firebase configuration
│   │
│   ├── pages/                  # Main application pages
│   │   ├── Home.jsx
│   │   ├── Login.jsx
│   │   ├── Upload.jsx
│   │   ├── Results.jsx
│   │   ├── Dashboard.jsx
│   │   └── ...
│   │
│   ├── services/
│   │   └── firestoreService.js # Firestore CRUD operations
│   │
│   ├── App.js                  # Main app component
│   └── index.js                # Entry point
│
├── .env                        # Environment variables (create this!)
├── .env.example                # Environment template
├── package.json
├── tailwind.config.js
└── README.md                   # This file
```

---

## 🎯 Usage Guide

### For End Users

#### 1. Sign In
- Navigate to `/login`
- Click "Sign in with Google"
- Authorize LexiGuard to access your Google account

#### 2. Analyze a Document
- Go to `/upload`
- Choose analysis type (Standard or Detailed)
- Upload file (PDF/DOCX) or paste text
- Click "Analyze Document"
- Wait for processing
- Results auto-save to your dashboard

#### 3. View Results
- Review document summary
- Check fairness score
- Examine flagged risky clauses
- Use additional features:
  - **Chat**: Ask questions about the document
  - **Negotiate**: Generate emails for risky clauses
  - **Email Report**: Send PDF summary to yourself

#### 4. Manage Your Analyses (Dashboard)
- Navigate to `/dashboard`
- View all past analyses
- **Search**: Type keywords in search bar
- **Filter**: By type (Standard/Detailed/Starred)
- **Sort**: By date, title, or risk level
- **Rename**: Click edit icon next to title
- **Star**: Click star icon to favorite
- **View**: Click "View Analysis" button
- **Delete**: Click trash icon (with confirmation)

---

## 🔌 API Integration

### Backend Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze-file` | POST | Upload & analyze document |
| `/analyze-clauses` | POST | Detailed clause analysis |
| `/chat` | POST | Chat with document |
| `/draft-negotiation` | POST | Generate negotiation email |
| `/send-document-review` | POST | Send PDF report via email |

### Example API Call

```javascript
const analyzeDocument = async (file, analysisType) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('analysis_type', analysisType);
  
  const response = await fetch(`${BACKEND_URL}/analyze-file`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

---

## 🐛 Troubleshooting

### Issue: "Firebase not initialized" Error

**Solutions:**
1. Verify all Firebase config values in `.env` file
2. Restart development server: `npm start`
3. Clear browser cache (Ctrl+Shift+Delete)
4. Check browser console for specific Firebase errors

### Issue: "Permission Denied" in Firestore

**Solutions:**
1. Verify Firestore Security Rules are deployed in Firebase Console
2. Check user is authenticated: `console.log(currentUser)`
3. Ensure `userID` field in documents matches `currentUser.uid`

### Issue: Analyses Not Appearing in Dashboard

**Solutions:**
1. Open browser DevTools console (F12)
2. Check for JavaScript errors
3. Verify Firestore data exists: Firebase Console → Firestore
4. Ensure documents have `uploadTimestamp` field
5. Create/verify Firestore composite index

### Issue: Google Sign-In Not Working

**Solutions:**
1. Verify Google provider is enabled in Firebase Console
2. Check authorized domains include `localhost`
3. Ensure Firebase API key is correct in `.env`
4. Try clearing cookies and cache

### Issue: Backend Connection Failed

**Solutions:**
1. Verify backend is running: `http://localhost:8000/docs`
2. Check `REACT_APP_BACKEND_URL` in `.env` file
3. Ensure CORS is enabled on backend
4. Check network tab in browser DevTools

---

## 🚀 Deployment to Google Cloud Run

### Step 1: Create Dockerfile

Create `Dockerfile` in `lexiguard-frontend/`:

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=build /app/build ./build
EXPOSE 3000
CMD ["serve", "-s", "build", "-l", "3000"]
```

### Step 2: Deploy to Cloud Run

```bash
# Deploy from source
gcloud run deploy lexiguard-frontend \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated
```

### Step 3: Update Firebase Configuration

1. Add your Cloud Run URL to Firebase Console → Authentication → Settings → Authorized domains
2. Update `.env` with production backend URL:
   ```env
   REACT_APP_BACKEND_URL=https://your-backend.run.app
   ```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Manual Testing Checklist

- [ ] Sign in with Google
- [ ] Upload document (PDF/DOCX/Text)
- [ ] View analysis results
- [ ] Chat with document
- [ ] Generate negotiation email
- [ ] Send email report
- [ ] View dashboard
- [ ] Search/filter analyses
- [ ] Rename analysis
- [ ] Star/unstar analysis
- [ ] Delete analysis
- [ ] Sign out

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🆘 Support

### Getting Help
1. Check **Troubleshooting** section above
2. Review [Firebase Documentation](https://firebase.google.com/docs)
3. Check browser console (F12) for errors
4. Review backend logs if API calls fail
5. Open an issue on GitHub

---

## 📞 Contact

- **Developers**: Dhriti Gandhi, Krisha Gandhi, Kavya Patel
- **Project**: LexiGuard - AI-Powered Legal Document Assistant
- **Repository**: https://github.com/krishagandhi0711/Lexiguard

---

## ✅ Quick Start Checklist

- [ ] Node.js 18+ installed
- [ ] Firebase project created
- [ ] Google Sign-In enabled in Firebase
- [ ] Firestore database created
- [ ] Security rules deployed
- [ ] Composite index created
- [ ] `.env` file configured
- [ ] Dependencies installed (`npm install`)
- [ ] Backend running (see Backend README)
- [ ] Frontend started (`npm start`)
- [ ] Test login works
- [ ] Test document upload works
- [ ] Test dashboard access works

---

**Built with ❤️ for LexiGuard**

*Last Updated: October 2025*
