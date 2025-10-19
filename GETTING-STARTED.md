# 🔒 Lexiguard - AI-Powered Legal Document Analyzer

An intelligent platform for analyzing legal documents, identifying unfair clauses, and providing contract negotiation assistance.

## 🚀 Quick Start (Windows)

### Option 1: Start Everything at Once (Easiest! ✨)
```powershell
# From the Lexiguard root directory
.\start-all.ps1
```

This automatically starts all services in separate terminal windows:
- ✅ Backend API (FastAPI) on http://localhost:8000
- ✅ Node.js server (Express) on http://localhost:5000  
- ✅ Frontend (React) on http://localhost:3000

### Option 2: Start Services Individually

**Backend (FastAPI)**
```powershell
cd lexiguard-backend
.\start-backend.ps1
```

**Frontend (React)**
```powershell
cd lexiguard-frontend
.\start-frontend.ps1
```

**Node Server (Express - Optional)**
```powershell
cd lexiguard-backend
node server.js
```

---

## ⚙️ First-Time Setup

### Prerequisites
- ✅ Python 3.11+ installed
- ✅ Node.js 14+ installed
- ✅ Google API Key ([Get it here](https://makersuite.google.com/app/apikey))

### 1️⃣ Backend Setup

```powershell
cd lexiguard-backend

# Create virtual environment (isolates Python packages)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install ALL dependencies in the venv (NOT globally)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Create .env file
@"
GOOGLE_API_KEY=your_actual_api_key_here
"@ | Out-File -FilePath .env -Encoding utf8
```

### 2️⃣ Frontend Setup

```powershell
cd lexiguard-frontend

# Install dependencies
npm install
```

### 3️⃣ Node Server Setup (Optional)

```powershell
cd lexiguard-backend

# Install Node dependencies
npm install
```

---

## 🔑 Environment Variables

**Required:** Create `.env` file in `lexiguard-backend/`:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

🔗 Get your API key: https://makersuite.google.com/app/apikey

---

## 📦 Installed Dependencies

### Python Backend (in `.venv` - isolated!)
```
fastapi              # Web framework
uvicorn[standard]    # ASGI server
google-generativeai  # Gemini AI (v0.8.5)
google-cloud-dlp     # PII redaction
pydantic            # Data validation
python-dotenv       # Environment variables
PyPDF2              # PDF processing
python-multipart    # File uploads
python-docx         # DOCX processing
```

### Node.js
```
express             # Web server
multer              # File uploads
cors                # Cross-origin
```

### React Frontend
```
react               # UI framework
tailwind-css        # Styling
react-router-dom    # Routing
```

---

## 🌐 Access Points

| Service | URL | Description |
|---------|-----|-------------|
| 🎨 **Frontend** | http://localhost:3000 | Main web interface |
| ⚡ **Backend API** | http://localhost:8000 | FastAPI endpoints |
| 📚 **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| 🗂️ **Node Server** | http://localhost:5000 | File upload handling |

---

## 🛠️ API Endpoints

### Document Analysis
```
POST /analyze-file          Upload & analyze document
POST /analyze-clauses       Detailed clause breakdown
POST /chat                  Chat with document
POST /draft-negotiation     Generate negotiation points
POST /draft-document-email  Draft email summary
```

---

## 🧪 Testing the Application

1. **Start all services:**
   ```powershell
   .\start-all.ps1
   ```

2. **Open browser:** Navigate to http://localhost:3000

3. **Upload document:** Use the sample file `sample_contract.txt` or any PDF/DOCX

4. **View analysis:** Check fairness score, unfair clauses, and recommendations

5. **Try features:**
   - 💬 Chat with document
   - 📝 Generate negotiation points
   - ✉️ Draft email summaries

---

## 📁 Project Structure

```
Lexiguard/
├── 📜 start-all.ps1                 # Start all services
├── 📜 Readme.md                     # This file
│
├── 📁 lexiguard-backend/            # Python FastAPI backend
│   ├── 🐍 .venv/                   # Virtual environment (ISOLATED!)
│   ├── 📄 main.py                  # FastAPI app (with DLP)
│   ├── 📄 app.py                   # Flask alternative (simpler)
│   ├── 📄 server.js                # Node.js upload server
│   ├── 📄 requirements.txt         # Python dependencies
│   ├── 🔒 .env                     # API keys (create this!)
│   ├── 📜 start-backend.ps1        # Backend startup script
│   └── 📁 uploads/                 # Uploaded files
│
└── 📁 lexiguard-frontend/           # React frontend
    ├── 📁 src/
    │   ├── 📁 components/          # UI components
    │   ├── 📁 pages/               # Page components
    │   ├── 📁 integrations/        # API calls
    │   └── 📄 App.js               # Main app
    ├── 📄 package.json
    └── 📜 start-frontend.ps1        # Frontend startup script
```

---

## 🔧 Troubleshooting

### ❌ "Module not found" Errors

**Solution:** Ensure packages are installed in `.venv`, not globally!

```powershell
cd lexiguard-backend

# Check what's in your venv
.\.venv\Scripts\python.exe -m pip list

# Reinstall if needed
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### ❌ Port Already in Use

```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process
taskkill /PID <process_id> /F
```

### ❌ Google API Errors

**Check these:**
1. ✅ API key is correct in `.env`
2. ✅ API key has Gemini API enabled
3. ✅ Billing enabled on Google Cloud project
4. ✅ Internet connection working

### ❌ Gemini Model Not Found

The backend tries these models in order:
1. `gemini-pro`
2. `gemini-1.0-pro`
3. `gemini-1.5-pro-latest`
4. `gemini-1.5-flash-latest`

**Check logs:** Look for `✅ Successfully initialized model` in backend terminal

### ❌ Virtual Environment Not Activating

```powershell
# May need to enable scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
cd lexiguard-backend
.\.venv\Scripts\Activate.ps1
```

---

## 🎯 Development Workflow

### Making Changes to Backend

```powershell
cd lexiguard-backend

# Always use venv Python!
.\.venv\Scripts\python.exe main.py

# Adding new packages
.\.venv\Scripts\python.exe -m pip install <package>
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

### Making Changes to Frontend

```powershell
cd lexiguard-frontend

npm start  # Auto-reloads on changes
```

---

## 📊 Architecture

```
┌─────────────┐
│   Browser   │ 
│ (Port 3000) │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────┐      ┌──────────────┐
│  React Frontend │      │ Node Server  │
│   (Tailwind)    │◄────►│  (Port 5000) │
└────────┬────────┘      └──────────────┘
         │                      │
         │ HTTP/JSON            │ File Uploads
         ▼                      ▼
┌──────────────────────────────────┐
│     FastAPI Backend (Port 8000)  │
│  ┌──────────┐  ┌──────────────┐  │
│  │  Gemini  │  │  Google DLP  │  │
│  │    AI    │  │ (PII Redact) │  │
│  └──────────┘  └──────────────┘  │
└──────────────────────────────────┘
```

---

## 🔒 Security Notes

- ✅ All Python packages isolated in `.venv`
- ✅ API keys stored in `.env` (never commit!)
- ✅ PII redaction via Google DLP
- ✅ CORS configured for localhost
- ✅ File upload validation

---

## 💡 Tips

1. **Always use the startup scripts** - They ensure correct environment activation
2. **Check terminal output** - Detailed logs show what's happening
3. **Use API docs** - Visit http://localhost:8000/docs to test endpoints
4. **Keep .venv local** - Never commit virtual environment to git
5. **Update dependencies in venv** - Use `.\.venv\Scripts\python.exe -m pip install`

---

## 🤝 Contributing

1. Create a feature branch
2. Make changes (test with `.\start-all.ps1`)
3. Update `requirements.txt` if adding Python packages
4. Test all features before committing
5. Never commit `.env` or `.venv/`

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🆘 Need Help?

1. Check **Troubleshooting** section above
2. Verify `.venv` has all packages: `.\.venv\Scripts\python.exe -m pip list`
3. Check browser console (F12) for frontend errors
4. Check terminal output for backend errors
5. Ensure API key is valid at https://console.cloud.google.com

---

## 🎉 You're Ready!

Run this command and start analyzing legal documents:

```powershell
.\start-all.ps1
```

Then open http://localhost:3000 in your browser! 🚀
