# 🧹 Lexiguard Cleanup Plan

## 📊 Current Project Analysis

### Active Components (DO NOT DELETE)
1. **lexiguard-backend/main.py** - ✅ PRIMARY FastAPI backend (CURRENTLY RUNNING)
2. **lexiguard-frontend/** - ✅ React frontend (ACTIVE)
3. **lexiguard-backend/server.js** - ✅ Node.js upload server (port 5000)
4. **start-all.ps1** - ✅ Startup script

### Alternative Implementations (KEEP - User Choice)
1. **lexiguard-backend/app.py** - Flask alternative (simpler, no DLP)
2. **fastapi_app/main.py** - Standalone FastAPI example
3. **flask_app/app.py** - Standalone Flask example

### Utility Components (USEFUL - KEEP)
1. **lexiguard_sdk/** - Python SDK for Lexiguard API
2. **sample_contract.txt** - Test document
3. **test_api.py**, **test_basic.py**, **test_upload.py** - API testing scripts
4. **Beginner_Tutorial.md** - Documentation
5. **setup.py** - SDK installation script

---

## 🗑️ Files That Can Be Safely Removed

### Category 1: Redundant/Unused Files

#### Root Level
- ❌ **private_doc.txt** - Personal/test document (not needed for production)
- ❌ **uploads/** (root level) - Duplicate of lexiguard-backend/uploads/
- ❌ **.env.example.py** - Seems to be test code, not a proper .env template

#### Backend
- ❌ **lexiguard-backend/dlp_service.py** - If not imported anywhere
- ❌ **lexiguard-backend/__pycache__/** - Python cache (regenerates automatically)
- ❌ **lexiguard-backend/Procfile** - Heroku deployment (not using?)
- ❌ **lexiguard-backend/runtime.txt** - Heroku Python version (not using?)
- ❌ **lexiguard-backend/vercel.json** - Vercel deployment (not using?)
- ❌ **lexiguard-backend/Dockerfile** - Docker config (not using?)
- ❌ **lexiguard-backend/README.md** - Redundant (have root README)

#### SDK
- ❌ **lexiguard_sdk/__pycache__/** - Python cache

---

## 📋 Recommended Actions

### Option 1: Minimal Cleanup (Safest)
**Remove only cache and obviously unused files**

```powershell
# Remove Python cache
Remove-Item -Recurse -Force "lexiguard-backend\__pycache__"
Remove-Item -Recurse -Force "lexiguard_sdk\__pycache__"

# Remove test/personal files
Remove-Item "private_doc.txt"
Remove-Item ".env.example.py"

# Remove root uploads folder (use backend/uploads instead)
Remove-Item -Recurse "uploads"
```

**Impact**: Zero - removes only cache and test files

---

### Option 2: Moderate Cleanup (Recommended)
**Remove deployment configs you're not using + cache**

```powershell
# All from Option 1 plus:

# Remove deployment configs (if not deploying)
Remove-Item "lexiguard-backend\Procfile"
Remove-Item "lexiguard-backend\runtime.txt"
Remove-Item "lexiguard-backend\vercel.json"
Remove-Item "lexiguard-backend\Dockerfile"

# Remove redundant README
Remove-Item "lexiguard-backend\README.md"
```

**Impact**: Low - removes deployment files you're not using locally

---

### Option 3: Aggressive Cleanup (Use with Caution)
**Remove alternative implementations if you're only using main.py**

```powershell
# All from Option 2 plus:

# If ONLY using lexiguard-backend/main.py:
Remove-Item -Recurse "fastapi_app"
Remove-Item -Recurse "flask_app"
Remove-Item "lexiguard-backend\app.py"

# If not using SDK:
Remove-Item -Recurse "lexiguard_sdk"
Remove-Item "setup.py"

# If not testing APIs:
Remove-Item "test_api.py"
Remove-Item "test_basic.py"
Remove-Item "test_upload.py"
```

**Impact**: Medium - removes alternative implementations (can't switch between Flask/FastAPI)

---

## 📁 Final Recommended Structure

After **Option 2 (Recommended)** cleanup:

```
Lexiguard/
├── 📄 Readme.md                         # Main documentation
├── 📄 GETTING-STARTED.md                # Setup guide
├── 📄 FIX-SUMMARY.md                    # Bug fix documentation
├── 📄 Beginner_Tutorial.md              # Tutorial
├── 📄 sample_contract.txt               # Test document
├── 📄 start-all.ps1                     # Startup script
├── 📄 .gitignore
│
├── 📁 lexiguard-backend/                # Main FastAPI backend
│   ├── 📄 main.py                      # ✅ PRIMARY - FastAPI (with DLP)
│   ├── 📄 app.py                       # Alternative Flask backend
│   ├── 📄 server.js                    # Node.js upload server
│   ├── 📄 start-backend.ps1            # Backend launcher
│   ├── 📄 requirements.txt             # Python dependencies
│   ├── 📄 package.json                 # Node dependencies
│   ├── 📄 .env                         # Environment variables
│   ├── 📄 .gitignore
│   ├── 📁 .venv/                       # Virtual environment
│   ├── 📁 uploads/                     # Uploaded files
│   └── 📁 node_modules/                # Node packages
│
├── 📁 lexiguard-frontend/               # React frontend
│   ├── 📄 package.json
│   ├── 📁 src/
│   ├── 📁 public/
│   └── 📁 node_modules/
│
├── 📁 fastapi_app/                      # Standalone FastAPI example
│   └── 📄 main.py
│
├── 📁 flask_app/                        # Standalone Flask example
│   └── 📄 app.py
│
├── 📁 lexiguard_sdk/                    # Python SDK
│   ├── 📄 __init__.py
│   ├── 📄 core.py
│   └── 📄 file_utils.py
│
├── 📄 setup.py                          # SDK installer
├── 📄 test_api.py                       # API tests
├── 📄 test_basic.py                     # Basic tests
└── 📄 test_upload.py                    # Upload tests
```

---

## 🎯 What I Recommend

**Go with Option 2** - It's the sweet spot:
- ✅ Removes unused deployment configs
- ✅ Keeps alternative backends (Flask/FastAPI) in case you need them
- ✅ Keeps SDK and tests (useful for development)
- ✅ Removes cache and test files
- ✅ Zero risk to current functionality

---

## ⚠️ Files to NEVER Delete

1. **lexiguard-backend/main.py** - Your current running backend
2. **lexiguard-backend/.venv/** - Virtual environment with all packages
3. **lexiguard-backend/.env** - Your API keys!
4. **lexiguard-backend/requirements.txt** - Dependency list
5. **lexiguard-frontend/** - Entire frontend folder
6. **start-all.ps1** - Your startup script

---

## 🤔 Questions Before Cleanup

1. **Are you deploying to Heroku/Vercel?** 
   - If NO: Delete Procfile, runtime.txt, vercel.json, Dockerfile
   - If YES: Keep them

2. **Do you need Flask alternative?**
   - If NO: Can delete `lexiguard-backend/app.py`, `flask_app/`
   - If YES: Keep them

3. **Do you use the SDK?**
   - If NO: Can delete `lexiguard_sdk/`, `setup.py`
   - If YES: Keep them

4. **Do you run API tests?**
   - If NO: Can delete `test_*.py` files
   - If YES: Keep them

---

## 📝 Next Steps

1. Review the options above
2. Tell me which option (1, 2, or 3) you want
3. Or tell me specific preferences (e.g., "keep SDK, remove Flask alternatives")
4. I'll create the exact cleanup commands for you

**My recommendation**: Start with **Option 2** - safe, clean, keeps flexibility.
