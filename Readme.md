## Local development (Windows)

This repo has a React frontend and two backend options:

- Python FastAPI service in `lexiguard-backend/main.py` (AI features; default port 8000)
- Simple Node/Express server in `lexiguard-backend/server.js` for listing static uploads (port 5000)

You can run either or both depending on which pages you use.

### Prerequisites

- Node.js 18+
- Python 3.10+

### 1) Run the AI backend (FastAPI)

In PowerShell:

```
cd lexiguard-backend
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
$env:GOOGLE_API_KEY = "YOUR_API_KEY"
python main.py
```

This starts FastAPI on http://localhost:8000

### 2) Run the uploads server (Node, optional)

For Dashboard uploads listing:

```
cd lexiguard-backend
npm install
npm start
```

This serves http://localhost:5000 and static files from `lexiguard-backend/uploads`.

### 3) Run the frontend (React)

Open a new PowerShell tab:

```
cd lexiguard-frontend
$env:REACT_APP_BACKEND_URL = "http://localhost:8000"
npm install
npm start
```

The app opens at http://localhost:3000

Notes:

- Pages `Upload` and `Results` call the FastAPI endpoints at port 8000.
- `Dashboard` fetches from http://localhost:5000/api/uploads (Node server).
- If you deploy the backend (Cloud Run), set `REACT_APP_BACKEND_URL` to that URL.

# 📁 LexiGuard SDK - Complete Directory Structure
# ⚖️ LexiGuard – AI-Powered Legal Document Assistant  

LexiGuard is our hackathon project that aims to **democratize legal understanding**. It helps users break down dense, jargon-heavy contracts into **clear, actionable insights**, while highlighting potential risks and fairness issues.  

---

## 🌟 What LexiGuard Does  

- **Summarizes** legal documents into plain English  
- **Flags risky clauses** with severity levels  
- Provides a **Fairness Score** against standard practices  
- Includes a **Negotiation Assistant** for contesting terms  
- Enables **Chat with Document** for context-aware Q&A  
- Built with a **modern, mobile-responsive frontend** and a **cloud-native backend API**  

---

## 📂 Project Structure  

- **Frontend (React + TailwindCSS):** User-facing web app with document upload, summaries, risk analysis, chat, and interactive features.  
- **Backend (FastAPI + Gemini AI):** Cloud-hosted API that powers document analysis, summaries, risk detection, and fairness scoring.  

For **in-depth setup, features, and technical details**, please check the individual READMEs:  
- [Frontend README](./lexiguard-frontend/README.md)  
- [Backend README](./lexiguard-backend/README.md)  

---

## 🚀 Live Deployments  

- **Frontend App:** 👉 [LexiGuard (Vercel)](https://lexiguard-frontend-9eiz1vrr4-krisha-gandhis-projects.vercel.app)  
- **Backend API:** 👉 [LexiGuard API (Google Cloud Run)](https://lexiguard-backend-798526808813.asia-south1.run.app)  

---

## 👩‍💻 Team  

- Dhriti Gandhi  
- Krisha Gandhi
- Kavya Patel
