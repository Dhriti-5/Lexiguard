# LexiGuard Backend API 🛡️

This repository contains the backend service for **LexiGuard**, an AI-powered legal document analysis tool. Our mission is to democratize legal understanding by transforming dense, jargon-filled contracts into simple, actionable insights for everyone.

---

## 🎯 The Problem

Legal documents—from rental agreements to terms of service—are often incomprehensible to the average person. This information gap creates significant risks, where individuals might agree to unfavorable terms without realizing it.

---

## 💡 Our Solution

The LexiGuard backend is a cloud-native API that serves as the "brain" of our application. It receives legal text, uses Google's powerful Gemini AI model to analyze it, and returns two key pieces of information:

1. **Plain-English Summary** of the document's purpose, terms, and obligations.  
2. **Risk Analysis** that flags potentially unfavorable or non-standard clauses, explained in simple terms.

---

## 🚀 Live API Endpoint

Our production API is deployed on **Google Cloud Run** and is available for the frontend to consume.

**Base URL:**  
```
[PASTE-YOUR-DEPLOYED-CLOUD-RUN-URL-HERE]
```

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.9+  
- **Framework:** FastAPI + Uvicorn  
- **AI Model:** Google Gemini 1.5 Flash (via Google AI Studio API)  
- **Containerization:** Docker  
- **Hosting:** Google Cloud Run + Artifact Registry  
- **Security:** API keys via environment variables (`.env`), never hardcoded  

---

## 📖 API Documentation

### `POST /analyze`

**Request Body (JSON):**
```json
{
  "text": "The full text of the legal document to be analyzed goes here..."
}
```

**Response (200 OK):**
```json
{
  "summary": "Plain-English summary of the document.",
  "risks": [
    {
      "clause_text": "The risky clause text.",
      "risk_explanation": "Explanation of the risk.",
      "severity": "High"
    },
    {
      "clause_text": "Another clause.",
      "risk_explanation": "Explanation of another risk.",
      "severity": "Medium"
    }
  ]
}
```

---

## 🏁 Getting Started (Local Setup)

### ✅ Prerequisites
- [Python 3.8+](https://www.python.org/downloads/)  
- [Git](https://git-scm.com/downloads/)  
- [VS Code](https://code.visualstudio.com/) (recommended)

### 📝 Setup Instructions

**1. Clone the repo**
```bash
git clone https://github.com/[Your-GitHub-Username]/lexiguard-backend.git
cd lexiguard-backend
```

**2. Create a virtual environment**
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your credentials**
- Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)  
- Create a `.env` file in the project root:  
  ```env
  GOOGLE_API_KEY="your-secret-api-key"
  ```

**5. Run the server**
```bash
uvicorn main:app --reload
```

Server will be available at:  
```
http://127.0.0.1:8000
```

**6. Test with cURL**
```bash
curl -X POST "http://127.0.0.1:8000/analyze" -H "Content-Type: application/json" -d '{"text": "The tenant agrees to a 12-month lease with auto-renewal unless terminated 90 days before expiry."}'
```

---

## ⚡ Quickstart for Hackathon Judges

To quickly test the **live deployed API**:

```bash
curl -X POST "[PASTE-YOUR-DEPLOYED-CLOUD-RUN-URL-HERE]/analyze" -H "Content-Type: application/json" -d '{"text": "This Agreement shall commence on the Effective Date and shall continue for a period of one (1) year. Thereafter, this Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term."}'
```

**Expected Output:**
```json
{
  "summary": "This is an agreement that lasts for one year and will automatically renew unless canceled 90 days before expiration.",
  "risks": [
    {
      "clause_text": "Thereafter, this Agreement shall automatically renew...",
      "risk_explanation": "This contract auto-renews yearly unless notice is given well in advance.",
      "severity": "High"
    }
  ]
}
```

---
## 🚀 Deployment

LexiGuard backend is deployed on [Google Cloud Run](https://lexiguard-backend-798526808813.asia-south1.run.app), ensuring scalable and secure serverless hosting.

Frontend is hosted on [Vercel](https://lexiguard-frontend-66oa228c3-krisha-gandhis-projects.vercel.app/) for fast and responsive access.

## 📌 Project Info

- **Frontend Repository:** [LexiGuard Frontend Repo](https://github.com/[Your-Teammate-Username]/lexiguard-frontend)  
- **Team Members:** [Your Name], [Teammate’s Name]  

---
