
# LexiGuard Backend API 🛡️

This repository contains the backend service for **LexiGuard**, an AI-powered legal document analysis tool.
Our mission is to **democratize legal understanding** by transforming dense, jargon-filled contracts into **simple, actionable insights** for everyone.

---

## 🎯 The Problem

Legal documents—from rental agreements to terms of service—are often incomprehensible to the average person.
This creates **information asymmetry**, where individuals unknowingly agree to unfavorable terms, exposing them to **financial and legal risks**.

---

## 💡 Our Solution

The LexiGuard backend is a **cloud-native API** that serves as the “brain” of our application. It:

* Generates **Plain-English Summaries** of document terms and obligations.
* Performs **Risk Analysis**, flagging unfavorable or non-standard clauses with explanations.
* Powers advanced features like **Fairness Score** and **Negotiation Assistant** (via Gemini AI).

---

## 🚀 Live API Endpoint

Our production API is deployed on **Google Cloud Run** and consumed by the frontend.

**Base URL:**

```
https://lexiguard-backend-798526808813.asia-south1.run.app
```

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python 3.9+
* **Framework:** FastAPI + Uvicorn
* **AI Model:** Google Gemini 1.5 Flash (via Google AI Studio API)
* **Containerization:** Docker
* **Hosting:** Google Cloud Run + Artifact Registry
* **Security:** API keys via `.env` (never hardcoded)
* **CORS:** Enabled for secure frontend-backend communication

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

* Python 3.8+
* Git
* VS Code (recommended)

### 📝 Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/krishagandhi0711/Lexiguard.git
   cd lexiguard/lexiguard-backend
   ```

2. **Create a virtual environment**

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your credentials**

   * Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   * Create a `.env` file in the project root:

     ```env
     GOOGLE_API_KEY="your-secret-api-key"
     ```

5. **Run the server**

   ```bash
   uvicorn main:app --reload
   ```

   Server available at:

   ```
   http://127.0.0.1:8000
   ```

6. **Test with cURL**

   ```bash
   curl -X POST "http://127.0.0.1:8000/analyze" \
   -H "Content-Type: application/json" \
   -d '{"text": "The tenant agrees to a 12-month lease with auto-renewal unless terminated 90 days before expiry."}'
   ```

---

## ☁️ Deployment on Google Cloud Run

1. **Build and push image**

   ```bash
   gcloud builds submit --tag gcr.io/lexiguard-hackathon-471911/lexiguard-backend
   ```
2. **Deploy to Cloud Run**

   ```bash
   gcloud run deploy lexiguard-backend \
     --image gcr.io/lexiguard-hackathon-471911/lexiguard-backend \
     --platform managed
   ```

---

## ⚡ Quickstart for Hackathon Judges

To quickly test the live deployed API:

```bash
curl -X POST "https://lexiguard-backend-798526808813.asia-south1.run.app/analyze" \
-H "Content-Type: application/json" \
-d '{"text": "This Agreement shall commence on the Effective Date and shall continue for a period of one (1) year. Thereafter, this Agreement shall automatically renew for successive one (1) year periods unless either party provides written notice of non-renewal at least ninety (90) days prior to the end of the then-current term."}'
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

## 🔐 Security & Privacy

* All documents are processed **in-memory** (no persistent storage).
* API keys are stored in `.env` files and never hardcoded.
* CORS enabled for secure frontend-backend communication.

---

## 📌 Project Info

* **Frontend Repository:** [LexiGuard Frontend Repo](https://github.com/krishagandhi0711/Lexiguard/tree/main/lexiguard-frontend)
* **Team Members:**

  * Krisha 
  * Dhriti Gandhi 
  * Kavya

---
