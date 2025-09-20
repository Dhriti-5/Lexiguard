# LexiGuard Frontend 💻

The **LexiGuard Frontend** is the user-facing interface for our AI-powered legal assistant. It provides a clean, intuitive platform where users can upload legal documents, view AI-powered insights, and take direct action with tools like the **Fairness Score** and **Negotiation Assistant**.

---

## 🎯 Features

- **Upload & Analyze Documents** – Supports PDF/DOCX uploads or plain text.  
- **Plain-English Summaries** – Converts dense legal text into simple explanations.  
- **Risk Analysis** – Flags unfavorable clauses with severity levels.  
- **Fairness Score** – Benchmarks contracts against fair-practice templates.  
- **Negotiation Assistant** – Drafts professional emails for contesting risky clauses.  
- **Chat with Document** – Ask plain-English questions, get context-aware answers.  
- **Modern UI** – Responsive design with TailwindCSS, smooth animations, and accessibility.  

---

## 🛠️ Tech Stack

- **Frontend:** React (v19.1.1), React Router DOM  
- **Styling & UI:** TailwindCSS, shadcn/ui, Framer Motion, Lucide React  
- **Content Rendering:** React Markdown  
- **Testing:** React Testing Library, Jest DOM, User Event Testing  
- **Backend Connection:** FastAPI + Uvicorn (LexiGuard Backend)  

---

## 🚀 Getting Started

### ✅ Prerequisites
- Node.js (v18+)  
- npm or yarn  
- Python 3.9+ (for backend)  

### 🔧 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/krishagandhi0711/Lexiguard.git
   cd lexiguard/lexiguard-frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure Backend Endpoint**
   - Ensure the backend (`lexiguard-backend`) is running locally with **Uvicorn**:
     ```bash
     cd ../lexiguard-backend
     uvicorn main:app --reload
     ```
     Backend will be available at:  
     ```
     http://127.0.0.1:8000
     ```

   - Create a `.env` file in the frontend root:
     ```env
     REACT_APP_BACKEND_URL=http://127.0.0.1:8000
     ```

4. **Start the frontend**
   ```bash
   npm start
   ```

   Frontend will be available at:  
   ```
   http://localhost:3000
   ```

---

## 📂 Project Structure

```
lexiguard-frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/           # Page-level views (Home, Dashboard, Upload, Results)
│   ├── integrations/    # API calls to backend
│   ├── App.js           # Main app entry
│   ├── index.js         # React entry point
│   └── utils.js         # Helper functions
└── package.json
```

---

## 🏆 Quickstart for Hackathon Judges

1. Run backend locally with **Uvicorn**.  
2. Start frontend with `npm start`.  
3. Upload a sample contract and explore:  
   - **Plain-English Summary**  
   - **Risk Analysis with severity badges**  
   - **Fairness Score gauge**  
   - **📧 “Help Me Negotiate”** button for high-risk clauses  

---

## 🔐 Security & Privacy

- No persistent file storage (in-memory processing only).  
- All frontend-backend communication is via secure HTTPS (when deployed).  

---

## 🌍 Deployment

- **Local:** Run backend with Uvicorn, frontend with `npm start`.  
- **Cloud:** Backend deployable on Google Cloud Run; frontend deployable on Vercel.

---
