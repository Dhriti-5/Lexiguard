# LexiGuard Backend API 🛡️⚖️

The **LexiGuard Backend** is a FastAPI-based cloud-native API that powers intelligent legal document analysis using Google's Gemini AI. It provides comprehensive document processing, risk analysis, PII redaction, and interactive chat capabilities.

---

## 🌟 Features

- **📄 Document Processing**: Supports PDF, DOCX, and TXT file uploads
- **🤖 AI-Powered Analysis**: Uses Google Gemini 2.5 Flash for intelligent document understanding
- **⚠️ Risk Detection**: Identifies and categorizes risky clauses with severity levels
- **📊 Fairness Scoring**: Evaluates contracts against standard legal practices
- **🔒 PII Redaction**: Automatic detection and redaction of Personally Identifiable Information using Google Cloud DLP
- **💬 Interactive Chat**: Context-aware Q&A about uploaded documents
- **🤝 Negotiation Assistant**: Provides strategic advice for contesting unfavorable terms
- **📧 Email Generation**: Creates professional emails for legal advisors
- **🌍 Translation Support**: Multi-language document analysis
- **📦 PDF Report Generation**: Generates detailed analysis reports

---

## 🏗️ Architecture

- **Framework**: FastAPI (Python)
- **AI Model**: Google Gemini 2.5 Flash
- **Document Processing**: PyPDF2, python-docx
- **PII Protection**: Google Cloud Data Loss Prevention (DLP) API
- **Cloud Storage**: Google Cloud Storage
- **Report Generation**: ReportLab
- **Deployment**: Google Cloud Run

---

## 📋 Prerequisites

- Python 3.9 or higher
- Google Cloud Platform account
- Google API Key for Gemini AI
- Google Cloud credentials with DLP API access

---

## 🚀 Quick Start

### 1. Clone and Navigate

```bash
cd lexiguard-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the `lexiguard-backend` directory:

```env
# Google Gemini API Key
GOOGLE_API_KEY=your_gemini_api_key_here

# Google Cloud Credentials
GOOGLE_APPLICATION_CREDENTIALS=./path-to-your-service-account-key.json

# GCP Project Configuration
GCP_PROJECT_ID=your_gcp_project_id
GCP_LOCATION=asia-south1

# SMTP Configuration (Optional - for email features)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### 4. Start the Server

**Using Python directly:**
```bash
python main.py
```

**Using PowerShell script:**
```powershell
.\start-backend.ps1
```

**Using Uvicorn:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

---

## 📡 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and API info |
| `POST` | `/analyze` | Analyze a legal document |
| `POST` | `/chat` | Chat with an analyzed document |
| `POST` | `/negotiation-advice` | Get negotiation strategies |
| `POST` | `/generate-email` | Generate professional emails |
| `POST` | `/redact-pii` | Redact PII from documents |
| `POST` | `/generate-report` | Generate PDF analysis report |
| `POST` | `/translate-document` | Translate documents |

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📤 Example Usage

### Analyze a Document

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@sample_contract.pdf" \
  -F "analysis_type=full"
```

### Chat with Document

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_123",
    "message": "What are the termination clauses?",
    "context": "Previous document analysis context"
  }'
```

---

## 🧪 Testing

Run the test suite:

```bash
# Test basic functionality
python test_basic.py

# Test API endpoints
python test_api.py

# Test PII redaction
python test_pii_redaction.py

# Test chat feature
python test_chat_feature.py

# Test fixed model
python test_fixed_model.py
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t lexiguard-backend .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_api_key \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  -v /path/to/credentials.json:/app/credentials.json \
  lexiguard-backend
```

---

## ☁️ Cloud Deployment

> **⚠️ Note**: The Cloud Run deployment is currently **offline** to manage costs during the development phase. As students, we've temporarily shut down the production instance and disabled certain GCP APIs to avoid ongoing charges. The deployment instructions below are preserved for future use or for those who wish to deploy their own instance.

### Deploy to Cloud Run

```bash
gcloud run deploy lexiguard-backend \
  --source . \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=your_api_key
```

---

## 📁 Project Structure

```
lexiguard-backend/
├── main.py                    # Main FastAPI application
├── app.py                     # Alternative Flask application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── translation_utils.py       # Translation utilities
├── check_models.py           # Model verification script
├── start-backend.ps1         # Windows startup script
├── test_*.py                 # Test files
└── uploads/                  # Temporary file storage
```

---

## 🔑 Key Components

### Document Processing
- Extracts text from PDF, DOCX, and TXT files
- Handles multi-page documents
- Supports various encoding formats

### AI Analysis
- **Summarization**: Generates concise document summaries
- **Risk Analysis**: Identifies concerning clauses with severity ratings
- **Fairness Scoring**: Evaluates contract fairness (0-100 scale)
- **Negotiation Advice**: Provides strategic recommendations

### Security Features
- PII Detection and Redaction using Google Cloud DLP
- Secure file handling and temporary storage
- CORS configuration for secure cross-origin requests

---

## 🛠️ Configuration

### CORS Settings

The API allows requests from:
- `http://localhost:3000` (Local development)
- `https://lexiguard-one.vercel.app` (Production frontend)
- All Vercel subdomains via regex

### Safety Settings

Gemini AI is configured with relaxed safety settings for legal content analysis:
```python
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "block_none",
    "HARM_CATEGORY_HATE_SPEECH": "block_none", 
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
}
```

---

## 🐛 Troubleshooting

### Model Not Found Error
If you encounter "models/gemini-2.5-flash is not found", ensure your API key is valid and has access to the latest Gemini models.

### Authentication Issues
Verify that `GOOGLE_APPLICATION_CREDENTIALS` points to a valid service account key file with DLP API permissions.

### Port Already in Use
Change the port in the startup command:
```bash
uvicorn main:app --port 8080
```

---

## 📊 Performance

- Average analysis time: 3-8 seconds per document
- Supports documents up to 16MB
- Handles concurrent requests via FastAPI async
- Cloud Run auto-scales based on traffic (when deployed)

### 💰 Cost Considerations

As a student project, we're mindful of cloud costs:
- Google Cloud Run charges based on request volume and compute time
- Google Cloud DLP API has per-request pricing
- Gemini API usage may incur costs depending on usage tier
- For development, we recommend running locally to minimize expenses
- Consider using free tier limits and setting up budget alerts in GCP

---

## 🔐 Security Best Practices

1. Never commit `.env` files or API keys to version control
2. Use service accounts with minimal required permissions
3. Regularly rotate API keys and credentials
4. Enable Cloud Run authentication for production
5. Implement rate limiting for public endpoints

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Gemini API](https://ai.google.dev/docs)
- [Google Cloud DLP](https://cloud.google.com/dlp/docs)
- [Frontend README](../lexiguard-frontend/README.md)

---

## 👥 Contributors

- Dhriti Gandhi
- Krisha Gandhi
- Kavya Patel

---

## 📄 License

This project is part of a hackathon submission. All rights reserved.

---

## 🆘 Support

For issues or questions, please refer to the main project documentation or contact the development team.
