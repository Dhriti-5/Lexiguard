# main.py - FIXED VERSION

import os
import json
import io
import logging
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types as dlp_types
import PyPDF2
from docx import Document

# --- 0. CONFIGURE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- 2. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="LexiGuard API",
    description="Analyzes legal documents (text, PDF, or DOCX) using Google's Gemini AI with PII Redaction.",
    version="1.4.0"
)

# --- 3. ENABLE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. CONFIGURE GOOGLE GEMINI ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env. Relying on ADC or Cloud Run credentials.")
else:
    genai.configure(api_key=API_KEY)

safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "block_none",
    "HARM_CATEGORY_HATE_SPEECH": "block_none",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
}

# Initialize Gemini model - using models from list_models()
# For v1beta API, must use FULL model name as returned by list_models()
MODEL_NAME = "models/gemini-2.5-flash"  # Latest Gemini Flash model
model = None

try:
    model = genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)
    logger.info(f"✅ Gemini model initialized successfully with {MODEL_NAME}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini model '{MODEL_NAME}': {e}")
    logger.error("Please check:")
    logger.error("  1. Your GOOGLE_API_KEY is set correctly in .env")
    logger.error("  2. The API key has Gemini API access enabled")
    logger.error("  3. Internet connection is working")
    model = None
    MODEL_NAME = None

# --- 5. CONFIGURE GOOGLE CLOUD DLP ---
# Initialize DLP client lazily to avoid multiprocessing issues
dlp_client = None
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

def get_dlp_client():
    """Lazy initialization of DLP client"""
    global dlp_client
    if dlp_client is None:
        try:
            dlp_client = dlp_v2.DlpServiceClient()
            logger.info("DLP client initialized successfully")
        except Exception as e:
            logger.warning(f"DLP client initialization failed: {e}")
            logger.warning("PII redaction will be disabled")
    return dlp_client

if not PROJECT_ID:
    logger.warning("GOOGLE_CLOUD_PROJECT not set; DLP may fail without ADC context.")


# Fix: Use correct DLP config objects for current google-cloud-dlp


INFO_TYPES_TO_REDACT = [
    dlp_types.InfoType(name="PERSON_NAME"),
    dlp_types.InfoType(name="EMAIL_ADDRESS"),
    dlp_types.InfoType(name="PHONE_NUMBER"),
    dlp_types.InfoType(name="STREET_ADDRESS"),
    dlp_types.InfoType(name="CREDIT_CARD_NUMBER"),
    dlp_types.InfoType(name="DATE_OF_BIRTH"),
    dlp_types.InfoType(name="US_SOCIAL_SECURITY_NUMBER"),
    # Removed INDIA_AADHAAR_ID_NUMBER - not available in all regions
]


info_type_transformations = dlp_types.InfoTypeTransformations(
    transformations=[
        {
            "info_types": INFO_TYPES_TO_REDACT,
            "primitive_transformation": dlp_types.PrimitiveTransformation(
                replace_with_info_type_config=dlp_types.ReplaceWithInfoTypeConfig()
            ),
        }
    ]
)

DEIDENTIFY_CONFIG = dlp_types.DeidentifyConfig(
    info_type_transformations=info_type_transformations
)

# --- 6. DATA MODELS ---
class DocumentRequest(BaseModel):
    text: str

class ChatRequest(BaseModel):
    message: str
    document_text: str

class NegotiationRequest(BaseModel):
    clause: str

class DocumentEmailRequest(BaseModel):
    document_summary: str
    risk_summary: str

class ExtendedAnalysisRequest(BaseModel):
    text: str

# --- 7. PROMPTS (FULL RESTORED) ---

SUMMARY_PROMPT = """
You are LexiGuard, an expert AI assistant that explains complex legal documents in simple terms.
Analyze the following contract. Provide a concise, bullet-point summary covering:
1. The primary purpose of the agreement.
2. The key responsibilities of each party.
3. The duration and key financial terms (like rent, salary, etc.).
Use clear, simple language suitable for a non-lawyer.
"""

RISK_ANALYSIS_PROMPT = """
You are a meticulous risk analysis AI. Scan the provided legal document.
Your task is to identify and extract any clauses that are potentially unfavorable, non-standard, or represent a significant risk.
Focus specifically on clauses related to: Indemnity, Limitation of Liability, Automatic Renewal, Termination Penalties, and Non-Compete agreements.
For each identified clause, you MUST provide the original text, a simple one-sentence explanation of the risk, and a severity level of either 'High' or 'Medium'.
You MUST respond ONLY with a valid JSON object. The structure of the JSON object must be:
{"risks": [{"clause_text": "...", "risk_explanation": "...", "severity": "..."}]}
If you find no risks, you MUST return: {"risks": []}
Do not add any text or formatting before or after the JSON object.
"""

NEGOTIATION_PROMPT = """
You are LexiGuard, an AI that helps users negotiate risky clauses politely.
Draft a short, professional, and polite email asking to revise or clarify the following clause:

Clause:
{clause}

Do not include placeholders like [Company]. Just write a natural, ready-to-send email body.
"""

DOCUMENT_EMAIL_PROMPT = """
You are LexiGuard, an AI assistant that helps users communicate about legal document reviews.

Generate a professional email to send to a legal advisor, counterparty, or stakeholder regarding a legal document review.

Document Summary:
{document_summary}

Identified Risks:
{risk_summary}

The email should:
1. Be professional, clear, and concise
2. Summarize the key findings from the document analysis
3. Highlight the most critical risks identified
4. Request clarification, revision, or discussion on the concerning clauses
5. Maintain a collaborative and constructive tone
6. Be ready to send with minimal editing

Generate ONLY the email body. Do not include:
- Subject line
- Sender/recipient names or addresses
- Placeholders like [Your Name] or [Company Name]

Start directly with a professional greeting and the email content.
"""

FAIRNESS_PROMPT = """
You are LexiGuard, a fairness evaluator.
Compare the following risky clause with a standard, balanced contract clause.
Return a JSON object strictly in this format:
{
  "standard_clause": "...",
  "risky_clause": "...",
  "fairness_score": 0-100,
  "explanation": "..."
}

Risky Clause:
{clause}
"""

DETAILED_CLAUSE_ANALYSIS_PROMPT = """
You are a legal expert analyzing contracts and agreements for LexiGuard. 
Analyze the following document and identify ALL risky or concerning clauses with deep explanations.

For EACH risky clause, provide:
1. **clause**: The exact clause text or relevant excerpt (keep it under 200 characters if too long)
2. **risk_level**: "High", "Medium", or "Low"
3. **impact**: Brief description of potential harm to the user (1 sentence)
4. **recommendation**: Specific actionable advice for negotiation (1-2 sentences)
5. **explanation**: Detailed plain-language explanation of why this clause is risky and what could go wrong (2-3 sentences)

Focus on these risk categories:
- Termination rights (sudden eviction, firing without cause)
- Financial liability (unlimited damages, penalties)
- Automatic renewal traps
- Non-compete restrictions
- Indemnification clauses
- Limitation of liability
- Unilateral changes by one party
- Waiver of legal rights

You MUST respond ONLY with a valid JSON array. Use this exact format:
[
    {
        "clause": "Original clause text here",
        "risk_level": "High",
        "impact": "Brief impact description",
        "recommendation": "What the user should do or negotiate",
        "explanation": "Detailed plain-language explanation of why this is risky"
    }
]

If no risks are found, return an empty array: []
CRITICAL: Respond ONLY with valid JSON, no additional text, no markdown.
"""

# --- 8. HELPER FUNCTIONS ---
# Model is already initialized above (lines 47-82) - using gemini-1.5-flash
# No need for duplicate get_working_model() function

def redact_text_with_dlp(text: str):
    if not text or not PROJECT_ID:
        logger.warning("DLP: Project ID not configured. Skipping redaction.")
        return text, False

    # Get DLP client (lazy initialization)
    client = get_dlp_client()
    if not client:
        logger.warning("DLP client not available. Skipping redaction.")
        return text, False

    parent_path = f"projects/{PROJECT_ID}/locations/global"
    item = {"value": text}

    try:
        response = client.deidentify_content(
            request={
                "parent": parent_path,
                "deidentify_config": DEIDENTIFY_CONFIG,
                "inspect_config": {"info_types": INFO_TYPES_TO_REDACT},
                "item": item,
            }
        )
        redacted = response.item.value
        changed = redacted != text
        logger.info("DLP Redaction complete.")
        return redacted, changed
    except Exception as e:
        logger.error(f"DLP failed: {e}")
        return text, False

def extract_text_from_pdf(file_stream):
    try:
        pdf = PyPDF2.PdfReader(file_stream)
        return "".join([p.extract_text() or "" for p in pdf.pages])
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Error extracting text from PDF.")

def extract_text_from_docx(file_stream):
    try:
        doc = Document(file_stream)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Error extracting text from DOCX.")

# --- 9. CORE ANALYSIS LOGIC ---
def analyze_text_internal(text: str):
    if model is None:
        raise HTTPException(status_code=503, detail="AI model not initialized. Check backend logs.")
    
    try:
        redacted_text, changed = redact_text_with_dlp(text)
        prompt = f"{SUMMARY_PROMPT}\n\nDocument:\n{redacted_text}"
        response = model.generate_content(prompt)
        summary = response.text.strip()

        risk_prompt = f"{RISK_ANALYSIS_PROMPT}\n\nDocument:\n{redacted_text}"
        risk_response = model.generate_content(risk_prompt)
        try:
            # Clean JSON response
            risks_text = risk_response.text.strip().replace("```json", "").replace("```", "").strip()
            risks = json.loads(risks_text)
        except Exception as e:
            logger.error(f"Risk JSON parse error: {e}")
            risks = {"risks": []}

        return {
            "summary": summary,
            "risks": risks,
            "pii_redacted": changed
        }
    except Exception as e:
        logger.error(f"Error in analyze_text_internal: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def analyze_clauses_detailed_internal(text: str):
    if model is None:
        raise HTTPException(status_code=503, detail="AI model not initialized. Check backend logs.")
    
    try:
        redacted_text, changed = redact_text_with_dlp(text)
        prompt = f"{DETAILED_CLAUSE_ANALYSIS_PROMPT}\n\nDocument:\n{redacted_text}"
        response = model.generate_content(prompt)
        try:
            # Clean JSON response
            risks_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            risks = json.loads(risks_text)
        except Exception as e:
            logger.error(f"Clause JSON parse error: {e}")
            risks = []
        return {"risks": risks, "pii_redacted": changed}
    except Exception as e:
        print(f"Error in detailed clause analysis: {str(e)}")
        return []

def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# --- 8. ROUTES ---
@app.get("/")
async def root():
    return {
        "message": "LexiGuard API is running.",
        "version": "1.3.0",
        "endpoints": [
            "/analyze",
            "/analyze-file (Standard Analysis with Negotiation)",
            "/analyze-clauses (Detailed Clause Analysis)",
            "/draft-negotiation (Generate negotiation emails)",
            "/draft-document-email (Generate comprehensive document review email)",
            "/analyze-extended (Fairness scoring)",
            "/chat"
        ]
    }

@app.post("/analyze")
async def analyze_document(request: DocumentRequest):
    return analyze_text_internal(request.text)

@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(None), text: str = Form(None)):
    """
    STANDARD ANALYSIS endpoint with negotiation support.
    Returns: summary, risks, suggestions, and file type.
    """
    if file:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            text_content = extract_text_from_pdf(file.file)
            file_type = "PDF"
        elif filename.endswith(".docx"):
            text_content = extract_text_from_docx(file.file)
            file_type = "DOCX"
        else:
            return {"error": "Unsupported file type. Only PDF or DOCX allowed."}
    elif text:
        document_text = text
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")

    result = analyze_text_internal(document_text)
    
    # Return in format expected by frontend
    response = {
        "filename": file.filename if file else "Text Input",
        "file_type": file.filename.split(".")[-1].upper() if file else "Text",
        "summary": result.get("summary", ""),
        "risks": result.get("risks", {}).get("risks", []),
        "pii_redacted": result.get("pii_redacted", False)
    }
    
    # Add privacy notice if PII was redacted
    if result.get("pii_redacted", False):
        response["privacy_notice"] = "✓ Your Personal Data Has Been Redacted for Privacy."
    
    return response

@app.post("/analyze-clauses")
async def analyze_clauses(file: UploadFile = File(None), text: str = Form(None)):
    """
    DETAILED CLAUSE ANALYSIS endpoint.
    Deep clause-by-clause analysis with detailed explanations,
    impact assessment, and recommendations.
    
    Returns:
    - filename (if file uploaded)
    - file_type (PDF, DOCX, or Text)
    - total_risky_clauses (count)
    - clauses (array of detailed clause objects)
    - document_preview (first 300 characters)
    """
    if file:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            text_content = extract_text_from_pdf(file.file)
            file_type = "PDF"
        elif filename.endswith(".docx"):
            text_content = extract_text_from_docx(file.file)
            file_type = "DOCX"
        else:
            return {"error": "Unsupported file type. Only PDF or DOCX allowed."}
        filename_display = file.filename
    elif text:
        document_text = text
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")
    
    result = analyze_clauses_detailed_internal(document_text)
    
    # Return in format expected by frontend
    response = {
        "filename": file.filename if file else "Text Input",
        "file_type": file.filename.split(".")[-1].upper() if file else "Text",
        "total_risky_clauses": len(result.get("risks", [])),
        "clauses": result.get("risks", []),
        "pii_redacted": result.get("pii_redacted", False)
    }
    
    # Add privacy notice if PII was redacted
    if result.get("pii_redacted", False):
        response["privacy_notice"] = "✓ Your Personal Data Has Been Redacted for Privacy."
    
    return response

@app.post("/negotiate-clause")
async def negotiate_clause(request: NegotiationRequest):
    prompt = NEGOTIATION_PROMPT.format(clause=request.clause)
    response = model.generate_content(prompt)
    return {"email": response.text.strip()}

@app.post("/generate-email")
async def generate_email(request: DocumentEmailRequest):
    prompt = DOCUMENT_EMAIL_PROMPT.format(
        document_summary=request.document_summary,
        risk_summary=request.risk_summary
    )
    response = model.generate_content(prompt)
    return {"email": response.text.strip()}

@app.post("/fairness-score")
async def fairness_score(request: NegotiationRequest):
    prompt = FAIRNESS_PROMPT.format(clause=request.clause)
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except Exception:
        return {"error": "Could not parse AI response"}

@app.get("/")
def root():
    return {"message": "LexiGuard API is running successfully 🚀"}

# --- CHAT ENDPOINT ---
@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """
    Chat with document endpoint - allows users to ask questions about the analyzed document
    """
    if not request.message.strip() or not request.document_text.strip():
        return {"reply": "Please provide a message and document text."}

    prompt = f"""
You are LexiGuard, a helpful AI assistant. Answer the user's messages based only on the provided legal document.
Maintain conversation context. Do not assume beyond the document.

Document:
{request.document_text}

User Message:
{request.message}

Respond concisely and clearly for a non-lawyer.
"""
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip() or "No answer could be generated."
    except Exception as e:
        logger.error(f"Chat error: {e}")
        answer = f"Error: {str(e)}"
    return {"reply": answer}


# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Vercel sets PORT automatically
    uvicorn.run("main:app", host="0.0.0.0", port=port)
