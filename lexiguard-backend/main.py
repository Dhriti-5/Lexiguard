# main.py - FIXED VERSION
import os
import json
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from docx import Document

# --- 1. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- 2. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="LexiGuard API",
    description="Analyzes legal documents (text, PDF, or DOCX) using Google's Gemini AI.",
    version="1.3.0"
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
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "block_none",
    "HARM_CATEGORY_HATE_SPEECH": "block_none",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
}

model = genai.GenerativeModel(
    "models/gemini-2.5-flash",
    safety_settings=safety_settings
)

# --- 5. DATA MODELS ---
class DocumentRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {"example": {"text": "This is a sample rental agreement..."}}

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

# --- 6. PROMPTS ---
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

# --- NEW: DETAILED CLAUSE EXPLANATION PROMPT ---
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

# --- 7. HELPERS ---
def analyze_text(document_text: str):
    """Analyze a legal document: generate summary, risk analysis, and suggestions."""
    
    # --- AI Call 1: Summary ---
    summary_prompt_with_formatting = f"""
You are LexiGuard, an expert AI assistant that explains complex legal documents in simple terms.
Analyze the following contract and return a properly formatted text summary.
Use numbered sections, line breaks, and clear headings.
Cover:

1. Primary Purpose
2. Key Responsibilities of each party
3. Duration and Financial Terms
4. Liability
5. Non-Compete (if any)
Do NOT use Markdown or bullets, just clean formatted text.
Document: {document_text}
"""
    try:
        summary_response = model.generate_content([summary_prompt_with_formatting])
        summary = getattr(summary_response, "text", "").strip() or "Summary could not be generated."
    except Exception as e:
        summary = f"Summary could not be generated. Error: {str(e)}"

    # Small delay for rate limits
    time.sleep(5)

    # --- AI Call 2: Risk Analysis ---
    try:
        risk_response = model.generate_content([RISK_ANALYSIS_PROMPT, document_text])
        risks_json_string = (
            risk_response.text.strip()
            .replace("```json", "")
            .replace("```", "")
        )
        risks_data = json.loads(risks_json_string)
        risks = risks_data.get("risks", [])
    except Exception as e:
        risks = []

    # --- Generate Suggestions based on High/Medium risks ---
    suggestions = []
    for risk in risks:
        clause = risk.get("clause_text", "")
        severity = risk.get("severity", "Medium")
        if severity == "High":
            suggestions.append(
                f"Review or negotiate the following clause to reduce risk: '{clause}'"
            )
        else:
            suggestions.append(
                f"Be aware of this clause: '{clause}'"
            )

    return {
        "summary": summary,
        "risks": risks,
        "suggestions": suggestions
    }

def analyze_clauses_detailed(document_text: str):
    """
    NEW FUNCTION: Perform deep clause-by-clause analysis with explanations, 
    impact assessment, and recommendations.
    """
    prompt = f"{DETAILED_CLAUSE_ANALYSIS_PROMPT}\n\nDocument:\n{document_text[:12000]}"
    
    try:
        response = model.generate_content([prompt])
        response_text = getattr(response, "text", "").strip()
        
        # Clean JSON response
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        clauses = json.loads(response_text)
        
        # Ensure it's a list
        if not isinstance(clauses, list):
            return []
        
        return clauses
    
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error in detailed analysis: {e}")
        print(f"Raw response: {response_text[:500]}")
        return []
    
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

def extract_text_from_txt(file) -> str:
    """NEW: Extract text from .txt files"""
    try:
        # Try UTF-8 first
        content = file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8')
        else:
            text = content
        return text
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        file.seek(0)
        content = file.read()
        return content.decode('latin-1')

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
        ],
        "supported_formats": ["PDF", "DOCX", "TXT", "Plain Text"]
    }

@app.post("/analyze")
async def analyze_document(request: DocumentRequest):
    result = analyze_text(request.text)
    result["file_type"] = "Text"
    result["privacy_notice"] = "Your file was processed in-memory and deleted after extraction."
    return result

@app.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(None),
    text: str = Form(None)
):
    """
    STANDARD ANALYSIS endpoint with negotiation support.
    NOW SUPPORTS: PDF, DOCX, and TXT files
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
        elif filename.endswith(".txt"):
            # FIXED: Now handles .txt files
            text_content = extract_text_from_txt(file.file)
            file_type = "TXT"
        else:
            return {"error": "Unsupported file type. Only PDF, DOCX, and TXT allowed."}
    elif text:
        text_content = text
        file_type = "Text"
    else:
        return {"error": "No file or text provided."}

    if not text_content.strip():
        return {"error": "No text could be extracted from the document."}

    result = analyze_text(text_content)
    result["file_type"] = file_type
    result["privacy_notice"] = "Your file was processed in-memory and deleted after extraction."
    return result

# --- NEW ENDPOINT: Detailed Clause Analysis ---
@app.post("/analyze-clauses")
async def analyze_clauses_endpoint(
    file: UploadFile = File(None),
    text: str = Form(None)
):
    """
    DETAILED CLAUSE ANALYSIS endpoint.
    Deep clause-by-clause analysis with detailed explanations,
    impact assessment, and recommendations.
    NOW SUPPORTS: PDF, DOCX, and TXT files
    
    Returns:
    - filename (if file uploaded)
    - file_type (PDF, DOCX, TXT, or Text)
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
        elif filename.endswith(".txt"):
            # FIXED: Now handles .txt files
            text_content = extract_text_from_txt(file.file)
            file_type = "TXT"
        else:
            return {"error": "Unsupported file type. Only PDF, DOCX, and TXT allowed."}
        filename_display = file.filename
    elif text:
        text_content = text
        file_type = "Text"
        filename_display = "Direct Text Input"
    else:
        return {"error": "No file or text provided."}

    if not text_content.strip():
        return {"error": "No text could be extracted from the document."}

    # Perform detailed clause analysis
    clauses = analyze_clauses_detailed(text_content)
    
    return {
        "filename": filename_display,
        "file_type": file_type,
        "total_risky_clauses": len(clauses),
        "clauses": clauses,
        "document_preview": text_content[:300],
        "privacy_notice": "Your file was processed in-memory and deleted after extraction."
    }

# --- RESTORED: NEGOTIATION ENDPOINT ---
@app.post("/draft-negotiation")
async def draft_negotiation(request: NegotiationRequest):
    """Drafts a polite negotiation email for a risky clause."""
    prompt = NEGOTIATION_PROMPT.format(clause=request.clause)
    try:
        resp = model.generate_content([prompt])
        email_text = getattr(resp, "text", "").strip() or "Could not generate email."
    except Exception as e:
        email_text = f"Error: {str(e)}"
    return {"negotiation_email": email_text}

# --- NEW: DOCUMENT EMAIL GENERATION ---
@app.post("/draft-document-email")
async def draft_document_email(request: DocumentEmailRequest):
    """
    Generates a comprehensive professional email covering all document findings.
    Used for both standard and detailed analysis results.
    """
    prompt = DOCUMENT_EMAIL_PROMPT.format(
        document_summary=request.document_summary[:2000],  # Limit length
        risk_summary=request.risk_summary[:2000]
    )
    
    try:
        resp = model.generate_content([prompt])
        email_text = getattr(resp, "text", "").strip() or "Could not generate email."
    except Exception as e:
        email_text = f"Error generating email: {str(e)}"
    
    return {"document_email": email_text}

# --- RESTORED: EXTENDED ANALYSIS WITH FAIRNESS SCORING ---
@app.post("/analyze-extended")
async def analyze_extended(request: ExtendedAnalysisRequest):
    """Performs extended analysis with clause comparison and fairness scoring."""
    base_result = analyze_text(request.text)

    fairness_results = []
    for risk in base_result.get("risks", []):
        clause_text = risk.get("clause_text", "")
        if not clause_text.strip():
            continue
        prompt = FAIRNESS_PROMPT.format(clause=clause_text)
        try:
            resp = model.generate_content([prompt])
            fairness_json = (
                resp.text.strip()
                .replace("```json", "")
                .replace("```", "")
            )
            parsed = json.loads(fairness_json)
        except Exception as e:
            parsed = {
                "standard_clause": "",
                "risky_clause": clause_text,
                "fairness_score": 50,
                "explanation": f"Error: {str(e)}"
            }
        fairness_results.append(parsed)

        # Respect rate limits
        time.sleep(2)

    base_result["fairness_analysis"] = fairness_results
    return base_result

# --- CHAT ENDPOINT ---
@app.post("/chat")
async def chat_with_document(request: ChatRequest):
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
        resp = model.generate_content([prompt])
        answer = getattr(resp, "text", "").strip() or "No answer could be generated."
    except Exception as e:
        answer = f"Error: {str(e)}"
    return {"reply": answer}

# --- 10. ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)