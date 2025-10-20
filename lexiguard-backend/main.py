# main.py - FIXED VERSION

import os
import json
import io
import logging
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types as dlp_types
import PyPDF2
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

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

class SendDocumentReviewRequest(BaseModel):
    filename: str
    document_summary: str
    risk_summary: str
    clauses: list
    user_email: str

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
You are LexiGuard, an AI that helps users politely negotiate risky contract clauses.
Draft a professional and polite email body requesting to amend or clarify the following clause.

Clause (PII may be redacted with placeholders like [PERSON_NAME]):
{clause}


Your task is to draft ONLY the email body. The email should:
1. Start professionally (e.g., "Regarding the draft agreement received...").
2. Clearly reference the clause in question (you can mention its content briefly).
3. Explain the user's concern politely based on the clause's meaning.
4. Suggest a discussion or a more balanced alternative.
5. Use standard placeholders like `[Your Name]`, `[Your Contact Info]`, `[Recipient Name/Company Name]`, and `[Date]` where specific personal or logistical details would normally go. Do NOT try to invent names or details.
6. End with a collaborative closing (e.g., "I look forward to discussing this further.").

Generate ONLY the email body text.
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
    
def extract_text_from_txt(file) -> str:
    try:
        # Read text file with UTF-8 encoding
        content = file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8')
        else:
            text = content
        return text
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            file.seek(0)
            content = file.read()
            text = content.decode('latin-1')
            return text
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise HTTPException(status_code=500, detail="Error extracting text from TXT file.")
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        raise HTTPException(status_code=500, detail="Error extracting text from TXT file.")

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
    "pii_redacted": changed,
    "redacted_text": redacted_text  # ✅ ADD THIS LINE
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
        return {
    "risks": risks, 
    "pii_redacted": changed,
    "redacted_text": redacted_text  # ✅ ADD THIS LINE
}
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
            document_text = extract_text_from_pdf(file.file)
            file_type = "PDF"
        elif filename.endswith(".docx"):
            document_text = extract_text_from_docx(file.file)
            file_type = "DOCX"
        elif filename.endswith(".txt"):  # ✅ ADD TXT SUPPORT
            document_text = extract_text_from_txt(file.file)
            file_type = "TXT"
        else:
            return {"error": "Unsupported file type. Only PDF, DOCX, or TXT allowed."}
    elif text:
        document_text = text
        file_type = "Text"  # ✅ ADD file_type for manual text input
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")

    # Redact PII from document for chat (privacy protection)
    redacted_document_text, _ = redact_text_with_dlp(document_text)
    
    result = analyze_text_internal(document_text)
    
    # Return in format expected by frontend
    response = {
        "filename": file.filename if file else "Text Input",
        "file_type": file.filename.split(".")[-1].upper() if file else "Text",
        "summary": result.get("summary", ""),
        "risks": result.get("risks", {}).get("risks", []),
        "pii_redacted": result.get("pii_redacted", False),
        "redacted_document_text": redacted_document_text
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
    - file_type (PDF, DOCX, TXT, or Text)
    - total_risky_clauses (count)
    - clauses (array of detailed clause objects)
    - document_preview (first 300 characters)
    """
    if file:
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            document_text = extract_text_from_pdf(file.file)
            file_type = "PDF"
        elif filename.endswith(".docx"):
            document_text = extract_text_from_docx(file.file)
            file_type = "DOCX"
        elif filename.endswith(".txt"):  # ✅ ADD TXT SUPPORT
            document_text = extract_text_from_txt(file.file)
            file_type = "TXT"
        else:
            return {"error": "Unsupported file type. Only PDF, DOCX, or TXT allowed."}
        filename_display = file.filename
    elif text:
        document_text = text
        file_type = "Text"  # ✅ ADD file_type for manual text input
        filename_display = "Text Input"
    else:
        raise HTTPException(status_code=400, detail="No file or text provided")

    # Redact PII from document for chat (privacy protection)
    redacted_document_text, _ = redact_text_with_dlp(document_text)
    
    result = analyze_clauses_detailed_internal(document_text)
    
    # Return in format expected by frontend
    response = {
        "filename": file.filename if file else "Text Input",
        "file_type": file.filename.split(".")[-1].upper() if file else "Text",
        "total_risky_clauses": len(result.get("risks", [])),
        "clauses": result.get("risks", []),
        "pii_redacted": result.get("pii_redacted", False),
        "redacted_text": redacted_document_text
    }
    
    # Add privacy notice if PII was redacted
    if result.get("pii_redacted", False):
        response["privacy_notice"] = "✓ Your Personal Data Has Been Redacted for Privacy."
    
    return response

@app.post("/negotiate-clause")
@app.post("/draft-negotiation")
async def negotiate_clause(request: NegotiationRequest):
    """Generate negotiation email for a risky clause (using redacted text)"""
    redacted_text, changed = redact_text_with_dlp(request.clause)
    prompt = NEGOTIATION_PROMPT.format(clause=redacted_text)
    response = model.generate_content(prompt)
    return {"negotiation_email": response.text.strip()}

@app.post("/generate-email")
@app.post("/draft-document-email")
async def generate_email(request: DocumentEmailRequest):
    """Generate comprehensive document review email"""
    prompt = DOCUMENT_EMAIL_PROMPT.format(
        document_summary=request.document_summary,
        risk_summary=request.risk_summary
    )
    response = model.generate_content(prompt)
    return {"document_email": response.text.strip()}

@app.post("/fairness-score")
async def fairness_score(request: NegotiationRequest):
    prompt = FAIRNESS_PROMPT.format(clause=request.clause)
    response = model.generate_content(prompt)
    try:
        return json.loads(response.text)
    except Exception:
        return {"error": "Could not parse AI response"}

# --- SEND DOCUMENT REVIEW EMAIL ENDPOINT ---
@app.post("/send-document-review")
async def send_document_review(request: SendDocumentReviewRequest):
    """
    Generate PDF document review report and send via email
    """
    try:
        # Create PDF in memory
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#064E3B'),
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("LexiGuard Document Review Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Add document info
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#0891B2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        story.append(Paragraph("Document Information", heading_style))
        story.append(Paragraph(f"<b>Filename:</b> {request.filename}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add document summary
        story.append(Paragraph("Document Summary", heading_style))
        story.append(Paragraph(request.document_summary, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add risk summary
        story.append(Paragraph("Risk Analysis", heading_style))
        story.append(Paragraph(request.risk_summary, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add risky clauses table
        if request.clauses:
            story.append(Paragraph("Identified Risky Clauses", heading_style))
            
            # Create table data
            table_data = [['Clause', 'Risk Level', 'Explanation']]
            for clause_item in request.clauses:
                clause_text = clause_item.get('clause', 'N/A')[:100] + '...' if len(clause_item.get('clause', '')) > 100 else clause_item.get('clause', 'N/A')
                risk = clause_item.get('risk', 'Unknown')
                explanation = clause_item.get('explanation', 'No explanation provided')[:150] + '...' if len(clause_item.get('explanation', '')) > 150 else clause_item.get('explanation', 'No explanation provided')
                
                table_data.append([clause_text, risk, explanation])
            
            # Create table with styling
            table = Table(table_data, colWidths=[2.5*inch, 1*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#064E3B')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(table)
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        pdf_data = pdf_buffer.read()
        
        # Send email using Gmail SMTP
        # Note: You'll need to set up Gmail App Password and add to .env
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("GMAIL_SENDER_EMAIL")  # Add to .env
        sender_password = os.getenv("GMAIL_APP_PASSWORD")  # Add Gmail App Password to .env
        
        if not sender_email or not sender_password:
            raise HTTPException(
                status_code=500,
                detail="Email configuration missing. Please set GMAIL_SENDER_EMAIL and GMAIL_APP_PASSWORD in .env file"
            )
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = request.user_email
        msg['Subject'] = f"LexiGuard Document Review: {request.filename}"
        
        # Email body
        body = f"""
Hello,

Please find attached your LexiGuard document review report for: {request.filename}

This report includes:
- Document Summary
- Risk Analysis
- Identified Risky Clauses

Thank you for using LexiGuard!

Best regards,
The LexiGuard Team
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        pdf_attachment = MIMEApplication(pdf_data, _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=f'LexiGuard_Review_{request.filename}.pdf')
        msg.attach(pdf_attachment)
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return {
            "success": True,
            "message": f"Document review email sent successfully to {request.user_email}"
        }
        
    except Exception as e:
        logger.error(f"Error sending document review email: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

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
You are LexiGuard, a helpful AI assistant. Answer the user's questions based *only* on the provided legal document context below.
This document has had Personal Identifiable Information (PII) like names and addresses replaced with placeholders (e.g., [PERSON_NAME], [STREET_ADDRESS]).
Focus your answer on explaining the rules, obligations, terms, and the overall context defined in the document, even with the placeholders.

Document Context (Redacted):
'''
{request.document_text} 
'''

User Message:
'''
{request.message}
'''

Respond concisely and clearly for a non-lawyer. If the answer depends entirely on specific PII that has been redacted, state that the specific detail is unavailable due to redaction, but explain the general rule or context.
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