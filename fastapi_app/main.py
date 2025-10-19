# fastapi_app/main.py
"""
FastAPI wrapper for LexiGuard SDK
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lexiguard_sdk import LexiGuard, FileParser, LexiGuardError

# Initialize FastAPI app
app = FastAPI(
    title="LexiGuard API",
    description="Legal Document Analysis API powered by LexiGuard SDK",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize SDK
try:
    lexiguard = LexiGuard(api_key=GEMINI_API_KEY)
except LexiGuardError as e:
    print(f"Warning: Failed to initialize LexiGuard: {e}")
    lexiguard = None


# Pydantic models for request/response
class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., description="Legal document text to analyze")


class AnalyzeClausesRequest(BaseModel):
    text: str = Field(..., description="Legal document text")
    clause_types: Optional[List[str]] = Field(None, description="Specific clause types to focus on")


class DraftNegotiationRequest(BaseModel):
    document_text: str = Field(..., description="The legal document text")
    concerns: List[str] = Field(..., description="List of concerns to address")
    recipient_name: str = Field(default="Recipient", description="Email recipient name")


class DraftDocumentEmailRequest(BaseModel):
    document_text: str = Field(..., description="The legal document text")
    review_notes: str = Field(..., description="Review notes or summary")
    recipient_name: str = Field(default="Recipient", description="Email recipient name")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    document_context: Optional[str] = Field(None, description="Optional document context")


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to LexiGuard API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze text",
            "analyze_file": "POST /analyze-file - Analyze uploaded file",
            "analyze_clauses": "POST /analyze-clauses - Detailed clause analysis",
            "analyze_extended": "POST /analyze-extended - Fairness scoring",
            "draft_negotiation": "POST /draft-negotiation - Generate negotiation email",
            "draft_document_email": "POST /draft-document-email - Generate review email",
            "chat": "POST /chat - Chat about document"
        }
    }


@app.post("/analyze")
async def analyze_text(request: AnalyzeTextRequest):
    """Analyze legal document text"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.analyze_text(request.text)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
    
    return result


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze uploaded file (PDF, DOCX, TXT)"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    # Read file bytes
    file_bytes = await file.read()
    
    # Parse file
    parse_result = FileParser.parse_uploaded_file(file_bytes, file.filename)
    
    if not parse_result["success"]:
        raise HTTPException(status_code=400, detail=parse_result["error"])
    
    # Analyze text
    analysis = lexiguard.analyze_text(parse_result["text"])
    
    if not analysis["success"]:
        raise HTTPException(status_code=400, detail=analysis.get("error", "Analysis failed"))
    
    # Add file info
    analysis["file_info"] = {
        "file_name": parse_result["file_name"],
        "file_type": parse_result["file_type"]
    }
    
    return analysis


@app.post("/analyze-clauses")
async def analyze_clauses(request: AnalyzeClausesRequest):
    """Perform detailed clause analysis"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.analyze_clauses(request.text, request.clause_types)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
    
    return result


@app.post("/analyze-extended")
async def analyze_extended(request: AnalyzeTextRequest):
    """Analyze document fairness and provide scoring"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.analyze_fairness(request.text)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Analysis failed"))
    
    return result


@app.post("/draft-negotiation")
async def draft_negotiation(request: DraftNegotiationRequest):
    """Generate negotiation email"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.draft_negotiation_email(
        request.document_text,
        request.concerns,
        request.recipient_name
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Email generation failed"))
    
    return result


@app.post("/draft-document-email")
async def draft_document_email(request: DraftDocumentEmailRequest):
    """Generate document review email"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.draft_document_review_email(
        request.document_text,
        request.review_notes,
        request.recipient_name
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Email generation failed"))
    
    return result


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat about document with context"""
    if not lexiguard:
        raise HTTPException(status_code=500, detail="LexiGuard SDK not initialized")
    
    result = lexiguard.chat(request.message, request.document_context)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Chat failed"))
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)