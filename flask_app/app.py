# flask_app/app.py
"""
Flask wrapper for LexiGuard SDK
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from lexiguard_sdk import LexiGuard, FileParser, LexiGuardError

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize SDK
try:
    lexiguard = LexiGuard(api_key=GEMINI_API_KEY)
except LexiGuardError as e:
    print(f"Warning: Failed to initialize LexiGuard: {e}")
    lexiguard = None


# Helper function for error responses
def error_response(message, status_code=400):
    """Create standardized error response"""
    return jsonify({
        "success": False,
        "error": message
    }), status_code


# Routes
@app.route("/", methods=["GET"])
def root():
    """Root endpoint"""
    return jsonify({
        "message": "Welcome to LexiGuard API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze text",
            "analyze_pdf": "POST /analyze-pdf - Analyze uploaded file",
            "analyze_clauses": "POST /analyze-clauses - Detailed clause analysis",
            "draft_negotiation": "POST /draft-negotiation - Generate negotiation email",
            "draft_document_email": "POST /draft-document-email - Generate review email",
            "chat": "POST /chat - Chat about document",
            "health": "GET /health - Health check"
        }
    })


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "sdk_initialized": lexiguard is not None
    })


@app.route("/analyze", methods=["POST"])
def analyze_text():
    """Analyze legal document text"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    data = request.get_json()
    
    if not data or "text" not in data:
        return error_response("Missing 'text' field in request")
    
    result = lexiguard.analyze_text(data["text"])
    
    if not result["success"]:
        return error_response(result.get("error", "Analysis failed"))
    
    return jsonify(result)


@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():
    """Analyze uploaded file (PDF, DOCX, TXT)"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    # Check if file is in request
    if "file" not in request.files:
        return error_response("No file uploaded")
    
    file = request.files["file"]
    
    if file.filename == "":
        return error_response("Empty filename")
    
    # Read file bytes
    file_bytes = file.read()
    
    # Parse file
    parse_result = FileParser.parse_uploaded_file(file_bytes, file.filename)
    
    if not parse_result["success"]:
        return error_response(parse_result["error"])
    
    # Analyze text
    analysis = lexiguard.analyze_text(parse_result["text"])
    
    if not analysis["success"]:
        return error_response(analysis.get("error", "Analysis failed"))
    
    # Add file info
    analysis["file_info"] = {
        "file_name": parse_result["file_name"],
        "file_type": parse_result["file_type"]
    }
    
    return jsonify(analysis)


@app.route("/analyze-clauses", methods=["POST"])
def analyze_clauses():
    """Perform detailed clause analysis"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    data = request.get_json()
    
    if not data or "text" not in data:
        return error_response("Missing 'text' field in request")
    
    clause_types = data.get("clause_types", None)
    
    result = lexiguard.analyze_clauses(data["text"], clause_types)
    
    if not result["success"]:
        return error_response(result.get("error", "Analysis failed"))
    
    return jsonify(result)


@app.route("/draft-negotiation", methods=["POST"])
def draft_negotiation():
    """Generate negotiation email"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    data = request.get_json()
    
    if not data or "document_text" not in data or "concerns" not in data:
        return error_response("Missing required fields: 'document_text' and 'concerns'")
    
    recipient_name = data.get("recipient_name", "Recipient")
    
    result = lexiguard.draft_negotiation_email(
        data["document_text"],
        data["concerns"],
        recipient_name
    )
    
    if not result["success"]:
        return error_response(result.get("error", "Email generation failed"))
    
    return jsonify(result)


@app.route("/draft-document-email", methods=["POST"])
def draft_document_email():
    """Generate document review email"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    data = request.get_json()
    
    if not data or "document_text" not in data:
        return error_response("Missing 'document_text' field")
    
    review_notes = data.get("review_notes", "")
    recipient_name = data.get("recipient_name", "Recipient")
    
    result = lexiguard.draft_document_review_email(
        data["document_text"],
        review_notes,
        recipient_name
    )
    
    if not result["success"]:
        return error_response(result.get("error", "Email generation failed"))
    
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    """Chat about document with context"""
    if not lexiguard:
        return error_response("LexiGuard SDK not initialized", 500)
    
    data = request.get_json()
    
    if not data or "message" not in data:
        return error_response("Missing 'message' field")
    
    document_context = data.get("document_context", None)
    
    result = lexiguard.chat(data["message"], document_context)
    
    if not result["success"]:
        return error_response(result.get("error", "Chat failed"))
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)