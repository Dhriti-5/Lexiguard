"""
Cloud Run Worker: Async Document Processor
Subscribes to Pub/Sub messages and processes documents asynchronously
"""

import os
import json
import time
import io
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import base64
import PyPDF2
from docx import Document as DocxDocument
import google.generativeai as genai
from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types as dlp_types

# Initialize Flask app
app = Flask(__name__)

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT', 'lexiguard-475609')
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'lexiguard-documents')
GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize clients
storage_client = storage.Client(project=PROJECT_ID)
db = firestore.Client(project=PROJECT_ID)

# Initialize DLP client
try:
    dlp_client = dlp_v2.DlpServiceClient()
    print("✅ DLP client initialized")
except Exception as e:
    print(f"⚠️ DLP client failed: {e}")
    dlp_client = None

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured with API key")
else:
    print("⚠️ No Gemini API key found")

try:
    model = genai.GenerativeModel(
        'models/gemini-2.0-flash-exp',
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
        }
    )
    print("✅ Gemini model initialized")
except Exception as e:
    print(f"⚠️ Gemini model init failed: {e}")
    model = None

print(f"\n{'='*60}")
print(f"✅ Cloud Run Worker Initialized")
print(f"   Project: {PROJECT_ID}")
print(f"   Bucket: {BUCKET_NAME}")
print(f"{'='*60}\n")

# DLP Configuration
INFO_TYPES_TO_REDACT = [
    dlp_types.InfoType(name="PERSON_NAME"),
    dlp_types.InfoType(name="EMAIL_ADDRESS"),
    dlp_types.InfoType(name="PHONE_NUMBER"),
    dlp_types.InfoType(name="STREET_ADDRESS"),
    dlp_types.InfoType(name="CREDIT_CARD_NUMBER"),
]

info_type_transformations = dlp_types.InfoTypeTransformations(
    transformations=[{
        "info_types": INFO_TYPES_TO_REDACT,
        "primitive_transformation": dlp_types.PrimitiveTransformation(
            replace_with_info_type_config=dlp_types.ReplaceWithInfoTypeConfig()
        ),
    }]
)

DEIDENTIFY_CONFIG = dlp_types.DeidentifyConfig(
    info_type_transformations=info_type_transformations
)

# AI Prompts
SUMMARY_PROMPT = """
You are LexiGuard, an expert AI assistant that explains complex legal documents in simple terms.

Analyze the following contract and provide a clear, well-structured summary.

Format your response using clean markdown:
- Use ## for main section headings
- Use **bold** for key terms, amounts, and dates
- Use simple bullet points (-) for lists
- Keep paragraphs short and readable

Structure your summary as follows:

## Document Overview
Brief explanation of what type of agreement this is and the parties involved.

## Key Responsibilities
**Party A Responsibilities:**
- List main obligations
- Highlight unusual terms

**Party B Responsibilities:**
- List main obligations
- Highlight unusual terms

## Duration and Important Dates
- Start date, end date, renewal terms, notice periods

## Financial Terms
- Payment amounts, fees, deposits, penalties

Use clear, simple language suitable for a non-lawyer.
"""

RISK_ANALYSIS_PROMPT = """
You are a meticulous risk analysis AI. Scan the provided legal document.
Identify potentially unfavorable, non-standard, or risky clauses.
Focus on: Indemnity, Limitation of Liability, Automatic Renewal, Termination Penalties, Non-Compete.

For each risk provide: clause_text, risk_explanation, severity (High/Medium/Low).

Respond ONLY with valid JSON:
{"risks": [{"clause_text": "...", "risk_explanation": "...", "severity": "High"}]}

If no risks: {"risks": []}
No markdown, no extra text.
"""


@app.route('/', methods=['POST'])
def process_pubsub_message():
    """Main endpoint for Pub/Sub push messages"""
    try:
        envelope = request.get_json()
        
        if not envelope or 'message' not in envelope:
            print("❌ Invalid Pub/Sub message format")
            return 'Bad Request: Invalid message', 400
        
        pubsub_message = envelope['message']
        
        if 'data' not in pubsub_message:
            print("❌ No data in Pub/Sub message")
            return 'Bad Request: No data', 400
        
        # Decode base64 message
        message_data = base64.b64decode(pubsub_message['data']).decode('utf-8')
        job_data = json.loads(message_data)
        
        print(f"\n{'='*60}")
        print(f"📨 Received Pub/Sub message")
        print(f"   Job ID: {job_data.get('jobId')}")
        print(f"   Document: {job_data.get('documentTitle')}")
        print(f"{'='*60}\n")
        
        # Process the job
        success = process_analysis_job(job_data)
        
        # Always return 200 to acknowledge message
        return 'OK', 200
            
    except Exception as e:
        print(f"❌ Error processing Pub/Sub message: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return 200 to prevent infinite retries
        return 'OK', 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'service': 'lexiguard-worker',
        'project': PROJECT_ID,
        'bucket': BUCKET_NAME,
        'dlp_available': dlp_client is not None,
        'gemini_available': model is not None
    }), 200


def process_analysis_job(job_data):
    """Process a single analysis job"""
    job_id = job_data.get('jobId')
    start_time = time.time()
    
    if not job_id:
        print("❌ No job ID in message")
        return False
    
    try:
        # Update status to processing
        print(f"🔄 Starting job {job_id}")
        update_job_status(job_id, 'processing', started_at=datetime.utcnow())
        
        print(f"   User: {job_data.get('userID')}")
        print(f"   Document: {job_data.get('documentTitle')}")
        print(f"   Analysis Type: {job_data.get('analysisType')}")
        print(f"   GCS Path: {job_data.get('gcsPath')}")
        
        # Step 1: Download from Cloud Storage
        print(f"\n⬇️  Step 1: Downloading from Cloud Storage...")
        file_bytes = download_from_gcs(job_data.get('gcsPath'))
        
        if not file_bytes:
            raise Exception("Failed to download file from Cloud Storage")
        
        print(f"   ✅ Downloaded {len(file_bytes)} bytes")
        
        # Step 2: Extract text based on file type
        print(f"\n📄 Step 2: Extracting text from {job_data.get('fileType')}...")
        document_text = extract_text(file_bytes, job_data.get('fileType'))
        
        if not document_text or len(document_text.strip()) < 50:
            raise Exception(f"Document text too short: {len(document_text)} characters")
        
        print(f"   ✅ Extracted {len(document_text)} characters")
        
        # Step 3: Redact PII with DLP
        print(f"\n🔒 Step 3: Redacting PII with Google DLP...")
        redacted_text, pii_found = redact_pii(document_text)
        print(f"   ✅ PII redaction: {pii_found}")
        
        # Step 4: Analyze with Gemini AI
        print(f"\n🤖 Step 4: Analyzing with Gemini AI...")
        analysis_results = analyze_with_gemini(
            redacted_text,
            job_data.get('analysisType', 'standard')
        )
        print(f"   ✅ Analysis complete")
        
        # Step 5: Save results to Firestore
        print(f"\n💾 Step 5: Saving results to Firestore...")
        analysis_id = save_analysis_results(
            job_data,
            redacted_text,
            pii_found,
            analysis_results
        )
        print(f"   ✅ Saved as: {analysis_id}")
        
        # Step 6: Update job status to completed
        processing_time = time.time() - start_time
        update_job_status(
            job_id,
            'completed',
            completed_at=datetime.utcnow(),
            result_analysis_id=analysis_id,
            processing_time_seconds=processing_time
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Job {job_id} completed successfully!")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Analysis ID: {analysis_id}")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        
        print(f"\n{'='*60}")
        print(f"❌ Job {job_id} FAILED")
        print(f"   Error: {error_message}")
        print(f"   Time: {processing_time:.2f}s")
        print(f"{'='*60}\n")
        
        import traceback
        print(traceback.format_exc())
        
        update_job_status(
            job_id,
            'failed',
            error_message=error_message,
            processing_time_seconds=processing_time
        )
        
        return False


def download_from_gcs(gcs_path):
    """Download file from Cloud Storage"""
    if not gcs_path:
        print("❌ No GCS path provided")
        return None
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        if not blob.exists():
            print(f"❌ File not found in GCS: {gcs_path}")
            return None
        
        content = blob.download_as_bytes()
        return content
        
    except Exception as e:
        print(f"❌ GCS download error: {e}")
        return None


def extract_text(file_bytes, file_type):
    """Extract text from file based on type"""
    if not file_bytes:
        raise Exception("No file bytes provided")
    
    try:
        file_type_lower = (file_type or 'txt').lower()
        
        if file_type_lower == 'pdf':
            pdf_file = io.BytesIO(file_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += page_text
                if page_num == 0:
                    print(f"   First page: {len(page_text)} chars")
            return text
            
        elif file_type_lower == 'docx':
            docx_file = io.BytesIO(file_bytes)
            doc = DocxDocument(docx_file)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            print(f"   Found {len(paragraphs)} paragraphs")
            return text
            
        elif file_type_lower == 'txt':
            try:
                return file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return file_bytes.decode('latin-1')
            
        else:
            raise Exception(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        print(f"❌ Text extraction error: {e}")
        raise Exception(f"Failed to extract text from {file_type}: {str(e)}")


def redact_pii(text):
    """Redact PII using Google DLP"""
    if not dlp_client:
        print("   ⚠️ DLP not available, skipping PII redaction")
        return text, False
    
    try:
        parent = f"projects/{PROJECT_ID}/locations/global"
        item = {"value": text[:50000]}  # Limit text size for DLP
        
        response = dlp_client.deidentify_content(
            request={
                "parent": parent,
                "deidentify_config": DEIDENTIFY_CONFIG,
                "inspect_config": {"info_types": INFO_TYPES_TO_REDACT},
                "item": item,
            }
        )
        
        redacted = response.item.value
        pii_found = redacted != text[:50000]
        
        # If text was truncated, append the rest
        if len(text) > 50000:
            redacted += text[50000:]
        
        return redacted, pii_found
        
    except Exception as e:
        print(f"   ⚠️ DLP error (continuing without redaction): {e}")
        return text, False


def analyze_with_gemini(text, analysis_type):
    """Analyze document with Gemini AI"""
    if not model:
        raise Exception("Gemini model not initialized")
    
    try:
        # Limit text length for API
        text_limited = text[:15000]
        
        # Generate summary
        print("   Generating summary...")
        summary_prompt = f"{SUMMARY_PROMPT}\n\nDocument:\n{text_limited}"
        summary_response = model.generate_content(summary_prompt)
        summary = summary_response.text.strip()
        print(f"   ✅ Summary: {len(summary)} chars")
        
        # Analyze risks
        print("   Analyzing risks...")
        risk_prompt = f"{RISK_ANALYSIS_PROMPT}\n\nDocument:\n{text_limited}"
        risk_response = model.generate_content(risk_prompt)
        
        risks = []
        try:
            risks_text = risk_response.text.strip()
            # Remove markdown formatting
            risks_text = risks_text.replace("```json", "").replace("```", "").strip()
            risks_data = json.loads(risks_text)
            risks = risks_data.get('risks', [])
            print(f"   ✅ Found {len(risks)} risks")
        except Exception as parse_error:
            print(f"   ⚠️ Failed to parse risks JSON: {parse_error}")
            risks = []
        
        # Generate recommendations
        recommendations = generate_recommendations(risks)
        print(f"   ✅ Generated {len(recommendations)} recommendations")
        
        return {
            'summary': summary,
            'risks': risks,
            'recommendations': recommendations,
            'clauseAnalysis': {}
        }
        
    except Exception as e:
        print(f"   ❌ Gemini analysis error: {e}")
        raise Exception(f"AI analysis failed: {str(e)}")


def generate_recommendations(risks):
    """Generate recommendations based on identified risks"""
    recommendations = []
    
    if not risks:
        return [
            "No significant risks detected in this document.",
            "Review all terms carefully before signing.",
            "Ensure you understand all obligations and responsibilities.",
            "Keep a signed copy for your records."
        ]
    
    high_risks = [r for r in risks if r.get('severity') == 'High']
    medium_risks = [r for r in risks if r.get('severity') == 'Medium']
    
    if high_risks:
        recommendations.append(f"⚠️ {len(high_risks)} high-risk clause(s) require immediate attention before signing.")
    
    if medium_risks:
        recommendations.append(f"Carefully review {len(medium_risks)} medium-risk clause(s) that may impact your rights.")
    
    recommendations.append("Consider consulting with legal counsel to review identified risks.")
    recommendations.append("Request written clarification on any unclear or concerning terms.")
    recommendations.append("Negotiate modifications to risky clauses before signing.")
    recommendations.append("Document all communications regarding contract modifications.")
    
    return recommendations


def save_analysis_results(job_data, redacted_text, pii_found, analysis_results):
    """Save analysis results to Firestore userAnalyses collection"""
    analysis_ref = db.collection('userAnalyses').document()
    analysis_id = analysis_ref.id
    
    analysis_doc = {
        'userID': job_data.get('userID'),
        'documentTitle': job_data.get('documentTitle'),
        'originalFilename': job_data.get('originalFilename', job_data.get('documentTitle')),
        'uploadTimestamp': firestore.SERVER_TIMESTAMP,
        'fileType': job_data.get('fileType'),
        'piiRedacted': pii_found,
        'redactedDocumentText': redacted_text[:50000],  # Limit size to avoid Firestore limits
        'analysisType': job_data.get('analysisType', 'standard'),
        'jobId': job_data.get('jobId'),
        
        # Analysis results from Gemini
        'summary': analysis_results.get('summary', ''),
        'risks': analysis_results.get('risks', []),
        'recommendations': analysis_results.get('recommendations', []),
        'clauseAnalysis': analysis_results.get('clauseAnalysis', {}),
    }
    
    analysis_ref.set(analysis_doc)
    return analysis_id


def update_job_status(job_id, status, **kwargs):
    """Update job status in Firestore analysisJobs collection"""
    try:
        job_ref = db.collection('analysisJobs').document(job_id)
        
        update_data = {
            'status': status,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }
        
        # Add optional fields
        if 'started_at' in kwargs:
            update_data['startedAt'] = kwargs['started_at']
        if 'completed_at' in kwargs:
            update_data['completedAt'] = kwargs['completed_at']
        if 'error_message' in kwargs:
            update_data['errorMessage'] = kwargs['error_message']
        if 'result_analysis_id' in kwargs:
            update_data['resultAnalysisId'] = kwargs['result_analysis_id']
        if 'processing_time_seconds' in kwargs:
            update_data['processingTimeSeconds'] = kwargs['processing_time_seconds']
        
        job_ref.update(update_data)
        
    except Exception as e:
        print(f"⚠️ Failed to update job status: {e}")


if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    print(f"🚀 Starting worker on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)