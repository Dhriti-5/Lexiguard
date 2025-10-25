"""
Cloud Run Worker: Async Document Processor
Subscribes to Pub/Sub messages and processes documents asynchronously
"""

import os
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
import base64

# Import local modules
from dlp_processor import DLPProcessor
from gemini_analyzer import GeminiAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Initialize clients
storage_client = storage.Client()
db = firestore.Client()

# Configuration
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'lexiguard-documents')
PROJECT_ID = os.environ.get('GCP_PROJECT')

# Initialize processors
dlp_processor = DLPProcessor(project_id=PROJECT_ID)
gemini_analyzer = GeminiAnalyzer()


@app.route('/', methods=['POST'])
def process_pubsub_message():
    """
    Main endpoint for Pub/Sub push messages
    Cloud Run receives POST requests from Pub/Sub subscription
    """
    try:
        # Extract Pub/Sub message
        envelope = request.get_json()
        
        if not envelope:
            return 'Bad Request: no Pub/Sub message received', 400
        
        if 'message' not in envelope:
            return 'Bad Request: invalid Pub/Sub message format', 400
        
        # Decode message data
        pubsub_message = envelope['message']
        
        if 'data' not in pubsub_message:
            return 'Bad Request: no data in Pub/Sub message', 400
        
        # Decode base64 message
        message_data = base64.b64decode(pubsub_message['data']).decode('utf-8')
        job_data = json.loads(message_data)
        
        print(f"📨 Received job: {job_data['jobId']}")
        
        # Process the job
        success = process_analysis_job(job_data)
        
        if success:
            return 'OK', 200
        else:
            # Return 200 to acknowledge message (prevents retry on permanent failures)
            # Errors are already logged in Firestore
            return 'Processed with errors', 200
            
    except Exception as e:
        print(f"❌ Error processing Pub/Sub message: {str(e)}")
        # Return 200 to prevent retries on malformed messages
        return f'Error: {str(e)}', 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy', 'service': 'lexiguard-worker'}), 200


def process_analysis_job(job_data):
    """
    Process a single analysis job
    
    Args:
        job_data (dict): Job details from Pub/Sub message
        
    Returns:
        bool: True if successful, False otherwise
    """
    job_id = job_data['jobId']
    start_time = time.time()
    
    try:
        # Update job status to 'processing'
        update_job_status(job_id, 'processing', started_at=datetime.utcnow())
        
        print(f"🔄 Processing job {job_id}")
        print(f"   User: {job_data['userID']}")
        print(f"   File: {job_data['documentTitle']}")
        print(f"   Type: {job_data['analysisType']}")
        
        # Step 1: Download document from Cloud Storage
        print(f"⬇️  Downloading from GCS: {job_data['gcsPath']}")
        document_content = download_from_gcs(job_data['gcsPath'])
        
        if not document_content:
            raise Exception("Failed to download document from Cloud Storage")
        
        # Step 2: Redact PII using Google DLP
        print(f"🔒 Redacting PII with DLP...")
        redacted_text, pii_found = dlp_processor.redact_pii(
            document_content,
            job_data['fileType']
        )
        
        print(f"   PII redacted: {pii_found}")
        
        # Step 3: Analyze with Gemini AI
        print(f"🤖 Analyzing with Gemini AI...")
        analysis_results = gemini_analyzer.analyze_document(
            redacted_text,
            job_data['analysisType'],
            job_data['documentTitle']
        )
        
        # Step 4: Save results to Firestore (userAnalyses collection)
        print(f"💾 Saving results to Firestore...")
        analysis_id = save_analysis_results(
            job_data=job_data,
            redacted_text=redacted_text,
            pii_found=pii_found,
            analysis_results=analysis_results
        )
        
        # Step 5: Update job status to 'completed'
        processing_time = time.time() - start_time
        update_job_status(
            job_id,
            'completed',
            completed_at=datetime.utcnow(),
            result_analysis_id=analysis_id,
            processing_time_seconds=processing_time
        )
        
        print(f"✅ Job {job_id} completed successfully in {processing_time:.2f}s")
        print(f"   Analysis ID: {analysis_id}")
        
        return True
        
    except Exception as e:
        # Log error and update job status
        processing_time = time.time() - start_time
        error_message = str(e)
        
        print(f"❌ Job {job_id} failed: {error_message}")
        
        update_job_status(
            job_id,
            'failed',
            error_message=error_message,
            processing_time_seconds=processing_time
        )
        
        return False


def download_from_gcs(gcs_path):
    """
    Download document from Cloud Storage
    
    Args:
        gcs_path (str): Path in GCS bucket (e.g., "uploads/file.pdf")
        
    Returns:
        bytes: Document content
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        
        # Download as bytes
        content = blob.download_as_bytes()
        
        return content
        
    except Exception as e:
        print(f"❌ Error downloading from GCS: {str(e)}")
        return None


def save_analysis_results(job_data, redacted_text, pii_found, analysis_results):
    """
    Save analysis results to Firestore userAnalyses collection
    
    Args:
        job_data (dict): Original job data
        redacted_text (str): PII-redacted document text
        pii_found (bool): Whether PII was detected
        analysis_results (dict): AI analysis results
        
    Returns:
        str: Analysis document ID
    """
    # Create new document in userAnalyses collection
    analysis_ref = db.collection('userAnalyses').document()
    analysis_id = analysis_ref.id
    
    analysis_doc = {
        'userID': job_data['userID'],
        'documentTitle': job_data['documentTitle'],
        'originalFilename': job_data.get('originalFilename', job_data['documentTitle']),
        'uploadTimestamp': firestore.SERVER_TIMESTAMP,
        'fileType': job_data['fileType'],
        'piiRedacted': pii_found,
        'redactedDocumentText': redacted_text,
        'analysisType': job_data['analysisType'],
        'jobId': job_data['jobId'],
        
        # Analysis results from Gemini
        'summary': analysis_results.get('summary', ''),
        'risks': analysis_results.get('risks', []),
        'recommendations': analysis_results.get('recommendations', []),
        'clauseAnalysis': analysis_results.get('clauseAnalysis', {}),
        
        # Metadata
        'processingTimeSeconds': None  # Will be updated by job
    }
    
    # Save to Firestore
    analysis_ref.set(analysis_doc)
    
    return analysis_id


def update_job_status(job_id, status, **kwargs):
    """
    Update job status in Firestore analysisJobs collection
    
    Args:
        job_id (str): Job document ID
        status (str): New status (processing, completed, failed)
        **kwargs: Additional fields to update (started_at, completed_at, error_message, etc.)
    """
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
    print(f"📝 Updated job {job_id} status to: {status}")


if __name__ == '__main__':
    # For local testing
    PORT = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=PORT, debug=True)