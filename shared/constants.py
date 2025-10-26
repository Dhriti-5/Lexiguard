"""
Shared constants for LexiGuard async processing
Used across Cloud Functions, Cloud Run workers, and backend
"""

# Job Status Constants
class JobStatus:
    PENDING = "pending"          # Job created, waiting for processing
    PROCESSING = "processing"    # Worker picked up the job
    COMPLETED = "completed"      # Analysis finished successfully
    FAILED = "failed"           # Job failed (with error details)

# Pub/Sub Configuration
PUBSUB_TOPIC_NAME = "document-analysis-jobs"
PUBSUB_SUBSCRIPTION_NAME = "document-analysis-worker"

# Firestore Collections
COLLECTION_USER_ANALYSES = "userAnalyses"
COLLECTION_ANALYSIS_JOBS = "analysisJobs"  # New collection for job tracking

# Cloud Storage
GCS_BUCKET_NAME = "lexiguard-documents"  # Update with your bucket name
UPLOAD_FOLDER = "uploads"                # Folder for user uploads
PROCESSED_FOLDER = "processed"           # Folder for processed docs

# File Processing
MAX_FILE_SIZE_MB = 10
SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt"]
ALLOWED_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain"
]

# Gemini Configuration
GEMINI_MODEL = "models/gemini-2.5-flash"
GEMINI_MAX_TOKENS = 8000

# DLP Configuration
DLP_MIN_LIKELIHOOD = "LIKELY"
DLP_MAX_FINDINGS = 0  # 0 means no limit

# Retry Configuration
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5