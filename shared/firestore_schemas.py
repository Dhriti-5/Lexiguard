"""
Firestore document schemas for LexiGuard
Defines the structure of documents in Firestore collections
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class AnalysisJob:
    """
    Schema for analysisJobs collection
    Tracks async processing jobs
    """
    jobId: str                          # Unique job identifier
    userID: str                         # Firebase Auth UID
    documentTitle: str                  # User-provided document name
    originalFilename: str               # Original file name
    fileType: str                       # pdf, docx, txt
    gcsPath: str                        # Cloud Storage path to uploaded file
    status: str                         # pending, processing, completed, failed
    createdAt: datetime                 # Job creation timestamp
    updatedAt: datetime                 # Last update timestamp
    analysisType: str                   # "standard" or "detailed"
    
    # Optional fields (populated during/after processing)
    startedAt: Optional[datetime] = None      # When worker started processing
    completedAt: Optional[datetime] = None    # When processing finished
    errorMessage: Optional[str] = None        # Error details if failed
    resultAnalysisId: Optional[str] = None    # Link to userAnalyses doc
    processingTimeSeconds: Optional[float] = None  # Total processing time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary"""
        data = asdict(self)
        # Convert datetime objects to Firestore timestamps
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value
        return data


@dataclass
class AnalysisResult:
    """
    Schema for userAnalyses collection (existing structure + new fields)
    Stores the final analysis results
    """
    analysisId: str                     # Unique analysis ID
    userID: str                         # Firebase Auth UID
    documentTitle: str                  # Document name
    originalFilename: str               # Original file name
    uploadTimestamp: datetime           # When uploaded
    fileType: str                       # pdf, docx, txt
    piiRedacted: bool                   # Whether PII was redacted
    redactedDocumentText: str           # Redacted content
    
    # Analysis results
    analysisType: str                   # "standard" or "detailed"
    summary: Optional[str] = None       # AI-generated summary
    risks: Optional[list] = None        # List of identified risks
    recommendations: Optional[list] = None  # Security recommendations
    clauseAnalysis: Optional[Dict] = None   # Detailed clause analysis
    
    # Processing metadata
    jobId: Optional[str] = None         # Link back to job
    processingTimeSeconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-compatible dictionary"""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value
        return data


# Pub/Sub Message Schema
@dataclass
class JobMessage:
    """
    Schema for Pub/Sub messages
    Published by Cloud Function, consumed by Cloud Run worker
    """
    jobId: str              # Reference to analysisJobs document
    userID: str             # User who created the job
    gcsPath: str           # Path to file in Cloud Storage
    analysisType: str      # "standard" or "detailed"
    documentTitle: str     # For display purposes
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)