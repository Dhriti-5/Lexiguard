"""
DLP Processor: Handles PII redaction using Google Cloud DLP
Extracts text from documents and redacts sensitive information
"""

import io
from typing import Tuple
import google.cloud.dlp_v2 as dlp
from PyPDF2 import PdfReader
from docx import Document


class DLPProcessor:
    """Google Cloud DLP processor for PII redaction"""
    
    def __init__(self, project_id: str):
        """
        Initialize DLP client
        
        Args:
            project_id (str): Google Cloud project ID
        """
        self.project_id = project_id
        self.dlp_client = dlp.DlpServiceClient()
        self.parent = f"projects/{project_id}"
        
        # Define info types to redact
        self.info_types = [
            {"name": "PERSON_NAME"},
            {"name": "PHONE_NUMBER"},
            {"name": "EMAIL_ADDRESS"},
            {"name": "CREDIT_CARD_NUMBER"},
            {"name": "US_SOCIAL_SECURITY_NUMBER"},
            {"name": "DATE_OF_BIRTH"},
            {"name": "PASSPORT"},
            {"name": "STREET_ADDRESS"},
            {"name": "US_DRIVER_LICENSE_NUMBER"},
            {"name": "IP_ADDRESS"},
            {"name": "IBAN_CODE"},
            {"name": "MEDICAL_RECORD_NUMBER"},
        ]
    
    def redact_pii(self, document_content: bytes, file_type: str) -> Tuple[str, bool]:
        """
        Extract text from document and redact PII
        
        Args:
            document_content (bytes): Raw document bytes
            file_type (str): File type (pdf, docx, txt)
            
        Returns:
            Tuple[str, bool]: (redacted_text, pii_found)
        """
        try:
            # Step 1: Extract text from document
            extracted_text = self._extract_text(document_content, file_type)
            
            if not extracted_text:
                return "Error: Could not extract text from document", False
            
            # Step 2: Redact PII using DLP
            redacted_text, pii_found = self._redact_with_dlp(extracted_text)
            
            return redacted_text, pii_found
            
        except Exception as e:
            print(f"❌ Error in DLP processing: {str(e)}")
            return f"Error processing document: {str(e)}", False
    
    def _extract_text(self, document_content: bytes, file_type: str) -> str:
        """
        Extract text from different file types
        
        Args:
            document_content (bytes): Raw document bytes
            file_type (str): File type
            
        Returns:
            str: Extracted text
        """
        try:
            if file_type.lower() == 'pdf':
                return self._extract_from_pdf(document_content)
            elif file_type.lower() == 'docx':
                return self._extract_from_docx(document_content)
            elif file_type.lower() == 'txt':
                return document_content.decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}")
            raise
    
    def _extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        
        return text.strip()
    
    def _extract_from_docx(self, docx_bytes: bytes) -> str:
        """Extract text from DOCX"""
        docx_file = io.BytesIO(docx_bytes)
        doc = Document(docx_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    
    def _redact_with_dlp(self, text: str) -> Tuple[str, bool]:
        """
        Redact PII from text using Google Cloud DLP
        
        Args:
            text (str): Text to redact
            
        Returns:
            Tuple[str, bool]: (redacted_text, pii_found)
        """
        try:
            # Prepare DLP request
            item = {"value": text}
            
            # Configure inspection
            inspect_config = {
                "info_types": self.info_types,
                "min_likelihood": dlp.Likelihood.LIKELY,
                "limits": {"max_findings_per_request": 0}  # No limit
            }
            
            # Configure deidentification (redaction)
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_config": {
                                    "new_value": {"string_value": "[REDACTED]"}
                                }
                            }
                        }
                    ]
                }
            }
            
            # Call DLP API
            response = self.dlp_client.deidentify_content(
                request={
                    "parent": self.parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": item,
                }
            )
            
            # Check if PII was found
            redacted_text = response.item.value
            pii_found = redacted_text != text
            
            if pii_found:
                print(f"✅ PII detected and redacted")
            else:
                print(f"ℹ️  No PII detected in document")
            
            return redacted_text, pii_found
            
        except Exception as e:
            print(f"❌ DLP API error: {str(e)}")
            # Return original text if DLP fails (don't block processing)
            return text, False