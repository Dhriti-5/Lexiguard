# lexiguard_sdk/core.py
"""
LexiGuard SDK - Core functionality for legal document analysis
"""

import google.generativeai as genai
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


class LexiGuardError(Exception):
    """Base exception for LexiGuard SDK"""
    pass


class LexiGuard:
    """
    Main SDK class for legal document analysis using Google Gemini AI.
    
    Usage:
        lg = LexiGuard(api_key="YOUR_API_KEY")
        result = lg.analyze_text("Contract text here...")
    """
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-2.5-flash"):
        """
        Initialize LexiGuard SDK.
        
        Args:
            api_key: Your Google Gemini API key
            model_name: Gemini model to use (default: gemini-1.5-flash)
        """
        if not api_key:
            raise LexiGuardError("API key is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def _generate_response(self, prompt: str) -> str:
        """
        Internal method to generate AI response.
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Generated text response
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise LexiGuardError(f"AI generation failed: {str(e)}")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze legal document text for key insights.
        
        Args:
            text: Legal document text to analyze
            
        Returns:
            Dictionary with analysis results including summary, risks, and recommendations
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Text cannot be empty"
            }
        
        prompt = f"""
        You are a legal document analysis expert. Analyze the following legal document text:
        
        {text}
        
        Provide a comprehensive analysis in JSON format with:
        1. "summary": Brief overview of the document
        2. "document_type": Type of legal document
        3. "key_clauses": List of important clauses
        4. "potential_risks": List of risks or concerning terms
        5. "recommendations": List of actionable recommendations
        6. "parties_involved": List of parties mentioned
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self._generate_response(prompt)
            # Clean response to extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            analysis = json.loads(response.strip())
            
            return {
                "success": True,
                "data": analysis
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse AI response: {str(e)}",
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_clauses(self, text: str, clause_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform detailed clause-by-clause analysis.
        
        Args:
            text: Legal document text
            clause_types: Optional list of specific clause types to focus on
            
        Returns:
            Dictionary with detailed clause analysis
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Text cannot be empty"
            }
        
        clause_focus = ""
        if clause_types:
            clause_focus = f"\nFocus especially on these clause types: {', '.join(clause_types)}"
        
        prompt = f"""
        Analyze the following legal document and break it down clause by clause:
        
        {text}
        {clause_focus}
        
        For each significant clause, provide:
        1. "clause_number": Sequential number
        2. "clause_title": Short descriptive title
        3. "clause_text": The actual clause text (excerpt)
        4. "analysis": Detailed analysis of what this clause means
        5. "risk_level": "low", "medium", or "high"
        6. "fairness_score": 1-10 (10 being most fair)
        7. "concerns": List of specific concerns if any
        
        Return as JSON with a "clauses" array.
        """
        
        try:
            response = self._generate_response(prompt)
            # Clean JSON response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            analysis = json.loads(response.strip())
            
            return {
                "success": True,
                "data": analysis,
                "total_clauses": len(analysis.get("clauses", []))
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_fairness(self, text: str) -> Dict[str, Any]:
        """
        Analyze document fairness and provide scoring.
        
        Args:
            text: Legal document text
            
        Returns:
            Dictionary with fairness scores and analysis
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Text cannot be empty"
            }
        
        prompt = f"""
        Analyze the fairness of this legal document:
        
        {text}
        
        Provide a fairness assessment in JSON format:
        1. "overall_fairness_score": 1-10 (10 being most fair)
        2. "balance_analysis": Analysis of balance between parties
        3. "one_sided_clauses": List of clauses that favor one party
        4. "red_flags": List of concerning terms or conditions
        5. "power_dynamics": Description of power balance
        6. "recommendations": How to improve fairness
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self._generate_response(prompt)
            # Clean JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            analysis = json.loads(response.strip())
            
            return {
                "success": True,
                "data": analysis
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def draft_negotiation_email(self, document_text: str, concerns: List[str], 
                                recipient_name: str = "Recipient") -> Dict[str, Any]:
        """
        Generate a professional negotiation email based on document concerns.
        
        Args:
            document_text: The legal document text
            concerns: List of specific concerns to address
            recipient_name: Name of the email recipient
            
        Returns:
            Dictionary with generated email content
        """
        if not document_text or not concerns:
            return {
                "success": False,
                "error": "Document text and concerns are required"
            }
        
        concerns_text = "\n".join([f"- {c}" for c in concerns])
        
        prompt = f"""
        Draft a professional, polite negotiation email regarding a legal document.
        
        Document context: {document_text[:500]}...
        
        Concerns to address:
        {concerns_text}
        
        The email should:
        1. Be addressed to {recipient_name}
        2. Be professional and diplomatic
        3. Clearly state the concerns
        4. Propose constructive solutions
        5. Request a discussion or revision
        
        Return JSON with:
        - "subject": Email subject line
        - "body": Full email body
        - "tone": The tone used (professional/diplomatic/etc)
        """
        
        try:
            response = self._generate_response(prompt)
            # Clean JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            email_data = json.loads(response.strip())
            
            return {
                "success": True,
                "data": email_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def draft_document_review_email(self, document_text: str, 
                                   review_notes: str,
                                   recipient_name: str = "Recipient") -> Dict[str, Any]:
        """
        Generate a professional email for document review sharing.
        
        Args:
            document_text: The legal document text
            review_notes: Your review notes or summary
            recipient_name: Name of the email recipient
            
        Returns:
            Dictionary with generated email content
        """
        if not document_text:
            return {
                "success": False,
                "error": "Document text is required"
            }
        
        prompt = f"""
        Draft a professional email to share a document review.
        
        Document context: {document_text[:500]}...
        Review notes: {review_notes}
        
        The email should:
        1. Be addressed to {recipient_name}
        2. Introduce the reviewed document
        3. Summarize key findings
        4. Be clear and actionable
        
        Return JSON with:
        - "subject": Email subject line
        - "body": Full email body
        """
        
        try:
            response = self._generate_response(prompt)
            # Clean JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            email_data = json.loads(response.strip())
            
            return {
                "success": True,
                "data": email_data
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def chat(self, message: str, document_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Interactive chat about a legal document.
        
        Args:
            message: User's question or message
            document_context: Optional document text for context
            
        Returns:
            Dictionary with chat response
        """
        if not message:
            return {
                "success": False,
                "error": "Message cannot be empty"
            }
        
        context_text = ""
        if document_context:
            context_text = f"\n\nDocument context:\n{document_context}\n\n"
        
        prompt = f"""
        You are a legal assistant helping users understand legal documents.
        {context_text}
        User question: {message}
        
        Provide a helpful, clear, and accurate response. If the question is about the document,
        reference specific parts of it. Be concise but thorough.
        """
        
        try:
            response = self._generate_response(prompt)
            
            return {
                "success": True,
                "data": {
                    "message": message,
                    "response": response,
                    "has_context": document_context is not None
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }