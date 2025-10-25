"""
Gemini Analyzer: AI-powered document analysis using Google Gemini
Performs contract analysis, risk assessment, and clause extraction
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Any


class GeminiAnalyzer:
    """Gemini AI analyzer for document processing"""
    
    def __init__(self):
        """Initialize Gemini AI with API key"""
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def analyze_document(
        self, 
        document_text: str, 
        analysis_type: str,
        document_title: str
    ) -> Dict[str, Any]:
        """
        Analyze document using Gemini AI
        
        Args:
            document_text (str): Redacted document text
            analysis_type (str): "standard" or "detailed"
            document_title (str): Document name for context
            
        Returns:
            Dict containing analysis results
        """
        try:
            if analysis_type == "detailed":
                return self._detailed_analysis(document_text, document_title)
            else:
                return self._standard_analysis(document_text, document_title)
                
        except Exception as e:
            print(f"❌ Gemini analysis error: {str(e)}")
            return {
                'summary': f"Error analyzing document: {str(e)}",
                'risks': [],
                'recommendations': [],
                'clauseAnalysis': {}
            }
    
    def _standard_analysis(self, text: str, title: str) -> Dict[str, Any]:
        """
        Standard analysis: Quick summary and key risks
        
        Args:
            text (str): Document text
            title (str): Document title
            
        Returns:
            Dict with summary and risks
        """
        prompt = f"""
You are a legal document analyst. Analyze the following document titled "{title}".

Document Content:
{text[:6000]}  # Limit to prevent token overflow

Provide a JSON response with the following structure:
{{
    "summary": "A concise 2-3 sentence summary of the document",
    "risks": [
        "Risk 1: Brief description",
        "Risk 2: Brief description",
        "Risk 3: Brief description"
    ],
    "recommendations": [
        "Recommendation 1",
        "Recommendation 2"
    ]
}}

Focus on:
- Key obligations and rights
- Potential legal or financial risks
- Important dates or deadlines
- Unusual or unfavorable terms

Return ONLY valid JSON, no additional text.
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean response (remove markdown if present)
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            analysis = json.loads(result_text)
            
            # Add empty clauseAnalysis for consistency
            analysis['clauseAnalysis'] = {}
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error, returning text response")
            return {
                'summary': response.text[:500],
                'risks': ['Unable to parse structured analysis'],
                'recommendations': [],
                'clauseAnalysis': {}
            }
    
    def _detailed_analysis(self, text: str, title: str) -> Dict[str, Any]:
        """
        Detailed analysis: Comprehensive clause-by-clause review
        
        Args:
            text (str): Document text
            title (str): Document title
            
        Returns:
            Dict with detailed analysis including clause breakdown
        """
        prompt = f"""
You are an expert legal analyst. Perform a comprehensive analysis of this document titled "{title}".

Document Content:
{text[:8000]}  # Allow more tokens for detailed analysis

Provide a detailed JSON response:
{{
    "summary": "Comprehensive 4-5 sentence summary covering purpose, parties, and key terms",
    "risks": [
        "High Risk: Description and impact",
        "Medium Risk: Description",
        "Low Risk: Description"
    ],
    "recommendations": [
        "Priority 1: Specific action item",
        "Priority 2: Specific action item",
        "Priority 3: Specific action item"
    ],
    "clauseAnalysis": {{
        "Payment Terms": {{
            "content": "Summary of payment terms",
            "risk_level": "Low/Medium/High",
            "concerns": ["Concern 1", "Concern 2"]
        }},
        "Liability": {{
            "content": "Summary of liability clauses",
            "risk_level": "Low/Medium/High",
            "concerns": ["Concern 1"]
        }},
        "Termination": {{
            "content": "Summary of termination clauses",
            "risk_level": "Low/Medium/High",
            "concerns": []
        }},
        "Confidentiality": {{
            "content": "Summary of confidentiality terms",
            "risk_level": "Low/Medium/High",
            "concerns": []
        }},
        "Intellectual Property": {{
            "content": "Summary of IP terms",
            "risk_level": "Low/Medium/High",
            "concerns": []
        }}
    }}
}}

Analyze key clauses including:
- Payment terms and amounts
- Liability and indemnification
- Termination conditions
- Confidentiality obligations
- Intellectual property rights
- Dispute resolution
- Governing law
- Force majeure
- Non-compete/Non-solicitation

Classify risks as High (immediate legal/financial exposure), Medium (potential issues), or Low (minor concerns).

Return ONLY valid JSON, no additional text.
"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean response
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            analysis = json.loads(result_text)
            
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error in detailed analysis")
            # Fallback to standard analysis
            return self._standard_analysis(text, title)
        except Exception as e:
            print(f"❌ Detailed analysis error: {str(e)}")
            return {
                'summary': f"Error performing detailed analysis: {str(e)}",
                'risks': [],
                'recommendations': [],
                'clauseAnalysis': {}
            }