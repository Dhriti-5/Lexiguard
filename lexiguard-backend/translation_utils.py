# translation_utils.py
import os
import logging
from google.cloud import translate_v2 as translate
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Initialize Translation client
translate_client = None

def get_translate_client():
    """Lazy initialization of Translation client"""
    global translate_client
    if translate_client is None:
        try:
            translate_client = translate.Client()
            logger.info("✅ Google Cloud Translation client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Translation client: {e}")
            logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set in .env")
    return translate_client

# Extended language codes mapping - All Indian Regional Languages + International
SUPPORTED_LANGUAGES = {
    # English
    'en': 'English',
    
    # Major Indian Languages (Official Languages)
    'hi': 'हिंदी (Hindi)',
    'bn': 'বাংলা (Bengali)',
    'te': 'తెలుగు (Telugu)',
    'mr': 'मराठी (Marathi)',
    'ta': 'தமிழ் (Tamil)',
    'gu': 'ગુજરાતી (Gujarati)',
    'kn': 'ಕನ್ನಡ (Kannada)',
    'ml': 'മലയാളം (Malayalam)',
    'pa': 'ਪੰਜਾਬੀ (Punjabi)',
    'or': 'ଓଡ଼ିଆ (Odia)',
    'as': 'অসমীয়া (Assamese)',
    'ur': 'اردو (Urdu)',
    
    # Additional Indian Languages
    'sd': 'سنڌي (Sindhi)',
    'ne': 'नेपाली (Nepali)',
    'sa': 'संस्कृत (Sanskrit)',
    'kok': 'कोंकणी (Konkani)',
    'mai': 'मैथिली (Maithili)',
    'doi': 'डोगरी (Dogri)',
    'mni': 'মৈতৈলোন্ (Manipuri)',
    'sat': 'ᱥᱟᱱᱛᱟᱲᱤ (Santali)',
    'ks': 'کٲشُر (Kashmiri)',
    
    # International Languages (commonly used in India)
    'es': 'Español (Spanish)',
    'fr': 'Français (French)',
    'de': 'Deutsch (German)',
    'pt': 'Português (Portuguese)',
    'ru': 'Русский (Russian)',
    'ja': '日本語 (Japanese)',
    'zh-CN': '中文 (Chinese Simplified)',
    'ar': 'العربية (Arabic)',
    'ko': '한국어 (Korean)',
    'it': 'Italiano (Italian)',
}

# Regional categorization for better UX
LANGUAGE_CATEGORIES = {
    'North Indian Languages': ['hi', 'pa', 'ur', 'ks', 'sd', 'ne', 'doi'],
    'East Indian Languages': ['bn', 'as', 'or', 'mni', 'sat', 'mai'],
    'West Indian Languages': ['gu', 'mr', 'kok'],
    'South Indian Languages': ['te', 'ta', 'kn', 'ml'],
    'Classical Languages': ['sa'],
    'International Languages': ['en', 'es', 'fr', 'de', 'pt', 'ru', 'ja', 'zh-CN', 'ar', 'ko', 'it'],
}

def translate_text(text: str, target_language: str, source_language: str = 'en') -> Optional[str]:
    """
    Translate text using Google Cloud Translation API
    
    Args:
        text: Text to translate
        target_language: Target language code (hi, bn, te, etc.)
        source_language: Source language code (default: en)
        
    Returns:
        Translated text or None if translation fails
    """
    if not text or not text.strip():
        return text
    
    # Don't translate if target is same as source
    if target_language == source_language:
        return text
    
    # Validate language code
    if target_language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language: {target_language}")
        return text
    
    try:
        client = get_translate_client()
        if not client:
            logger.error("Translation client not available")
            return text
        
        # Perform translation
        result = client.translate(
            text,
            target_language=target_language,
            source_language=source_language,
            format_='text'
        )
        
        translated_text = result['translatedText']
        logger.info(f"✅ Translated text to {target_language}")
        return translated_text
        
    except Exception as e:
        logger.error(f"❌ Translation error for {target_language}: {e}")
        return text  # Return original text if translation fails

def translate_summary(summary: str, target_language: str) -> str:
    """Translate document summary"""
    return translate_text(summary, target_language) or summary

def translate_risks(risks: List[Dict], target_language: str) -> List[Dict]:
    """
    Translate risks array (standard analysis format)
    
    Args:
        risks: List of risk objects with clause_text and risk_explanation
        target_language: Target language code
        
    Returns:
        List of risk objects with translated fields
    """
    translated_risks = []
    
    for risk in risks:
        translated_risk = risk.copy()
        
        # Translate clause_text
        if 'clause_text' in risk and risk['clause_text']:
            translated_risk['clause_text'] = translate_text(
                risk['clause_text'], 
                target_language
            ) or risk['clause_text']
        
        # Translate risk_explanation
        if 'risk_explanation' in risk and risk['risk_explanation']:
            translated_risk['risk_explanation'] = translate_text(
                risk['risk_explanation'], 
                target_language
            ) or risk['risk_explanation']
        
        translated_risks.append(translated_risk)
    
    return translated_risks

def translate_clauses(clauses: List[Dict], target_language: str) -> List[Dict]:
    """
    Translate clauses array (detailed analysis format)
    
    Args:
        clauses: List of clause objects with multiple text fields
        target_language: Target language code
        
    Returns:
        List of clause objects with translated fields
    """
    translated_clauses = []
    
    for clause in clauses:
        translated_clause = clause.copy()
        
        # Translate each text field
        text_fields = ['clause', 'impact', 'recommendation', 'explanation']
        
        for field in text_fields:
            if field in clause and clause[field]:
                translated_clause[field] = translate_text(
                    clause[field], 
                    target_language
                ) or clause[field]
        
        translated_clauses.append(translated_clause)
    
    return translated_clauses

def translate_negotiation_email(email_text: str, target_language: str) -> str:
    """Translate negotiation email"""
    return translate_text(email_text, target_language) or email_text

def translate_suggestions(suggestions: List[str], target_language: str) -> List[str]:
    """Translate suggestions array"""
    translated_suggestions = []
    
    for suggestion in suggestions:
        translated = translate_text(suggestion, target_language)
        translated_suggestions.append(translated or suggestion)
    
    return translated_suggestions

def translate_analysis_content(analysis_data: Dict, target_language: str) -> Dict:
    """
    Translate all AI-generated content in an analysis document
    
    Args:
        analysis_data: Complete analysis document from Firestore
        target_language: Target language code
        
    Returns:
        Dictionary with all translated content
    """
    translated_content = {}
    
    # Translate summary
    if 'summary' in analysis_data and analysis_data['summary']:
        translated_content['summary'] = translate_summary(
            analysis_data['summary'], 
            target_language
        )
    
    # Translate risks (standard analysis)
    if 'risks' in analysis_data and analysis_data['risks']:
        translated_content['risks'] = translate_risks(
            analysis_data['risks'], 
            target_language
        )
    
    # Translate clauses (detailed analysis)
    if 'clauses' in analysis_data and analysis_data['clauses']:
        translated_content['clauses'] = translate_clauses(
            analysis_data['clauses'], 
            target_language
        )
    
    # Translate suggestions
    if 'suggestions' in analysis_data and analysis_data['suggestions']:
        translated_content['suggestions'] = translate_suggestions(
            analysis_data['suggestions'], 
            target_language
        )
    
    logger.info(f"✅ Translated analysis content to {target_language}")
    return translated_content

def get_language_categories():
    """Get languages organized by categories for better UI"""
    return LANGUAGE_CATEGORIES

def get_supported_languages_list():
    """Get list of all supported languages"""
    return [
        {"code": code, "name": name, "category": get_language_category(code)}
        for code, name in SUPPORTED_LANGUAGES.items()
    ]

def get_language_category(language_code: str) -> str:
    """Get category for a language code"""
    for category, codes in LANGUAGE_CATEGORIES.items():
        if language_code in codes:
            return category
    return 'Other'