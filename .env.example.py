# .env.example
# Copy this file to .env and fill in your actual values
# NEVER commit .env to git!

GEMINI_API_KEY=your_google_gemini_api_key_here
MODEL_NAME=gemini-1.5-flash
DEBUG=True

# Server Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

# Optional: Rate Limiting
MAX_REQUESTS_PER_MINUTE=60

# Optional: File Upload Limits (in MB)
MAX_FILE_SIZE=10

# ========================================
# config.py - Configuration Management
# ========================================

"""
Configuration management for LexiGuard SDK
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration"""
    
    # API Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    
    # Server Configuration
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10"))  # MB
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    MAX_REQUESTS_PER_MINUTE = 30  # Stricter in production


# Factory function to get config
def get_config(env: Optional[str] = None) -> Config:
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name ('development', 'production')
        
    Returns:
        Configuration object
    """
    if env is None:
        env = os.getenv("ENVIRONMENT", "development")
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
    }
    
    config_class = configs.get(env.lower(), DevelopmentConfig)
    config = config_class()
    config.validate()
    
    return config


# ========================================
# example_usage.py - Usage Examples
# ========================================

"""
Example usage scripts for LexiGuard SDK
"""

from lexiguard_sdk import LexiGuard, FileParser, analyze_file_quick
from config import get_config


def example_1_basic_analysis():
    """Example 1: Basic text analysis"""
    print("="*50)
    print("Example 1: Basic Text Analysis")
    print("="*50)
    
    config = get_config()
    lg = LexiGuard(api_key=config.GEMINI_API_KEY)
    
    sample_text = """
    This Software License Agreement is made on January 1, 2025.
    The Licensee agrees to pay $10,000 annually for software usage.
    The term is 2 years with automatic renewal. Either party may
    terminate with 30 days notice.
    """
    
    result = lg.analyze_text(sample_text)
    
    if result["success"]:
        print("✅ Analysis successful!")
        print(f"Document Type: {result['data']['document_type']}")
        print(f"Summary: {result['data']['summary']}")
    else:
        print(f"❌ Error: {result['error']}")


def example_2_file_analysis():
    """Example 2: File analysis"""
    print("\n" + "="*50)
    print("Example 2: File Analysis")
    print("="*50)
    
    config = get_config()
    
    # Method 1: Manual parsing
    parse_result = FileParser.parse_file("sample_contract.pdf")
    
    if parse_result["success"]:
        lg = LexiGuard(api_key=config.GEMINI_API_KEY)
        analysis = lg.analyze_text(parse_result["text"])
        print(f"✅ File analyzed: {parse_result['file_name']}")
    
    # Method 2: Quick analysis
    result = analyze_file_quick(
        api_key=config.GEMINI_API_KEY,
        file_path="sample_contract.pdf"
    )
    
    if result["success"]:
        print("✅ Quick analysis complete!")


def example_3_clause_analysis():
    """Example 3: Detailed clause analysis"""
    print("\n" + "="*50)
    print("Example 3: Clause Analysis")
    print("="*50)
    
    config = get_config()
    lg = LexiGuard(api_key=config.GEMINI_API_KEY)
    
    contract = "Your detailed contract text..."
    
    result = lg.analyze_clauses(
        text=contract,
        clause_types=["payment", "termination", "liability"]
    )
    
    if result["success"]:
        print(f"✅ Found {result['total_clauses']} clauses")
        
        for clause in result['data']['clauses']:
            risk = clause.get('risk_level', 'unknown')
            print(f"  [{risk.upper()}] {clause['clause_title']}")


def example_4_fairness_scoring():
    """Example 4: Fairness analysis"""
    print("\n" + "="*50)
    print("Example 4: Fairness Scoring")
    print("="*50)
    
    config = get_config()
    lg = LexiGuard(api_key=config.GEMINI_API_KEY)
    
    contract = "Your contract text..."
    
    result = lg.analyze_fairness(contract)
    
    if result["success"]:
        score = result['data']['overall_fairness_score']
        print(f"✅ Fairness Score: {score}/10")
        
        if score < 5:
            print("⚠️  WARNING: This contract may be one-sided!")


def example_5_email_generation():
    """Example 5: Generate negotiation email"""
    print("\n" + "="*50)
    print("Example 5: Email Generation")
    print("="*50)
    
    config = get_config()
    lg = LexiGuard(api_key=config.GEMINI_API_KEY)
    
    concerns = [
        "Payment terms are unclear",
        "Termination clause favors the vendor",
        "Liability cap is too low"
    ]
    
    result = lg.draft_negotiation_email(
        document_text="Contract text...",
        concerns=concerns,
        recipient_name="John Smith"
    )
    
    if result["success"]:
        print("✅ Email generated!")
        print(f"\nSubject: {result['data']['subject']}")
        print(f"\nBody:\n{result['data']['body']}")


def example_6_batch_processing():
    """Example 6: Batch process multiple files"""
    print("\n" + "="*50)
    print("Example 6: Batch Processing")
    print("="*50)
    
    config = get_config()
    lg = LexiGuard(api_key=config.GEMINI_API_KEY)
    
    files = ["contract1.pdf", "contract2.pdf", "contract3.pdf"]
    
    results = []
    for file in files:
        print(f"Processing {file}...")
        result = analyze_file_quick(config.GEMINI_API_KEY, file)
        
        if result["success"]:
            results.append({
                "file": file,
                "type": result["data"]["document_type"],
                "risks": len(result["data"]["potential_risks"])
            })
    
    # Summary report
    print("\n📊 Summary Report:")
    for r in results:
        print(f"  {r['file']}: {r['type']} ({r['risks']} risks)")


if __name__ == "__main__":
    # Run all examples
    try:
        example_1_basic_analysis()
        # example_2_file_analysis()  # Uncomment if you have sample files
        example_3_clause_analysis()
        example_4_fairness_scoring()
        example_5_email_generation()
        # example_6_batch_processing()  # Uncomment if you have multiple files
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


# ========================================
# .gitignore - Git Ignore File
# ========================================

# Copy this content to .gitignore file

"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/

# API Keys (extra safety)
*api_key*
*secret*
credentials.json

# Temporary files
tmp/
temp/
*.tmp

# OS
Thumbs.db
"""