# 📁 LexiGuard SDK - Complete Directory Structure

This document shows the complete file structure for your LexiGuard project.

## 🗂️ Full Directory Tree

```
lexiguard-project/
│
├── lexiguard_sdk/                 # Core SDK (framework-agnostic)
│   ├── __init__.py               # Package initialization & exports
│   ├── core.py                   # Main LexiGuard class with AI logic
│   └── file_utils.py             # File parsing utilities (PDF, DOCX, TXT)
│
├── fastapi_app/                   # FastAPI web server wrapper
│   ├── __init__.py               # Makes it a package
│   └── main.py                   # FastAPI endpoints
│
├── flask_app/                     # Flask web server wrapper
│   ├── __init__.py               # Makes it a package
│   └── app.py                    # Flask endpoints
│
├── tests/                         # Unit tests (optional but recommended)
│   ├── __init__.py
│   ├── test_core.py              # Tests for core SDK
│   ├── test_file_utils.py        # Tests for file parsing
│   ├── test_fastapi.py           # Tests for FastAPI endpoints
│   └── test_flask.py             # Tests for Flask endpoints
│
├── examples/                      # Usage examples
│   ├── example_usage.py          # SDK usage examples
│   ├── batch_analysis.py         # Batch processing example
│   └── simple_frontend.html      # Simple web interface
│
├── docs/                          # Documentation
│   ├── API.md                    # API reference
│   ├── TUTORIAL.md               # Beginner tutorial
│   └── DEPLOYMENT.md             # Deployment guide
│
├── .env                          # Environment variables (DO NOT COMMIT!)
├── .env.example                  # Example env file (safe to commit)
├── .gitignore                    # Git ignore rules
├── config.py                     # Configuration management
├── setup.py                      # Package setup for distribution
├── requirements.txt              # Core dependencies
├── requirements-fastapi.txt      # FastAPI dependencies
├── requirements-flask.txt        # Flask dependencies
├── requirements-dev.txt          # Development dependencies
├── README.md                     # Main documentation
├── LICENSE                       # License file (e.g., MIT)
└── CHANGELOG.md                  # Version history
```

## 📝 File-by-File Breakdown

### 1. Core SDK Files

#### `lexiguard_sdk/__init__.py`
```python
# Purpose: Makes lexiguard_sdk a Python package
# Exports: LexiGuard, FileParser, etc.
# Size: ~20 lines
```

#### `lexiguard_sdk/core.py`
```python
# Purpose: Main SDK class with all AI analysis methods
# Contains:
#   - LexiGuard class
#   - analyze_text()
#   - analyze_clauses()
#   - analyze_fairness()
#   - draft_negotiation_email()
#   - draft_document_review_email()
#   - chat()
# Size: ~300 lines
```

#### `lexiguard_sdk/file_utils.py`
```python
# Purpose: File parsing utilities
# Contains:
#   - FileParser class
#   - parse_pdf()
#   - parse_docx()
#   - parse_txt()
#   - parse_file()
#   - parse_uploaded_file()
# Size: ~200 lines
```

### 2. Web Server Files

#### `fastapi_app/main.py`
```python
# Purpose: FastAPI web server
# Contains: All REST API endpoints
# Features:
#   - Async support
#   - Automatic documentation
#   - Request validation with Pydantic
# Size: ~200 lines
```

#### `flask_app/app.py`
```python
# Purpose: Flask web server
# Contains: All REST API endpoints
# Features:
#   - Synchronous
#   - Simple and straightforward
#   - CORS support
# Size: ~200 lines
```

### 3. Configuration Files

#### `.env` (Create this, don't commit!)
```bash
GEMINI_API_KEY=your_actual_api_key_here
MODEL_NAME=gemini-1.5-flash
DEBUG=True
```

#### `.env.example` (Safe to commit)
```bash
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-1.5-flash
DEBUG=True
```

#### `config.py`
```python
# Purpose: Configuration management
# Features:
#   - Load from .env
#   - Development/Production configs
#   - Validation
# Size: ~100 lines
```

### 4. Dependency Files

#### `requirements.txt` (Core SDK)
```
google-generativeai>=0.3.0
PyPDF2>=3.0.0
python-docx>=0.8.11
```

#### `requirements-fastapi.txt` (FastAPI Server)
```
google-generativeai>=0.3.0
PyPDF2>=3.0.0
python-docx>=0.8.11
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6
```

#### `requirements-flask.txt` (Flask Server)
```
google-generativeai>=0.3.0
PyPDF2>=3.0.0
python-docx>=0.8.11
flask>=2.3.0
flask-cors>=4.0.0
```

#### `requirements-dev.txt` (Development)
```
# All core requirements plus:
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.7.0
isort>=5.12.0
mypy>=1.4.0
```

### 5. Setup & Distribution

#### `setup.py`
```python
# Purpose: Package distribution setup
# Use: pip install -e .
# Features:
#   - Defines package metadata
#   - Lists dependencies
#   - Enables installation
# Size: ~60 lines
```

### 6. Documentation Files

#### `README.md`
```markdown
# Purpose: Main documentation
# Contains:
#   - Overview
#   - Installation instructions
#   - Quick start guide
#   - API reference
#   - Examples
# Size: ~500 lines
```

#### `LICENSE`
```
# Purpose: Software license
# Suggested: MIT License (open source)
```

#### `CHANGELOG.md`
```markdown
# Purpose: Track version changes
# Format:
## [1.0.0] - 2025-01-01
### Added
- Initial release
```

### 7. Git Configuration

#### `.gitignore`
```
# Purpose: Files to exclude from Git
# Contains:
#   - __pycache__/
#   - .env
#   - *.pyc
#   - venv/
# Size: ~50 lines
```

## 🚀 Setup Steps (In Order)

### Step 1: Create Directory Structure
```bash
mkdir -p lexiguard-project/{lexiguard_sdk,fastapi_app,flask_app,tests,examples,docs}
cd lexiguard-project
```

### Step 2: Create Core Files
```bash
touch lexiguard_sdk/{__init__.py,core.py,file_utils.py}
touch fastapi_app/{__init__.py,main.py}
touch flask_app/{__init__.py,app.py}
```

### Step 3: Create Configuration
```bash
touch .env.example .gitignore config.py
```

### Step 4: Create Requirements
```bash
touch requirements.txt requirements-fastapi.txt requirements-flask.txt requirements-dev.txt
```

### Step 5: Create Documentation
```bash
touch README.md LICENSE CHANGELOG.md setup.py
```

### Step 6: Initialize Git
```bash
git init
git add .gitignore .env.example README.md
git commit -m "Initial commit"
```

## 📦 Installation Commands

### For End Users (SDK Only)
```bash
# Clone repository
git clone https://github.com/yourusername/lexiguard-sdk.git
cd lexiguard-sdk

# Install core dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### For FastAPI Development
```bash
# Install with FastAPI dependencies
pip install -r requirements-fastapi.txt

# Set API key
export GEMINI_API_KEY="your_key"

# Run server
cd fastapi_app
python main.py
```

### For Flask Development
```bash
# Install with Flask dependencies
pip install -r requirements-flask.txt

# Set API key
export GEMINI_API_KEY="your_key"

# Run server
cd flask_app
python app.py
```

### For Contributors (Full Development)
```bash
# Install all dependencies including dev tools
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

## 🔧 Optional Files to Add

### Docker Support

#### `Dockerfile` (FastAPI)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

COPY lexiguard_sdk/ ./lexiguard_sdk/
COPY fastapi_app/ ./fastapi_app/

ENV GEMINI_API_KEY=""
EXPOSE 8000

CMD ["python", "fastapi_app/main.py"]
```

#### `docker-compose.yml`
```yaml
version: '3.8'
services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./lexiguard_sdk:/app/lexiguard_sdk
      - ./fastapi_app:/app/fastapi_app
```

### CI/CD

#### `.github/workflows/test.yml`
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/
      - name: Check code style
        run: |
          black --check .
          isort --check .
```

### Testing

#### `tests/test_core.py`
```python
import pytest
from lexiguard_sdk import LexiGuard

def test_lexiguard_initialization():
    lg = LexiGuard(api_key="test_key")
    assert lg is not None

def test_analyze_text_empty():
    lg = LexiGuard(api_key="test_key")
    result = lg.analyze_text("")
    assert result["success"] == False
    assert "error" in result
```

#### `pytest.ini`
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

## 📊 File Size Summary

| Component | Files | Total Lines | Purpose |
|-----------|-------|-------------|---------|
| Core SDK | 3 | ~520 | Main functionality |
| FastAPI | 1 | ~200 | Web API (async) |
| Flask | 1 | ~200 | Web API (sync) |
| Config | 1 | ~100 | Configuration |
| Setup | 1 | ~60 | Package setup |
| Tests | 4 | ~200 | Quality assurance |
| Docs | 3 | ~1000 | Documentation |
| **Total** | **14** | **~2280** | Full project |

## 🎯 Development Workflow

### 1. Daily Development
```bash
# Start your day
cd lexiguard-project
source venv/bin/activate  # or venv\Scripts\activate on Windows
git pull

# Make changes to files
# ...

# Test your changes
pytest tests/

# Format code
black .
isort .

# Commit
git add .
git commit -m "Description of changes"
git push
```

### 2. Adding a New Feature

**Example: Add document comparison feature**

1. **Update SDK** (`lexiguard_sdk/core.py`):
```python
def compare_documents(self, doc1: str, doc2: str) -> Dict[str, Any]:
    """Compare two documents"""
    prompt = f"Compare these documents:\nDoc 1: {doc1}\nDoc 2: {doc2}"
    response = self._generate_response(prompt)
    return {"success": True, "data": response}
```

2. **Add FastAPI endpoint** (`fastapi_app/main.py`):
```python
class CompareRequest(BaseModel):
    doc1: str
    doc2: str

@app.post("/compare")
async def compare_documents(request: CompareRequest):
    result = lexiguard.compare_documents(request.doc1, request.doc2)
    return result
```

3. **Add Flask endpoint** (`flask_app/app.py`):
```python
@app.route("/compare", methods=["POST"])
def compare_documents():
    data = request.get_json()
    result = lexiguard.compare_documents(data["doc1"], data["doc2"])
    return jsonify(result)
```

4. **Add tests** (`tests/test_core.py`):
```python
def test_compare_documents():
    lg = LexiGuard(api_key="test_key")
    result = lg.compare_documents("Doc 1", "Doc 2")
    assert result["success"] == True
```

5. **Update documentation** (`README.md`):
```markdown
### Compare Documents
```python
result = lg.compare_documents(doc1, doc2)
```
```

### 3. Creating a Release

```bash
# Update version in setup.py
# version="1.1.0"

# Update CHANGELOG.md
echo "## [1.1.0] - 2025-01-15" >> CHANGELOG.md
echo "### Added" >> CHANGELOG.md
echo "- Document comparison feature" >> CHANGELOG.md

# Tag release
git tag -a v1.1.0 -m "Version 1.1.0: Document comparison"
git push origin v1.1.0

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (if public)
pip install twine
twine upload dist/*
```

## 🔐 Security Best Practices

### 1. Never Commit Secrets
```bash
# Always check before committing
git status
git diff

# If you accidentally committed .env:
git rm .env --cached
git commit -m "Remove .env file"
```

### 2. Use Environment Variables
```python
# Good
api_key = os.getenv("GEMINI_API_KEY")

# Bad
api_key = "AIzaSy..."  # NEVER do this!
```

### 3. Rotate API Keys Regularly
- Change your API key every 90 days
- Use separate keys for dev/prod
- Revoke old keys immediately

## 📈 Scaling Tips

### 1. Add Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def analyze_text_cached(self, text: str):
    return self.analyze_text(text)
```

### 2. Add Rate Limiting
```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_text(request: AnalyzeTextRequest):
    # ...
```

### 3. Add Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Analyzing document...")
```

### 4. Add Database (for user storage)
```python
# Add to requirements
# sqlalchemy
# alembic

from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    text = Column(String)
    analysis = Column(String)
```

## 🎓 Learning Path

### Week 1: Basics
- [ ] Set up the project structure
- [ ] Install dependencies
- [ ] Run basic examples
- [ ] Make your first API call

### Week 2: Core SDK
- [ ] Understand core.py
- [ ] Understand file_utils.py
- [ ] Write your own analysis
- [ ] Handle errors properly

### Week 3: Web Servers
- [ ] Run FastAPI server
- [ ] Run Flask server
- [ ] Make API requests with curl
- [ ] Test with Postman

### Week 4: Advanced
- [ ] Add new features
- [ ] Write tests
- [ ] Deploy to cloud
- [ ] Build a frontend

## 🚀 Quick Reference Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-fastapi.txt
export GEMINI_API_KEY="your_key"

# Run FastAPI
cd fastapi_app && python main.py

# Run Flask
cd flask_app && python app.py

# Test
pytest tests/ -v

# Format
black . && isort .

# Build package
python setup.py sdist bdist_wheel

# Install locally
pip install -e .

# Generate docs
pdoc --html lexiguard_sdk
```

## ✅ Checklist for Production

- [ ] All tests passing
- [ ] Code formatted (black, isort)
- [ ] No hardcoded secrets
- [ ] Environment variables set
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Documentation complete
- [ ] API documentation (FastAPI: /docs)
- [ ] Rate limiting enabled
- [ ] HTTPS enabled
- [ ] CORS configured properly
- [ ] Backup strategy in place
- [ ] Monitoring set up
- [ ] Load testing done
- [ ] Security audit complete

## 🆘 Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `API key not found` | Set `GEMINI_API_KEY` environment variable |
| `Port already in use` | Change port or kill existing process |
| `Cannot import lexiguard_sdk` | Run from project root or install with `pip install -e .` |
| `Tests failing` | Check if API key is set for tests |
| `File parsing error` | Verify file format and install parsers |

---

**Congratulations!** You now have a complete understanding of the LexiGuard SDK project structure. This modular design allows you to:

✅ Use the SDK standalone  
✅ Deploy with FastAPI or Flask  
✅ Easily add new features  
✅ Scale efficiently  
✅ Maintain code quality  

Happy coding! 🎉