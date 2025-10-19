# 🛡️ LexiGuard SDK - Complete User Testing Guide

**Welcome!** This guide will help you set up and test LexiGuard from scratch, even if you're new to programming. Follow each step carefully, and you'll have a working legal document analysis system in under 30 minutes.

---

## 📋 Table of Contents

1. [What is LexiGuard?](#what-is-lexiguard)
2. [Prerequisites](#prerequisites)
3. [Installation Guide](#installation-guide)
4. [Getting Your API Key](#getting-your-api-key)
5. [Testing the SDK](#testing-the-sdk)
6. [Testing FastAPI Server](#testing-fastapi-server)
7. [Testing Flask Server](#testing-flask-server)
8. [Advanced Testing](#advanced-testing)
9. [Troubleshooting](#troubleshooting)
10. [What's Next?](#whats-next)

---

## 🎯 What is LexiGuard?

LexiGuard is a **legal document analysis toolkit** powered by Google's Gemini AI. It helps you:

- ✅ Analyze legal documents and contracts
- ✅ Identify risks and unfair clauses
- ✅ Get recommendations for negotiations
- ✅ Generate professional emails
- ✅ Extract key information from PDFs and Word documents

**Three Ways to Use It:**
1. **SDK** - Import into your Python scripts
2. **FastAPI** - Modern REST API with auto-documentation
3. **Flask** - Traditional web API

---

## ✅ Prerequisites

Before starting, you need:

### 1. Python 3.8 or Higher

**Check if you have Python:**

Open Terminal (Mac/Linux) or Command Prompt (Windows) and type:

```bash
python --version
```

**Expected output:** `Python 3.8.x` or higher

**If Python is not installed:**
- Download from: https://www.python.org/downloads/
- During installation, **check** "Add Python to PATH"

### 2. Text Editor

Any of these will work:
- **VS Code** (recommended): https://code.visualstudio.com/
- **Notepad++** (Windows): https://notepad-plus-plus.org/
- **TextEdit** (Mac) - built-in
- **Notepad** (Windows) - built-in

### 3. Web Browser

Chrome, Firefox, Safari, or Edge - any modern browser works!

### 4. Internet Connection

You'll need internet to:
- Download packages
- Get API key
- Make AI analysis calls

---

## 🚀 Installation Guide

### Step 1: Download/Clone the Project

If you already have the `lexiguard` folder with all files, **skip to Step 2**.

Otherwise:
```bash
git clone https://github.com/yourusername/lexiguard.git
cd lexiguard
```

### Step 2: Open Terminal/Command Prompt

**Windows:**
1. Press `Windows Key + R`
2. Type `cmd` and press Enter

**Mac:**
1. Press `Command + Space`
2. Type `Terminal` and press Enter

**Linux:**
- Press `Ctrl + Alt + T`

### Step 3: Navigate to Your Project Folder

In the terminal, type:

```bash
cd C:\Users\YourName\path\to\lexiguard
```

**💡 Pro Tip:** Type `cd ` (with a space), then drag your `lexiguard` folder into the terminal. It will auto-fill the path!

**Verify you're in the right place:**

Type:
```bash
dir     # Windows
ls      # Mac/Linux
```

**You should see these folders:**
- `lexiguard_sdk`
- `fastapi_app`
- `flask_app`
- `setup.py`
- `README.md`

### Step 4: Install Required Packages

Copy this **entire command** (all one line) and paste into terminal:

```bash
pip install google-generativeai PyPDF2 python-docx fastapi uvicorn python-multipart flask flask-cors requests
```

Press Enter and wait. You'll see lots of text scrolling - that's normal!

**Expected output:** You should see "Successfully installed..." messages.

**If you see errors:**

Try this instead:
```bash
python -m pip install google-generativeai PyPDF2 python-docx fastapi uvicorn python-multipart flask flask-cors requests
```

Or on Mac/Linux:
```bash
python3 -m pip install google-generativeai PyPDF2 python-docx fastapi uvicorn python-multipart flask flask-cors requests
```

**Verify installation:**
```bash
pip list | findstr google-generativeai     # Windows
pip list | grep google-generativeai        # Mac/Linux
```

You should see: `google-generativeai` with a version number.

---

## 🔑 Getting Your API Key

LexiGuard uses Google's Gemini AI, which requires a free API key.

### Step 1: Go to Google AI Studio

Open your browser and visit:
```
https://makersuite.google.com/app/apikey
```

### Step 2: Sign In

- Use your **Google account** (Gmail)
- If you don't have one, create a free account

### Step 3: Create API Key

1. Click the **"Create API Key"** button
2. Choose **"Create API key in new project"**
3. **Copy** the entire key (it looks like: `AIzaSyD...`)

**⚠️ IMPORTANT:** 
- Save this key somewhere safe!
- Don't share it publicly
- Don't commit it to GitHub

### Step 4: Set Your API Key

In your terminal (the same one from earlier), run **ONE** of these commands:

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=AIzaSyD...paste_your_actual_key_here...
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="AIzaSyD...paste_your_actual_key_here..."
```

**Mac/Linux:**
```bash
export GEMINI_API_KEY="AIzaSyD...paste_your_actual_key_here..."
```

**⚠️ Replace `AIzaSyD...` with your ACTUAL API key!**

**To verify it's set:**
```bash
echo %GEMINI_API_KEY%              # Windows CMD
echo $env:GEMINI_API_KEY           # Windows PowerShell
echo $GEMINI_API_KEY               # Mac/Linux
```

You should see your API key printed.

---

## 🧪 Testing the SDK

This is the **simplest test** - it doesn't need a web server!

### Step 1: Create Test File

In your `lexiguard` folder, create a new file called `test_basic.py`

**How to create:**
- Right-click in folder → New → Text Document
- Rename to `test_basic.py` (not .txt!)
- Open it with your text editor

### Step 2: Add Test Code

Copy and paste this code into `test_basic.py`:

```python
print("🚀 Starting LexiGuard SDK Test...")
print("-" * 50)

# Test 1: Check if we can import the SDK
print("\n✓ Test 1: Importing SDK...")
try:
    from lexiguard_sdk import LexiGuard
    print("   SUCCESS! SDK imported.")
except Exception as e:
    print(f"   ERROR: {e}")
    print("   Make sure you're running from the lexiguard folder!")
    exit()

# Test 2: Check API key
print("\n✓ Test 2: Checking API key...")
import os
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("   ERROR: API key not found!")
    print("   Did you run the 'set' or 'export' command?")
    api_key = input("\n   Paste your API key here: ").strip()
    os.environ["GEMINI_API_KEY"] = api_key

print(f"   API key found: {api_key[:10]}...")

# Test 3: Initialize SDK
print("\n✓ Test 3: Initializing LexiGuard...")
try:
    lg = LexiGuard(api_key=api_key)
    print("   SUCCESS! LexiGuard initialized.")
except Exception as e:
    print(f"   ERROR: {e}")
    exit()

# Test 4: Analyze simple text
print("\n✓ Test 4: Analyzing sample contract...")
sample_text = """
This is a simple employment agreement between ABC Corp and John Doe.
The employee will work full-time for $80,000 per year.
The term is 2 years with 30 days notice for termination.
The employee receives health insurance and 15 days paid vacation.
"""

try:
    result = lg.analyze_text(sample_text)
    
    if result["success"]:
        print("   SUCCESS! Analysis complete!")
        print("\n📄 Results:")
        print(f"   Document Type: {result['data'].get('document_type', 'N/A')}")
        print(f"   Summary: {result['data'].get('summary', 'N/A')[:100]}...")
        
        key_clauses = result['data'].get('key_clauses', [])
        print(f"\n   Key Clauses Found: {len(key_clauses)}")
        
        risks = result['data'].get('potential_risks', [])
        print(f"   Potential Risks: {len(risks)}")
        
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown error')}")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "-" * 50)
print("Test complete!")
```

**Save the file.**

### Step 3: Run the Test

In your terminal (make sure you're in the `lexiguard` folder):

```bash
python test_basic.py
```

### Step 4: Check Results

**✅ SUCCESS looks like:**
```
🚀 Starting LexiGuard SDK Test...
--------------------------------------------------

✓ Test 1: Importing SDK...
   SUCCESS! SDK imported.

✓ Test 2: Checking API key...
   API key found: AIzaSyDAFG...

✓ Test 3: Initializing LexiGuard...
   SUCCESS! LexiGuard initialized.

✓ Test 4: Analyzing sample contract...
   SUCCESS! Analysis complete!

📄 Results:
   Document Type: Employment Agreement
   Summary: This employment agreement establishes...

   Key Clauses Found: 4
   Potential Risks: 2

🎉 ALL TESTS PASSED!
```

**Note:** You might see some WARNING messages about "ALTS creds" or "absl::InitializeLog" - **ignore these**. They're internal Google messages and don't affect functionality.

**❌ If you see errors, jump to the [Troubleshooting](#troubleshooting) section.**

---

## 🌐 Testing FastAPI Server

Now let's test the web API!

### Step 1: Fix Import Issues

Before starting, we need to fix a path issue.

**Open** `fastapi_app/main.py` in your text editor.

**At the very top** (before any other imports), add these lines:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

So the top of your file looks like:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
# ... rest of the imports
```

**Save the file.**

### Step 2: Change Port (If Needed)

At the **bottom** of `fastapi_app/main.py`, find this line:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

If port 8000 is already in use on your computer, change it to:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Save the file.**

### Step 3: Start the Server

In your terminal:

```bash
cd fastapi_app
python main.py
```

**✅ SUCCESS looks like:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```

**⚠️ Keep this terminal open!** The server needs to stay running.

### Step 4: Test in Browser

Open your web browser and go to:

```
http://localhost:8001
```

(Use `8000` if you didn't change the port)

**You should see:** A welcome JSON message with endpoint information!

### Step 5: Test Interactive API Docs

Go to:
```
http://localhost:8001/docs
```

**You should see:** A beautiful interactive API documentation page!

**Try it out:**
1. Click on **POST /analyze**
2. Click **"Try it out"**
3. In the Request body, paste:
   ```json
   {
     "text": "This Employment Agreement is between XYZ Corp and Jane Smith. Salary is $95,000 per year. The term is 2 years with 60 days notice."
   }
   ```
4. Click **"Execute"**
5. Scroll down to see the **Response**!

**You should see:** A detailed analysis with document type, summary, clauses, risks, and recommendations!

### Step 6: Test from Python Script

**Open a NEW terminal** (keep the server running in the first one!).

Navigate to your `lexiguard` folder and create `test_api.py`:

```python
import requests
import json

print("🧪 Testing FastAPI Server...")
print("-" * 60)

# IMPORTANT: Change port to 8001 if you changed it
API_URL = "http://localhost:8001"  # or 8000

# Test 1: Check if server is running
print("\n✓ Test 1: Server Health Check...")
try:
    response = requests.get(API_URL)
    if response.status_code == 200:
        print("   ✅ SUCCESS! Server is running!")
        data = response.json()
        print(f"   Message: {data['message']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"   ⚠️  Got status code {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("   Make sure the FastAPI server is running!")
    print(f"   Try: cd fastapi_app && python main.py")
    exit()

# Test 2: Analyze text
print("\n✓ Test 2: Testing /analyze endpoint...")
contract_data = {
    "text": """
    This Software License Agreement is made on January 15, 2025.
    The Client agrees to pay $15,000 annually for software usage.
    The term is 2 years with automatic renewal unless either party 
    provides 90 days written notice. The software is provided 
    "as-is" without warranty. Liability is limited to fees paid.
    Confidential information must be protected for 5 years.
    """
}

try:
    print("   Sending contract for analysis...")
    response = requests.post(
        f"{API_URL}/analyze",
        json=contract_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("   ✅ SUCCESS! Analysis complete!\n")
        
        print("   📄 RESULTS:")
        print(f"   Document Type: {result['data'].get('document_type', 'N/A')}")
        print(f"   \n   Summary: {result['data'].get('summary', 'N/A')[:200]}...")
        
        key_clauses = result['data'].get('key_clauses', [])
        print(f"\n   📝 Key Clauses ({len(key_clauses)}):")
        for i, clause in enumerate(key_clauses[:3], 1):
            print(f"      {i}. {clause}")
        
        risks = result['data'].get('potential_risks', [])
        print(f"\n   ⚠️  Potential Risks ({len(risks)}):")
        for i, risk in enumerate(risks[:3], 1):
            print(f"      {i}. {risk}")
            
        recommendations = result['data'].get('recommendations', [])
        print(f"\n   💡 Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")
    else:
        print(f"   ❌ ERROR: Status code {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 3: Test clause analysis
print("\n✓ Test 3: Testing /analyze-clauses endpoint...")
clause_data = {
    "text": contract_data["text"],
    "clause_types": ["payment", "termination", "warranty"]
}

try:
    response = requests.post(
        f"{API_URL}/analyze-clauses",
        json=clause_data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ SUCCESS! Found {result.get('total_clauses', 0)} clauses")
        
        # Show first clause
        clauses = result['data'].get('clauses', [])
        if clauses:
            first_clause = clauses[0]
            print(f"   \n   Example Clause:")
            print(f"      Title: {first_clause.get('clause_title')}")
            print(f"      Risk Level: {first_clause.get('risk_level')}")
            print(f"      Fairness: {first_clause.get('fairness_score')}/10")
    else:
        print(f"   ⚠️  Status code {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 4: Test fairness analysis
print("\n✓ Test 4: Testing /analyze-extended endpoint...")
try:
    response = requests.post(
        f"{API_URL}/analyze-extended",
        json={"text": contract_data["text"]},
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        score = result['data'].get('overall_fairness_score', 'N/A')
        print(f"   ✅ SUCCESS! Fairness Score: {score}/10")
        
        if isinstance(score, (int, float)) and score < 6:
            print(f"   ⚠️  WARNING: This contract may be one-sided!")
    else:
        print(f"   ⚠️  Status code {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "-" * 60)
print("🎉 Testing complete!")
print(f"\n💡 TIP: Visit {API_URL}/docs to test all endpoints interactively!")
```

**Run it:**
```bash
python test_api.py
```

**✅ You should see all tests passing!**

### Step 7: Stop the Server

When you're done testing, go back to the terminal running the server and press:

```
Ctrl + C
```

---

## 🐍 Testing Flask Server

Let's test the Flask alternative!

### Step 1: Fix Import Issues

**Open** `flask_app/app.py` in your text editor.

**At the very top** (before any other imports), add these lines:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Save the file.**

### Step 2: Start Flask Server

**Make sure FastAPI is stopped first!** (Press Ctrl+C in that terminal)

In your terminal:

```bash
cd flask_app
python app.py
```

**✅ SUCCESS looks like:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
 * Press CTRL+C to quit
```

### Step 3: Test Health Endpoint

Open browser and go to:
```
http://localhost:5000/health
```

**You should see:**
```json
{
  "status": "healthy",
  "sdk_initialized": true
}
```

### Step 4: Test from Python

Create `test_flask.py` in your `lexiguard` folder:

```python
import requests

print("🧪 Testing Flask Server...")
print("-" * 50)

API_URL = "http://localhost:5000"

# Test 1: Health check
print("\n✓ Test 1: Health Check...")
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Status: {data['status']}")
        print(f"   SDK Initialized: {data['sdk_initialized']}")
    else:
        print(f"   ❌ Error: {response.status_code}")
        exit()
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    print("   Make sure Flask server is running!")
    exit()

# Test 2: Analyze endpoint
print("\n✓ Test 2: Testing /analyze endpoint...")
data = {
    "text": "This is a rental agreement for $2,000 per month. Lease term is 12 months. Security deposit is $4,000."
}

try:
    response = requests.post(f"{API_URL}/analyze", json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Analysis successful!")
        print(f"   Document Type: {result['data'].get('document_type')}")
        print(f"   Summary: {result['data'].get('summary', '')[:100]}...")
    else:
        print(f"   ❌ Error: {response.text}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "-" * 50)
print("🎉 Flask testing complete!")
```

**Run it:**
```bash
python test_flask.py
```

---

## 🚀 Advanced Testing

### Test 1: File Upload

Create a simple text file called `sample_contract.txt`:

```
EMPLOYMENT AGREEMENT

This Agreement is made on January 20, 2025, between:
- Employer: TechStart Inc.
- Employee: Michael Johnson

Position: Senior Software Engineer
Salary: $120,000 per year
Start Date: February 1, 2025
Employment Term: 2 years

Benefits:
- Health insurance (fully covered)
- 401(k) matching (4% of salary)
- 25 days paid time off
- Remote work allowed 3 days per week

Termination:
Either party may terminate this agreement with 30 days written notice.

Non-Compete:
Employee agrees not to work for direct competitors for 6 months after 
termination within a 50-mile radius of company headquarters.

Confidentiality:
All company information must remain confidential during and after employment.
```

**Test uploading it** - create `test_upload.py`:

```python
import requests

print("📁 Testing File Upload...")
print("-" * 50)

# Use FastAPI (make sure it's running!)
API_URL = "http://localhost:8001"  # or 8000
file_path = "sample_contract.txt"

print(f"\nUploading file: {file_path}")

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        response = requests.post(
            f"{API_URL}/analyze-file",
            files=files,
            timeout=30
        )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ File analyzed successfully!\n")
        
        file_info = result.get('file_info', {})
        print(f"📄 File Information:")
        print(f"   Name: {file_info.get('file_name')}")
        print(f"   Type: {file_info.get('file_type')}")
        
        print(f"\n📋 Analysis:")
        print(f"   Document Type: {result['data'].get('document_type')}")
        print(f"   Summary: {result['data'].get('summary', '')[:200]}...")
        
        risks = result['data'].get('potential_risks', [])
        print(f"\n⚠️  Identified {len(risks)} potential risks:")
        for i, risk in enumerate(risks[:3], 1):
            print(f"   {i}. {risk}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    print("Make sure you created the sample_contract.txt file!")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "-" * 50)
```

**Run it** (make sure FastAPI server is running):
```bash
python test_upload.py
```

### Test 2: Email Generation

Create `test_email.py`:

```python
import requests

print("📧 Testing Email Generation...")
print("-" * 50)

API_URL = "http://localhost:8001"  # Adjust if needed

# Test negotiation email
print("\n✓ Generating negotiation email...")
data = {
    "document_text": "This contract has a 90-day notice period and unlimited liability clause.",
    "concerns": [
        "90-day notice period is too long",
        "Unlimited liability is too risky",
        "No dispute resolution clause"
    ],
    "recipient_name": "Sarah Johnson"
}

try:
    response = requests.post(
        f"{API_URL}/draft-negotiation",
        json=data,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        email = result['data']
        
        print("✅ Email generated successfully!\n")
        print("=" * 60)
        print(f"Subject: {email.get('subject')}")
        print("=" * 60)
        print(email.get('body'))
        print("=" * 60)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "-" * 50)
```

**Run it:**
```bash
python test_email.py
```

### Test 3: Interactive Chat

Create `test_chat.py`:

```python
import requests

print("💬 Testing Document Chat...")
print("-" * 50)

API_URL = "http://localhost:8001"

document = """
Employment Agreement for Software Engineer position.
Salary: $100,000 per year. Start date: March 1, 2025.
Benefits include health insurance, 401k, and 20 days PTO.
Non-compete period: 6 months after termination.
"""

questions = [
    "What is the salary?",
    "How many vacation days do I get?",
    "Is there a non-compete clause?",
    "What benefits are included?"
]

print(f"\nDocument: {document[:100]}...\n")

for question in questions:
    print(f"❓ Q: {question}")
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": question,
                "document_context": document
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result['data'].get('response', '')
            print(f"💡 A: {answer}\n")
        else:
            print(f"❌ Error: {response.status_code}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")

print("-" * 50)
```

**Run it:**
```bash
python test_chat.py
```

---

## 🔧 Troubleshooting

### Problem: "Module not found: lexiguard_sdk"

**Solution:**
1. Make sure you're running from the `lexiguard` folder
2. Add the path fix to the top of your files:
   ```python
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

### Problem: "API key is required" or "API key not found"

**Solution:**
1. Set the API key again in the SAME terminal:
   ```bash
   # Windows
   set GEMINI_API_KEY=your_key_here
   
   # Mac/Linux
   export GEMINI_API_KEY="your_key_here"
   ```
2. Verify it's set:
   ```bash
   echo %GEMINI_API_KEY%        # Windows
   echo $GEMINI_API_KEY         # Mac/Linux
   ```

### Problem: "Cannot connect to server"

**Solutions:**
1. Check if the server is actually running (look for "Uvicorn running" message)
2. Make sure you're using the correct port (8000, 8001, or 5000)
3. Try `127.0.0.1` instead of `localhost`:
   ```
   http://127.0.0.1:8001
   ```

### Problem: "Port already in use" (Error 10048)

**Solution:**
1. Change the port in your code:
   ```python
   # In fastapi_app/main.py
   uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
   
   # In flask_app/app.py
   app.run(host="0.0.0.0", port=5001)  # Changed from 5000
   ```

**Or find and stop the process using the port:**

Windows:
```cmd
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

Mac/Linux:
```bash
lsof -i :8000
kill -9 <process_id>
```

### Problem: "404 models/gemini-1.5-flash is not found"

**Solution:**
Change the model name in `lexiguard_sdk/core.py`:

```python
def __init__(self, api_key: str, model_name: str = "gemini-pro"):  # Changed from gemini-1.5-flash
```

### Problem: "pip is not recognized"

**Solution:**
```bash
python -m pip install package-name
```

### Problem: Files not parsing (PDF/DOCX errors)

**Solution:**
Reinstall the parsing libraries:
```bash
pip install --upgrade PyPDF2 python-docx
```

🔧 Troubleshooting
Problem: "Module not found: lexiguard_sdk"
Solution:

Make sure you're running from the lexiguard folder
Add the path fix to the top of your files:

python   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Problem: "API key is required" or "API key not found"
Solution:

Set the API key again in the SAME terminal:

bash   # Windows
   set GEMINI_API_KEY=your_key_here
   
   # Mac/Linux
   export GEMINI_API_KEY="your_key_here"

Verify it's set:

bash   echo %GEMINI_API_KEY%        # Windows
   echo $GEMINI_API_KEY         # Mac/Linux
Problem: "Cannot connect to server"
Solutions:

Check if the server is actually running (look for "Uvicorn running" message)
Make sure you're using the correct port (8000, 8001, or 5000)
Try 127.0.0.1 instead of localhost:

   http://127.0.0.1:8001
Problem: "Port already in use" (Error 10048)
Solution:

Change the port in your code:

python   # In fastapi_app/main.py
   uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed from 8000
   
   # In flask_app/app.py
   app.run(host="0.0.0.0", port=5001)  # Changed from 5000
Or find and stop the process using the port:
Windows:
cmdnetstat -ano | findstr :8000
taskkill /PID <process_id> /F
Mac/Linux:
bashlsof -i :8000
kill -9 <process_id>
Problem: "404 models/gemini-1.5-flash is not found"
Solution:
Change the model name in lexiguard_sdk/core.py:
pythondef __init__(self, api_key: str, model_name: str = "gemini-pro"):  # Changed from gemini-1.5-flash
Problem: "pip is not recognized"
Solution:
bashpython -m pip install package-name
Problem: Files not parsing (PDF/DOCX errors)
Solution:
Reinstall the parsing libraries:
bashpip install --upgrade PyPDF2 python-docx
Problem: Requests module not found
Solution:
bashpip install requests

🎉 What's Next?
Congratulations! You now have a fully functional legal document analysis system!
Things You Can Do Now
1. Build Your Own Applications 🏗️
Simple Desktop App:
pythonfrom lexiguard_sdk import LexiGuard
import os

def analyze_my_contract():
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Load your contract
    with open("my_contract.txt", "r") as f:
        text = f.read()
    
    # Analyze it
    result = lg.analyze_text(text)
    
    # Save results
    with open("analysis_report.txt", "w") as f:
        f.write(f"Document Type: {result['data']['document_type']}\n\n")
        f.write(f"Summary:\n{result['data']['summary']}\n\n")
        f.write(f"Risks:\n")
        for risk in result['data']['potential_risks']:
            f.write(f"- {risk}\n")
    
    print("Analysis saved to analysis_report.txt!")

analyze_my_contract()
Batch Processor:
pythonimport os
from lexiguard_sdk import LexiGuard

def analyze_all_contracts(folder_path):
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.txt', '.pdf', '.docx')):
            print(f"Analyzing {filename}...")
            
            result = lg.analyze_file(os.path.join(folder_path, filename))
            
            # Save individual reports
            report_name = f"{filename}_analysis.json"
            with open(report_name, "w") as f:
                import json
                json.dump(result, f, indent=2)
            
            print(f"✅ Saved {report_name}")

analyze_all_contracts("./contracts")
2. Integrate with Existing Systems 🔗
Add to Your Django/Flask App:
python# In your views.py
from lexiguard_sdk import LexiGuard
from django.http import JsonResponse

def analyze_contract_view(request):
    if request.method == 'POST':
        contract_text = request.POST.get('contract')
        lg = LexiGuard(api_key=settings.GEMINI_API_KEY)
        result = lg.analyze_text(contract_text)
        return JsonResponse(result)
Create a Slack Bot:
pythonfrom slack_sdk import WebClient
from lexiguard_sdk import LexiGuard

def handle_contract_upload(file_url):
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    # Download and analyze
    result = lg.analyze_file(file_url)
    # Send results back to Slack
    return format_for_slack(result)
3. Customize the Analysis ⚙️
Focus on Specific Clauses:
pythonlg = LexiGuard(api_key=api_key)

# Look for specific clause types
result = lg.analyze_clauses(
    contract_text,
    clause_types=["termination", "payment", "intellectual_property"]
)

# Get fairness scores
fairness = lg.analyze_fairness(contract_text)
print(f"Overall Fairness: {fairness['data']['overall_fairness_score']}/10")
Extract Specific Information:
python# Extract key terms
result = lg.extract_entities(contract_text)

# Ask specific questions
answer = lg.chat(
    message="What is the notice period for termination?",
    document_context=contract_text
)
4. Build a Web Interface 🌐
Simple HTML + JavaScript Frontend:
html<!DOCTYPE html>
<html>
<head>
    <title>LexiGuard Analyzer</title>
</head>
<body>
    <h1>Contract Analyzer</h1>
    <textarea id="contract" rows="10" cols="50"></textarea>
    <button onclick="analyze()">Analyze</button>
    <div id="results"></div>
    
    <script>
        async function analyze() {
            const text = document.getElementById('contract').value;
            const response = await fetch('http://localhost:8001/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const result = await response.json();
            document.getElementById('results').innerHTML = 
                `<h2>${result.data.document_type}</h2>
                 <p>${result.data.summary}</p>`;
        }
    </script>
</body>
</html>
5. Deploy to Production 🚀
Deploy FastAPI to Heroku:
bash# Create Procfile
echo "web: uvicorn fastapi_app.main:app --host=0.0.0.0 --port=${PORT}" > Procfile

# Create requirements.txt
pip freeze > requirements.txt

# Deploy
heroku create my-lexiguard-app
git push heroku main
Deploy to AWS Lambda:
bash# Use Mangum for serverless
pip install mangum

# In your main.py
from mangum import Mangum
handler = Mangum(app)
Deploy with Docker:
dockerfileFROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
Learning Resources 📚
Master the SDK

Read the source code in lexiguard_sdk/core.py
Experiment with different document types
Try different Gemini models (gemini-pro, gemini-1.5-flash)

Improve Your Skills

Python: https://docs.python.org/3/tutorial/
FastAPI: https://fastapi.tiangolo.com/tutorial/
Flask: https://flask.palletsprojects.com/
Gemini AI: https://ai.google.dev/docs

API Documentation

FastAPI auto-docs: http://localhost:8001/docs
Alternative docs: http://localhost:8001/redoc

Real-World Use Cases 💼
1. Employment Contract Review
Help employees understand:

Compensation packages
Non-compete clauses
Termination terms
Benefits and perks

2. Rental Agreement Analysis
Identify issues with:

Security deposits
Maintenance responsibilities
Lease terms
Tenant rights

3. Vendor Contract Management
Review:

Service level agreements (SLAs)
Payment terms
Liability clauses
Termination conditions

4. Software License Review
Understand:

Usage restrictions
Liability limitations
Data privacy terms
Renewal conditions

5. Freelance Contract Evaluation
Check for:

Payment schedules
Intellectual property rights
Scope of work clarity
Dispute resolution

Pro Tips 💡
1. API Key Security
Never hardcode your API key! Use environment variables:
python# ❌ BAD
lg = LexiGuard(api_key="AIzaSyD...")

# ✅ GOOD
import os
lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
For production, use:

AWS Secrets Manager
Azure Key Vault
Environment variables in your hosting platform

2. Handle Rate Limits
Add retry logic for API calls:
pythonimport time

def analyze_with_retry(lg, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return lg.analyze_text(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise e
3. Cache Results
Don't re-analyze the same document:
pythonimport hashlib
import json

def get_cached_analysis(text):
    # Create hash of document
    doc_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = f"cache/{doc_hash}.json"
    
    # Check if cached
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    
    # Analyze and cache
    result = lg.analyze_text(text)
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    
    return result
4. Better Error Handling
Provide helpful error messages:
pythontry:
    result = lg.analyze_text(contract_text)
except Exception as e:
    if "API key" in str(e):
        print("❌ Invalid API key. Get one at: https://makersuite.google.com")
    elif "quota" in str(e).lower():
        print("❌ API quota exceeded. Wait a bit or upgrade your plan.")
    elif "timeout" in str(e).lower():
        print("❌ Request timed out. Try with a shorter document.")
    else:
        print(f"❌ Unexpected error: {e}")
5. Optimize Performance
For large documents:
python# Split long documents
def split_document(text, max_length=5000):
    words = text.split()
    chunks = []
    current = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_length:
            chunks.append(' '.join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += len(word) + 1
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks

# Analyze each chunk
chunks = split_document(long_contract)
results = [lg.analyze_text(chunk) for chunk in chunks]
Community & Support 🤝
Get Help

GitHub Issues: Report bugs and request features
Stack Overflow: Tag questions with lexiguard
Documentation: Check the README.md file

Contribute
Want to make LexiGuard better?

Fork the repository
Create a feature branch
Make your improvements
Submit a pull request

Share Your Projects
Built something cool with LexiGuard? Share it!

Tweet with #LexiGuard
Write a blog post
Create a tutorial video

Advanced Features to Explore 🔬
1. Multi-Language Support
python# Analyze contracts in different languages
result = lg.analyze_text(spanish_contract, language="es")
2. Comparison Mode
python# Compare two versions of a contract
result = lg.compare_documents(old_version, new_version)
3. Custom Prompts
python# Use your own analysis prompts
result = lg.analyze_text(
    text,
    custom_prompt="Focus on financial obligations and deadlines"
)
4. Webhook Integration
python# Get notified when analysis completes
@app.post("/webhook")
async def handle_analysis_complete(data: dict):
    # Send email, Slack notification, etc.
    pass
Keep Your API Key Safe! 🔐
Create a .env file:
bash# .env
GEMINI_API_KEY=your_api_key_here
Load it in your code:
pythonfrom dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
Install python-dotenv:
bashpip install python-dotenv
Add to .gitignore:
.env
*.pyc
__pycache__/
Final Checklist ✅
Before going to production:

 API key stored securely (not in code)
 Error handling implemented
 Rate limiting considered
 Input validation added
 Results caching implemented (optional)
 Logging configured
 Tests written
 Documentation updated
 Security audit completed
 Performance optimized


🎊 You're Ready!
You've successfully:

✅ Installed LexiGuard
✅ Tested the SDK
✅ Set up FastAPI server
✅ Tested Flask server
✅ Explored advanced features
✅ Learned best practices

Now go build something amazing! 🚀

📞 Need More Help?
Quick Reference Commands
Start FastAPI Server:
bashcd fastapi_app
python main.py
Start Flask Server:
bashcd flask_app
python app.py
Set API Key (remember to do this in EVERY new terminal):
bash# Windows CMD
set GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Mac/Linux
export GEMINI_API_KEY="your_key_here"
Install All Dependencies:
bashpip install google-generativeai PyPDF2 python-docx fastapi uvicorn python-multipart flask flask-cors requests python-dotenv
Common Gotchas to Avoid ⚠️

Forgetting to set API key in new terminals

Solution: Create a .env file or add to your shell profile


Running from wrong directory

Solution: Always run from the lexiguard folder


Port conflicts

Solution: Change ports or stop conflicting processes


Module import errors

Solution: Add the path fix at the top of your files


Timeout errors on large files

Solution: Increase timeout or split documents into chunks



Performance Tips 🏎️
Expected Response Times:

Simple text analysis: 2-5 seconds
File upload (1 page): 3-7 seconds
Large documents (10+ pages): 10-30 seconds
Clause analysis: 5-10 seconds
Email generation: 3-5 seconds

If responses are slow:

Check your internet connection
Try a shorter document first
Use gemini-1.5-flash model (faster, slightly less detailed)
Implement caching for repeated analyses

API Rate Limits 📊
Free Tier (Gemini API):

60 requests per minute
1,500 requests per day
1 million tokens per day

If you hit limits:

Wait a minute before retrying
Implement exponential backoff
Cache results when possible
Consider upgrading to paid tier

Sample Project Ideas 💡
Beginner Projects

Contract Comparison Tool - Compare old vs new versions
Risk Highlighter - Highlight risky clauses in documents
Email Template Generator - Auto-generate negotiation emails
Simple Web Dashboard - Upload and analyze contracts

Intermediate Projects

Multi-Document Analyzer - Batch process entire folders
Contract Database - Store and search analyzed contracts
Slack Integration - Analyze contracts in Slack channels
PDF Report Generator - Create professional PDF reports
Contract Templates - Generate contract templates with AI

Advanced Projects

Real-time Collaboration Tool - Multiple users reviewing together
Mobile App - React Native or Flutter app
Chrome Extension - Analyze contracts on any website
Enterprise Dashboard - Team management and analytics
AI Contract Negotiator - Suggest counter-proposals automatically

Code Snippets Library 📝
Generate PDF Report
pythonfrom reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from lexiguard_sdk import LexiGuard
import os

def generate_pdf_report(contract_text, output_file="report.pdf"):
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    result = lg.analyze_text(contract_text)
    
    c = canvas.Canvas(output_file, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Contract Analysis Report")
    
    c.setFont("Helvetica", 12)
    y = 700
    
    c.drawString(100, y, f"Document Type: {result['data']['document_type']}")
    y -= 30
    
    c.drawString(100, y, "Summary:")
    y -= 20
    c.setFont("Helvetica", 10)
    
    # Wrap text
    summary = result['data']['summary']
    for line in summary.split('\n'):
        if y < 100:
            c.showPage()
            y = 750
        c.drawString(120, y, line[:80])
        y -= 15
    
    c.save()
    print(f"✅ Report saved to {output_file}")

# Usage
generate_pdf_report("This is a sample contract...")
Email Notification System
pythonimport smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_analysis_email(recipient, analysis_result):
    sender = "your-email@gmail.com"
    password = "your-app-password"
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = "Contract Analysis Complete"
    
    body = f"""
    Your contract analysis is complete!
    
    Document Type: {analysis_result['data']['document_type']}
    
    Summary: {analysis_result['data']['summary']}
    
    Number of Risks: {len(analysis_result['data']['potential_risks'])}
    
    View full report: [link to your web app]
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()
    
    print(f"✅ Email sent to {recipient}")
Webhook Integration
pythonfrom fastapi import FastAPI, BackgroundTasks
import requests

app = FastAPI()

def notify_webhook(url: str, data: dict):
    """Send analysis results to a webhook"""
    try:
        response = requests.post(url, json=data, timeout=10)
        print(f"✅ Webhook notified: {response.status_code}")
    except Exception as e:
        print(f"❌ Webhook failed: {e}")

@app.post("/analyze-with-webhook")
async def analyze_and_notify(
    text: str, 
    webhook_url: str,
    background_tasks: BackgroundTasks
):
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    result = lg.analyze_text(text)
    
    # Send webhook in background
    background_tasks.add_task(notify_webhook, webhook_url, result)
    
    return {"message": "Analysis complete, webhook will be notified"}
Database Integration (SQLite)
pythonimport sqlite3
import json
from datetime import datetime

def save_analysis_to_db(contract_text, analysis_result):
    conn = sqlite3.connect('lexiguard_analyses.db')
    c = conn.cursor()
    
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         contract_text TEXT,
         document_type TEXT,
         analysis_data TEXT,
         created_at TIMESTAMP)
    ''')
    
    # Insert analysis
    c.execute('''
        INSERT INTO analyses (contract_text, document_type, analysis_data, created_at)
        VALUES (?, ?, ?, ?)
    ''', (
        contract_text,
        analysis_result['data']['document_type'],
        json.dumps(analysis_result),
        datetime.now()
    ))
    
    conn.commit()
    conn.close()
    print("✅ Analysis saved to database")

def get_recent_analyses(limit=10):
    conn = sqlite3.connect('lexiguard_analyses.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT id, document_type, created_at 
        FROM analyses 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (limit,))
    
    results = c.fetchall()
    conn.close()
    
    return results
Scheduled Analysis (Cron Job)
pythonimport schedule
import time
import os
from lexiguard_sdk import LexiGuard

def analyze_daily_contracts():
    """Run this daily to analyze new contracts"""
    lg = LexiGuard(api_key=os.getenv("GEMINI_API_KEY"))
    contracts_folder = "./new_contracts"
    
    for filename in os.listdir(contracts_folder):
        if filename.endswith(('.pdf', '.docx', '.txt')):
            print(f"Analyzing {filename}...")
            
            file_path = os.path.join(contracts_folder, filename)
            result = lg.analyze_file(file_path)
            
            # Save results
            save_analysis_to_db(filename, result)
            
            # Move to processed folder
            os.rename(file_path, f"./processed/{filename}")
    
    print("✅ Daily analysis complete!")

# Schedule daily at 9 AM
schedule.every().day.at("09:00").do(analyze_daily_contracts)

print("📅 Scheduler started. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(60)
Testing Checklist ✓
Use this checklist when testing your implementation:
Basic Functionality:

 SDK imports correctly
 API key is recognized
 Simple text analysis works
 Results are properly formatted
 Error handling works

File Operations:

 Can upload .txt files
 Can upload .pdf files
 Can upload .docx files
 Large files (5+ MB) work
 Invalid files are rejected

API Endpoints:

 /analyze endpoint works
 /analyze-file endpoint works
 /analyze-clauses endpoint works
 /analyze-extended endpoint works
 /chat endpoint works
 /draft-negotiation endpoint works
 Error responses are helpful

Performance:

 Responses within acceptable time
 No memory leaks
 Handles concurrent requests
 Proper timeout handling

Security:

 API key not exposed in logs
 Input validation implemented
 File size limits enforced
 CORS configured properly
 Rate limiting works

Deployment Checklist 🚀
Before deploying to production:
Environment Setup:

 Environment variables configured
 API keys stored securely
 Dependencies listed in requirements.txt
 Python version specified

Code Quality:

 Code is well-documented
 Error handling comprehensive
 Logging implemented
 Tests passing
 No hardcoded secrets

Security:

 HTTPS enabled
 Input sanitization
 Rate limiting configured
 CORS settings correct
 File upload restrictions

Monitoring:

 Error tracking (Sentry, etc.)
 Performance monitoring
 Usage analytics
 Log aggregation
 Health check endpoint

Backup & Recovery:

 Database backups scheduled
 Disaster recovery plan
 Rollback procedure documented

Version History 📜
Keep track of changes:
v1.0.0 - Initial Release

Basic text analysis
PDF/DOCX support
FastAPI and Flask servers
Simple web interface

v1.1.0 - Enhanced Features (You can add this!)

Improved accuracy
Faster processing
Better error messages
Caching support

Your Next Version:

What will you build?
What features will you add?
How will you improve it?


🎓 Graduation Certificate
Congratulations! 🎉
You have successfully completed the LexiGuard SDK User Testing Guide!
You now know how to:

✅ Install and configure LexiGuard
✅ Analyze legal documents with AI
✅ Build REST APIs with FastAPI and Flask
✅ Handle file uploads and parsing
✅ Implement error handling and logging
✅ Deploy to production
✅ Build custom applications

You're now a LexiGuard Developer! 👨‍💻👩‍💻

📬 Stay Connected
Share Your Success

Built something cool? Share it!
Found a bug? Report it!
Have a suggestion? We want to hear it!

Keep Learning

Experiment with different document types
Try advanced AI prompts
Build your own features
Contribute to the project

Final Words
Remember: The best way to learn is by doing. Start small, experiment often, and don't be afraid to make mistakes. Every great developer started exactly where you are now.
Happy coding! 🚀

Last Updated: October 2025
Version: 1.0.0
Made with ❤️ for developers and legal professionals

📄 Appendix A: Complete API Reference
Core SDK Methods
python# Initialize
lg = LexiGuard(api_key="your-key", model_name="gemini-pro")

# Analyze text
result = lg.analyze_text(text: str) -> dict

# Analyze file
result = lg.analyze_file(file_path: str) -> dict

# Analyze specific clauses
result = lg.analyze_clauses(text: str, clause_types: list) -> dict

# Extended fairness analysis
result = lg.analyze_fairness(text: str) -> dict

# Extract entities
result = lg.extract_entities(text: str) -> dict

# Chat about document
result = lg.chat(message: str, document_context: str) -> dict

# Draft negotiation email
result = lg.draft_negotiation_email(
    document_text: str,
    concerns: list,
    recipient_name: str
) -> dict

# Compare documents
result = lg.compare_documents(doc1: str, doc2: str) -> dict
FastAPI Endpoints
GET  /                    # Welcome message
POST /analyze             # Analyze text
POST /analyze-file        # Upload and analyze file
POST /analyze-clauses     # Analyze specific clauses
POST /analyze-extended    # Extended analysis with fairness
POST /extract-entities    # Extract key entities
POST /chat                # Ask questions about document
POST /draft-negotiation   # Generate negotiation email
POST /compare             # Compare two documents
GET  /docs                # API documentation
GET  /redoc               # Alternative documentation
Response Format
All endpoints return:
json{
  "success": true,
  "data": {
    "document_type": "Employment Agreement",
    "summary": "...",
    "key_clauses": [...],
    "potential_risks": [...],
    "recommendations": [...]
  },
  "timestamp": "2025-01-20T10:30:00Z"
}
Error Format
json{
  "success": false,
  "error": "Error message here",
  "timestamp": "2025-01-20T10:30:00Z"
}

📄 Appendix B: Environment Variables
bash# Required
GEMINI_API_KEY=your_api_key_here

# Optional
GEMINI_MODEL=gemini-pro
API_TIMEOUT=30
MAX_FILE_SIZE=10485760  # 10 MB in bytes
ENABLE_CACHING=true
CACHE_DIR=./cache
LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# Database (if using)
DATABASE_URL=sqlite:///lexiguard.db

# Email (if using)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Deployment
PORT=8000
HOST=0.0.0.0
WORKERS=4

📄 Appendix C: Common Error Codes
CodeMeaningSolution400Bad RequestCheck your input format401UnauthorizedVerify API key403ForbiddenCheck API key permissions404Not FoundCheck endpoint URL413File Too LargeReduce file size429Rate LimitWait before retrying500Server ErrorCheck server logs503Service UnavailableGemini API might be down

🎊 THE END - GO BUILD SOMETHING AMAZING! 🎊