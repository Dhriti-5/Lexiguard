# LexiGuard Email Features Guide

## Overview
LexiGuard now includes two powerful email features to streamline your workflow:

1. **Negotiation Email - Gmail Integration**: Quickly draft negotiation emails for risky clauses and open them directly in Gmail
2. **Document Review Email - PDF Reports**: Send comprehensive document review reports as PDF attachments via email

---

## Feature 1: Negotiation Email with Gmail Integration

### What It Does
- Generate AI-powered negotiation emails for specific risky clauses
- Opens Gmail with pre-filled subject and body
- Allows quick editing before sending

### How to Use
1. **Analyze a document** (Upload → Analyze)
2. **Find a risky clause** in the results
3. **Click "Draft Negotiation Email"** button on any clause
4. **Wait for AI** to generate the email
5. **Choose an option**:
   - **Draft in Gmail**: Opens Gmail with the email pre-filled (recommended)
   - **Copy to Clipboard**: Copy the text to paste elsewhere

### Technical Details
- **Frontend**: `Results.jsx` - `handleDraftInGmail()` function
- **Backend**: `/draft-negotiation` endpoint (alias: `/draft-negotiation-email`)
- **AI Model**: Google Gemini 2.5 Flash
- **Privacy**: Uses DLP-redacted text for all AI calls

---

## Feature 2: Document Review Email with PDF Attachment

### What It Does
- Generates comprehensive PDF document review reports
- Includes:
  - Document summary
  - Risk analysis
  - Detailed table of risky clauses
- Sends report as PDF attachment via email

### Setup Required (IMPORTANT!)

#### Step 1: Enable Gmail App Password
Since this uses Gmail SMTP, you need to create an **App Password** (not your regular Gmail password):

1. **Go to your Google Account**: https://myaccount.google.com/
2. **Navigate to Security** → **2-Step Verification** (enable if not already)
3. **Scroll down** to **App passwords**
4. **Generate new app password**:
   - App: "Mail"
   - Device: "Windows Computer" (or your preference)
5. **Copy the 16-character password** (e.g., `abcd efgh ijkl mnop`)

#### Step 2: Update Backend `.env` File
Add these two lines to `lexiguard-backend/.env`:

```env
# Existing variables (keep these)
GOOGLE_API_KEY=your_existing_api_key
GOOGLE_CLOUD_PROJECT=your_project_id

# NEW EMAIL CONFIGURATION
GMAIL_SENDER_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=abcdefghijklmnop
```

**Example:**
```env
GMAIL_SENDER_EMAIL=john.smith@gmail.com
GMAIL_APP_PASSWORD=abcd efgh ijkl mnop
```

⚠️ **Important Notes:**
- Use your **actual Gmail address** for `GMAIL_SENDER_EMAIL`
- Use the **16-character App Password** (no spaces) for `GMAIL_APP_PASSWORD`
- **Never commit** `.env` file to Git (it's in `.gitignore`)

#### Step 3: Restart Backend Server
After updating `.env`:

```powershell
cd lexiguard-backend
# Stop the current server (Ctrl+C)
# Start it again
python main.py
```

### How to Use

1. **Analyze a document** (Upload → Analyze)
2. **Click "Generate Document Email"** button at the bottom
3. **Wait for email preview** to generate
4. **Enter your email address** in the input field
5. **Click "Send PDF via Email"**
6. **Check your inbox** for the PDF report

### Email Format
- **Subject**: `LexiGuard Document Review: [filename]`
- **Body**: Professional greeting with summary
- **Attachment**: `LexiGuard_Review_[filename].pdf`

### PDF Report Contents
- **Document Information**: Filename, analysis date
- **Document Summary**: Key points from analysis
- **Risk Analysis**: Overall risk assessment
- **Risky Clauses Table**: 
  - Clause text (truncated to 100 chars)
  - Risk level (High/Medium/Low)
  - Detailed explanation (truncated to 150 chars)

---

## Technical Architecture

### Frontend Changes (`Results.jsx`)

#### New State Variables
```javascript
const [userEmail, setUserEmail] = useState("");
const [emailSending, setEmailSending] = useState(false);
const [emailSent, setEmailSent] = useState(false);
```

#### New Functions
1. **`handleDraftInGmail()`**: Opens Gmail with mailto link
2. **`handleSendDocumentEmail()`**: Sends PDF report via backend
3. Updated **`handleGenerateDocumentEmail()`**: Now generates email preview first

#### UI Changes
- **Negotiation Modal**: Added "Draft in Gmail" button
- **Document Email Modal**: 
  - Added email input field
  - Added "Send PDF via Email" button
  - Added success/loading states

### Backend Changes (`main.py`)

#### New Dependencies
```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
```

#### New Request Model
```python
class SendDocumentReviewRequest(BaseModel):
    filename: str
    document_summary: str
    risk_summary: str
    clauses: list
    user_email: str
```

#### New Endpoint
- **Route**: `POST /send-document-review`
- **Purpose**: Generate PDF and send via Gmail SMTP
- **Input**: SendDocumentReviewRequest
- **Output**: `{"success": true, "message": "..."}`
- **Error Handling**: Returns 500 with error detail if email config missing

---

## Troubleshooting

### Issue: "Email configuration missing"
**Solution**: Make sure you've added `GMAIL_SENDER_EMAIL` and `GMAIL_APP_PASSWORD` to `.env` file and restarted the backend.

### Issue: "Authentication failed"
**Solutions**:
1. Verify you're using **App Password**, not regular Gmail password
2. Ensure 2-Step Verification is enabled on your Google Account
3. Check that App Password has no spaces: `abcdefghijklmnop` (not `abcd efgh ijkl mnop`)

### Issue: Email not received
**Check**:
1. Spam/Junk folder
2. Gmail "Social" or "Promotions" tabs
3. Backend terminal for error messages
4. User email input is correct

### Issue: PDF generation fails
**Check**:
1. `reportlab` is installed: `pip install reportlab`
2. Clauses array format is correct
3. Backend terminal for detailed error logs

### Issue: Gmail mailto link doesn't work
**Solutions**:
1. Set Gmail as default email client in Windows Settings
2. Manually copy the email text and paste in Gmail
3. Use the "Copy to Clipboard" button instead

---

## Privacy & Security

### PII Redaction (Existing Feature)
- All document text is redacted using **Google Cloud DLP API**
- 7 PII types protected:
  - `PERSON_NAME`, `EMAIL_ADDRESS`, `PHONE_NUMBER`
  - `STREET_ADDRESS`, `CREDIT_CARD_NUMBER`
  - `DATE_OF_BIRTH`, `US_SOCIAL_SECURITY_NUMBER`
- AI calls use **only redacted text**
- Privacy badge shown in UI when PII is detected

### Email Security
- Gmail SMTP uses **TLS encryption** (port 587)
- App Passwords are **more secure** than regular passwords
- Email addresses are **not stored** in backend
- PDFs are generated **in-memory** (not saved to disk)

### Best Practices
1. **Never commit** `.env` file to version control
2. **Rotate App Passwords** periodically
3. **Use different App Passwords** for different applications
4. **Review Google Account activity** regularly

---

## Testing

### Test Negotiation Email
1. Upload `sample_contract.txt`
2. Wait for analysis
3. Click "Draft Negotiation Email" on any clause
4. Verify Gmail opens with correct subject/body

### Test Document Review Email
1. **Ensure `.env` is configured**
2. Upload `sample_contract.txt`
3. Click "Generate Document Email"
4. Enter your email address
5. Click "Send PDF via Email"
6. **Check inbox for PDF**

---

## Future Enhancements

Potential improvements:
- Support for **other email providers** (Outlook, SendGrid, etc.)
- **Customizable email templates**
- **Multiple recipient support**
- **Email scheduling**
- **PDF customization** (colors, fonts, logo)
- **Attachment of original document** (with PII redacted)

---

## Developer Notes

### File Locations
- **Frontend**: `lexiguard-frontend/src/pages/Results.jsx`
- **Backend**: `lexiguard-backend/main.py`
- **Config**: `lexiguard-backend/.env`
- **Dependencies**: `lexiguard-backend/requirements.txt`

### Testing Endpoint Directly
```bash
# Test /send-document-review endpoint
curl -X POST http://localhost:8000/send-document-review \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.pdf",
    "document_summary": "This is a test summary",
    "risk_summary": "Found 2 risks",
    "clauses": [
      {
        "clause": "Test clause text",
        "risk": "High",
        "explanation": "This is risky because..."
      }
    ],
    "user_email": "your-email@gmail.com"
  }'
```

### Environment Variables Reference
```env
# Google AI
GOOGLE_API_KEY=your_gemini_api_key

# Google Cloud DLP
GOOGLE_CLOUD_PROJECT=your_gcp_project_id

# Email Configuration (NEW)
GMAIL_SENDER_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=your_16_char_app_password
```

---

## Quick Start Checklist

- [ ] Generate Gmail App Password
- [ ] Add `GMAIL_SENDER_EMAIL` to `.env`
- [ ] Add `GMAIL_APP_PASSWORD` to `.env`
- [ ] Install reportlab: `pip install reportlab`
- [ ] Restart backend server
- [ ] Test negotiation email feature
- [ ] Test document review email feature
- [ ] Verify PDF received in inbox

---

## Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review backend terminal logs
3. Verify all environment variables are set
4. Ensure all dependencies are installed

**Version**: 1.0.0  
**Last Updated**: January 2025
