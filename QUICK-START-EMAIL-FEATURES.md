# 🚀 Quick Start: New Email Features

## What's New?

### 1. Gmail Integration for Negotiation Emails
Click a button and Gmail opens with your negotiation email ready to send!

### 2. PDF Document Review Reports
Send professional PDF reports to any email address.

---

## 📋 Setup (5 Minutes)

### Step 1: Get Gmail App Password

1. Go to: https://myaccount.google.com/security
2. Enable **2-Step Verification** (if not already enabled)
3. Scroll to **App passwords**
4. Create new password:
   - App: **Mail**
   - Device: **Windows Computer**
5. Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

### Step 2: Update `.env` File

Open `lexiguard-backend/.env` and add these lines:

```env
GMAIL_SENDER_EMAIL=your-email@gmail.com
GMAIL_APP_PASSWORD=abcdefghijklmnop
```

**Example:**
```env
GMAIL_SENDER_EMAIL=john.smith@gmail.com
GMAIL_APP_PASSWORD=abcdefghijklmnop
```

### Step 3: Install reportlab

```powershell
cd lexiguard-backend
pip install reportlab
```

### Step 4: Restart Backend

Stop the backend (Ctrl+C) and start again:
```powershell
python main.py
```

---

## ✅ Test It!

### Test Gmail Integration
1. Upload and analyze a document
2. Click **"Draft Negotiation Email"** on any risky clause
3. Click **"Draft in Gmail"**
4. Gmail opens - ready to send! ✉️

### Test PDF Email Reports
1. Upload and analyze a document
2. Click **"Generate Document Email"**
3. Enter your email address
4. Click **"Send PDF via Email"**
5. Check your inbox for the PDF! 📧

---

## 🔒 Privacy Maintained

- All AI calls still use **DLP-redacted text**
- Your PII is **always protected**
- Email addresses are **not stored**
- PDFs are generated **in-memory**

---

## 📚 Full Documentation

- **User Guide**: `EMAIL-FEATURES-GUIDE.md`
- **Technical Summary**: `EMAIL-FEATURES-SUMMARY.md`
- **Config Template**: `lexiguard-backend/.env.example`

---

## 🆘 Troubleshooting

**"Email configuration missing"**
→ Check `.env` file has `GMAIL_SENDER_EMAIL` and `GMAIL_APP_PASSWORD`

**"Authentication failed"**
→ Use App Password (16 chars), not regular Gmail password

**Gmail link doesn't open**
→ Use "Copy to Clipboard" button instead

**PDF not received**
→ Check spam folder and backend terminal for errors

---

## 🎉 You're Ready!

Enjoy streamlined email workflows with LexiGuard!

**Questions?** Check `EMAIL-FEATURES-GUIDE.md` for detailed help.
