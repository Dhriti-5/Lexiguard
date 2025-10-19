# Lexiguard Backend Startup Script
# This script ensures the backend runs with the correct virtual environment

Write-Host "🚀 Starting Lexiguard Backend..." -ForegroundColor Cyan

# Check if .venv exists
if (-not (Test-Path ".\.venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment and install dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  Warning: .env file not found!" -ForegroundColor Yellow
    Write-Host "Please create a .env file with your GOOGLE_API_KEY" -ForegroundColor Yellow
}

# Start the backend server
Write-Host "🎯 Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host "" 
& .\.venv\Scripts\python.exe main.py
