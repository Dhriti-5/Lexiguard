# Lexiguard - Complete Application Startup Script
Write-Host "Starting Lexiguard Services..." -ForegroundColor Cyan

function Start-Service {
    param([string]$Name, [string]$Path, [string]$Command)
    Write-Host "Starting $Name..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Path'; $Command"
    Start-Sleep -Seconds 2
}

if (-not (Test-Path ".\lexiguard-backend\.env")) {
    Write-Host "WARNING: .env file not found in lexiguard-backend!" -ForegroundColor Yellow
}

Write-Host ""
Start-Service -Name "Backend (Port 8000)" -Path "$PWD\lexiguard-backend" -Command "& .\.venv\Scripts\python.exe main.py"

if (Test-Path ".\lexiguard-backend\server.js") {
    Start-Service -Name "Node Server (Port 5000)" -Path "$PWD\lexiguard-backend" -Command "node server.js"
}

Start-Service -Name "Frontend (Port 3000)" -Path "$PWD\lexiguard-frontend" -Command "npm start"

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
