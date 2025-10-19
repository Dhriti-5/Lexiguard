# Lexiguard Frontend Startup Script

Write-Host "🚀 Starting Lexiguard Frontend..." -ForegroundColor Cyan

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing npm dependencies..." -ForegroundColor Yellow
    npm install
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
}

# Start the React development server
Write-Host "🎯 Starting React app on http://localhost:3000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
npm start
