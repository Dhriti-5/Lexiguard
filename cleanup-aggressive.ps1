# Lexiguard Cleanup Script - Option 3 (Aggressive)
# This removes alternative implementations if you're ONLY using main.py

Write-Host "🧹 Lexiguard Aggressive Cleanup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  WARNING: This will remove alternative backends!" -ForegroundColor Yellow
Write-Host "Only use this if you're CERTAIN you only need main.py" -ForegroundColor Yellow
Write-Host ""

$itemsToRemove = @()

# All from Option 2
$itemsToRemove += "lexiguard-backend\__pycache__"
$itemsToRemove += "lexiguard_sdk\__pycache__"
$itemsToRemove += "private_doc.txt"
$itemsToRemove += ".env.example.py"
$itemsToRemove += "uploads"
$itemsToRemove += "lexiguard-backend\Procfile"
$itemsToRemove += "lexiguard-backend\runtime.txt"
$itemsToRemove += "lexiguard-backend\vercel.json"
$itemsToRemove += "lexiguard-backend\Dockerfile"
$itemsToRemove += "lexiguard-backend\dlp_service.py"
$itemsToRemove += "lexiguard-backend\README.md"

# Alternative implementations
$itemsToRemove += "lexiguard-backend\app.py"
$itemsToRemove += "fastapi_app"
$itemsToRemove += "flask_app"

# SDK (if not needed)
# Uncomment these if you don't use the SDK:
# $itemsToRemove += "lexiguard_sdk"
# $itemsToRemove += "setup.py"

# Test scripts (if not needed)
# Uncomment these if you don't run tests:
# $itemsToRemove += "test_api.py"
# $itemsToRemove += "test_basic.py"
# $itemsToRemove += "test_upload.py"

Write-Host "⚠️  Items to be REMOVED:" -ForegroundColor Yellow
Write-Host ""
foreach ($item in $itemsToRemove) {
    if (Test-Path $item) {
        Write-Host "  ❌ $item" -ForegroundColor Red
    } else {
        Write-Host "  ⚠️  $item (not found)" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "Items that will be KEPT:" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/main.py (FastAPI - PRIMARY)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/.venv/ (virtual environment)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/.env (API keys)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/server.js (Node server)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-frontend/ (React app)" -ForegroundColor Green
Write-Host "  ✅ lexiguard_sdk/ (SDK) - if not commented out above" -ForegroundColor Green
Write-Host "  ✅ test_*.py (tests) - if not commented out above" -ForegroundColor Green
Write-Host "  ✅ start-all.ps1 (launcher)" -ForegroundColor Green
Write-Host ""

Write-Host "⚠️  Are you ABSOLUTELY SURE?" -ForegroundColor Yellow
Write-Host "This will remove Flask alternative and example apps!" -ForegroundColor Yellow
Write-Host ""
$confirmation = Read-Host "Type 'DELETE' to confirm aggressive cleanup"

if ($confirmation -eq "DELETE") {
    Write-Host ""
    Write-Host "Starting aggressive cleanup..." -ForegroundColor Cyan
    Write-Host ""
    
    $removed = 0
    $notFound = 0
    
    foreach ($item in $itemsToRemove) {
        if (Test-Path $item) {
            try {
                Remove-Item -Path $item -Recurse -Force -ErrorAction Stop
                Write-Host "  ✅ Removed: $item" -ForegroundColor Green
                $removed++
            } catch {
                Write-Host "  ❌ Failed to remove: $item" -ForegroundColor Red
                Write-Host "     Error: $_" -ForegroundColor DarkRed
            }
        } else {
            $notFound++
        }
    }
    
    Write-Host ""
    Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "✅ Aggressive cleanup complete!" -ForegroundColor Green
    Write-Host "   Removed: $removed items" -ForegroundColor White
    if ($notFound -gt 0) {
        Write-Host "   Not found: $notFound items" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "Your main.py backend will continue to work!" -ForegroundColor Green
    
} else {
    Write-Host ""
    Write-Host "❌ Cleanup cancelled. No files were removed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
