# Lexiguard Cleanup Script - Option 2 (Recommended)
# This removes unused files while keeping all functionality

Write-Host "🧹 Lexiguard Project Cleanup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$itemsToRemove = @()

# Python cache files
$itemsToRemove += "lexiguard-backend\__pycache__"
$itemsToRemove += "lexiguard_sdk\__pycache__"

# Test/personal files
$itemsToRemove += "private_doc.txt"
$itemsToRemove += ".env.example.py"

# Duplicate uploads folder
$itemsToRemove += "uploads"

# Unused deployment configs
$itemsToRemove += "lexiguard-backend\Procfile"
$itemsToRemove += "lexiguard-backend\runtime.txt"
$itemsToRemove += "lexiguard-backend\vercel.json"
$itemsToRemove += "lexiguard-backend\Dockerfile"
$itemsToRemove += "lexiguard-backend\dlp_service.py"

# Redundant README
$itemsToRemove += "lexiguard-backend\README.md"

Write-Host "The following items will be removed:" -ForegroundColor Yellow
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
Write-Host "  ✅ lexiguard-backend/app.py (Flask alternative)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/.venv/ (virtual environment)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-backend/.env (API keys)" -ForegroundColor Green
Write-Host "  ✅ lexiguard-frontend/ (React app)" -ForegroundColor Green
Write-Host "  ✅ fastapi_app/ (example)" -ForegroundColor Green
Write-Host "  ✅ flask_app/ (example)" -ForegroundColor Green
Write-Host "  ✅ lexiguard_sdk/ (SDK)" -ForegroundColor Green
Write-Host "  ✅ test_*.py (test scripts)" -ForegroundColor Green
Write-Host "  ✅ start-all.ps1 (launcher)" -ForegroundColor Green
Write-Host ""

$confirmation = Read-Host "Do you want to proceed with cleanup? (yes/no)"

if ($confirmation -eq "yes" -or $confirmation -eq "y") {
    Write-Host ""
    Write-Host "Starting cleanup..." -ForegroundColor Cyan
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
    Write-Host "✅ Cleanup complete!" -ForegroundColor Green
    Write-Host "   Removed: $removed items" -ForegroundColor White
    if ($notFound -gt 0) {
        Write-Host "   Not found: $notFound items" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "Your application will continue to work normally!" -ForegroundColor Green
    Write-Host "All active components have been preserved." -ForegroundColor Green
    
} else {
    Write-Host ""
    Write-Host "❌ Cleanup cancelled. No files were removed." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
