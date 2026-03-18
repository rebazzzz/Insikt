$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Command-Exists($name) {
    return $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

function Ensure-WingetPackage($commandName, $packageId, $friendlyName) {
    if (Command-Exists $commandName) {
        Write-Host "$friendlyName is already installed." -ForegroundColor Green
        return
    }
    if (-not (Command-Exists "winget")) {
        Write-Host "winget was not found. Please install $friendlyName manually." -ForegroundColor Yellow
        return
    }
    Write-Step "Installing $friendlyName"
    winget install --id $packageId -e --accept-package-agreements --accept-source-agreements --silent
}

Write-Step "Checking Python"
if (-not (Command-Exists "py") -and -not (Command-Exists "python")) {
    if (Command-Exists "winget") {
        winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements --silent
        if (-not (Command-Exists "py") -and -not (Command-Exists "python")) {
            Write-Host "Python was installed, but Windows may need a fresh terminal session before it is available." -ForegroundColor Yellow
            Write-Host "Please run INSTALL_FOR_TESTER.bat once more." -ForegroundColor Yellow
            exit 0
        }
    } else {
        throw "Python 3.11+ is required and winget is not available to install it automatically."
    }
}

$pythonCmd = if (Command-Exists "py") { "py" } else { "python" }
$pythonArgs = if ($pythonCmd -eq "py") { @("-3.11") } else { @() }

Write-Step "Creating local environment"
if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    & $pythonCmd @pythonArgs -m venv venv
}

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Write-Step "Installing Insikt dependencies"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Ensure-WingetPackage "ollama" "Ollama.Ollama" "Ollama"
Ensure-WingetPackage "tesseract" "UB-Mannheim.TesseractOCR" "Tesseract OCR"

Write-Step "Preparing recommended model"
if (Command-Exists "ollama") {
    ollama pull llama3.2:latest
} else {
    Write-Host "Ollama is not available yet. You can install the model later from inside the app." -ForegroundColor Yellow
}

Write-Step "Ready"
Write-Host "Double-click START_INSIKT.bat to launch the app." -ForegroundColor Green
