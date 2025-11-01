#requires -version 5.1
<#
PowerShell Script to Install Dependencies for Manual Car Driving Script (Windows)
Author: [Your Name]
Date: (Get-Date -Format "yyyy-MM-dd")
#>

$ErrorActionPreference = "Stop"

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)] [string] $Name,
        [string] $InstallHint = ""
    )
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        if ($InstallHint) {
            throw "Required command '$Name' not found. $InstallHint"
        } else {
            throw "Required command '$Name' not found."
        }
    }
}

function Install-WithWinget {
    param(
        [Parameter(Mandatory = $true)] [string] $Id,
        [string] $Name = $null,
        [ValidateSet("User","Machine")] [string] $Scope = "User"
    )
    $pkg = if ($Name) { $Name } else { $Id }
    Write-Host "Ensuring $pkg is installed..."

    $scopeArg = ""
    if ($Scope -eq "User") { $scopeArg = "--scope=user" }
    if ($Scope -eq "Machine") { $scopeArg = "--scope=machine" }

    & winget install -e --id $Id --source winget --accept-package-agreements --accept-source-agreements $scopeArg | Out-Null
}

Write-Host "Starting installation of packages..."

# 1) Ensure winget is available
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    throw "winget is not available. Install 'App Installer' from Microsoft Store, then re-run this script."
}

# 2) Ensure Python 3 is installed (prefer user scope to avoid admin)
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Install-WithWinget -Id "Python.Python.3.12" -Name "Python 3.12" -Scope "User"
    # Refresh PATH for current session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
}

# 3) Optional native tool: SWIG
try {
    if (-not (Get-Command swig -ErrorAction SilentlyContinue)) {
        Install-WithWinget -Id "SWIG.SWIG" -Name "SWIG" -Scope "User"
    }
} catch {
    Write-Warning "SWIG installation skipped or failed. Prebuilt wheels may satisfy Box2D on your Python version."
}

# 4) Verify python and pip
Assert-Command -Name "python" -InstallHint "Install Python 3 from Microsoft Store or https://www.python.org."
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# 5) Create virtual environment
Write-Host "Creating a virtual environment..."
python -m venv venv

# 6) Activate the virtual environment
Write-Host "Activating the virtual environment..."
$activate = Join-Path -Path (Resolve-Path ".\venv").Path -ChildPath "Scripts\Activate.ps1"
. $activate

# 7) Install Python packages
Write-Host "Installing Python packages..."
# gym[box2d] pulls Box2D extras; box2d-py may require a compatible wheel.
python -m pip install --upgrade numpy
python -m pip install --upgrade "gym[box2d]"
python -m pip install --upgrade pyglet
python -m pip install --upgrade box2d-py
python -m pip install --upgrade pygame

# 8) Deactivate venv
Write-Host "Deactivating the virtual environment..."
deactivate

Write-Host "Installation complete."
Write-Host ""
Write-Host "To activate the virtual environment, run:"
Write-Host ".\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then, to run your script, execute:"
Write-Host "python manual_car_driving.py"
