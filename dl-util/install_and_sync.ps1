#!/usr/bin/env pwsh
# Repo setup script: install uv if needed, sync env, and optionally start Jupyter Lab.
# This script lives in dl-util/ but operates from the repository root so relative paths work.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# Resolve repo root (parent of this script's directory)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir
Set-Location $repoRoot

Write-Host "[setup] Working directory: $repoRoot"
Write-Host "[setup] Checking for uv..."

function Install-Uv {
    try {
        Write-Host "[setup] uv not found. Installing via official script..."
        # Install in the current PowerShell session so uv becomes immediately available here.
        # Using Invoke-RestMethod (irm) piped to Invoke-Expression (iex) avoids spawning a new process
        # and reduces quoting/ExecutionPolicy pitfalls.
        irm https://astral.sh/uv/install.ps1 | iex
    } catch {
        Write-Error "[error] Failed to install uv. Please check your network or permissions."
        throw
    }
}

$addedPath = $false
function Ensure-UvPathInSession {
    # Add common user-local bin directories to PATH for this session so newly installed uv is discoverable
    $candidates = @()
    foreach ($p in @("$HOME\.local\bin", "$env:USERPROFILE\.local\bin", "$HOME\.cargo\bin", "$env:USERPROFILE\.cargo\bin")) {
        if ($p -and -not ($candidates -contains $p)) { $candidates += $p }
    }
    foreach ($dir in $candidates) {
        if (Test-Path $dir) {
            $parts = ($env:Path -split ';')
            $exists = $false
            foreach ($pp in $parts) { if ($pp.TrimEnd('\\') -ieq $dir.TrimEnd('\\')) { $exists = $true; break } }
            if (-not $exists) {
                $script:addedPath = $true
                $env:Path = "$dir;$env:Path"
                Write-Host "[setup] Added $dir to current session PATH"
            }
        }
    }
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) { Install-Uv; Ensure-UvPathInSession }

# Re-resolve after install (may require new session on some setups)
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    if (-not $addedPath) { Ensure-UvPathInSession; $uv = Get-Command uv -ErrorAction SilentlyContinue }
    if (-not $uv) {
        Write-Warning "[warn] uv command not found after installation. You may need to open a new PowerShell window."
        Write-Host   "[info] After restarting, run: uv sync"
    }
} else {
    Write-Host "[setup] Syncing Python environment with uv..."
    try { uv sync } catch { Write-Error "[error] uv sync failed. Check pyproject.toml or run manually." }

    # Optional: register a kernel name tied to this environment for convenience
    try {
        Write-Host "[setup] Registering IPython kernel (optional)..."
        uv run ipython kernel install --user --name="project"
    } catch {
        Write-Warning "[warn] ipython kernel registration skipped."
    }
}

function Should-Launch {
    $val = $env:START_JUPYTER
    switch -Regex ($val) {
        '^(1|yes|true)$'  { return $true }
        '^(0|no|false)$'  { return $false }
    }
    # Show a helpful hint before prompting
    Write-Host ''
    Write-Host 'New to using uv to manage python projects? Check the "uv Usage hints" in the README of this project' -ForegroundColor Cyan
    $ans = Read-Host "Start Jupyter Lab now? [Y/n]"
    if ($ans -match '^(n|no)$' -or $ans -match '^[Nn]') { return $false }
    return $true
}

if (Should-Launch) {
    Write-Host "[setup] Launching Jupyter Lab... (Ctrl+C to stop)"
    uv run --with jupyter jupyter lab
} else {
    Write-Host "[setup] Skipping Jupyter Lab launch. You can start it later with: uv run --with jupyter jupyter lab"
}
