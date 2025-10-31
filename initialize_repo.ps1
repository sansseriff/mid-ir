#!/usr/bin/env pwsh
[CmdletBinding()]
param(
    [Alias('u')]
    [string]$User,
    [Alias('r')]
    [string]$Repo,
    [Alias('d')]
    [string]$Domain,
    [Alias('y')]
    [switch]$Yes,
    [Alias('regen')]
    [switch]$RegenReadme,
    # Guard: absorb any stray `-and` passed by shells or tooling; ignored by script logic
    [switch]$and
)

$ErrorActionPreference = 'Stop'

# Basic trap to surface unexpected errors with context
trap {
    Write-Host "[trap] $($_.Exception.GetType().FullName): $($_.Exception.Message)"
    Write-Host "[trap] At: $($_.InvocationInfo.PositionMessage)"
    break
}

function Write-Usage {
@'
Usage: pwsh ./initialize_repo.ps1 [-User <github_user>] [-Repo <repo_name>] [-Domain <custom.domain>] [-Yes] [-RegenReadme]

Options:
  -User         GitHub username or org (default: inferred from git remote if available)
  -Repo         Repository name (default: inferred from git remote or current folder)
  -Domain       Custom domain to use for GitHub Pages (overrides CNAME / github.io URL)
  -Yes          Non-interactive; accept inferred defaults without prompting
  -RegenReadme  Force regenerating README from template
'@
}

# Resolve repo root (script directory's parent)
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

$STATE_FILE = 'dl-util/.s2j-state.json'

function Infer-From-Git {
    try {
        git rev-parse --is-inside-work-tree *> $null
        if ($LASTEXITCODE -eq 0) {
            $url = git config --get remote.origin.url 2>$null
            if ($url) {
                $m = [regex]::Match($url, 'github\.com[:/]{1}([^/]+)/([^/.]+)')
                if ($m.Success) {
                    if (-not $script:User) { $script:User = $m.Groups[1].Value }
                    if (-not $script:Repo) { $script:Repo = $m.Groups[2].Value }
                }
            }
        }
    } catch {}
}

Infer-From-Git

# Fallbacks
if (-not $User) { $User = (git config user.name 2>$null) }
if (-not $Repo) { $Repo = Split-Path -Leaf (Get-Location) }

# Read CNAME if present and no explicit domain
if (-not $Domain) {
    $cnamePath = if (Test-Path 'dl-util/CNAME') { 'dl-util/CNAME' } elseif (Test-Path 'CNAME') { 'CNAME' } else { $null }
    if ($cnamePath) {
        $line = Get-Content $cnamePath | Where-Object { $_ -and ($_ -notmatch '^[#\s]') } | Select-Object -First 1
        if ($line) { $Domain = $line.Trim() }
    }
}

if (-not $Yes) {
    $tmp = Read-Host "GitHub user/organization [$User]"; if ($tmp) { $User = $tmp }
    $tmp = Read-Host "Repository name [$Repo]"; if ($tmp) { $Repo = $tmp }
    $tmp = Read-Host "Custom domain for Pages (blank to use github.io) [$Domain]"; if ($tmp -ne $null) { if ($tmp -ne '') { $Domain = $tmp } }
}

if (-not $User -or -not $Repo) {
    Write-Error "Could not determine GitHub user or repo name."; exit 1
}

$RepoUrl = "https://github.com/$User/$Repo.git"
$PagesBase = if ($Domain) { "https://$Domain" } else { "https://$User.github.io/$Repo" }

$DownloadCmd = "curl -fsSL $PagesBase/dl.sh | bash"
$WgetCmd    = "wget -qO- $PagesBase/dl.sh | bash"
# Build a Windows one-liner string with proper quoting for PowerShell 5.1+
# Use -f formatting to avoid nested quote-escaping issues
$WinPsCmd   = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "iwr -UseBasicParsing {0}/dl.ps1 | iex"' -f $PagesBase
$Sha256DlSh = ''

Write-Host "`nConfiguring with:"
Write-Host "  GitHub repo: $RepoUrl"
Write-Host "  Pages base:  $PagesBase"
Write-Host "  One-liner:   $DownloadCmd"

function Ensure-Files-Exist {
    foreach ($p in 'dl-util','dl-util/dl.sh.template','dl-util/dl.ps1.template','dl-util/README.template') {
        if (-not (Test-Path $p)) { Write-Error "Required file not found: $p"; exit 1 }
    }
}

function Replace-FirstToken {
    param(
        [Parameter(Mandatory)] [string]$Content,
        [Parameter(Mandatory)] [string]$Token,
        [Parameter(Mandatory)] [string]$Replacement
    )
    $parts = $Content -split [regex]::Escape($Token), 2
    if ($parts.Count -eq 1) { return $Content }
    return ($parts[0] + $Replacement + $parts[1])
}

function Write-Utf8NoBomLF {
    param(
        [Parameter(Mandatory)] [string]$Path,
        [Parameter(Mandatory)] [string]$Content
    )
    # Normalize to LF endings, write UTF8 without BOM
    $lf = $Content -replace "\r\n?", "`n"
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $lf, $enc)
}

function Update-DlSh {
    $tpl = 'dl-util/dl.sh.template'
    $out = 'dl-util/dl.sh'
    $content = Get-Content -Raw -Encoding UTF8 $tpl
    $content = Replace-FirstToken -Content $content -Token '__REPO_URL__' -Replacement $RepoUrl
    Write-Utf8NoBomLF -Path $out -Content $content
    try { & git update-index --chmod=+x $out *> $null } catch {}
}

function Update-DlPs1 {
    $tpl = 'dl-util/dl.ps1.template'
    $out = 'dl-util/dl.ps1'
    $content = Get-Content -Raw -Encoding UTF8 $tpl
    $content = Replace-FirstToken -Content $content -Token '__REPO_URL__' -Replacement $RepoUrl
    Write-Utf8NoBomLF -Path $out -Content $content
}

function Write-RepoUrlTxt {
    $out = 'dl-util/repo_url.txt'
    Write-Utf8NoBomLF -Path $out -Content $RepoUrl
}

function Compute-Sha {
    try {
        $global:Sha256DlSh = (Get-FileHash -Algorithm SHA256 'dl-util/dl.sh').Hash.ToLower()
    } catch {
        $global:Sha256DlSh = '(unable to compute, Get-FileHash missing)'
    }
}

function Rewrite-IndexHtml {
    $html = @"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>$Repo - bootstrap</title>
    <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
        code, pre { background: #f6f8fa; padding: .2rem .4rem; border-radius: 4px; }
        pre { padding: .75rem 1rem; overflow-x: auto; }
        .muted { color: #6a737d; }
    </style>
    <link rel="canonical" href="$PagesBase/" />
    <meta name="robots" content="noindex" />
    <meta name="description" content="Bootstrap script for $User/$Repo" />
    <meta property="og:title" content="$Repo - bootstrap" />
    <meta property="og:description" content="Run a single command to clone and set up the repo." />
    <meta property="og:url" content="$PagesBase/" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary" />
    <meta name="twitter:title" content="$Repo - bootstrap" />
    <meta name="twitter:description" content="Run a single command to clone and set up the repo." />
    <link rel="icon" href="data:;base64,iVBORw0KGgo=" />
    <meta http-equiv="X-Content-Type-Options" content="nosniff" />
    <meta http-equiv="Referrer-Policy" content="no-referrer" />
    <meta http-equiv="X-Frame-Options" content="DENY" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';" />
    <meta name="X-Robots-Tag" content="noarchive, noimageindex" />
    <meta name="X-Content-Security-Policy" content="default-src 'none'" />
</head>
<body>
    <h1>Bootstrap $Repo</h1>
    <h2>Quick install</h2>
    <p>macOS/Linux (curl):</p>
    <pre><code>$DownloadCmd</code></pre>
    <p>macOS/Linux (wget):</p>
    <pre><code>$WgetCmd</code></pre>
    <p>Windows (PowerShell):</p>
    <pre><code>$WinPsCmd</code></pre>
    <p class="muted">Integrity (SHA256 of dl.sh): <code>$Sha256DlSh</code></p>
    <p class="muted">This downloads <code>dl.sh</code> from GitHub Pages and runs it.</p>
    <p>
        - <a href="$PagesBase/dl.sh">dl.sh</a> &nbsp;-&nbsp;
        <a href="$PagesBase/dl.ps1">dl.ps1</a> &nbsp;-&nbsp;
        <a href="https://github.com/$User/$Repo">Repository on GitHub</a>
    </p>
</body>
</html>
"@
    Write-Utf8NoBomLF -Path 'dl-util/index.html' -Content $html
}

function Get-QuickInstallBlock {
    @(
        '<!-- QUICK_INSTALL_START -->',
        'Once configured and published, anyone can bootstrap your analysis with:',
        '',
        'macOS/Linux (curl):',
        '```zsh',
        $DownloadCmd,
        '```',
        '',
        'macOS/Linux (wget):',
        '```zsh',
        $WgetCmd,
        '```',
        '',
        'Windows (PowerShell):',
        '```powershell',
        $WinPsCmd,
        '```',
        '',
        'Integrity (SHA256 of dl.sh):',
        '',
        '```text',
        $Sha256DlSh,
        '```',
        '',
        'This line is auto-generated by `initialize_repo.ps1` after you personalize the repo.',
        '<!-- QUICK_INSTALL_END -->'
    ) -join "`n"
}

function Update-Readme-Placeholders {
    $file = 'README.md'
    if (-not (Test-Path $file)) { Write-Warning "README.md not found"; return }
    $text = Get-Content -Raw -Encoding UTF8 $file
    $rawBase = "https://raw.githubusercontent.com/$User/$Repo/main"

    if ($text -match '__REPO_NAME__') {
        # Use literal replacements to avoid introducing backslashes from regex escaping
        $text = $text.Replace('__REPO_NAME__', $Repo)
        $text = $text.Replace('__DOWNLOAD_CMD__', $DownloadCmd)
        $text = $text.Replace('__WGET_CMD__', $WgetCmd)
        $text = $text.Replace('__WIN_POWERSHELL_CMD__', $WinPsCmd)
        $text = $text.Replace('__SHA256_DL_SH__', $Sha256DlSh)
        $text = $text.Replace('__RAW_BASE_URL__', $rawBase)
        Write-Utf8NoBomLF -Path $file -Content $text
    } elseif ($text -match '<!-- QUICK_INSTALL_START -->' -and $text -match '<!-- QUICK_INSTALL_END -->') {
        $block = Get-QuickInstallBlock
        $text  = [regex]::Replace($text, '<!-- QUICK_INSTALL_START -->[\s\S]*?<!-- QUICK_INSTALL_END -->', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $block })
        # Replace any remaining raw URL placeholders globally
        $text = $text.Replace('__RAW_BASE_URL__', $rawBase)
        Write-Utf8NoBomLF -Path $file -Content $text
    }
}

function Maybe-Replace-Readme {
    # PS 5.1 can misparse `Test-Path $STATE_FILE -and ...` as passing `-and` to Test-Path.
    # Parenthesize each side so `-and` operates on expressions, not command arguments.
    if ((Test-Path $STATE_FILE) -and (-not $RegenReadme)) {
    Write-Host "[info] Detected bootstrap state at $STATE_FILE - updating README placeholders only."
        Update-Readme-Placeholders
        return
    }
    $doShort = if ($Yes) { 'y' } else { Read-Host "`nReplace the template README with a short project README (and save the template as README.template.md)? [Y/n]" }
    if ($doShort -match '^(n|no)$') {
        Write-Host "[info] Keeping the existing README and updating its Quick install block."
        $block = Get-QuickInstallBlock
        $file  = 'README.md'
        $text = Get-Content -Raw -Encoding UTF8 $file
        if ($text -match '<!-- QUICK_INSTALL_START -->' -and $text -match '<!-- QUICK_INSTALL_END -->') {
            $text  = [regex]::Replace($text, '<!-- QUICK_INSTALL_START -->[\s\S]*?<!-- QUICK_INSTALL_END -->', [System.Text.RegularExpressions.MatchEvaluator]{ param($m) $block })
        } else {
            $text += "`n## Quick install`n`n$block"
        }
        Write-Utf8NoBomLF -Path $file -Content $text
    } else {
        Write-Host "[info] Creating a short README and saving the template as README.template.md"
        if (Test-Path 'README.md') {
            if (Test-Path 'README.template.md') { Move-Item -Force 'README.template.md' 'README.template.md.bak' }
            Move-Item -Force 'README.md' 'README.template.md'
        }
        Copy-Item -Force 'dl-util/README.template' 'README.md'
        Update-Readme-Placeholders
    }
}

function Write-State {
    $obj = [ordered]@{
        timestamp   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
        gh_user     = $User
        repo_name   = $Repo
        repo_url    = $RepoUrl
        pages_base  = $PagesBase
        dl_sh_sha256= $Sha256DlSh
    }
    $json = $obj | ConvertTo-Json -Depth 3
    Write-Utf8NoBomLF -Path $STATE_FILE -Content $json
}

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
                $env:Path = "$dir;$env:Path"
                Write-Host "[setup] Added $dir to current session PATH"
            }
        }
    }
}

function Setup-UvEnvironment {
    if ($Yes) { $should='y' } else { $should = Read-Host "`nThis template uses the uv Python package manager. Install uv (if needed) and initialize a local environment? [Y/n]" }
    if ($should -match '^(n|no)$') { Write-Host "[info] Skipping uv setup."; return }

    Write-Host "[setup] Checking for uv..."
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        Write-Host "[setup] uv not found. Installing via official script..."
        try {
            # Install in current session; avoids nested process and keeps PATH changes visible here
            irm https://astral.sh/uv/install.ps1 | iex
            Ensure-UvPathInSession
        } catch {
            Write-Warning "[error] Failed to install uv. Install manually: https://docs.astral.sh/uv/"
            return
        }
    }
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        try { uv sync } catch { Write-Warning "[error] uv sync failed. Run manually later." }
    Write-Host "[setup] Python environment synchronized."
        Write-Host "[tip] Run Python with: uv run your_script.py"
    } else {
        Ensure-UvPathInSession
        $uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($uv) {
            try { uv sync } catch { Write-Warning "[error] uv sync failed. Run manually later." }
            Write-Host "[setup] Python environment synchronized."
            Write-Host "[tip] Run Python with: uv run your_script.py"
        } else {
            Write-Warning "[warn] uv not found after installation. You may need to start a new PowerShell session. Then run: uv sync"
        }
    }
}

Ensure-Files-Exist
Write-Host "[step] Ensure-Files-Exist completed"
Update-DlSh
Write-Host "[step] Update-DlSh completed"
Update-DlPs1
Write-Host "[step] Update-DlPs1 completed"
Write-RepoUrlTxt
Write-Host "[step] Write-RepoUrlTxt completed"
Compute-Sha
Write-Host "[step] Compute-Sha completed (SHA: $Sha256DlSh)"
Rewrite-IndexHtml
Write-Host "[step] Rewrite-IndexHtml completed"
Maybe-Replace-Readme
Write-Host "[step] Maybe-Replace-Readme completed"
Write-State
Write-Host "[step] Write-State completed"
Setup-UvEnvironment
Write-Host "[step] Setup-UvEnvironment completed"

Write-Host "`nDone. Next steps:"
Write-Host "  1) Commit and push these changes to GitHub."
Write-Host "  2) Ensure GitHub Pages is set to 'Deploy from GitHub Actions'."
Write-Host "  3) Wait for the 'Deploy dl.sh to GitHub Pages' workflow to finish."
Write-Host "  4) Learn how to add packages (uv add ...) and manage the Python env: https://docs.astral.sh/uv/guides/projects/"
Write-Host "  5) You can remove packages by editing pyproject.toml and running 'uv sync'"
Write-Host ("  6) Share this one-liner:`n`n   {0}`n" -f $DownloadCmd)