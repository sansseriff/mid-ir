#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running under bash even if invoked as `sh initialize_repo.sh`
if [ -z "${BASH_VERSION:-}" ]; then
	exec bash "$0" "$@"
fi

# Bootstrap this template for your own GitHub repo and GitHub Pages URL.
#
# What this does:
# - Detects or asks for your GitHub username and repo name
# - Optionally uses a custom domain from CNAME (or a --domain flag)
# - Sets the REPO_URL inside dl-util/dl.sh so it clones YOUR repo
# - Rewrites dl-util/index.html with links and instructions
# - Updates README.md "Quick install" section with the correct curl one-liner
# - Writes dl-util/repo_url.txt for reference
#
# After running this, push to GitHub. The included GitHub Action will publish
# dl-util/ to GitHub Pages. Set Pages to "Deploy from GitHub Actions" if prompted.

usage() {
	cat <<'USAGE'
Usage: ./initialize_repo.sh [--user <github_user>] [--repo <repo_name>] [--domain <custom.domain>] [--yes]

Options:
	--user     GitHub username or org (default: inferred from git remote if available)
	--repo     Repository name (default: inferred from git remote or current folder)
	--domain   Custom domain to use for GitHub Pages (overrides CNAME / github.io URL)
	--yes      Non-interactive; accept inferred defaults without prompting
USAGE
}

GH_USER=""
REPO_NAME=""
CUSTOM_DOMAIN=""
ASSUME_YES=false
REGEN_README=false
STATE_FILE="dl-util/.s2j-state.json"

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help) usage; exit 0 ;;
		--user) GH_USER=${2:-}; shift 2 ;;
		--repo) REPO_NAME=${2:-}; shift 2 ;;
		--domain) CUSTOM_DOMAIN=${2:-}; shift 2 ;;
		-y|--yes) ASSUME_YES=true; shift ;;
		--regen-readme) REGEN_README=true; shift ;;
		*) echo "Unknown argument: $1"; usage; exit 1 ;;
	esac
done

# Infer from git remote if possible
infer_from_git() {
	if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
		local url
		url=$(git config --get remote.origin.url || true)
		# Supported forms: https://github.com/user/repo.git or git@github.com:user/repo.git
		if [[ -n "$url" ]]; then
			if [[ "$url" =~ github.com[:/]{1}([^/]+)/([^/.]+) ]]; then
				GH_USER=${GH_USER:-"${BASH_REMATCH[1]}"}
				REPO_NAME=${REPO_NAME:-"${BASH_REMATCH[2]}"}
			fi
		fi
	fi
}

infer_from_git

# Fallbacks
GH_USER=${GH_USER:-"$(git config user.name 2>/dev/null || true)"}
REPO_NAME=${REPO_NAME:-"$(basename "$(pwd)")"}

# Read CNAME if present (prefer dl-util/CNAME) and no explicit --domain
if [[ -z "$CUSTOM_DOMAIN" ]]; then
	if [[ -f dl-util/CNAME ]]; then
		CUSTOM_DOMAIN=$(grep -E '^[^#[:space:]].*$' dl-util/CNAME | head -n1 || true)
	elif [[ -f CNAME ]]; then
		CUSTOM_DOMAIN=$(grep -E '^[^#[:space:]].*$' CNAME | head -n1 || true)
	fi
fi

if ! $ASSUME_YES; then
	read -r -p "GitHub user/organization [$GH_USER]: " tmp || true
	GH_USER=${tmp:-$GH_USER}
	read -r -p "Repository name [$REPO_NAME]: " tmp || true
	REPO_NAME=${tmp:-$REPO_NAME}
	read -r -p "Custom domain for Pages (blank to use github.io) [$CUSTOM_DOMAIN]: " tmp || true
	CUSTOM_DOMAIN=${tmp:-$CUSTOM_DOMAIN}
fi

if [[ -z "$GH_USER" || -z "$REPO_NAME" ]]; then
	echo "Error: Could not determine GitHub user or repo name." >&2
	exit 1
fi

REPO_URL="https://github.com/$GH_USER/$REPO_NAME.git"

if [[ -n "$CUSTOM_DOMAIN" ]]; then
	PAGES_BASE="https://$CUSTOM_DOMAIN"
else
	PAGES_BASE="https://$GH_USER.github.io/$REPO_NAME"
fi

	DOWNLOAD_CMD="curl -fsSL $PAGES_BASE/dl.sh | bash"
	WGET_CMD="wget -qO- $PAGES_BASE/dl.sh | bash"
	# PowerShell one-liner for Windows
	WIN_PS_CMD="powershell -NoProfile -ExecutionPolicy Bypass -Command \"iwr -UseBasicParsing $PAGES_BASE/dl.ps1 | iex\""
SHA256_DL_SH=""

printf "\nConfiguring with:\n"
printf "  GitHub repo: %s\n" "$REPO_URL"
printf "  Pages base:  %s\n" "$PAGES_BASE"
printf "  One-liner:   %s\n" "$DOWNLOAD_CMD"

update_dl_sh() {
	local tpl="dl-util/dl.sh.template"
	local out="dl-util/dl.sh"
	[[ -f "$tpl" ]] || { echo "Error: $tpl not found" >&2; exit 1; }
	# Replace only the first occurrence of the token to catch the assignment, preserve conditionals
	local content
	content=$(cat "$tpl")
	content=${content/__REPO_URL__/$REPO_URL}
	printf "%s" "$content" > "$out"
	chmod +x "$out" || true
}

update_dl_ps1() {
	local tpl="dl-util/dl.ps1.template"
	local out="dl-util/dl.ps1"
	[[ -f "$tpl" ]] || { echo "Error: $tpl not found" >&2; exit 1; }
	local content
	content=$(cat "$tpl")
	content=${content/__REPO_URL__/$REPO_URL}
	printf "%s" "$content" > "$out"
}

write_repo_url_txt() {
	echo "$REPO_URL" > dl-util/repo_url.txt
}

rewrite_index_html() {
	cat > dl-util/index.html <<HTML
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<title>${REPO_NAME} — bootstrap</title>
	<style>
		body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
		code, pre { background: #f6f8fa; padding: .2rem .4rem; border-radius: 4px; }
		pre { padding: .75rem 1rem; overflow-x: auto; }
		.muted { color: #6a737d; }
	</style>
	<link rel="canonical" href="$PAGES_BASE/" />
	${CUSTOM_DOMAIN:+<meta http-equiv="refresh" content="0; url=$PAGES_BASE/">}
	<meta name="robots" content="noindex" />
	<meta name="description" content="Bootstrap script for $GH_USER/$REPO_NAME" />
	<meta property="og:title" content="$REPO_NAME — bootstrap" />
	<meta property="og:description" content="Run a single command to clone and set up the repo." />
	<meta property="og:url" content="$PAGES_BASE/" />
	<meta property="og:type" content="website" />
	<meta name="twitter:card" content="summary" />
	<meta name="twitter:title" content="$REPO_NAME — bootstrap" />
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
	<h1>Bootstrap ${REPO_NAME}</h1>
	<h2>Quick install</h2>
	<p>macOS/Linux (curl):</p>
	<pre><code>${DOWNLOAD_CMD}</code></pre>
	<p>macOS/Linux (wget):</p>
	<pre><code>${WGET_CMD}</code></pre>
	<p>Windows (PowerShell):</p>
	<pre><code>${WIN_PS_CMD}</code></pre>
	<p class="muted">Integrity (SHA256 of dl.sh): <code>${SHA256_DL_SH}</code></p>
	<p class="muted">This downloads <code>dl.sh</code> from GitHub Pages and runs it.</p>
	<p>
		• <a href="$PAGES_BASE/dl.sh">dl.sh</a> &nbsp;•&nbsp;
		<a href="$PAGES_BASE/dl.ps1">dl.ps1</a> &nbsp;•&nbsp;
		<a href="https://github.com/${GH_USER}/${REPO_NAME}">Repository on GitHub</a>
	</p>
</body>
</html>
HTML
}

update_readme_block() {
	local file="README.md"
	local start='<!-- QUICK_INSTALL_START -->'
	local end='<!-- QUICK_INSTALL_END -->'

	# Build the replacement block in a temp file, avoiding heredoc issues
	local tmp
	tmp=$(mktemp)
	
	# Write line by line to avoid heredoc expansion issues
	{
		echo "${start}"
		echo "Once configured and published, anyone can bootstrap your analysis with:"
		echo ""
		echo "macOS/Linux (curl):"
		echo '```zsh'
		echo "${DOWNLOAD_CMD}"
		echo '```'
		echo ""
		echo "macOS/Linux (wget):"
		echo '```zsh'
		echo "${WGET_CMD}"
		echo '```'
		echo ""
		echo "Windows (PowerShell):"
		echo '```powershell'
		echo "${WIN_PS_CMD}"
		echo '```'
		echo ""
		echo "Integrity (SHA256 of dl.sh):"
		echo ""
		echo '```text'
		echo "${SHA256_DL_SH}"
		echo '```'
		echo ""
		echo "This line is auto-generated by \`initialize_repo.sh\` after you personalize the repo."
		echo "${end}"
	} > "$tmp"

	if grep -q "$start" "$file" && grep -q "$end" "$file"; then
		# Replace the range between start and end (inclusive) with the temp block
		awk -v start="$start" -v end="$end" -v replfile="$tmp" '
			$0 ~ start { replacing=1; while ((getline line < replfile) > 0) print line; next }
			replacing && $0 ~ end { replacing=0; next }
			!replacing { print }
		' "$file" > "$file.tmp"
		mv "$file.tmp" "$file"
	else
		# Append a new section if markers are missing
		{
			printf "\n## Quick install\n\n"; cat "$tmp"
		} >> "$file"
	fi

	rm -f "$tmp"
}

ensure_files_exist() {
	[[ -d dl-util ]] || { echo "Error: dl-util/ directory not found" >&2; exit 1; }
	[[ -f dl-util/dl.sh.template ]] || { echo "Error: dl-util/dl.sh.template not found" >&2; exit 1; }
	[[ -f dl-util/dl.ps1.template ]] || { echo "Error: dl-util/dl.ps1.template not found" >&2; exit 1; }
}

compute_sha() {
	if command -v shasum >/dev/null 2>&1; then
		SHA256_DL_SH="$(shasum -a 256 dl-util/dl.sh | awk '{print $1}')"
	elif command -v sha256sum >/dev/null 2>&1; then
		SHA256_DL_SH="$(sha256sum dl-util/dl.sh | awk '{print $1}')"
	else
		SHA256_DL_SH="(unable to compute; install shasum or sha256sum)"
	fi
}

is_bootstrapped() {
	[[ -f "$STATE_FILE" ]]
}

write_state() {
	# Write or update bootstrap state with current metadata
	ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
	cat > "$STATE_FILE" <<JSON
{
  "timestamp": "$ts",
  "gh_user": "$GH_USER",
  "repo_name": "$REPO_NAME",
  "repo_url": "$REPO_URL",
  "pages_base": "$PAGES_BASE",
  "dl_sh_sha256": "$SHA256_DL_SH"
}
JSON
}

update_readme_placeholders() {
	# Replace placeholders in README.md with actual values (first run only)
	# For subsequent runs, use update_readme_block which updates the Quick install section
	local file="README.md"
	if [[ ! -f "$file" ]]; then
		echo "Error: $file not found" >&2
		return 1
	fi
	
	# Only replace placeholders if they still exist (i.e., first run)
	if grep -q "__REPO_NAME__" "$file"; then
		if sed --version >/dev/null 2>&1; then
			# GNU sed
			sed -i "s@__REPO_NAME__@${REPO_NAME}@g" "$file"
			sed -i "s@__DOWNLOAD_CMD__@${DOWNLOAD_CMD}@g" "$file"
			sed -i "s@__WGET_CMD__@${WGET_CMD}@g" "$file"
			sed -i "s@__WIN_POWERSHELL_CMD__@${WIN_PS_CMD}@g" "$file"
			sed -i "s@__SHA256_DL_SH__@${SHA256_DL_SH}@g" "$file"
		else
			# BSD sed (macOS)
			sed -i '' "s@__REPO_NAME__@${REPO_NAME}@g" "$file"
			sed -i '' "s@__DOWNLOAD_CMD__@${DOWNLOAD_CMD}@g" "$file"
			sed -i '' "s@__WGET_CMD__@${WGET_CMD}@g" "$file"
			sed -i '' "s@__WIN_POWERSHELL_CMD__@${WIN_PS_CMD}@g" "$file"
			sed -i '' "s@__SHA256_DL_SH__@${SHA256_DL_SH}@g" "$file"
		fi
	else
		# Placeholders already replaced; update the Quick install block instead
		if grep -q "<!-- QUICK_INSTALL_START -->" "$file"; then
			update_readme_block
		fi
	fi
}

maybe_replace_readme() {
	# If already bootstrapped and not forcing regeneration, only update placeholders
	if is_bootstrapped && ! $REGEN_README; then
		echo "[info] Detected bootstrap state at $STATE_FILE — updating README placeholders only."
		if [[ -f README.md ]]; then
			update_readme_placeholders
		else
			echo "[warn] README.md not found. Run with --regen-readme to regenerate from template."
		fi
		return
	fi

	local do_short=""
	if $ASSUME_YES; then
		do_short="y"
	else
		printf "\nReplace the template README with a short project README (and save the template as README.template.md)? [Y/n]: "
		read -r do_short
	fi

	case "$do_short" in
		n|N|no|No|NO)
			echo "[info] Keeping the existing README and updating its Quick install block."
			update_readme_block
			;;
		*)
			echo "[info] Creating a short README and saving the template as README.template.md"
			# Backup existing README if present
			if [[ -f README.md ]]; then
				if [[ -f README.template.md ]]; then
					mv -f README.template.md README.template.md.bak
				fi
				mv README.md README.template.md
			fi
			
			# Copy from dl-util template
			if [[ ! -f dl-util/README.template ]]; then
				echo "Error: dl-util/README.template not found" >&2
				exit 1
			fi
			
			cp dl-util/README.template README.md
			update_readme_placeholders
			;;
	esac
}

setup_uv_environment() {
	# Offer to install uv and sync the Python environment
	local should_setup=""
	
	if $ASSUME_YES; then
		should_setup="y"
	else
		printf "\nThis template repo uses the uv Python package manager for environment and dependency management.\n"
		printf "Install uv (if not already installed) and initialize a local Python environment? [Y/n]: "
		read -r should_setup
	fi
	
	case "$should_setup" in
		n|N|no|No|NO)
			echo "[info] Skipping uv setup. You can run 'dl-util/install_and_sync.sh' later to set up the environment."
			return
			;;
	esac
	
	echo "[setup] Checking for uv..."
	if ! command -v uv >/dev/null 2>&1; then
		echo "[setup] uv not found. Installing via official script..."
		if command -v curl >/dev/null 2>&1; then
			curl -fsSL https://astral.sh/uv/install.sh | sh || {
				echo "[error] Failed to install uv. You can install it manually later.";
				return 1;
			}
			# Make uv available immediately in this shell session
			export PATH="$HOME/.local/bin:$PATH"
		elif command -v wget >/dev/null 2>&1; then
			wget -qO- https://astral.sh/uv/install.sh | sh || {
				echo "[error] Failed to install uv. You can install it manually later.";
				return 1;
			}
			# Make uv available immediately in this shell session
			export PATH="$HOME/.local/bin:$PATH"
		else
			echo "[error] Neither curl nor wget is available to install uv."
			echo "[info] Please install uv manually from https://docs.astral.sh/uv/"
			return 1
		fi
	else
		echo "[setup] uv is already installed."
	fi
	
	echo "[setup] Syncing Python environment with uv..."
	if command -v uv >/dev/null 2>&1; then
		uv sync || {
			echo "[error] uv sync failed. You may need to check pyproject.toml or run it manually later.";
			return 1;
		}
		echo "[setup] ✓ Python environment synchronized."
		echo "[info] Learn more about using uv here: https://docs.astral.sh/uv/guides/projects/"
		echo "[tip] Run Python with: uv run your_script.py"
	else
		echo "[warn] uv command not found after installation. You may need to restart your shell."
		echo "[info] After restarting, run: uv sync"
	fi
}

ensure_files_exist
update_dl_sh
update_dl_ps1
write_repo_url_txt
compute_sha
rewrite_index_html
maybe_replace_readme
write_state
setup_uv_environment

printf "\nDone. Next steps:\n"
printf "  1) Commit and push these changes to GitHub.\n"
printf "  2) Ensure GitHub Pages is set to 'Deploy from GitHub Actions'.\n"
printf "  3) Wait for the 'Deploy dl.sh to GitHub Pages' workflow to finish.\n"
printf "  4) Learn how to add packages (uv add ...) and manage the Python env: https://docs.astral.sh/uv/guides/projects/\n"
printf "  5) You can remove packages by editing pyproject.toml and running 'uv sync'.\n"
printf "  6) Share this one-liner:\n\n   %s\n\n" "$DOWNLOAD_CMD"