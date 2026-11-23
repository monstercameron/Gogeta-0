# Gogeta-0 cockpit from Windows PowerShell / Windows Terminal. Runs everything inside WSL Ubuntu-24.04.
#   .\cockpit\cockpit.ps1            TUI
#   .\cockpit\cockpit.ps1 --mcp      MCP server (stdio)
#   .\cockpit\cockpit.ps1 --setup | --test | --ingest | --watchdog | --cloud ...
# Forward slashes before wslpath: backslashes are eaten on the way into WSL (PowerShell 7), which used to leave $repo empty.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$win = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path -replace '\\', '/'
$repo = (wsl -d Ubuntu-24.04 -- wslpath -a "$win" | Out-String).Trim()
if (-not $repo) { Write-Error "cockpit.ps1: could not map $win into WSL (wslpath)"; exit 2 }
wsl -d Ubuntu-24.04 --cd "$repo" -- bash cockpit/cockpit.sh @Rest
exit $LASTEXITCODE
