# Gogeta-0 cockpit from Windows PowerShell / Windows Terminal. Runs everything inside WSL Ubuntu-24.04.
#   .\cockpit\cockpit.ps1            TUI
#   .\cockpit\cockpit.ps1 --mcp      MCP server (stdio)
#   .\cockpit\cockpit.ps1 --setup | --test | --ingest
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest)
$sh = (wsl -d Ubuntu-24.04 -- wslpath -a ($PSScriptRoot + "\cockpit.sh")).Trim()
wsl -d Ubuntu-24.04 -- bash $sh @Rest
