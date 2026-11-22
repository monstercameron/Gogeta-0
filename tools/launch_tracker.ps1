<#
.SYNOPSIS
  Start / stop / check the local Trackio dashboard (runs in WSL) from Windows PowerShell.

.EXAMPLE
  .\tools\launch_tracker.ps1 status
  .\tools\launch_tracker.ps1 start -Open
  .\tools\launch_tracker.ps1 start -Port 7862
  .\tools\launch_tracker.ps1 stop -Port 7862
  .\tools\launch_tracker.ps1 url

  Thin wrapper over tools/launch_tracker.sh (same subcommands), run through `wsl -d Ubuntu-24.04`.
  Localhost only; -Open opens http://127.0.0.1:<port> in the default browser once it answers.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][ValidateSet('start', 'stop', 'status', 'url')][string]$Command = 'status',
    [int]$Port = 0,
    [string]$Project = '',
    [int]$Timeout = 0,
    [switch]$Open,
    [string]$Distro = 'Ubuntu-24.04'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$shArgs = @($Command)
if ($Port -gt 0) { $shArgs += @('--port', "$Port") }
if ($Project) { $shArgs += @('--project', $Project) }
if ($Timeout -gt 0) { $shArgs += @('--timeout', "$Timeout") }

$out = & wsl.exe -d $Distro --cd $repo --exec bash tools/launch_tracker.sh @shArgs 2>&1 | ForEach-Object { "$_" }
$code = $LASTEXITCODE
$out | ForEach-Object { Write-Output $_ }

if ($Open -and $code -eq 0) {
    $url = $out | Where-Object { $_ -match '^http://127\.0\.0\.1:\d+$' } | Select-Object -Last 1
    if ($url) { Start-Process $url } else { Write-Warning 'No dashboard URL in the output; nothing opened.' }
}
exit $code
