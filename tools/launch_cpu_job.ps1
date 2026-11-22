<#
.SYNOPSIS
  Run a small CPU-only job (in WSL, detached) within the compute budget, from Windows PowerShell.

.EXAMPLE
  .\tools\launch_cpu_job.ps1 -Name k4 -- /root/.venvs/gogeta/bin/python experiments/checks/k4_ternary_gap_vs_size.py
  .\tools\launch_cpu_job.ps1 -DryRun -- python3 -c "print(42)"

  Thin wrapper over tools/launch_cpu_job.sh, run through `wsl -d Ubuntu-24.04`: no GPU
  (CUDA_VISIBLE_DEVICES=""), 2 threads, at most 8 CPU jobs alive. Everything after `--` is the
  command, run from the repo root inside WSL (use Linux paths). Arguments pass through PowerShell
  and wsl.exe quoting; for commands with tricky quoting, call the .sh from a WSL shell instead.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Name = '',
    [switch]$DryRun,
    [string]$Distro = 'Ubuntu-24.04',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Command
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$Command = @($Command | Where-Object { $_ -ne '--' })
if ($Command.Count -eq 0) { Write-Error 'Give the command after --, e.g. -- python3 -c "print(1)"'; exit 2 }

$shArgs = @()
if ($Name) { $shArgs += @('--name', $Name) }
if ($DryRun) { $shArgs += '--dry-run' }
$shArgs += '--'
$shArgs += $Command

& wsl.exe -d $Distro --cd $repo --exec bash tools/launch_cpu_job.sh @shArgs
exit $LASTEXITCODE
