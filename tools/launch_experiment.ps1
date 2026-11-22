<#
.SYNOPSIS
  Launch a registered experiment's training run (in WSL) from Windows PowerShell.

.EXAMPLE
  .\tools\launch_experiment.ps1 E3 -DryRun
  .\tools\launch_experiment.ps1 E3 -Seed 1
  .\tools\launch_experiment.ps1 E3 -Seed 2 -WaitCool
  .\tools\launch_experiment.ps1 E9 -Foreground

  Thin wrapper over tools/launch_experiment.sh, run through `wsl -d Ubuntu-24.04`. Detached runs
  keep going after this window closes (setsid nohup inside WSL). All preflight checks (prediction,
  clean code, dataset, GPU slot, GPU temperature, dashboard) happen in the .sh.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidatePattern('^[Ee]\d+$')][string]$Experiment,
    [int]$Seed = -1,
    [switch]$DryRun,
    [switch]$Foreground,
    [switch]$NoCompile,
    [long]$MaxTokens = 0,
    [switch]$WaitCool,
    [string]$Distro = 'Ubuntu-24.04'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

$shArgs = @($Experiment.ToUpper())
if ($Seed -ge 0) { $shArgs += @('--seed', "$Seed") }
if ($DryRun) { $shArgs += '--dry-run' }
if ($Foreground) { $shArgs += '--foreground' }
if ($NoCompile) { $shArgs += '--no-compile' }
if ($MaxTokens -gt 0) { $shArgs += @('--max-tokens', "$MaxTokens") }
if ($WaitCool) { $shArgs += '--wait-cool' }

& wsl.exe -d $Distro --cd $repo --exec bash tools/launch_experiment.sh @shArgs
exit $LASTEXITCODE
