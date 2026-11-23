param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$BriefFile,
    [string]$Model = "gpt-6-luna",
    [string]$Effort = "high"
)
# Runs one read-only Codex review (default GPT-6 Luna, high; -Model/-Effort override) in the Gogeta-0 repo; its final message is saved to $Out.
$codex = (Get-ChildItem "$env:LOCALAPPDATA\OpenAI\Codex\bin" -Recurse -Filter codex.exe |
          Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
$brief = Get-Content -Raw -LiteralPath $BriefFile
$sw = [Diagnostics.Stopwatch]::StartNew()
& $codex exec -m $Model -c model_reasoning_effort=$Effort -s read-only -C "C:\Users\mreca\Desktop\Gogeta-0" --ephemeral -o $Out $brief 2>&1 | Select-Object -Last 4
$rc = $LASTEXITCODE
$ok = (Test-Path $Out) -and ((Get-Item $Out).Length -gt 200)
"codex_exit=$rc ok=$ok seconds=$([math]::Round($sw.Elapsed.TotalSeconds,1)) out=$Out"
