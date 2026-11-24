param([Parameter(Mandatory = $true)][string]$Label)
# Destroy ONE of our labelled boxes after its results are pulled (AGENTS 0h). Refuses any label we did not create.
# Hardened 2026-11-23: under Windows PowerShell 5.1, ConvertFrom-Json does not enumerate arrays, so a Where-Object
# filter matched the WHOLE instance list and the destroy got every id. Now: force enumeration, require exactly one
# exact-label match with a single numeric id, and refuse otherwise.
$v = "C:\Users\mreca\AppData\Roaming\Python\Python312-arm64\Scripts\vastai.exe"
if ($Label -notlike "gogeta-cockpit-*") { "REFUSED: not a cockpit label"; exit 1 }
$all = @(& $v show instances --raw | ConvertFrom-Json) | ForEach-Object { $_ }
$m = @($all | Where-Object { [string]$_.label -ceq $Label })
if ($m.Count -eq 0) { "no live instance labelled $Label"; exit 0 }
if ($m.Count -ne 1) { "REFUSED: $($m.Count) instances match $Label"; exit 2 }
$i = $m[0]
$id = [string]$i.id
if ($id -notmatch '^\d+$') { "REFUSED: bad id '$id'"; exit 3 }
$up = ((Get-Date).ToUniversalTime() - [DateTimeOffset]::FromUnixTimeSeconds([int64]$i.start_date).UtcDateTime).TotalHours
"$Label id=$id uptime_h=$([math]::Round($up,2)) est_cost=$([math]::Round($up*$i.dph_total,3))"
& $v destroy instance $id -y
