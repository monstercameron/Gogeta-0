$v = "C:\Users\mreca\AppData\Roaming\Python\Python312-arm64\Scripts\vastai.exe"
$offers = 49024363, 61346829, 56293992, 59216359, 61199767, 61700448, 37232172, 55395683
$i = 1
$out = @()
foreach ($o in $offers) {
  $label = "gogeta-cockpit-w1-$i"
  $r = & $v create instance $o --image pytorch/pytorch:latest --disk 60 --ssh --direct --label $label --raw 2>&1 | Out-String
  try { $j = $r | ConvertFrom-Json; $out += "$label offer=$o contract=$($j.new_contract) ok=$($j.success)" } catch { $out += "$label offer=$o FAILED: $($r.Substring(0, [Math]::Min(200, $r.Length)))" }
  $i++
}
$out
