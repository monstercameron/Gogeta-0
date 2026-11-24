V=/mnt/c/Users/mreca/AppData/Roaming/Python/Python312-arm64/Scripts/vastai.exe
S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
timeout 120 $V show instances --raw > /tmp/c1_inst.json 2>&1
python3 - <<'PY'
import json
d=json.load(open('/tmp/c1_inst.json'))
rows=[]
for x in sorted(d,key=lambda x:x.get('label') or ''):
    p=(x.get('ports') or {}).get('22/tcp')
    port=p[0]['HostPort'] if p else None
    print(x.get('label'),x['id'],x.get('actual_status'),x.get('public_ipaddr'),port,round(x.get('dph_total',0),3))
    if x.get('actual_status')=='running' and port and (x.get('label') or '').startswith('gogeta-cockpit-c1-'):
        rows.append(f"{x['label']} {x['id']} {x['public_ipaddr']} {port} {round(x.get('dph_total',0),3)}")
open('/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad/c1/endpoints.txt','w').write("\n".join(rows)+("\n" if rows else ""))
print(len(rows),"running with ssh")
PY
