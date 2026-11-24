S=/mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad
while read -r label id host port dph; do
  echo "$label: $(ssh -n -i /root/.ssh/gogeta_vast -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=12 -p $port root@$host 'grep -E "^(RUN|SCORE)" /root/fusiondance-data/logs/worker.status 2>/dev/null | cut -c1-110 | tr "\n" " "; f=$(ls -t /root/fusiondance-data/logs/a*.log /root/fusiondance-data/logs/e*.log 2>/dev/null | head -1); echo "[$(basename $f .log): $(grep -o "step *[0-9]*/6104" $f | tail -1)]"' 2>/dev/null)"
done < $S/c1/endpoints_all.txt
