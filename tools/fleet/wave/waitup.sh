for i in $(seq 1 40); do
  bash /mnt/c/Users/mreca/AppData/Local/Temp/claude/49d4f5bb8a4c5013/3e443edc-7678-4608-9bc9-766d9924b1a9/scratchpad/c1/endpoints.sh > /tmp/c1ep.txt 2>&1
  if grep -q "^1 running with ssh" /tmp/c1ep.txt; then cat /tmp/c1ep.txt; exit 0; fi
  sleep 15
done
cat /tmp/c1ep.txt; echo "NOT UP"
