SHA=b494ff8c6383ae31190905b3bd124bbe854fc040
cd /root
[ -d Gogeta-0-v2 ] || git clone -q https://github.com/monstercameron/Gogeta-0.git Gogeta-0-v2
cd Gogeta-0-v2 && git fetch -q origin && git checkout -q --detach $SHA && git rev-parse --short HEAD
ls -la /root/fusiondance-data/datasets/code-py-v1/docs_train.jsonl | awk '{print "docs_train bytes", $5}'
cat > /root/v2build.sh <<EOF
. /root/.venvs/gogeta/bin/activate
export GOGETA_DATA=/root/fusiondance-data CUDA_VISIBLE_DEVICES=
cd /root/Gogeta-0-v2
t0=\$(date +%s)
python experiments/data_integrity/build_v2.py --manifest /root/fusiondance-data/logs/v2.manifest.json > /root/fusiondance-data/logs/build_v2.log 2>&1
echo "V2BUILD rc=\$? wall_s=\$(( \$(date +%s) - t0 ))" >> /root/fusiondance-data/logs/build_v2.log
EOF
bash /root/v2build.sh; tail -n 2 /root/fusiondance-data/logs/build_v2.log
