#!/usr/bin/env bash
# Gogeta-0 remote dataset fetch: rebuild code-py-v1 ON THE INSTANCE from the public Hugging Face source and prove it
# is byte-identical to ours, instead of uploading 525 MB over the operator's home link (SQ220: 494 s at ~1 MB/s).
#
#   ssh -p PORT root@HOST 'bash -s' < cockpit/remote/fetch_data.sh          # after bootstrap.sh
#
# Steps:
#   1. copy the COMMITTED tokenizer (gogeta/assets/tokenizer-code-py-v1.json) into place, so it is never retrained;
#   2. tools/prepare_data.py download,filter,tokenize (fixed seed 1234, sorted parquet order, sha1 dedupe);
#   3. verify every file against experiments/data_integrity/dataset.manifest.json (sha256) and check the repo
#      checkout is still clean (prepare_data rewrites gogeta/assets/*; identical bytes leave it clean).
# The last line is "DATA OK ..." or "DATA MISMATCH <file>". On a mismatch (for example the upstream HF dataset changed),
# fall back to the sha256-checked upload (cockpit fleet push-data). A mismatch is never used for training.
set -uo pipefail
. "$HOME/gogeta.env" || { echo "DATA FAILED: run bootstrap.sh first"; exit 1; }
export GOGETA_DATA="${GOGETA_DATA_DIR:-$GOGETA_DATA}"   # GOGETA_DATA_DIR=/scratch/x: rebuild elsewhere (testing)
mkdir -p "$GOGETA_DATA/logs"
OUT="$GOGETA_DATA/datasets/code-py-v1"
MAN=experiments/data_integrity/dataset.manifest.json
t0=$(date +%s)
exec > >(tee "$GOGETA_DATA/logs/fetch_data.log") 2>&1

mkdir -p "$OUT"
cp gogeta/assets/tokenizer-code-py-v1.json "$OUT/tokenizer.json"

# Already verified? (idempotent re-run)
verify() {
  python - "$MAN" "$OUT" <<'PY'
import hashlib, json, pathlib, sys
man, out = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
want = {"tokenizer.json": man["tokenizer"]["sha256"]}
for sp in man["splits"].values():
    for name, f in (sp.get("files") or {}).items():    # some split entries (e.g. the frozen test split) carry no files
        if name != "docs_train.jsonl":          # only needed to build train.bin; checked via train.bin
            want[name] = f["sha256"]
bad = []
for name, sha in sorted(want.items()):
    p = out / name
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
    print(f"  {'ok ' if got == sha else 'BAD'} {name} {got[:16]}")
    if got != sha: bad.append(name)
sys.exit(1 if bad else 0)
PY
}
if verify >/dev/null 2>&1; then echo "DATA OK (already present and verified)"; exit 0; fi

echo "== rebuild from Hugging Face ($(nproc) CPUs)"
# -P: don't put tools/ on sys.path, where tools/queue.py would shadow the stdlib `queue` (pyarrow imports it)
python -P tools/prepare_data.py --steps download,filter,tokenize || { echo "DATA FAILED: prepare_data error"; exit 1; }
echo "== verify against $MAN"
if ! verify; then echo "DATA MISMATCH: see above; use the sha256-checked upload instead"; exit 3; fi
if [ -n "$(git status --porcelain)" ]; then git status --porcelain; echo "DATA MISMATCH: repo assets changed"; exit 3; fi
rm -f "$OUT/docs_train.jsonl"          # 1 GB intermediate, not needed for training
echo "DATA OK rebuilt + verified in $(( $(date +%s) - t0 ))s"
