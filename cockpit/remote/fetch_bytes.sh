#!/usr/bin/env bash
# Gogeta-0 remote byte-dataset build: build code-py-v1-bytes ON THE INSTANCE from code-py-v1's docs_{train,val}.jsonl
# and prove it is byte-identical to ours (dataset-bytes.manifest.json, sha256). Running this on two boxes is SQ221
# (byte-dataset rebuild determinism).
#
#   ssh -p PORT root@HOST 'bash -s' < cockpit/remote/fetch_data.sh   # run with KEEP=1 so docs_train.jsonl stays
#   ssh -p PORT root@HOST 'bash -s' < cockpit/remote/fetch_bytes.sh   # then this (CPU only, ~30-60 s)
#
# Steps:
#   1. require a clean checkout and code-py-v1/docs_{train,val}.jsonl (fetch_data.sh with KEEP=1);
#   2. snapshot the COMMITTED manifest, then run build_bytes.py (it checks the docs against dataset.manifest.json
#      and its own round-trip checks; it also rewrites the repo manifest, which is restored from git afterwards);
#   3. verify train.bin, val.bin and both doc-offset files against the snapshot (sha256), and check the repo is clean.
# The last line is "BYTES OK ..." or "BYTES MISMATCH <file>" / "BYTES FAILED: <why>". A mismatch is never used for training.
set -uo pipefail
. "$HOME/gogeta.env" || { echo "BYTES FAILED: run bootstrap.sh first"; exit 1; }
export GOGETA_DATA="${GOGETA_DATA_DIR:-$GOGETA_DATA}"   # GOGETA_DATA_DIR=/scratch/x: rebuild elsewhere (testing)
mkdir -p "$GOGETA_DATA/logs"
SRC="$GOGETA_DATA/datasets/code-py-v1"
OUT="$GOGETA_DATA/datasets/code-py-v1-bytes"
MAN=experiments/byte_model/dataset-bytes.manifest.json
t0=$(date +%s)
exec > >(tee "$GOGETA_DATA/logs/fetch_bytes.log") 2>&1

[ -z "$(git status --porcelain)" ] || { git status --porcelain; echo "BYTES FAILED: checkout not clean"; exit 1; }
for f in docs_train.jsonl docs_val.jsonl; do
  [ -s "$SRC/$f" ] || { echo "BYTES FAILED: $SRC/$f missing (run fetch_data.sh with KEEP=1 first)"; exit 1; }
done
SNAP=$(mktemp /tmp/dataset-bytes.manifest.XXXXXX.json)
cp "$MAN" "$SNAP"   # the committed manifest, kept because build_bytes.py overwrites $MAN with a fresh one

verify() {
  python -P - "$SNAP" "$OUT" <<'PY'
import hashlib, json, pathlib, sys
man, out = json.load(open(sys.argv[1])), pathlib.Path(sys.argv[2])
bad = []
for split, sp in sorted(man["splits"].items()):
    for name, f in sorted(sp["files"].items()):
        p = out / name
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        size_ok = p.exists() and p.stat().st_size == f["bytes"]
        ok = got == f["sha256"] and size_ok
        print(f"  {'ok ' if ok else 'BAD'} {name} {got[:16]} {p.stat().st_size if p.exists() else 0} B")
        if not ok: bad.append(name)
sys.exit(1 if bad else 0)
PY
}
restore_manifest() { git checkout -q -- "$MAN" 2>/dev/null; }
if verify >/dev/null 2>&1; then echo "BYTES OK (already present and verified)"; restore_manifest; exit 0; fi

echo "== build code-py-v1-bytes ($(nproc) CPUs)"
# -P (and the script path, not -m): don't put tools/ or the cwd on sys.path; tools/queue.py would shadow stdlib `queue`
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 python -P experiments/byte_model/build_bytes.py --src "$SRC" --out "$OUT"
rc=$?
restore_manifest
[ $rc -eq 0 ] || { echo "BYTES FAILED: build_bytes.py exit $rc"; exit 1; }
echo "== verify against the committed $MAN"
if ! verify; then echo "BYTES MISMATCH: see above"; exit 3; fi
if [ -n "$(git status --porcelain)" ]; then git status --porcelain; echo "BYTES MISMATCH: repo changed"; exit 3; fi
echo "BYTES OK built + verified in $(( $(date +%s) - t0 ))s ($(python -P -c "import json;print(json.load(open('$SNAP'))['splits']['train']['sha256'][:16])") train.bin)"
