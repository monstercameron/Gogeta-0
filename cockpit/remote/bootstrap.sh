#!/usr/bin/env bash
# Gogeta-0 remote bootstrap: set up a fresh rented GPU box in one go (apt + uv + pinned venv + repo + smoke test).
#
#   ssh -p PORT root@HOST 'bash -s -- <commit-sha>' < cockpit/remote/bootstrap.sh        # foreground, streams log
#   scp -P PORT cockpit/remote/bootstrap.sh root@HOST: && ssh ... 'bash bootstrap.sh <sha> --detach'   # background
#
# Idempotent: re-running skips what is already done. Everything is pulled on the instance's own (datacenter) link;
# nothing large goes over the operator's upload. The code comes only from `git clone` of the public repo at a
# pushed commit (AGENTS §0a / SQ223: no local files are copied). The dataset is NOT handled here (see cockpit fleet
# push-data, sha256-verified). Log: ~/setup.log. The last line is "ENV OK ..." or "SETUP FAILED at <step>".
#
# Pins: keep in sync with the N1X training env (`notes/reproducibility.md`). transformers and einops are needed by
# E10 (hf_llama) and fla, though requirements.txt omits them (SQ220 finding).
set -uo pipefail
SHA="${1:?usage: bootstrap.sh <commit-sha> [--detach]}"
if [ "${2:-}" = "--detach" ]; then
  [ -f "$0" ] || { echo "--detach needs the script as a file (scp it first); stdin mode runs in the foreground"; exit 2; }
  nohup setsid bash "$0" "$SHA" > /dev/null 2>&1 < /dev/null &
  echo "BOOTSTRAP_DETACHED pid $! log $HOME/setup.log"; exit 0
fi

REPO_URL="https://github.com/monstercameron/Gogeta-0.git"
VENV="$HOME/.venvs/gogeta"
DATA="$HOME/fusiondance-data"
TORCH_PIN='torch==2.14.1'; TORCH_INDEX='https://download.pytorch.org/whl/cu130'
PIP_PINS=(triton==3.8.0 flash-linear-attention==0.6.0 trackio==0.45.1 transformers==5.24.0 einops==0.8.2
          tokenizers numpy pyarrow)

exec > >(tee "$HOME/setup.log") 2>&1
t0=$(date +%s); STEP=start
fail() { echo "SETUP FAILED at $STEP (after $(( $(date +%s) - t0 ))s); see $HOME/setup.log"; exit 1; }
step() { STEP="$1"; echo "== [$(( $(date +%s) - t0 ))s] $1"; }
echo "== bootstrap start $(date -Is) commit $SHA"

step "apt: compiler + git + curl"   # Triton/Inductor need a C compiler (missing on pytorch/pytorch images)
if ! command -v gcc >/dev/null || ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential git curl ca-certificates || fail
fi
gcc --version | head -1

step "uv"
if ! command -v uv >/dev/null; then
  (python3 -m pip install -q uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) || fail
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || fail

step "python 3.12 venv"
[ -x "$VENV/bin/python" ] || { uv python install 3.12 && uv venv -p 3.12 "$VENV"; } || fail
. "$VENV/bin/activate"

step "torch ($TORCH_PIN, cu130)"
python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.14.1') else 1)" 2>/dev/null \
  || uv pip install "$TORCH_PIN" --index-url "$TORCH_INDEX" || fail

step "pinned deps"
uv pip install "${PIP_PINS[@]}" || fail

step "repo @ $SHA"
mkdir -p "$DATA"/{datasets,logs,runs,run,checkpoints}
[ -d "$HOME/Gogeta-0/.git" ] || git clone --quiet "$REPO_URL" "$HOME/Gogeta-0" || fail
cd "$HOME/Gogeta-0" && git fetch --quiet origin && git checkout --quiet --detach "$SHA" || fail
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "HEAD $(git rev-parse HEAD) != $SHA"; fail; }
[ -z "$(git status --porcelain)" ] || { echo "checkout not clean"; fail; }

step "env file"
cat > "$HOME/gogeta.env" <<EOF
. "$VENV/bin/activate"
export GOGETA_DATA="$DATA" GOGETA_TRACKER=none PYTHONUNBUFFERED=1
cd "$HOME/Gogeta-0"
EOF

step "smoke test (CUDA, torch.compile via gcc, fla, transformers)"
python - <<'PY' || fail
import sys, torch, triton, fla, transformers
assert torch.cuda.is_available(), "no CUDA"
f = torch.compile(lambda x: (x * 2).sin().sum())
y = f(torch.randn(1024, device="cuda")); torch.cuda.synchronize()
print("ENV OK python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "triton", triton.__version__, "fla", fla.__version__, "transformers", transformers.__version__,
      "gpu", torch.cuda.get_device_name(0), "compile ok")
PY
pip freeze > "$DATA/logs/pip-freeze.txt" 2>/dev/null || uv pip freeze > "$DATA/logs/pip-freeze.txt"
echo "== bootstrap done in $(( $(date +%s) - t0 ))s  (source ~/gogeta.env to use)"
