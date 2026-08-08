#!/bin/bash
# One-shot setup for a fresh RunPod pod (A6000 48GB recommended).
# Usage:  bash pod_setup.sh
# Idempotent: safe to re-run; downloads skip if already present.
set -u

echo "=== [1/4] Python deps (transformers pinned per reproducibility notes) ==="
pip install -q --no-warn-script-location \
  "transformers==5.7.0" accelerate peft pyyaml pytest numpy datasets matplotlib \
  "huggingface_hub[hf_transfer]"

mkdir -p /workspace/models /workspace/results /workspace/logs

echo "=== [2/4] Model downloads (ungated only — no HF token required) ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
dl () {  # dl <repo_id> <local_dir>
  if [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ]; then
    echo "  already present: $2"
  else
    echo "  downloading $1 -> $2"
    hf download "$1" --local-dir "$2" || echo "  WARNING: download failed for $1"
  fi
}
dl Qwen/Qwen2.5-7B                       /workspace/models/qwen2.5-7b
dl Qwen/Qwen2.5-1.5B                     /workspace/models/qwen2.5-1.5b
dl NousResearch/Meta-Llama-3.1-8B        /workspace/models/llama-3.1-8b
dl mistralai/Mistral-7B-v0.1             /workspace/models/mistral-7b
# Fallback ungated Mistral mirror if the official repo rejects anonymous pulls:
if [ ! -f /workspace/models/mistral-7b/config.json ]; then
  dl unsloth/mistral-7b-v0.3             /workspace/models/mistral-7b
fi

echo "=== [3/4] Warm-up corpus (WikiText-2, 400 passages) ==="
python - <<'PY'
import json, os
path = '/workspace/sleep-research/experiments/data/warmup_corpus.json'
if os.path.exists(path):
    print('  corpus already present')
else:
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    sents = []
    for row in ds:
        t = row['text'].strip()
        if 80 < len(t) < 400 and not t.startswith('='):
            sents.append(t)
        if len(sents) >= 400: break
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(sents, open(path, 'w'))
    print('  corpus sentences:', len(sents))
PY

echo "=== [4/4] Sanity: GPU + code tree ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls /workspace/sleep-research/ 2>/dev/null || echo "  NOTE: upload code tree to /workspace/sleep-research first"
echo "SETUP_DONE"
