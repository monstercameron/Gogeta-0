cd /mnt/c/Users/mreca/Desktop/Gogeta-0
ls ~/fusiondance-data/datasets/ ~/fusiondance-data/datasets/* | head -30
~/.venvs/gogeta/bin/python - <<'P'
import json
d=json.load(open('gogeta/assets/tokenizer-code-py-v1.json',encoding='utf8'))
print(d['model']['type'],len(d['model']['vocab']),d['model'].get('byte_fallback'),d['model'].get('unk_token'))
print(d['pre_tokenizer']);print(d['normalizer']);print(d['decoder']);print(d['added_tokens']);print(d['post_processor'])
P
~/.venvs/gogeta/bin/python -c "import tokenizers,sys;print(tokenizers.__version__)"; ~/.venvs/gogeta/bin/python -c "import sentencepiece" 2>&1|tail -1
ls ~/fusiondance-data/hf-cache/*/snapshots/*/ | head -30
nproc; free -m|head -2
