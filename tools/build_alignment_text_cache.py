"""Cache a common frozen text encoder over TRAIN vocabulary descriptions only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(ROOT))


def build(root, model_name, revision=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    prepared = json.loads((root / 'prepared.json').read_text(encoding='utf-8'))
    keys = prepared['identity']['vocab_keys']
    prompts = [f'{key.split("::", 1)[0].replace("_", " ")}: {key.split("::", 1)[1]}.' for key in keys]
    path = root / 'text_cache.pt'
    manifest = root / 'text_cache.json'
    identity = {'model': model_name, 'requested_revision': revision, 'keys': keys, 'prompts': prompts,
                'max_length': 48, 'pooling': 'attention_masked_mean', 'frozen': True}
    if path.exists() or manifest.exists():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if previous['identity'] != identity or hashlib.sha256(path.read_bytes()).hexdigest() != previous['sha256']:
            raise ValueError('Text cache differs or is incomplete; use a new campaign directory')
        return path
    encoder = AutoModel.from_pretrained(model_name, revision=revision).cpu().eval()
    resolved_revision = getattr(encoder.config, '_commit_hash', None)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision or resolved_revision)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=48, return_tensors='pt')
    with torch.no_grad():
        tokens = encoder(**encoded).last_hidden_state
        mask = encoded['attention_mask'].bool()
        pooled = (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    torch.save({'global': pooled.cpu(), 'tokens': tokens.cpu(), 'mask': mask.cpu(), 'keys': keys}, path)
    from tools.run_pasa_server import write_json
    write_json(manifest, {'identity': identity, 'resolved_revision': resolved_revision,
                         'hidden_size': int(tokens.shape[-1]), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign_root', required=True)
    parser.add_argument('--text_model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--text_revision', default=None)
    args = parser.parse_args()
    print(build(Path(os.path.abspath(args.campaign_root)), args.text_model, args.text_revision))


if __name__ == '__main__':
    main()
