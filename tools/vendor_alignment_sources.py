"""Copy pinned academic source modules from locally verified official repositories."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parent.parent


def main():
    sources = ROOT / 'output' / 'sota_upstream'
    expected = {'CARZero': 'fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d',
                'RadZero': '656ae5f1af3f106e96c95542ce3ee5c0ee8777fc',
                'transformers': 'a22a4378d97d06b7a1d9abad6e0086d30fdea199'}
    for name, commit in expected.items():
        actual = subprocess.check_output(['git', '-C', str(sources / name), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise ValueError(f'{name} checkout is not the verified pinned commit {commit}')
    destination = ROOT / 'third_party'
    records = {}
    for project in ('CARZero', 'RadZero'):
        target = destination / project.lower()
        target.mkdir(parents=True, exist_ok=True)
        (target / '__init__.py').write_text('', encoding='utf-8')
        (target / 'LICENSE').write_bytes((sources / project / 'LICENSE').read_bytes())
    decoder = sources / 'CARZero' / 'CARZero' / 'models' / 'transformer_decoder.py'
    cleaned_decoder = '\n'.join(line.rstrip() for line in decoder.read_text(encoding='utf-8').splitlines()) + '\n'
    (destination / 'carzero' / 'transformer_decoder.py').write_text(cleaned_decoder, encoding='utf-8')
    dqn = sources / 'CARZero' / 'CARZero' / 'models' / 'dqn_wo_self_atten.py'
    text = dqn.read_text(encoding='utf-8')
    for line in ('import torchvision.models as models\n', 'import ipdb\n',
                 'from transformers import AutoModel,BertConfig,AutoTokenizer\n'):
        text = text.replace(line, '')
    text = text.replace('from ..models.transformer_decoder import *', 'from .transformer_decoder import *')
    text = '\n'.join(line.rstrip() for line in text.splitlines()) + '\n'
    (destination / 'carzero' / 'dqn.py').write_text(text, encoding='utf-8')
    losses = sources / 'RadZero' / 'exp' / 'cxr_pt' / 'model' / 'losses.py'
    original = losses.read_text(encoding='utf-8')
    tree = ast.parse(original)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SimilarityLogit')
    snippet = '\n'.join(original.splitlines()[node.lineno - 1:node.end_lineno])
    snippet = snippet.replace(').squeeze()', ').squeeze(-1).squeeze(-1)')
    header = 'import math\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n\n'
    (destination / 'radzero' / 'similarity.py').write_text(header + snippet + '\n', encoding='utf-8')
    reference_names = {'multi_positive_nce_loss', 'get_row_loss', 'get_col_loss'}
    reference_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in reference_names]
    reference = '\n\n'.join('\n'.join(original.splitlines()[node.lineno - 1:node.end_lineno]) for node in reference_nodes)
    (destination / 'radzero' / 'reference_losses.py').write_text('import torch\n\n' + reference + '\n', encoding='utf-8')
    hf_source = sources / 'transformers' / 'src' / 'transformers' / 'models' / 'dinov2' / 'modeling_dinov2.py'
    hf_text = hf_source.read_text(encoding='utf-8')
    names = {'Dinov2SelfAttention', 'Dinov2SelfOutput', 'Dinov2Attention', 'Dinov2LayerScale',
             'drop_path', 'Dinov2DropPath', 'Dinov2MLP', 'Dinov2SwiGLUFFN', 'Dinov2Layer'}
    nodes = [node for node in ast.parse(hf_text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    chunks = ['\n'.join(hf_text.splitlines()[node.lineno - 1:node.end_lineno]) for node in nodes]
    chunks.insert(-1, "DINOV2_ATTENTION_CLASSES = {'eager': Dinov2Attention}")
    hf_header = '# Copyright 2023 Meta Platforms, Inc. and affiliates and HuggingFace Inc.\n'
    hf_header += '# Licensed under Apache-2.0; extracted unmodified forward blocks from transformers v4.49.0.\n'
    hf_header += 'from __future__ import annotations\nimport math\nimport torch\nfrom torch import nn\nACT2FN = {"gelu": nn.functional.gelu}\n\n'
    (destination / 'radzero' / 'dinov2_blocks.py').write_text(hf_header + '\n\n'.join(chunks) + '\n', encoding='utf-8')
    (destination / 'radzero' / 'TRANSFORMERS_LICENSE').write_bytes((sources / 'transformers' / 'LICENSE').read_bytes())
    records['CARZero'] = {'repository': 'https://github.com/laihaoran/CARZero',
        'commit': 'fa4e09cdfe7801e36e83d96815e3fcf66ffdd13d', 'license': 'Apache-2.0',
        'upstream_files': {'dqn_wo_self_atten.py': hashlib.sha256(dqn.read_bytes()).hexdigest(),
                           'transformer_decoder.py': hashlib.sha256(decoder.read_bytes()).hexdigest()},
        'changes': 'Remove unused heavyweight imports; fix package-relative import; trim trailing whitespace. Alignment computation unchanged.'}
    records['RadZero'] = {'repository': 'https://github.com/deepnoid-ai/RadZero',
        'commit': '656ae5f1af3f106e96c95542ce3ee5c0ee8777fc', 'license': 'CC-BY-NC-4.0',
        'upstream_files': {'losses.py': hashlib.sha256(losses.read_bytes()).hexdigest()},
        'changes': 'Extract SimilarityLogit and reference loss functions; preserve singleton batch/candidate axes by explicit squeeze(-1).'}
    records['DINOv2_blocks'] = {'repository': 'https://github.com/huggingface/transformers',
        'commit': 'a22a4378d97d06b7a1d9abad6e0086d30fdea199', 'tag': 'v4.49.0', 'license': 'Apache-2.0',
        'sha256': hashlib.sha256(hf_source.read_bytes()).hexdigest(),
        'changes': 'Extract eager non-pruned GELU forward blocks; stack two as the official RadZero visual adapter. No external HF import during training.'}
    (destination / 'ALIGNMENT_SOURCES.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
