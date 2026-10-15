"""Create strict JSON viewer copies without modifying experiment evidence or running jobs."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def repair(root, pattern='history.json'):
    root = Path(os.path.abspath(root))
    if not root.is_dir():
        raise ValueError('Experiment directory does not exist')
    destination = root / 'json_view'
    files = [path for path in root.rglob(pattern) if path.is_file() and destination not in path.parents]
    converted, skipped = [], []
    for path in files:
        constants = []
        def replace_constant(token):
            constants.append(token)
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=replace_constant)
        except (json.JSONDecodeError, UnicodeError) as error:
            skipped.append(str(path))
            print(f'SKIP possibly incomplete file: {path}: {error}')
            continue
        if not constants:
            continue
        output = destination / path.relative_to(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                             suffix='.tmp', delete=False) as file:
                temporary = file.name
                json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        converted.append(str(output))
        print(f'VIEW COPY ({len(constants)} nonfinite metrics -> null): {output}')
    return {'converted': converted, 'skipped': skipped}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--pattern', default='history.json', help='Use "*.json" to convert other result files too')
    args = parser.parse_args()
    result = repair(args.root, args.pattern)
    print(f'Converted {len(result["converted"])} files; originals unchanged; skipped {len(result["skipped"])}.')
