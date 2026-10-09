"""Package only required source/docs; never include data, credentials or old runs."""
import hashlib
import argparse
import os
from pathlib import Path
import zipfile


ROOT = Path(os.path.abspath(__file__)).parent.parent
FILES = ['train_semantic_alignment.py', 'semantic_evaluation.py', 'experiment_model.py',
         'experiment_dataset.py', 'dataset.py', 'semantic_graph_visualize.py', 'requirements.txt',
         'tools/run_pasa_server.py', 'tools/package_pasa_server.py', 'utils/bootstrap_semantic_5seed.py',
         'tests/test_semantic_review_revision.py', 'tests/test_pasa_server_runner.py',
         'SERVER_RUN.md', 'PASA_EVALUATION_PROTOCOL.md',
         'revision/pasa_review_20261009/REVISION_PLAN.md',
         'revision/pasa_review_20261009/CODE_CHANGES.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sota', action='store_true')
    args = parser.parse_args()
    files = list(FILES)
    if args.sota:
        files += ['recent_alignment.py', 'train_recent_alignment.py', 'requirements_alignment.txt',
                  'tools/run_recent_alignment.py', 'tools/build_alignment_text_cache.py',
                  'tools/summarize_recent_alignment.py', 'tests/test_recent_alignment.py', 'SOTA_ALIGNMENT_RUN.md',
                  'research/sota_alignment_20261009/CONFERENCE_SEARCH.md',
                  'research/sota_alignment_20261009/JOURNAL_SEARCH.md',
                  'research/sota_alignment_20261009/IMPLEMENTATION_VALIDATION.md']
        files += [str(path.relative_to(ROOT)).replace('\\', '/') for path in (ROOT / 'third_party').rglob('*')
                  if path.is_file() and '__pycache__' not in path.parts]
    target = ROOT / 'artifacts' / ('PASA_ALIGNMENT_SOTA.zip' if args.sota else 'PASA_SERVER_V2.zip')
    target.parent.mkdir(exist_ok=True)
    checksums = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            contents = (ROOT / name).read_bytes()
            archive.writestr('brats_fusion/' + name, contents)
            checksums.append(hashlib.sha256(contents).hexdigest() + '  ' + name)
        archive.writestr('brats_fusion/SHA256SUMS.txt', '\n'.join(checksums) + '\n')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid archive')
    print(target)
    print('SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
