"""Package only required source/docs; never include data, credentials or old runs."""
import hashlib
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
    target = ROOT / 'artifacts' / 'PASA_SERVER_V2.zip'
    target.parent.mkdir(exist_ok=True)
    checksums = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
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
