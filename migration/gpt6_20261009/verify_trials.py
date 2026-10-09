"""Independent synthetic checks; no training imports or patient data."""
import ast
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_targets():
    source = ast.parse((ROOT / 'train_semantic_alignment.py').read_text(encoding='utf-8'))
    names = {'clean_value', 'canonical_field', 'anchor_source', 'anchor_type',
             'make_anchor', 'semantic_anchors', 'target_anchor_keys'}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {'PATHOLOGY_FIELDS': ('Tumor Grade', 'Tumor Type'),
                 'MOLECULAR_FIELDS': ('IDH', 'MGMT', '1p19Q CODEL'),
                 'CLINICAL_FIELDS': ('Age at Histological Diagnosis', 'Gender')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<extracted targets>', 'exec'), namespace)
    patient = {'Tumor Grade': '4', 'Tumor Type': 'Glioblastoma', 'IDH': 'wildtype',
               'MGMT': 'unknown', '1p19Q CODEL': 'non-codeleted'}
    actual = {region: namespace['target_anchor_keys'](patient, region, 'region_rules')
              for region in ('enhancing', 'edema', 'necrotic')}
    expected = {
        'enhancing': ['tumor_grade::4', 'tumor_type::glioblastoma'],
        'edema': ['idh::wildtype', 'tumor_type::glioblastoma', 'tumor_grade::4'],
        'necrotic': ['tumor_grade::4', '1p19q_codel::non-codeleted', 'tumor_type::glioblastoma'],
    }
    assert actual == expected, (actual, expected)
    assert namespace['target_anchor_keys']({'IDH': 'wildtype'}, 'enhancing', 'region_rules') == ['idh::wildtype']
    return actual


def verify_metrics(path):
    func = runpy.run_path(str(path))['masked_retrieval_metrics']
    cases = [
        ((['u', 'p1', 'n', 'p2'], {'p1', 'p2'}, {'n'}, 1), (1.0, .5, 1.0)),
        ((['n', 'p1'], {'p1', 'p2'}, {'n'}, 1), (0.0, 0.0, .5)),
        ((['p1'], {'p1', 'p2'}, set(), 5), (1.0, .5, 1.0)),
        ((['n'], {'p'}, {'n'}, 1), (0.0, 0.0, 0.0)),
        (([], {'p'}, set(), 1), (0.0, 0.0, 0.0)),
        ((['u'], set(), set(), 1), (None, None, None)),
    ]
    keys = ('hit_at_k', 'recall_at_k', 'reciprocal_rank')
    for args, expected in cases:
        actual = func(*args)
        assert tuple(actual[k] for k in keys) == expected, (path, args, actual)
    invalid = [(['p'], {'p'}, {'p'}, 1), (['u', 'u'], {'p'}, set(), 1)]
    invalid.extend((['p'], {'p'}, set(), k) for k in (0, -1, 1.5, True, '1'))
    for args in invalid:
        try:
            func(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError((path, 'accepted invalid input', args))
    return {'behavior_cases': len(cases), 'invalid_cases': len(invalid), 'status': 'pass'}


if __name__ == '__main__':
    result = {'synthetic_patient': verify_targets(), 'trials': {}}
    for model in ('sol_low', 'astra_low'):
        result['trials'][model] = verify_metrics(Path(__file__).parent / model / 'metrics.py')
    print(json.dumps(result, indent=2, ensure_ascii=False))
