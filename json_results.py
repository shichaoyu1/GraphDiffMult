"""Strict, atomic result JSON: undefined numeric metrics are null, never NaN."""
import json
import math
import numbers
import os
from pathlib import Path
import tempfile


def finite_payload(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {key: finite_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_payload(item) for item in value]
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, payload):
    path = Path(path)
    contents = json.dumps(finite_payload(payload), ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temporary = file.name
            file.write(contents)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
