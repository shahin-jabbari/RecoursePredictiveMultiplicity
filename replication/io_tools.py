"""Study settings and access to freshly generated local inputs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / 'studies.json').read_text())

def read(path):
    return json.loads(Path(path).read_text())

def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=serial, allow_nan=False, indent=2) + '\n')

def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, default=serial, allow_nan=False) + '\n')

def unpack(study, directory, prefix):
    from raw_cache import copy_part
    return copy_part(study, directory, prefix)

def snapshot(key):
    return ROOT / CATALOG[key]

def setup_source(study, folder='source'):
    import sys
    directory = snapshot(study) / folder
    sys.path[:0] = [str(directory), str(directory / 'frozen_utils'), str(ROOT)]
    return directory
