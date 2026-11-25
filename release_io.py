"""Filesystem conventions shared by the command-line entry points."""
import hashlib
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))


def sequence_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', name):
        raise ValueError('Use a sequence name consisting of letters, numbers, dots, underscores or hyphens')
    return name


def independent(left, right):
    a, b = Path(left).resolve(), Path(right).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(f'Paths must be separate: {a} and {b}')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@contextmanager
def new_directory(path):
    """Publish a complete new directory on success; leave no partial result on failure."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix='.' + path.name + '-') as tmp:
        staging = Path(tmp) / 'output'
        staging.mkdir()
        yield staging
        os.rename(staging, path)
