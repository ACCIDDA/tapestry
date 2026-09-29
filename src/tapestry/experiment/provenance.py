"""Run provenance and atomic JSON writes, shared by every experiment tool.

Moved unchanged from `models/provenance.py`: model-agnostic already.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

SLURM = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')


def save(path, value):
    """Write JSON through a temporary file, so a killed job leaves no half file."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def git_state():
    """Commit of the checkout running this code; None outside a git checkout."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'],
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit=None, git_dirty=None)
    return dict(git_commit=commit, git_dirty=bool(changes))


def torch_state():
    try:
        import torch
        return dict(torch_version=torch.__version__)
    except ImportError:
        return dict(torch_version=None)


def environment():
    return dict(host=socket.gethostname(), slurm={name: os.environ.get(name) for name in SLURM},
                **torch_state(), **git_state())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
