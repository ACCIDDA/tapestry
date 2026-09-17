"""Run provenance and atomic JSON writes, shared by every model and tool.

Experiments are not locked to a code or data version, so what a run recorded
about itself is the only reliable account of how it was produced. These helpers
have no project dependencies, so any module can record provenance without
importing the experiment manager.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
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
    """The wheel that produced the numbers; CUDA builds differ in supported architectures."""
    try:
        import torch
        return dict(torch_version=torch.__version__)
    except ImportError:
        return dict(torch_version=None)


def environment():
    return dict(host=socket.gethostname(), slurm={name: os.environ.get(name) for name in SLURM},
                **torch_state(), **git_state())
