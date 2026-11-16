"""Filesystem locations used by the data and evaluation tools.

All motion data lives under one data root:

* ``$SUPERMIMIC_DATA`` if set, otherwise
* ``<repo>/data`` (a directory or a symlink to wherever the data actually is).

Expected layout (see README "Data")::

  <data root>/
    lafan1_g1/{csv,npz}/
    bones_seed/{g1,metadata,npz_name_filtered,npz_final,packed_final}/
    loco/{npz,packed}/           # walk/run library for the velocity interface
    derived/                     # small generated clips (static poses, ...)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "SUPERMIMIC_DATA"


def data_root() -> Path:
  env = os.environ.get(ENV_VAR)
  return Path(env).expanduser() if env else REPO_ROOT / "data"


def data_path(*parts: str) -> Path:
  """``data_root() / parts``; does not check existence."""
  return data_root().joinpath(*parts)


def require(path: Path, what: str) -> Path:
  """Return ``path`` or exit with a message pointing at the data root."""
  if not path.exists():
    raise SystemExit(
      f"{what} not found: {path}\n"
      f"Data root is {data_root()} (set ${ENV_VAR} or create <repo>/data)."
    )
  return path
