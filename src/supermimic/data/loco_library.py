"""Walk/run library for the velocity interface (paper Appendix G.1: "a walking
and running library filtered from LAFAN1 and BONES-SEED").

Selection (ours; the paper does not list it): BONES-SEED clips in the final set
whose category is "Basic Locomotion Neutral" and whose movement type mentions
walking, jogging, running, turning or standing idle, plus the LAFAN1 walk*,
run* and sprint* clips. Links them into ``<data>/loco/npz`` and packs them into
``<data>/loco/packed``. Our run: 24,644 clips, 50.8 h.

Usage: uv run sm-loco-library
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tyro

from supermimic.paths import data_path, require

TYPE_PATTERN = "walk|jog|run|turning|standing idle"
LAFAN_PREFIXES = ("walk", "run", "sprint")


@dataclass
class Args:
  metadata: str | None = None
  """Default <data>/bones_seed/metadata/seed_metadata_v004.parquet."""
  keep_list: str | None = None
  """Default <data>/bones_seed/keep_final.txt (from sm-bones-finalize)."""
  bones_npz: str | None = None
  """Default <data>/bones_seed/npz_final."""
  lafan_npz: str | None = None
  """Default <data>/lafan1_g1/npz."""
  out: str | None = None
  """Default <data>/loco."""
  pack: bool = True


def _link(src: Path, dst: Path) -> None:
  if not dst.is_symlink() and not dst.exists():
    dst.symlink_to(src.resolve())


def main(a: Args) -> None:
  meta = require(
    Path(
      a.metadata or data_path("bones_seed", "metadata", "seed_metadata_v004.parquet")
    ),
    "BONES-SEED metadata",
  )
  keep_file = require(
    Path(a.keep_list or data_path("bones_seed", "keep_final.txt")), "final keep list"
  )
  bones = require(
    Path(a.bones_npz or data_path("bones_seed", "npz_final")), "BONES npz dir"
  )
  lafan = require(Path(a.lafan_npz or data_path("lafan1_g1", "npz")), "LAFAN1 npz dir")
  out = Path(a.out or data_path("loco"))
  (out / "npz").mkdir(parents=True, exist_ok=True)

  m = pd.read_parquet(meta)
  keep = set(keep_file.read_text().splitlines())
  m = m[m.filename.isin(keep) & m.category.eq("Basic Locomotion Neutral")]
  types = m.content_type_of_movement.str.lower().fillna("")
  sel = m.filename[types.str.contains(TYPE_PATTERN, regex=True)]
  for c in sel:
    _link(bones / f"{c}.npz", out / "npz" / f"{c}.npz")
  n_lafan = 0
  for f in sorted(lafan.glob("*.npz")):
    if f.name.split("_")[0].rstrip("0123456789") in LAFAN_PREFIXES:
      _link(f, out / "npz" / f"lafan1_{f.name}")
      n_lafan += 1
  print(f"[INFO] {len(sel)} BONES-SEED + {n_lafan} LAFAN1 clips -> {out / 'npz'}")
  if a.pack:
    subprocess.run(
      [
        sys.executable,
        "-m",
        "mjlab.scripts.pack_motions",
        "--input",
        str(out / "npz" / "*.npz"),
        "--output",
        str(out / "packed"),
        "--workers",
        str(min(64, os.cpu_count() or 8)),
      ],
      check=True,
    )


def cli() -> None:
  """Console entry point ``sm-loco-library``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
