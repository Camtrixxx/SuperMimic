"""Final BONES-SEED clip set: drop clips flagged by supermimic.data.feasibility.

Writes <data>/keep_final.txt and <data>/removed_by_feasibility.csv, and links
the kept clips into <data>/npz_final/ (pack them with mjlab.scripts.pack_motions).

Usage: uv run sm-bones-finalize --data data/bones_seed
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import tyro


@dataclass
class Args:
  data: str
  feasibility: str = "feasibility.parquet"
  src_dir: str = "npz_name_filtered"
  out_dir: str = "npz_final"


def main(a: Args) -> None:
  d = pd.read_parquet(os.path.join(a.data, a.feasibility))
  keep = d.loc[~d.infeasible, "clip"]
  keep.to_csv(os.path.join(a.data, "keep_final.txt"), index=False, header=False)
  d.loc[d.infeasible, ["clip", "elevated", "penetration", "floating"]].to_csv(
    os.path.join(a.data, "removed_by_feasibility.csv"), index=False
  )
  out = os.path.join(a.data, a.out_dir)
  os.makedirs(out, exist_ok=True)
  for c in keep:
    dst = os.path.join(out, f"{c}.npz")
    if not os.path.lexists(dst):
      os.symlink(os.path.abspath(os.path.join(a.data, a.src_dir, f"{c}.npz")), dst)
  print(f"[INFO] kept {len(keep)} of {len(d)} clips -> {out}")


def cli() -> None:
  """Console entry point ``sm-bones-finalize``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
