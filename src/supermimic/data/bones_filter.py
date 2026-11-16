"""Paper Appendix E name filter for BONES-SEED (G1 retarget).

Keyword list = SONIC's default list in
GR00T-WholeBodyControl/gear_sonic/data_process/filter_and_copy_bones_data.py
(case-insensitive substring match on "<date_dir>/<filename>", as SONIC does)
plus EXTRA terms for terrain/object contact the paper's examples name but
SONIC misses ("jump off 50cm", "look over fence").

Counts on seed_metadata_v004 (142,220 clips):
  SONIC list            keeps 129,785
  + EXTRA (this file)   keeps 129,679   (paper: 129,667, i.e. 12 clips = 0.01% off)
The paper's exact extra terms are not published; we did not tune further.

Usage: uv run sm-bones-name-filter --metadata .../seed_metadata_v004.parquet \
         --out-keep keep_names.txt
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import tyro

SONIC_KEYWORDS = [
  "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat",
  "table", "_sit_", "sit_", "ladder", "crutch", "_bed_", "_ride_", "scooter",
  "stepdown", "acrobatics_", "box_HSPU", "cartwheel", "50cm_box_", "on_box",
  "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
  "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
  "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump", "safety_roll",
  "box_dips", "walking_on_edge", "push_obstacle",
]  # fmt: skip
EXTRA_KEYWORDS = ["jump_off", "jump_on_50cm", "look_over_fence"]


@dataclass
class Args:
  metadata: str
  out_keep: str
  out_removed: str | None = None


def main(a: Args) -> None:
  m = pd.read_parquet(a.metadata)
  key = (m.move_g1_path.str.split("/").str[-2] + "/" + m.filename).str.lower()
  kws = [k.lower() for k in SONIC_KEYWORDS + EXTRA_KEYWORDS]
  removed = key.apply(lambda s: any(k in s for k in kws))
  keep = m.filename[~removed]
  keep.to_csv(a.out_keep, index=False, header=False)
  if a.out_removed:
    m.filename[removed].to_csv(a.out_removed, index=False, header=False)
  print(f"[INFO] {len(m)} clips, removed {int(removed.sum())}, kept {len(keep)}")


def cli() -> None:
  """Console entry point ``sm-bones-name-filter``."""
  main(tyro.cli(Args))


if __name__ == "__main__":
  cli()
