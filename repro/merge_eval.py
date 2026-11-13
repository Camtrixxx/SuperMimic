"""Merge sharded evaluate.py outputs into one report (paper Appendix F.1).

Prints the overall normalized return / failed count, and failed motions per
motion group in the style of the paper's Table A.8 (groups matched on clip
names; our patterns, the paper does not publish its grouping).

Usage: uv run repro/merge_eval.py --pattern 'eval/full_it*_shard*.json' --out merged.json
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import tyro

# Approximation of the paper's Table A.8 groups.
GROUPS = {
  "side rolls": r"side_roll",
  "flips (360/180/90)": r"flip",
  "fire-in-the-hole turns": r"fire_in_the_hole",
  "crossed legs": r"cross(ed)?_leg",
  "lying / resting / standing up": r"(lying|lie_down|rest|stand_up|get_up|getup)",
  "broad jumps": r"broad_jump",
  "dance routine": r"dancing_routine",
}


@dataclass
class Args:
  pattern: str
  out: str | None = None


def main(a: Args) -> None:
  files = sorted(glob.glob(a.pattern))
  by_it: dict[object, list[dict]] = defaultdict(list)
  meta: dict[object, dict] = {}
  for f in files:
    for entry in json.loads(open(f).read()):
      by_it[entry["iteration"]] += entry["per_clip"]
      meta[entry["iteration"]] = {"wall_time_h": entry["wall_time_h"]}
  report = []
  for it, rows in sorted(by_it.items(), key=lambda kv: (kv[0] is None, kv[0])):
    n = len(rows)
    failed = [r["clip"] for r in rows if r["failed"]]
    ret = sum(r["normalized_return"] for r in rows) / n
    groups = {}
    for name, pat in GROUPS.items():
      members = [r for r in rows if re.search(pat, r["clip"], re.I)]
      groups[name] = (sum(r["failed"] for r in members), len(members))
    print(
      f"it={it} t={meta[it]['wall_time_h']} clips={n} "
      f"ret={ret:.4f} failed={len(failed)}"
    )
    for name, (k, m) in groups.items():
      print(f"   {name:32s} {k}/{m}")
    report.append(
      {
        "iteration": it,
        **meta[it],
        "clips": n,
        "normalized_return_mean": ret,
        "failed": len(failed),
        "failed_clips": failed,
        "groups": groups,
      }
    )
  if a.out:
    open(a.out, "w").write(json.dumps(report, indent=1))


if __name__ == "__main__":
  main(tyro.cli(Args))
