"""Summarize per-snapshot eval JSONs (repro/watch_eval.sh) across seeds.

Usage: uv run repro/summarize_evals.py --target 0.82 <run_dir>...
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
  runs: tuple[str, ...]
  target: float = 0.82


def load(run: Path) -> list[dict]:
  rows = []
  for f in sorted((run / "eval").glob("it*.json")):
    entry = json.loads(f.read_text())[0]
    rows.append(entry)
  return sorted(rows, key=lambda r: r["iteration"])


def main(a: Args) -> None:
  fail_counter: Counter[str] = Counter()
  print(f"{'run':<34} {'first>=tgt':>16} {'best ret':>16} {'first 0-fail':>16} {'final':>18}")
  for run in map(Path, a.runs):
    rows = load(run)
    first = next((r for r in rows if r["normalized_return_mean"] >= a.target), None)
    best = max(rows, key=lambda r: r["normalized_return_mean"])
    zero = next((r for r in rows if r["failed"] == 0), None)
    last = rows[-1]

    def fmt(r: dict | None) -> str:
      if r is None:
        return "never"
      return f"{r['normalized_return_mean']:.3f}@{r['wall_time_h']:.2f}h"

    print(
      f"{run.name:<34} {fmt(first):>16} {fmt(best):>16} {fmt(zero):>16} "
      f"{fmt(last)} f={last['failed']}"
    )
    for r in rows:
      if r["iteration"] >= 10000:
        fail_counter.update(r["failed_clips"])
    print(
      "   curve: "
      + " ".join(
        f"{r['iteration'] // 1000}k:{r['normalized_return_mean']:.2f}/{r['failed']}"
        for r in rows
      )
    )
  print("\nclips failing most often at >=10k iterations (count of checkpoints):")
  for clip, n in fail_counter.most_common(10):
    print(f"  {clip}: {n}")


if __name__ == "__main__":
  main(tyro.cli(Args))
