"""The experiment scoreboard — turns a run into the numbers that decide the design.

The p2e training loop feeds this each step with the wrapper's per-env `on_tv` flag
and `player_pos` (both already emitted by CrafterTVWrapper). It accumulates the
behavioural lie-detector metrics and writes one JSON per arm to SCOREBOARD_DIR.

`report()` then loads every arm's JSON and prints the comparison table, using the
`random` arm as the CHANCE baseline (what an indifferent agent spends on the TVs).

What it answers (Q1, the noise-trap lie detector):
  * time_on_tv      = steps standing on a TV tile / total steps.
                      drive-arm ~= chance  -> ignores the noise (the win).
                      drive-arm >> chance  -> hypnotised / trapped.
  * return_ratio    = repeat visits to the SAME tv tile / total tv encounters.
                      high -> keeps going back to stare (hypnosis signature).
  * distinct_tvs    = how many different TVs it ever stepped on (coverage sanity).

Memory's effect on prediction (Q2) is read separately from the sheeprl loss logs
(world_model_loss with MEM_AUG=1 vs =0); this module owns the behavioural side.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np


class Scoreboard:
    """Accumulates per-step TV behaviour for one run (one arm)."""

    def __init__(self, arm: str, out_dir: str) -> None:
        self.arm = arm
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"arm_{arm}.json")
        self.total_steps = 0
        self.steps_on_tv = 0
        self._distinct_tv_cells: set = set()
        self._tv_cell_visits: Dict[str, int] = {}
        self._last_on_tv = False
        self.tv_entries = 0  # number of times it STEPPED ONTO a tv (rising edges)

    def update(self, on_tv: bool, pos: Optional[tuple] = None) -> None:
        self.total_steps += 1
        if on_tv:
            self.steps_on_tv += 1
            if pos is not None:
                key = f"{int(pos[0])},{int(pos[1])}"
                self._distinct_tv_cells.add(key)
                self._tv_cell_visits[key] = self._tv_cell_visits.get(key, 0) + 1
            if not self._last_on_tv:
                self.tv_entries += 1  # rising edge = a fresh approach to a TV
        self._last_on_tv = on_tv

    def metrics(self) -> Dict:
        tot = max(self.total_steps, 1)
        # a "return" = an entry onto a tv cell that has been entered before.
        repeat_entries = sum(max(v - 1, 0) for v in self._tv_cell_visits.values())
        return {
            "arm": self.arm,
            "total_steps": self.total_steps,
            "steps_on_tv": self.steps_on_tv,
            "time_on_tv": self.steps_on_tv / tot,
            "tv_entries": self.tv_entries,
            "distinct_tvs": len(self._distinct_tv_cells),
            "return_ratio": (repeat_entries / self.tv_entries) if self.tv_entries else 0.0,
        }

    def dump(self) -> None:
        with open(self.path, "w") as fh:
            json.dump(self.metrics(), fh, indent=2)


def report(out_dir: str) -> str:
    """Load every arm_*.json in out_dir and render the comparison table."""
    arms: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(out_dir, "arm_*.json"))):
        with open(p) as fh:
            arms.append(json.load(fh))
    if not arms:
        return f"(no arm_*.json found in {out_dir})"

    # the random arm is the chance baseline (arm name may be seed-tagged, e.g. random_s42)
    _rand = [a["time_on_tv"] for a in arms if a["arm"].split("_s")[0] == "random"]
    chance = (sum(_rand) / len(_rand)) if _rand else None

    lines = []
    lines.append("=" * 72)
    lines.append("SCOREBOARD — Q1: does the drive resist the seekable noise trap?")
    lines.append("=" * 72)
    if chance is not None:
        lines.append(f"CHANCE baseline (random arm): time_on_tv = {chance:.3%}  <- the ruler")
    else:
        lines.append("CHANCE baseline (random arm): MISSING — run the `random` arm or numbers have no scale")
    lines.append("")
    hdr = f"{'arm':<14}{'steps':>8}{'time_on_tv':>12}{'vs_chance':>11}{'return_ratio':>14}{'distinct':>10}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for a in arms:
        if chance and chance > 0 and a["arm"].split("_s")[0] != "random":
            ratio = f"{a['time_on_tv'] / chance:>9.2f}x"
        else:
            ratio = f"{'—':>10}"
        lines.append(
            f"{a['arm']:<14}{a['total_steps']:>8}{a['time_on_tv']:>11.3%} {ratio:>10}"
            f"{a['return_ratio']:>14.2f}{a['distinct_tvs']:>10}"
        )
    lines.append("")
    lines.append("READ: vs_chance ~1.0  -> ignores the noise (PASS for the drive).")
    lines.append("      vs_chance >>1    -> seeking/trapped.  high return_ratio -> hypnosis.")
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCOREBOARD_DIR", ".")
    print(report(d))
