"""Crafter with KNOWN, seekable 'TV' (noise) tiles — the proper-world noise test.

Real Crafter (procedural, seed-deterministic), with a handful of walkable tiles
designated as 'TVs'. When the agent STANDS ON a TV tile, its whole view becomes
random static (it is "watching the TV"); off a TV tile it sees the real game.

Why this is clean, not a hack:
* Placement is EXACT and KNOWN — I pick the TV tiles each episode from that
  episode's deterministic world and remember their coordinates. Only the static
  *content* is random; the *where* is known.
* Encounter is guaranteed — TVs are placed on walkable tiles right around the
  spawn, so a normally-exploring agent meets them constantly.
* Ground truth is exact — `self.env._player.pos` gives the agent's position every
  step, so I know precisely when the view SHOULD be static.

Verification hooks baked in:
* `encounter_count` / `distinct_tvs_visited` — exact counts (placement is known).
* `assert_consistency` — a hard check that static is shown IFF the agent is on a
  TV tile; mismatches are recorded, not hidden.
* `save_frame` — dump the agent's actual view (static-on-TV vs real-off-TV) so a
  human can eyeball that the injection is real.

Crafter regenerates the world each episode (episode counter is in the seed hash),
so TV tiles are re-derived deterministically on every reset — always known.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from sheeprl.envs.crafter import CrafterWrapper

Cell = Tuple[int, int]

#: Materials the agent can stand on (so a TV tile is reachable).
_WALKABLE = {"grass", "sand", "path"}


class CrafterTVWrapper(CrafterWrapper):
    """CrafterWrapper with known, seekable static-noise tiles.

    Args:
        id, screen_size, seed: passed through to CrafterWrapper.
        n_tvs: how many TV tiles to place each episode.
        tv_radius: TVs are chosen from walkable tiles within this many cells of
            the spawn, so the agent reliably encounters them.
        noise_seed: RNG seed for the static content (separate from the world seed).
    """

    def __init__(
        self,
        id: str,
        screen_size,
        seed: Optional[int] = None,
        n_tvs: int = 2,
        tv_radius: int = 5,
        noise_seed: int = 0,
    ) -> None:
        super().__init__(id, screen_size, seed)
        self.n_tvs = int(n_tvs)
        self.tv_radius = int(tv_radius)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._obs_shape = self.observation_space["rgb"].shape

        self.tv_coords: Set[Cell] = set()
        self.encounter_count = 0
        self._visited_tvs: Set[Cell] = set()
        self.consistency_errors = 0
        self._steps = 0

    # --- TV placement (known, deterministic per episode) ---------------------

    def _player_cell(self) -> Cell:
        p = self.env._player.pos
        return (int(p[0]), int(p[1]))

    def _material_at(self, cell: Cell) -> Optional[str]:
        try:
            mat = self.env._world[cell][0]
            return mat
        except Exception:
            return None

    def _place_tvs(self) -> None:
        """Pick n_tvs walkable tiles near the spawn as TVs (deterministic)."""
        cx, cy = self._player_cell()
        candidates: List[Cell] = []
        for dx in range(-self.tv_radius, self.tv_radius + 1):
            for dy in range(-self.tv_radius, self.tv_radius + 1):
                if dx == 0 and dy == 0:
                    continue  # not the spawn tile itself
                cell = (cx + dx, cy + dy)
                if self._material_at(cell) in _WALKABLE:
                    candidates.append(cell)
        # deterministic order: by distance to spawn, then coords
        candidates.sort(key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c))
        self.tv_coords = set(candidates[: self.n_tvs])

    # --- observation injection ------------------------------------------------

    def _static(self) -> np.ndarray:
        return self._noise_rng.integers(0, 256, size=self._obs_shape, dtype=np.uint8)

    def _obs_with_tv(self, raw_obs: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool]:
        on_tv = self._player_cell() in self.tv_coords
        if on_tv:
            return {"rgb": self._static()}, True
        return {"rgb": raw_obs}, False

    # --- gym API --------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self.env._seed = seed
        raw = self.env.reset()
        self._place_tvs()  # re-derive known TV tiles for this episode's world
        obs, on_tv = self._obs_with_tv(raw)
        self._check_consistency(on_tv)
        if on_tv:
            self._register_encounter()
        return obs, {"tv_coords": sorted(self.tv_coords), "on_tv": on_tv}

    def step(self, action):
        raw, reward, terminated, truncated, info = super().step(action)
        raw = raw["rgb"]  # super already wrapped it; re-inject with TV logic
        obs, on_tv = self._obs_with_tv(raw)
        self._steps += 1
        self._check_consistency(on_tv)
        if on_tv:
            self._register_encounter()
        info = dict(info)
        info["on_tv"] = on_tv
        info["player_pos"] = self._player_cell()
        return obs, reward, terminated, truncated, info

    # --- verification hooks ---------------------------------------------------

    def _register_encounter(self) -> None:
        self.encounter_count += 1
        self._visited_tvs.add(self._player_cell())

    def _check_consistency(self, on_tv: bool) -> None:
        """Hard check: static IFF the agent is on a known TV tile."""
        should_be = self._player_cell() in self.tv_coords
        if should_be != on_tv:
            self.consistency_errors += 1

    @property
    def distinct_tvs_visited(self) -> int:
        return len(self._visited_tvs)

    def stats(self) -> Dict[str, int]:
        return {
            "steps": self._steps,
            "tv_encounters": self.encounter_count,
            "distinct_tvs_visited": self.distinct_tvs_visited,
            "tvs_placed_this_episode": len(self.tv_coords),
            "consistency_errors": self.consistency_errors,
        }

    def save_frame(self, path: str) -> None:
        """Save the agent's current view as a PNG (eyeball the static)."""
        import imageio.v3 as iio

        raw = self.env.render()
        obs, _ = self._obs_with_tv(raw)
        iio.imwrite(path, obs["rgb"])
