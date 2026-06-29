"""Noisy Crafter — the noise-trap test for Plan2Explore (POC validation, Stage 2).

Wraps SheepRL's CrafterWrapper and overlays a patch of pure random static onto a
fixed corner of EVERY observation. That corner is unpredictable forever — the
classic "noisy TV". The question this answers on a real neural world model:

    Does Plan2Explore's curiosity (ensemble disagreement) get dominated by the
    unlearnable noise, or does it ignore it and still learn to play Crafter?

We measure it by comparing achievements unlocked WITH the noise patch vs a clean
run. If noise tanks exploration -> the noisy-TV problem is real here and your
learning-progress drive has something to solve. If competence holds -> ensemble
disagreement already handles it.

All injection happens in ``_convert_obs`` (both reset and step route through it),
so the rest of SheepRL is untouched. Select it via hydra:

    env.wrapper._target_=noise_crafter.NoisyCrafterWrapper
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from sheeprl.envs.crafter import CrafterWrapper


class NoisyCrafterWrapper(CrafterWrapper):
    """CrafterWrapper with a random-static patch burned into every frame.

    Args:
        id: Crafter env id (passed through from config).
        screen_size: Observation size (passed through from config).
        seed: Env seed (passed through from config).
        noise_frac: Side length of the square noise patch as a fraction of the
            frame, placed in the top-left corner. 0.25 => a quarter-width patch.
        noise_seed: Seed for the noise RNG (kept separate from the env seed so the
            static is independent of game dynamics).
    """

    def __init__(
        self,
        id: str,
        screen_size: Sequence[int] | int,
        seed: int | None = None,
        noise_frac: float = 0.25,
        noise_seed: int = 0,
    ) -> None:
        super().__init__(id, screen_size, seed)
        self._noise_rng = np.random.default_rng(noise_seed)
        h, w = self.observation_space["rgb"].shape[:2]
        self._ph = max(1, int(h * noise_frac))
        self._pw = max(1, int(w * noise_frac))
        self._channels = self.observation_space["rgb"].shape[2]
        self._dtype = self.observation_space["rgb"].dtype

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        out = np.array(obs, copy=True)
        static = self._noise_rng.integers(
            0, 256, size=(self._ph, self._pw, self._channels), dtype=self._dtype
        )
        out[: self._ph, : self._pw] = static
        return {"rgb": out}
