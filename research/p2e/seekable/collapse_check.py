"""Collapse check (rung-1 viability test) for the neural seekable-noise experiment.

The previous neural attempt FAILED for one reason: the world was (x,y) coordinates,
whose dynamics (move right = x+1) a neural net learns globally from almost no data,
so prediction error collapsed to ~0 EVERYWHERE and there was no curiosity signal.

This checks whether a *textured* pixel world fixes that. Each cell has its own
fixed random "view" (texture). To predict the next view the model must have
actually VISITED that cell — there is no global rule to generalise. So:

* a cell the model has seen  -> low error (it learned that view)   = mastered
* a cell it has NOT seen      -> high error (it cannot guess the view) = novelty

If unseen cells stay surprising while seen cells get learned, the novelty signal
survives a neural model -> the full experiment is viable. If everything collapses
to ~0 (unseen == seen), the world is still too trivial and we stop and rethink.

Procedure: train a numpy MLP forward model online while an agent random-walks in
ONLY the LEFT half of a 6x6 grid, then compare prediction error on the trained
LEFT half vs the never-visited RIGHT half.

Run:  python research/p2e/seekable/collapse_check.py
"""

from __future__ import annotations

import numpy as np

N = 6                      # grid side
TEX = 48                   # texture (view) dimensionality per cell
ACTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up/down/left/right
HIDDEN = 256
LR = 0.03
TRAIN_STEPS = 60_000


def build_world(seed: int = 0) -> np.ndarray:
    """Return an (N, N, TEX) array: each cell's fixed high-contrast BINARY view.

    Binary (0/1) views on purpose: a model that hasn't seen a cell can only fall
    back on guessing ~0.5, which is badly wrong on binary data (MSE ~0.25). So an
    unvisited cell stays genuinely surprising and can't be cheaply "averaged away"
    the way smooth random textures could. This is the richness fix after the first
    collapse check showed only a 2.1x signal with smooth textures.
    """
    rng = np.random.default_rng(seed)
    return (rng.random((N, N, TEX)) < 0.5).astype(np.float32)


def obs(world: np.ndarray, pos) -> np.ndarray:
    return world[pos[0], pos[1]]


def step(pos, action):
    dx, dy = ACTIONS[action]
    return (min(N - 1, max(0, pos[0] + dx)), min(N - 1, max(0, pos[1] + dy)))


class MLP:
    """One-hidden-layer tanh MLP: [view ++ one_hot(action)] -> next view. Manual SGD."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        din, dout = TEX + len(ACTIONS), TEX
        self.W1 = (rng.standard_normal((din, HIDDEN)) / np.sqrt(din)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN, np.float32)
        self.W2 = (rng.standard_normal((HIDDEN, dout)) / np.sqrt(HIDDEN)).astype(np.float32)
        self.b2 = np.zeros(dout, np.float32)

    def _feat(self, view, action):
        a = np.zeros(len(ACTIONS), np.float32); a[action] = 1.0
        return np.concatenate([view, a])

    def predict(self, view, action):
        x = self._feat(view, action)
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def error(self, view, action, target):
        d = self.predict(view, action) - target
        return float(np.mean(d * d))

    def update(self, view, action, target):
        x = self._feat(view, action)
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        d = out - target
        err = float(np.mean(d * d))
        dout = (2.0 / TEX) * d
        self.W2 -= LR * np.outer(h, dout); self.b2 -= LR * dout
        dh = (dout @ self.W2.T) * (1 - h * h)
        self.W1 -= LR * np.outer(x, dh); self.b1 -= LR * dh
        return err


def mean_error_over(model, world, cols):
    """Mean prediction error over all (cell in cols, action) transitions."""
    errs = []
    for x in cols:
        for y in range(N):
            for a in range(len(ACTIONS)):
                nxt = step((x, y), a)
                errs.append(model.error(world[x, y], a, world[nxt[0], nxt[1]]))
    return float(np.mean(errs))


def main():
    world = build_world(seed=0)
    model = MLP(seed=1)
    rng = np.random.default_rng(2)

    left_cols = [0, 1, 2]     # the half we train in
    right_cols = [3, 4, 5]    # never visited

    before = mean_error_over(model, world, left_cols)

    # Random-walk + online-train, confined to the LEFT half.
    pos = (0, 0)
    for _ in range(TRAIN_STEPS):
        a = int(rng.integers(len(ACTIONS)))
        nxt = step(pos, a)
        if nxt[0] in left_cols:           # stay in the left half
            model.update(world[pos[0], pos[1]], a, world[nxt[0], nxt[1]])
            pos = nxt

    visited = mean_error_over(model, world, left_cols)
    unvisited = mean_error_over(model, world, right_cols)
    ratio = unvisited / visited if visited > 0 else float("inf")

    print("=== COLLAPSE CHECK (textured pixel world) ===")
    print(f"  error before training (left):     {before:.4f}")
    print(f"  error AFTER training, VISITED:     {visited:.4f}   (should drop a lot)")
    print(f"  error AFTER training, UNVISITED:   {unvisited:.4f}   (should stay high)")
    print(f"  novelty ratio unvisited/visited:   {ratio:.1f}x")
    print()
    if ratio >= 3.0 and visited < 0.5 * before:
        print("  VERDICT: NO COLLAPSE. Unseen cells stay surprising while seen cells")
        print("           are learned -> real novelty signal survives a neural model.")
        print("           Green light for the full seekable-noise experiment.")
    else:
        print("  VERDICT: COLLAPSE / WEAK SIGNAL. The world is still too easy to")
        print("           generalise, or the model didn't learn. STOP and make the")
        print("           world richer before building the comparison.")


if __name__ == "__main__":
    main()
