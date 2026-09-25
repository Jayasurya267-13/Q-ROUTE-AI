"""Discrete, quantum-inspired particle swarm search over feasible routes."""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


class QPSOOptimizer:
    def __init__(self, particles: int = 30, iterations: int = 100, alpha: float = 0.75,
                 seed: int | None = 42):
        if particles < 1 or iterations < 0 or alpha <= 0:
            raise ValueError("particles must be positive, iterations non-negative, alpha positive")
        self.particles, self.iterations, self.alpha = particles, iterations, alpha
        self.rng = random.Random(seed)

    def optimize(self, routes: Sequence[Sequence[str]], traffic_data=None,
                 fitness_fn: Callable[[Sequence[str]], float] | None = None):
        """Return (best_route, best_fitness). QPSO updates route-index positions.

        Every position decodes to an explicitly feasible route. The logarithmic
        quantum-behaved update contracts particles toward their personal/global
        attractors; integer decoding keeps the search valid for discrete paths.
        """
        feasible = [list(route) for route in routes if route and len(set(route)) == len(route)]
        if not feasible:
            raise ValueError("QPSO requires at least one valid, loop-free route")
        score = fitness_fn or (lambda route: float(route[-1]) if isinstance(route[-1], (int, float)) else len(route))
        n = len(feasible)
        pos = [self.rng.uniform(0, n - 1) for _ in range(self.particles)]
        pbest = pos[:]
        pscore = [float(score(feasible[round(x)])) for x in pos]
        best_i = min(range(len(pos)), key=lambda i: pscore[i])
        gbest, gscore = pbest[best_i], pscore[best_i]
        for _ in range(self.iterations):
            mbest = sum(pbest) / len(pbest)
            for i in range(len(pos)):
                attractor = self.rng.random() * pbest[i] + (1 - self.rng.random()) * gbest
                u = max(self.rng.random(), 1e-12)
                sign = -1 if self.rng.random() < 0.5 else 1
                pos[i] = max(0.0, min(n - 1.0, attractor + sign * self.alpha * abs(mbest - pos[i]) * math.log(1 / u)))
                candidate = round(pos[i])
                value = float(score(feasible[candidate]))
                if value < pscore[i]:
                    pbest[i], pscore[i] = pos[i], value
                if value < gscore:
                    gbest, gscore = pos[i], value
        return feasible[round(gbest)].copy(), gscore
