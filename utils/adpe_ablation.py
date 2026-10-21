"""Deterministic attribute masks for the optional evaluation diagnostic."""

import itertools
import math
import random


def make_ablation_masks(num_attributes, ratios, repeats, exact_max, seed):
    if num_attributes <= 0 or repeats <= 0 or exact_max <= 0:
        raise ValueError("Attribute count, mask repeats and exact_max must be positive")
    rng = random.Random(seed)
    groups = []
    for ratio in ratios:
        if not 0 <= ratio <= 1:
            raise ValueError("Attribute keep ratios must be in [0, 1]")
        kept = int(math.floor(num_attributes * ratio + 0.5))
        if math.comb(num_attributes, kept) <= exact_max:
            selections = list(itertools.combinations(range(num_attributes), kept))
        else:
            selections = [tuple(sorted(rng.sample(range(num_attributes), kept))) for _ in range(repeats)]
        masks = [[int(i in selection) for i in range(num_attributes)] for selection in selections]
        groups.append((ratio, kept, masks))
    if not groups:
        raise ValueError("At least one keep ratio is required")
    return groups
