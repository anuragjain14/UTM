"""Deterministic random streams (requirement X-2).

Every run gets independent streams for each concern, derived from (base_seed, run_index), so a run
is reproducible on any machine regardless of how runs are sharded or ordered.
"""

import numpy as np

STREAMS = ("design", "plant", "sensors", "corruption", "window", "detectors")


def run_streams(base_seed: int, run_index: int) -> dict:
    ss = np.random.SeedSequence([int(base_seed), int(run_index)])
    return {name: np.random.Generator(np.random.PCG64(child))
            for name, child in zip(STREAMS, ss.spawn(len(STREAMS)))}


def stream_seeds(base_seed: int, run_index: int) -> dict:
    """Integer fingerprints of each stream, recorded in the run summary for audit."""
    ss = np.random.SeedSequence([int(base_seed), int(run_index)])
    return {name: int(child.generate_state(1)[0])
            for name, child in zip(STREAMS, ss.spawn(len(STREAMS)))}
