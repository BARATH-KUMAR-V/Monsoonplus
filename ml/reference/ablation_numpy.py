"""Dependency-free modality ablation. Thin launcher for ml/training/linear_probe.py.

    python ml/reference/ablation_numpy.py            # from anywhere

Pure NumPy: no PyTorch, no GPU, no network, about a second. It was MonsoonPlus_v3's
standalone script; it now scores the project's own held-out test windows (same
generator, same chronological split, same targets as the neural model) instead of a
private copy of the generator, so its numbers sit in the same table as the network's.
See ml/training/linear_probe.py for the method and its limits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.training.linear_probe import main  # noqa: E402

if __name__ == "__main__":
    main()
