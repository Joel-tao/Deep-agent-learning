"""SAC fixed condition; all hyperparameters come from the shared runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train import main


if __name__ == "__main__":
    # Select fixed shaping while keeping training settings shared across conditions.
    main("fixed")
