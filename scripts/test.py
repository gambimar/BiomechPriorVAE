"""Inspect a saved scaler: ``python scripts/test.py PATH``."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """Print the saved scaler dimensions."""
    from src.scaler import load_scaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scaler_path")
    scaler = load_scaler(parser.parse_args().scaler_path)
    print(f"Scaler expects {len(scaler['mean_'])} features")
    print(f"Scaler scale shape: {scaler['scale_'].shape}")


if __name__ == "__main__":
    main()
