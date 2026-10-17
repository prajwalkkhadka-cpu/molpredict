"""Command-line single-molecule predictor."""

from __future__ import annotations

import argparse
import sys

from molpredict.report import PRESET_MOLECULES, load_predictor, render_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict LogP for a molecule")
    parser.add_argument("--smiles", help="SMILES string")
    parser.add_argument("--name", help="Preset drug name (aspirin, caffeine, ...)")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args(argv)

    name, smiles, known = "Custom molecule", args.smiles, None
    if args.name:
        presets = {v[0].lower(): v for v in PRESET_MOLECULES.values()}
        hit = presets.get(args.name.lower())
        if hit is None:
            print(f"Unknown name '{args.name}'. Available: {', '.join(v[0] for v in PRESET_MOLECULES.values())}",
                  file=sys.stderr)
            return 2
        name, smiles, _, known = hit

    if not smiles:
        parser.print_usage(sys.stderr)
        print("error: provide --smiles or --name", file=sys.stderr)
        return 2

    ok = render_report(name, smiles, load_predictor(args.checkpoint), known)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
