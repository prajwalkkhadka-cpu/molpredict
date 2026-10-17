"""Interactive terminal demo: pick a preset drug or enter any SMILES."""

import argparse

from molpredict.report import (BOLD, CYAN, GREEN, PRESET_MOLECULES, RED, RESET,
                               load_predictor, render_report)


def main() -> None:
    parser = argparse.ArgumentParser(description="MolPredict interactive demo")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    predictor = load_predictor(args.checkpoint)

    print(f"{BOLD}{CYAN}MolPredict - molecular LogP prediction{RESET}")
    print("Model:", "trained MLP loaded" if predictor else "no checkpoint (RDKit values only)")
    while True:
        print(f"\n{BOLD}Choose an option:{RESET}")
        for key, (name, *_rest) in PRESET_MOLECULES.items():
            print(f"  [{key}] {name}")
        print("  [7] Enter a custom SMILES")
        print(f"  {RED}[0] Exit{RESET}")
        try:
            choice = input(f"\n{GREEN}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "0":
            break
        if choice in PRESET_MOLECULES:
            name, smiles, desc, known = PRESET_MOLECULES[choice]
            print(f"\n{desc}")
            render_report(name, smiles, predictor, known)
        elif choice == "7":
            smi = input("SMILES (e.g. CCO or c1ccccc1): ").strip()
            if smi:
                render_report("Custom molecule", smi, predictor)
        else:
            print(f"{RED}Please choose 0-7.{RESET}")


if __name__ == "__main__":
    main()
