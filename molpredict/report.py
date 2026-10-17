"""Terminal report rendering shared by predict.py and demo.py."""

from __future__ import annotations

import sys
from typing import Dict, Optional

from molpredict.chem import lipinski_rule_of_five, mol_properties, parse_smiles
from molpredict.inference import Predictor

_USE_COLOR = sys.stdout.isatty()


def _c(code: str) -> str:
    return code if _USE_COLOR else ""


RESET, BOLD = _c("\033[0m"), _c("\033[1m")
CYAN, GREEN, YELLOW, RED, BLUE, MAGENTA = (
    _c("\033[36m"), _c("\033[32m"), _c("\033[33m"), _c("\033[31m"), _c("\033[34m"), _c("\033[35m"),
)

PRESET_MOLECULES: Dict[str, tuple] = {
    "1": ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "Analgesic & anti-inflammatory", 1.19),
    "2": ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CNS stimulant", -0.07),
    "3": ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", "Antipyretic & analgesic", 0.46),
    "4": ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "NSAID", 3.72),
    "5": ("Nicotine", "CN1CCCC1c2cccnc2", "Alkaloid stimulant", 1.17),
    "6": ("Vitamin C", "C1=C(C(=O)OC1C(CO)O)O", "Essential nutrient / antioxidant", -1.85),
}


def load_predictor(path: str) -> Optional[Predictor]:
    """Load the trained model, or return None if no checkpoint exists."""
    try:
        return Predictor.from_checkpoint(path)
    except FileNotFoundError:
        return None


def render_report(name: str, smiles: str, predictor: Optional[Predictor],
                  known_logp: Optional[float] = None) -> bool:
    """Print a report. Returns False if the SMILES is invalid."""
    mol = parse_smiles(smiles)
    print(f"\n{BOLD}{MAGENTA}== {name}{RESET}")
    print(f"  SMILES: {smiles}")
    if mol is None:
        print(f"  {RED}Invalid SMILES - RDKit could not parse it.{RESET}")
        return False

    props = mol_properties(mol)
    model_logp = predictor.predict_one(smiles) if predictor else None

    if model_logp is not None:
        logp, logp_label = model_logp, "MLP prediction"
    else:
        logp, logp_label = props["crippen_logp"], "RDKit Crippen estimate"

    print(f"\n  {BOLD}Properties{RESET}")
    print(f"    LogP ({logp_label}): {GREEN}{logp:.2f}{RESET}")
    if model_logp is not None:
        print(f"    LogP (RDKit Crippen)       : {props['crippen_logp']:.2f}")
    if known_logp is not None:
        print(f"    LogP (reference value)     : {known_logp:.2f}   |error| = {abs(logp - known_logp):.2f}")
    print(f"    Molecular weight           : {props['mw']:.2f} g/mol")
    print(f"    TPSA                       : {props['tpsa']:.1f} A^2")
    print(f"    H-bond donors / acceptors  : {props['hbd']} / {props['hba']}")
    print(f"    Rotatable bonds            : {props['rotatable_bonds']}")
    if predictor is None:
        print(f"  {YELLOW}No trained model found; showing RDKit values only. "
              f"Run `python scripts/train.py` to enable the MLP.{RESET}")

    rules, violations = lipinski_rule_of_five(props["mw"], logp, props["hbd"], props["hba"])
    print(f"\n  {BOLD}Lipinski rule of five{RESET}")
    for desc, passed, val in rules:
        tag = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"    {tag}  {desc:<28} (value: {val})")
    verdict = {0: f"{GREEN}0 violations{RESET}", 1: f"{YELLOW}1 violation (usually still acceptable){RESET}"}
    print(f"    -> {verdict.get(violations, f'{RED}{violations} violations{RESET}')}")
    return True
