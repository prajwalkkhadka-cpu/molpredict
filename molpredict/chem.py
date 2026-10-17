"""RDKit-based chemistry helpers shared by the CLI, demo and tests."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string; return None for empty or invalid input."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def mol_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Real (RDKit-computed) physicochemical properties of a molecule."""
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "crippen_logp": float(Crippen.MolLogP(mol)),
    }


def lipinski_rule_of_five(
    mw: float, logp: float, hbd: int, hba: int
) -> Tuple[List[Tuple[str, bool, str]], int]:
    """Evaluate Lipinski's rule of five. Returns (rule results, #violations)."""
    rules = [
        ("Molecular weight <= 500 Da", mw <= 500, f"{mw:.1f}"),
        ("LogP <= 5", logp <= 5.0, f"{logp:.2f}"),
        ("H-bond donors <= 5", hbd <= 5, f"{hbd}"),
        ("H-bond acceptors <= 10", hba <= 10, f"{hba}"),
    ]
    violations = sum(1 for _, passed, _ in rules if not passed)
    return rules, violations
