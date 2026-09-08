"""SMILES parsing and validation using RDKit."""

from __future__ import annotations

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


class InvalidSMILESError(Exception):
    """Raised when a SMILES string cannot be parsed by RDKit."""


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string into an RDKit molecule.

    Raises InvalidSMILESError if the string is not a valid molecule.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise InvalidSMILESError(f"SMILES string is empty or not a string: {smiles!r}")

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise InvalidSMILESError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def is_valid_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse this SMILES string, False otherwise."""
    try:
        parse_smiles(smiles)
        return True
    except InvalidSMILESError:
        return False


def canonicalize(smiles: str) -> str:
    """Return the RDKit canonical form of a SMILES string."""
    mol = parse_smiles(smiles)
    return Chem.MolToSmiles(mol)
