"""Compute molecular descriptors from SMILES strings using RDKit.

These descriptors are the input features for the solubility model. The
measured target value is never used as a feature, so there is no leakage
from label to input.
"""

from __future__ import annotations

import pandas as pd
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from .molecules import parse_smiles, InvalidSMILESError

DESCRIPTOR_COLUMNS = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotatableBonds",
    "RingCount",
    "HeavyAtoms",
    "FractionCSP3",
]


def compute_descriptors(smiles: str) -> dict:
    """Compute a fixed set of RDKit descriptors for one SMILES string.

    Raises InvalidSMILESError if the SMILES cannot be parsed.
    """
    mol = parse_smiles(smiles)
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def build_feature_table(df: pd.DataFrame, smiles_column: str = "smiles",
                         target_column: str | None = "target") -> pd.DataFrame:
    """Build a descriptor feature table from a dataframe of SMILES strings.

    Rows with SMILES that RDKit cannot parse are dropped and reported.
    If target_column is given, it is carried through unchanged as the label
    column named 'Target', but it is not used to compute any descriptor.
    """
    records = []
    dropped = 0
    for _, row in df.iterrows():
        smi = row[smiles_column]
        try:
            desc = compute_descriptors(smi)
        except InvalidSMILESError:
            dropped += 1
            continue
        desc["SMILES"] = smi
        if target_column is not None:
            desc["Target"] = row[target_column]
        records.append(desc)

    if dropped:
        print(f"Dropped {dropped} rows because RDKit could not parse the SMILES")

    columns = ["SMILES"] + DESCRIPTOR_COLUMNS
    if target_column is not None:
        columns.append("Target")

    return pd.DataFrame.from_records(records, columns=columns)
