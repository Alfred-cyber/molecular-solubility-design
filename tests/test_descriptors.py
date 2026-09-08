import pandas as pd
import pytest

from molecular_engineering.descriptors import compute_descriptors, build_feature_table, DESCRIPTOR_COLUMNS
from molecular_engineering.molecules import InvalidSMILESError


def test_compute_descriptors_ethanol():
    desc = compute_descriptors("CCO")
    assert set(desc.keys()) == set(DESCRIPTOR_COLUMNS)
    assert desc["MW"] == pytest.approx(46.07, abs=0.1)
    assert desc["HeavyAtoms"] == 3
    assert desc["HBD"] == 1


def test_compute_descriptors_invalid_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        compute_descriptors("not_valid!!!")


def test_build_feature_table_drops_invalid_rows():
    df = pd.DataFrame({
        "smiles": ["CCO", "invalid_smiles_xyz", "c1ccccc1"],
        "target": [0.1, -3.0, -2.1],
    })
    features = build_feature_table(df)
    assert len(features) == 2
    assert "invalid_smiles_xyz" not in features["SMILES"].values


def test_build_feature_table_does_not_use_target_as_descriptor():
    df = pd.DataFrame({"smiles": ["CCO"], "target": [0.1]})
    features = build_feature_table(df)
    assert "Target" in features.columns
    assert "target" not in [c.lower() for c in DESCRIPTOR_COLUMNS]


def test_build_feature_table_without_target_column():
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
    features = build_feature_table(df, target_column=None)
    assert "Target" not in features.columns
    assert len(features) == 2
