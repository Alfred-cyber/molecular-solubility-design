import pytest

from molecular_engineering.molecules import (
    parse_smiles,
    is_valid_smiles,
    canonicalize,
    InvalidSMILESError,
)


def test_parse_valid_smiles():
    mol = parse_smiles("CCO")
    assert mol is not None
    assert mol.GetNumAtoms() == 3


def test_parse_invalid_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        parse_smiles("not_a_smiles_string(((")


def test_parse_empty_string_raises():
    with pytest.raises(InvalidSMILESError):
        parse_smiles("")


def test_is_valid_smiles_true_for_ethanol():
    assert is_valid_smiles("CCO") is True


def test_is_valid_smiles_false_for_garbage():
    assert is_valid_smiles("XyZ123!!!") is False


def test_canonicalize_is_idempotent():
    canon = canonicalize("OCC")
    assert canonicalize(canon) == canon
