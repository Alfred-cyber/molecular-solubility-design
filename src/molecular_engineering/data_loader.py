"""Load and clean the Delaney/ESOL solubility dataset from Hugging Face.

The dataset is downloaded through the `datasets` library and cached locally
by that library (usually under ~/.cache/huggingface). Nothing here is
hard-coded or bundled with the repository.
"""

from __future__ import annotations

import pandas as pd


class DatasetLoadError(Exception):
    """Raised when the dataset cannot be downloaded or is malformed."""


REQUIRED_COLUMNS = ("smiles", "target")


def _to_dataframe(hf_split, smiles_column: str, target_column: str) -> pd.DataFrame:
    df = hf_split.to_pandas()
    missing = [c for c in (smiles_column, target_column) if c not in df.columns]
    if missing:
        raise DatasetLoadError(
            f"Dataset is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )
    df = df.rename(columns={smiles_column: "smiles", target_column: "target"})
    return df[["smiles", "target"]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"].notna() & (df["smiles"] != "") & (df["smiles"] != "nan")]
    df = df[df["target"].notna()]
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    return df


def load_delaney_splits(
    dataset_name: str = "zpn/delaney",
    smiles_column: str = "smiles",
    target_column: str = "target",
) -> dict[str, pd.DataFrame]:
    """Download the Delaney/ESOL dataset and return cleaned train/validation/test splits.

    Raises DatasetLoadError if the download fails or the dataset does not have
    the expected structure.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetLoadError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from exc

    try:
        raw = load_dataset(dataset_name)
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to download dataset '{dataset_name}' from Hugging Face: {exc}"
        ) from exc

    available_splits = list(raw.keys())
    split_map = {}
    for wanted in ("train", "validation", "test"):
        if wanted in available_splits:
            split_map[wanted] = wanted
        elif wanted == "validation" and "valid" in available_splits:
            split_map[wanted] = "valid"

    if "train" not in split_map:
        raise DatasetLoadError(
            f"Dataset '{dataset_name}' does not provide a train split. "
            f"Available splits: {available_splits}"
        )

    result = {}
    for name, hf_key in split_map.items():
        df = _to_dataframe(raw[hf_key], smiles_column, target_column)
        df = _clean(df)
        if df.empty:
            raise DatasetLoadError(f"Split '{name}' is empty after cleaning")
        result[name] = df

    if "validation" not in result:
        result["train"], result["validation"] = _split_off_validation(result["train"])

    if "test" not in result:
        raise DatasetLoadError(
            f"Dataset '{dataset_name}' does not provide a test split. "
            f"Available splits: {available_splits}"
        )

    return result


def _split_off_validation(train_df: pd.DataFrame, fraction: float = 0.1, seed: int = 42):
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(train_df, test_size=fraction, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
