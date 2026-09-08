"""Project configuration loading and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_CONFIG = {
    "dataset": "zpn/delaney",
    "smiles_column": "smiles",
    "target_column": "target",
    "target_logS": -2.0,
    "top_n": 10,
    "model": "random_forest",
    "random_seed": 42,
    "models_dir": "models",
    "results_dir": "results",
    "model_filename": "solubility_model.joblib",
}


@dataclass
class Config:
    dataset: str = DEFAULT_CONFIG["dataset"]
    smiles_column: str = DEFAULT_CONFIG["smiles_column"]
    target_column: str = DEFAULT_CONFIG["target_column"]
    target_logS: float = DEFAULT_CONFIG["target_logS"]
    top_n: int = DEFAULT_CONFIG["top_n"]
    model: str = DEFAULT_CONFIG["model"]
    random_seed: int = DEFAULT_CONFIG["random_seed"]
    models_dir: str = DEFAULT_CONFIG["models_dir"]
    results_dir: str = DEFAULT_CONFIG["results_dir"]
    model_filename: str = DEFAULT_CONFIG["model_filename"]

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def model_path(self) -> Path:
        return Path(self.models_dir) / self.model_filename


def load_config(path: str | None = None) -> Config:
    """Load configuration from a JSON or YAML file, falling back to defaults.

    Any keys missing from the file are filled in with DEFAULT_CONFIG values.
    """
    if path is None:
        return Config()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = file_path.read_text()
    if file_path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    merged = dict(DEFAULT_CONFIG)
    merged.update(data or {})
    return Config(**merged)
