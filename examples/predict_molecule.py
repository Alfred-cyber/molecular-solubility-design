"""Example: predict logS for a few example molecules using the saved model.

Run with:
    python examples/predict_molecule.py

Requires a trained model. Run examples/train_model.py first if
models/solubility_model.joblib does not exist yet.
"""

import pandas as pd

from molecular_engineering.config import load_config
from molecular_engineering.descriptors import compute_descriptors
from molecular_engineering.models import SolubilityPredictor

EXAMPLE_MOLECULES = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "acetic acid": "CC(=O)O",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
}


def main():
    config = load_config()
    predictor = SolubilityPredictor.load(config.model_path)

    for name, smiles in EXAMPLE_MOLECULES.items():
        desc = compute_descriptors(smiles)
        row = pd.DataFrame([desc])
        prediction = predictor.predict(row)[0]
        print(f"{name} ({smiles}): predicted logS = {prediction:.3f} (model output, not a measurement)")


if __name__ == "__main__":
    main()
