# Molecular Engineering: Solubility Prediction and Inverse Design

A research-prototype Python project that predicts aqueous solubility from
molecular structure and uses that model to search for candidate molecules
close to a target solubility.

This is **not** a drug-discovery tool and does not claim to generate novel
molecules. It is a scientifically honest, end-to-end demonstration of the
workflow:

```
Real molecular data (Hugging Face)
  -> RDKit molecular descriptors
  -> machine-learning property prediction
  -> inverse molecular search over known structures
  -> candidate ranking
  -> (optional) simulated closed-loop feedback
```

## Motivation

A common question in computational chemistry is: *given a molecule's
structure, can we predict a physical property like solubility without
running a physical measurement?* And the follow-up question that matters
for design work: *given a property we want, which molecules are worth
looking at first?*

This project answers both questions with real experimental data, a
transparent baseline model comparison, and a search procedure that is
explicit about what it does and does not do.

## Dataset

**Source:** [`zpn/delaney`](https://huggingface.co/datasets/zpn/delaney) on
Hugging Face — the Delaney (ESOL) aqueous solubility dataset.

The dataset provides SMILES strings and experimentally measured log
solubility (logS, mol/L) values, with `train`, `validation`, and `test`
splits. It is downloaded on demand through the `datasets` library and
cached locally by that library (typically under `~/.cache/huggingface`).
**No dataset files are bundled with this repository.**

Citation:

> Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from
> Molecular Structure. *Journal of Chemical Information and Computer
> Sciences*, 44(3), 1000-1005. Hosted on Hugging Face as `zpn/delaney`.

The measured `target` values in this dataset are experimental
measurements. Anything this project computes from a trained model is a
**prediction**, not a new measurement — that distinction is preserved
throughout the code and its output.

## Project structure

```
molecular_engineering/
  README.md
  pyproject.toml
  requirements.txt
  config.example.json

  src/molecular_engineering/
    __init__.py
    data_loader.py       download and clean the Hugging Face dataset
    molecules.py          SMILES parsing and validation (RDKit)
    descriptors.py        RDKit descriptor calculation
    models.py             SolubilityPredictor (fit/predict/evaluate/save/load)
    evaluation.py         regression metrics and baseline comparison
    inverse_design.py     search a candidate pool for a target property
    optimization.py       generic objective/constraint scoring
    active_learning.py    simulated closed-loop retraining demo
    visualization.py      matplotlib plots
    pipeline.py           end-to-end training pipeline
    cli.py                command-line interface
    config.py             configuration loading

  tests/                  pytest test suite
  examples/               runnable example scripts
  models/                 trained model is saved here (.gitkeep only, no binaries committed)
  results/                metrics, summaries, and plots are written here
```

## Installation

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd molecular_engineering
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### RDKit

RDKit is installed automatically from PyPI via `requirements.txt`
(`rdkit` package, prebuilt wheels are available for Linux, macOS, and
Windows on standard Python versions). If the PyPI wheel does not work on
your platform, install it through conda instead:

```bash
conda install -c conda-forge rdkit
```

### Hugging Face access

The first run of training, evaluation, or search will download the
`zpn/delaney` dataset automatically. You need an internet connection for
that first run; after that, the dataset is cached locally and subsequent
runs do not re-download it.

No Hugging Face account or API token is required for this public dataset.

## Configuration

Parameters are read from a JSON (or YAML) config file, with sensible
defaults if you don't provide one. See `config.example.json`:

```json
{
  "dataset": "zpn/delaney",
  "smiles_column": "smiles",
  "target_column": "target",
  "target_logS": -2.0,
  "top_n": 10,
  "model": "random_forest",
  "random_seed": 42,
  "models_dir": "models",
  "results_dir": "results",
  "model_filename": "solubility_model.joblib"
}
```

Copy it and edit as needed:

```bash
cp config.example.json config.json
```

Pass `--config config.json` to any CLI command to use it. Without
`--config`, defaults are used (equivalent to the file above).

Note: `pipeline.run_training_pipeline` always trains and compares all
three baseline models (linear regression, random forest, gradient
boosting) and selects the one with the best validation performance, so
the model actually deployed may differ from the `"model"` value in the
config file. That field is kept for record-keeping and for future
single-model retraining workflows.

## Training

```bash
python -m molecular_engineering.cli train
```

or, equivalently:

```bash
python examples/train_model.py
```

This will:

1. Download and clean the `zpn/delaney` dataset (train / validation / test
   splits preserved as provided).
2. Compute RDKit descriptors for every molecule (molecular weight, LogP,
   TPSA, H-bond donors/acceptors, rotatable bonds, ring count, heavy atom
   count, fraction sp3).
3. Train three baseline regressors (linear regression, random forest,
   histogram gradient boosting) on the training split.
4. Compare them on the **validation** split (MAE, RMSE, R²) and select the
   best one by validation MAE — never by training performance.
5. Evaluate the selected model once on the held-out **test** split.
6. Save the trained model to `models/solubility_model.joblib`.
7. Write a JSON summary (dataset, model, metrics, seed, timestamp) to
   `results/training_summary.json`.
8. Print model-level feature importance for the selected model, labeled
   explicitly as a model artifact rather than a causal explanation.

Example output:

```
Model comparison (sorted by validation MAE):
             model  train_MAE  ...   val_MAE   val_RMSE    val_R2
      random_forest      0.28  ...      0.62       0.81      0.89
  gradient_boosting      0.31  ...      0.65       0.85      0.87
  linear_regression      0.75  ...      0.98       1.20      0.71

Selected model: random_forest
Test set performance:
  MAE: 0.63
  RMSE: 0.82
  R2: 0.88

Model-level feature importance (does not imply causation):
LogP              0.41
TPSA              0.19
MW                0.15
...
```

(Actual numbers depend on the current dataset content and random seed.)

## Prediction

```bash
python -m molecular_engineering.cli predict "CCO"
```

```
SMILES: CCO
Predicted logS (model output, not a measurement): -0.14
```

Or from Python:

```python
from molecular_engineering.config import load_config
from molecular_engineering.descriptors import compute_descriptors
from molecular_engineering.models import SolubilityPredictor
import pandas as pd

config = load_config()
predictor = SolubilityPredictor.load(config.model_path)
row = pd.DataFrame([compute_descriptors("CCO")])
print(predictor.predict(row))
```

## Evaluation

```bash
python -m molecular_engineering.cli evaluate
```

Re-evaluates the saved model on the untouched test split and prints MAE,
RMSE, and R².

## Inverse design

```bash
python -m molecular_engineering.cli search --target -2.0 --top 10
```

This searches the Delaney dataset itself for molecules whose **predicted**
logS is closest to the target, and reports each candidate's SMILES,
predicted logS, measured logS (for comparison), prediction error,
molecular weight, LogP, and TPSA.

```python
from molecular_engineering.optimization import optimize, Objective, Constraints
```

`optimization.py` generalizes this into an objective/constraint system:

- Objectives: `target`, `minimize`, `maximize`
- Constraints: max molecular weight, max LogP, min TPSA, max ring count,
  allowed elements

```python
result = optimize(
    candidate_pool,
    predictor,
    Objective(kind="target", target_value=-2.0, tolerance=0.5),
    constraints=Constraints(max_mw=200, max_logp=3),
    top_n=10,
)
```

### Important scientific limitation

This inverse-design step **searches known molecules and ranks them** — it
does not generate new molecular structures:

```
target property -> search known chemical structures -> predict -> rank
```

A more advanced future version, not implemented here, would look like:

```
target property -> generate new molecular structures -> validate
  -> predict -> optimize -> test
```

`inverse_design.MoleculeGenerator` defines the interface a future
generative component would implement (`generate(n) -> list[SMILES]`). The
current implementation, `DatasetMoleculeGenerator`, only samples from an
existing dataset. A real generator could later use evolutionary
algorithms, SELFIES-based sampling, molecular transformers, graph neural
networks, diffusion models, or reinforcement learning — none of that is
implemented or faked here.

## Simulated closed-loop / active learning

`active_learning.py` demonstrates the shape of a closed-loop design cycle:

```
train model -> search candidates -> select candidates
  -> simulated feedback -> add to training data -> retrain -> search again
```

The "feedback" is a **simulated measurement**: the dataset's known value
for a molecule plus random noise, standing in for what a real lab assay
would return. It is not, and is never labeled as, a real experimental
result. This is meant to illustrate the architecture of active learning,
not to claim state-of-the-art performance.

```python
from molecular_engineering.active_learning import run_active_learning_loop

result = run_active_learning_loop(train_df, pool_df, target_logS=-2.0, rounds=3)
```

## Visualizations

`visualization.py` produces:

- Actual vs. predicted logS (test set)
- Prediction residuals
- Distribution of measured logS
- Molecular weight vs. logS
- LogP vs. logS
- Top inverse-design candidates vs. target

Run `python examples/train_model.py` and `python examples/inverse_design.py`
to generate these into `results/`.

## Running the tests

```bash
pytest tests/ -v
```

The test suite covers dataset loading and cleaning (with a mocked Hugging
Face call, so tests don't require network access), SMILES validation,
descriptor generation, missing-data handling, model training and
prediction, evaluation metrics, inverse-design ranking, optimization
constraints, and reproducibility with a fixed random seed.

## Reproducibility

Every training run records: dataset name, model type, hyperparameter
seed, validation/test metrics, number of samples per split, and a
timestamp, written to `results/training_summary.json`. The default random
seed is `42` and is configurable through the config file.

## Limitations

- The property predicted is limited to aqueous solubility, from a single
  dataset of about 1,100 molecules.
- Inverse design searches existing molecules; it does not invent new
  chemical structures.
- The active-learning feedback loop uses simulated (noisy, dataset-derived)
  measurements, never real laboratory data.
- Descriptor-based baseline models (linear regression, random forest,
  gradient boosting) are used; no graph neural networks or deep learning
  architectures are included in this version.
- Model predictions carry no guaranteed accuracy for molecules very
  different in structure from the training data.

## Future work

- Plug in additional real datasets (larger solubility sets, toxicity,
  blood-brain-barrier permeability, bioactivity, materials, QM, or
  PubChem-derived collections) behind the same `data_loader` interface.
- Implement an actual `MoleculeGenerator` (e.g., a SELFIES-based sampler
  or graph-based generative model) and validate its output before using
  it for design.
- Add graph-based or learned molecular representations alongside the
  current RDKit descriptor features.
- Extend `optimization.py` to multi-objective scoring across several
  predicted properties at once.


