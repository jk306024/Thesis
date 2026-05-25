# Predictive Machine Learning Models in Football
### Analysing Player Shots and Understanding Gambling Losses

MSc thesis project. Predicts the number of shots a Premier League player will take in a match using pre-match tabular features. Six models are compared: a naive baseline, a rolling-5 mean, a Poisson GLM, two XGBoost variants and a neural network.

---

## Repository structure

```
pipeline.py            training pipeline — feature engineering, CV, calibration, evaluation
make_all_figures.py    generates all thesis figures from pipeline outputs
thesis_figures_clean/  final figures used in the thesis
requirements.txt       pinned Python dependencies
```

## How to run

Install dependencies:
```
pip install -r requirements.txt
```

Run the pipeline (requires `FINALDATASET.csv` in the same directory):
```
python pipeline.py
```

Generate figures:
```
python make_all_figures.py
```

## Models

| Model | Holdout MAE |
|---|---|
| Naive (global mean) | 1.010 |
| Rolling 5-match mean | 0.902 |
| Poisson GLM | 0.900 |
| XGBoost S3 | 0.875 |
| Neural Net (PyTorch) | 0.889 |
| **XGBoost Tuned** | **0.847** |

All four count-based models output a predicted rate λ (expected shots). Poisson tail probabilities are derived from λ and calibrated using isotonic regression trained on out-of-fold predictions.

## Features

48 features including rolling shot averages (3/5/10 match windows), exponentially weighted means, shot-rate per 90 minutes, opponent defensive strength, Elo ratings, position encoding and home/away context.

## Notes

- Train/test split is player-based (80/20) to prevent data leakage
- Cross-validation uses GroupKFold grouped by player
- All random seeds fixed to 42 for reproducibility
- Figures use the Okabe-Ito colorblind-safe palette
