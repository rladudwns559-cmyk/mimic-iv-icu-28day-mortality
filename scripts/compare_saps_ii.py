"""Evaluate an authorized SAPS II score export against the tuned XGBoost."""
import argparse
from pathlib import Path

from predict_tuned_models import ROOT, predict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

SPLITS = ROOT / 'data/processed/error_cleaned/splits'
TUNING = ROOT / 'results/three_models_nested_tuning'
OUT = ROOT / 'results/saps_ii_comparison'
KEYS = ['subject_id', 'hadm_id', 'stay_id']


def compare(source, output):
    scores = pd.read_csv(source, usecols=[*KEYS, 'sapsii'])
    if scores[KEYS].isna().any().any() or not scores.stay_id.is_unique:
        raise ValueError('SAPS II export must have one row per ICU stay with all three identifiers.')
    scores = scores.dropna(subset=['sapsii'])
    if not (scores.sapsii.between(0, 163) & scores.sapsii.mod(1).eq(0)).all():
        raise ValueError('Invalid SAPS II score.')

    coverage = []
    matches = {}
    for split in ('train', 'validation', 'test'):
        folder = SPLITS / split
        ids = pd.read_csv(folder / 'row_ids.csv')
        labels = pd.read_csv(folder / 'y_target.csv').mortality_28d
        if len(ids) != len(labels) or not labels.isin([0, 1]).all():
            raise ValueError(f'Invalid label file for {split}.')
        rows = ids.assign(mortality_28d=labels, feature_row=np.arange(len(ids)))
        matched = rows.merge(scores, on=KEYS, validate='one_to_one')
        matches[split] = matched
        coverage.append(dict(split=split, total=len(rows), matched=len(matched),
                             deaths=int(matched.mortality_28d.sum())))

    output.mkdir(parents=True, exist_ok=False)
    counts = pd.DataFrame(coverage)
    counts.to_csv(output / 'coverage.csv', index=False)
    print(counts.to_string(index=False))
    development = pd.concat([matches['train'], matches['validation']])
    test = matches['test']
    if development.mortality_28d.nunique() < 2 or test.mortality_28d.nunique() < 2:
        raise ValueError('Both development and test matches need survivors and deaths.')

    calibration = LogisticRegression(C=np.inf).fit(
        development[['sapsii']], development.mortality_28d)
    test_x = pd.read_csv(SPLITS / 'test' / 'X_features.csv')
    if len(test_x) != counts.loc[counts.split.eq('test'), 'total'].item():
        raise ValueError('Test features and identifiers have different row counts.')
    # Predict the complete pre-defined test batch before matching scores.
    xgb = predict(test_x, 'xgboost', TUNING)[test.feature_row.to_numpy()]
    saps = calibration.predict_proba(test[['sapsii']])[:, 1]
    constant = np.full(len(test), development.mortality_28d.mean())
    y = test.mortality_28d
    summary = pd.DataFrame([
        dict(model=name, n=len(y), deaths=int(y.sum()), roc_auc=roc_auc_score(y, rank),
             average_precision=average_precision_score(y, rank), brier=brier_score_loss(y, prob))
        for name, rank, prob in [
            ('saps_ii', test.sapsii, saps),
            ('tuned_xgboost_mice', xgb, xgb),
            ('development_prevalence', constant, constant),
        ]
    ])
    summary.to_csv(output / 'test_metrics.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    compare(args.scores, args.output)
