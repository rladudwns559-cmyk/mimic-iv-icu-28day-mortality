"""Evaluate charted APACHE II scores against the tuned MICE XGBoost model."""
import csv
import shutil
import subprocess
from pathlib import Path

from predict_tuned_models import ROOT, predict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

RAW = ROOT / 'data/raw/chartevents.csv'
SPLITS = ROOT / 'data/processed/error_cleaned/splits'
OUT = ROOT / 'results/apache_ii_comparison'
TUNING = ROOT / 'results/three_models_nested_tuning'
ITEMS = (226743, 226744, 226745, 226746, 226747, 226748)


def scan():
    executable = shutil.which('rg')
    if executable is None:
        raise RuntimeError('rg is required to scan the raw chart file')
    score_path = OUT / 'charted_rows.csv'
    temp = score_path.with_suffix('.tmp')
    pattern = ',(' + '|'.join(map(str, ITEMS)) + '),'
    with subprocess.Popen(
        [executable, '--no-config', '--no-filename', '--no-line-number', '--color=never',
         '-e', pattern, str(RAW)],
        stdout=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
    ) as scan_process, temp.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['subject_id', 'hadm_id', 'stay_id', 'charttime', 'storetime', 'itemid', 'valuenum'])
        for row in csv.reader(scan_process.stdout):
            if len(row) == 11 and row[6].isdigit() and int(row[6]) in ITEMS:
                writer.writerow([*row[:3], *row[4:7], row[8]])
        if scan_process.wait() not in (0, 1):
            raise RuntimeError('Raw chart scan failed')
    temp.replace(score_path)
    return score_path


def main():
    OUT.mkdir(exist_ok=True)
    source = OUT / 'charted_rows.csv'
    if not source.exists():
        print('Scanning raw chart data for recorded APACHE II', flush=True)
        source = scan()
    charted = pd.read_csv(source)
    cohort = pd.read_csv(ROOT / 'data/processed/cohort_base.csv',
                         usecols=['stay_id', 'intime', 'prediction_time'])
    rows = charted[charted.itemid.eq(226743)].merge(cohort, on='stay_id', validate='many_to_one')
    for c in ('intime', 'prediction_time', 'charttime', 'storetime'):
        rows[c] = pd.to_datetime(rows[c], errors='coerce')
    rows = rows[rows.charttime.ge(rows.intime) & rows.charttime.le(rows.prediction_time)
                & rows.storetime.le(rows.prediction_time)
                & rows.valuenum.between(0, 71) & rows.valuenum.mod(1).eq(0)]
    rows = rows.sort_values(['charttime', 'storetime', 'valuenum'])
    keys = ['subject_id', 'hadm_id', 'stay_id']
    scores = rows.drop_duplicates(keys, keep='last')[keys + ['valuenum']].rename(
        columns={'valuenum': 'apache_ii'})
    partitions = {}
    coverage = []
    for split in ('train', 'validation', 'test'):
        ids = pd.read_csv(SPLITS / split / 'row_ids.csv')
        labels = pd.read_csv(SPLITS / split / 'y_target.csv').mortality_28d
        assert len(ids) == len(labels) and ids.stay_id.is_unique
        base = ids.assign(mortality_28d=labels, feature_row=np.arange(len(ids)))
        matched = base.merge(scores, on=keys, validate='one_to_one')
        partitions[split] = matched
        coverage.append(dict(split=split, total=len(base), matched=len(matched),
                             deaths=int(matched.mortality_28d.sum())))
    counts = pd.DataFrame(coverage)
    counts.to_csv(OUT / 'coverage.csv', index=False)
    print('Raw records by item:', charted.itemid.value_counts().sort_index().to_dict())
    print(counts.to_string(index=False))
    development = pd.concat([partitions['train'], partitions['validation']])
    test = partitions['test']
    if development.mortality_28d.nunique() < 2 or test.mortality_28d.nunique() < 2:
        print('Insufficient APACHE II score coverage for comparative evaluation.')
        return
    calibration = LogisticRegression(penalty=None).fit(
        development[['apache_ii']], development.mortality_28d)
    test_x = pd.read_csv(SPLITS / 'test' / 'X_features.csv')
    xgb = predict(test_x, 'xgboost', TUNING)[test.feature_row.to_numpy()]
    apache = calibration.predict_proba(test[['apache_ii']])[:, 1]
    y = test.mortality_28d
    constant = np.full(len(test), development.mortality_28d.mean())
    summary = pd.DataFrame([
        dict(model=name, n=len(y), deaths=int(y.sum()), roc_auc=roc_auc_score(y, ranks),
             average_precision=average_precision_score(y, ranks), brier=brier_score_loss(y, prob))
        for name, ranks, prob in [('apache_ii', test.apache_ii, apache),
                                  ('tuned_xgboost_mice', xgb, xgb),
                                  ('development_prevalence', constant, constant)]
    ])
    summary.to_csv(OUT / 'test_metrics.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
