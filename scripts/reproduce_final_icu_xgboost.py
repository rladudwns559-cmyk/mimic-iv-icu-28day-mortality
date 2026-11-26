"""Rebuild and evaluate the final first-ICU XGBoost using locally authorized data."""
import hashlib
import json
import argparse
from pathlib import Path

from tune_three_models_nested_cv import ROOT, SOURCE, XGB_SETTINGS, metrics, prepare
import numpy as np
import pandas as pd
import xgboost as xgb

IDS = ['subject_id', 'hadm_id', 'stay_id']
OUT = ROOT / 'results/final_icu_xgboost'


def main(source=SOURCE, raw_dir=ROOT / 'data/raw', output=OUT):
    config = json.loads((ROOT / 'schemas/final_model_inputs.json').read_text())
    report = ROOT / 'reports/xgboost_indicator_icu_comparison/final_icu_test_metrics.json'
    reference = json.loads(report.read_text())
    columns = config['input_features']
    icu = pd.read_csv(raw_dir / 'icustays.csv', usecols=[*IDS, 'first_careunit'])
    splits = {}
    for split in ('train', 'validation', 'test'):
        folder = source / split
        for name in ('X_features.csv', 'y_target.csv', 'row_ids.csv'):
            expected = (reference['test_sha256'][name] if split == 'test'
                        else config['input_sha256'][f'{split}/{name}'])
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                print(f'Input differs from the recorded experiment: {split}/{name}')
        x = pd.read_csv(folder / 'X_features.csv', usecols=columns)[columns].copy()
        y = pd.read_csv(folder / 'y_target.csv').mortality_28d
        ids = pd.read_csv(folder / 'row_ids.csv')
        linked = ids.merge(icu, on=IDS, how='left', validate='one_to_one')
        pd.testing.assert_frame_equal(linked[IDS], ids)
        if not (len(x) == len(y) == len(ids) and x.gender.isin(['F', 'M']).all()
                and y.isin([0, 1]).all() and linked.first_careunit.notna().all()):
            raise ValueError(f'Invalid inputs for {split}')
        x['gender'] = x.gender.map({'F': 0, 'M': 1})
        splits[split] = (x, y, ids, linked.first_careunit)
    a, y, ids, units = [pd.concat(parts, ignore_index=True)
                        for parts in zip(splits['train'], splits['validation'])]
    b, z, test_ids, test_units = splits['test']
    if not (ids.subject_id.is_unique and test_ids.subject_id.is_unique
            and set(ids.subject_id).isdisjoint(test_ids.subject_id)):
        raise ValueError('Development and test must contain disjoint patients.')
    a, b, prep = prepare(a, b, columns, 'native')
    categories = sorted(units.unique())
    model = xgb.XGBClassifier(**{**XGB_SETTINGS, **reference['xgboost']})
    model.fit(np.column_stack([a, units.to_numpy()[:, None] == categories]), y)
    design = np.column_stack([b, test_units.to_numpy()[:, None] == categories])
    probabilities = model.predict_proba(design)[:, 1].astype(float)
    baseline = float(np.mean((z - y.mean()) ** 2))
    scores = metrics(z, probabilities)
    results = dict(metrics=scores, train_n=len(y), test_n=len(z),
                   brier_reference=baseline, brier_relative_reduction=1-scores['brier_score']/baseline,
                   icu_categories=categories, clinical_features=columns, std_columns=prep['std_columns'])
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(output / 'model.ubj')
    (output / 'test_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results['metrics'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, default=SOURCE)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.splits, args.raw_dir, args.output)
