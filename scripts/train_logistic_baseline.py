"""Fit a train-only preprocessing pipeline; evaluate validation, reserve test."""
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    log_loss, roc_auc_score, precision_recall_curve, auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/error_cleaned/splits'
OUT = ROOT / 'results/logistic_baseline'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'model.joblib').exists():
        raise FileExistsError('Baseline already exists; preserve it before rerunning.')
    frames, labels, ids, hashes = {}, {}, {}, {}
    for split in ['train', 'validation']:
        for filename in ['X_features.csv', 'y_target.csv', 'row_ids.csv']:
            path = SOURCE / split / filename
            hashes[f'{split}/{filename}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames[split] = pd.read_csv(SOURCE / split / 'X_features.csv')
        labels[split] = pd.read_csv(SOURCE / split / 'y_target.csv').mortality_28d
        ids[split] = pd.read_csv(SOURCE / split / 'row_ids.csv')
        assert len(frames[split]) == len(labels[split]) == len(ids[split])
        assert set(labels[split]) == {0, 1}
        assert ids[split].subject_id.is_unique
    assert not set(ids['train'].subject_id) & set(ids['validation'].subject_id)
    columns = json.loads((SOURCE / 'model_feature_columns.json').read_text())
    assert all(frame.columns.tolist() == columns for frame in frames.values())
    numeric = [c for c in columns if c != 'gender']
    assert frames['train'][numeric].select_dtypes(include='number').shape[1] == len(numeric)
    assert not frames['train'][numeric].isna().all().any()
    for frame in frames.values():
        assert not np.isinf(frame[numeric].to_numpy()).any()
        assert frame.gender.isin(['F', 'M']).all()
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric),
        ('gender', OneHotEncoder(drop='if_binary', handle_unknown='error',
                                 sparse_output=False), ['gender']),
    ])
    model = Pipeline([
        ('preprocessing', preprocessing),
        # C selects L2 regularization strength; lbfgs uses L2 by default.
        ('classifier', LogisticRegression(C=1.0, solver='lbfgs',
                                          max_iter=5000, random_state=42)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(frames['train'], labels['train'])
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Model failed to converge: ' + '; '.join(convergence))
    metrics = {}
    for split in frames:
        probability = model.predict_proba(frames[split])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        y = labels[split]
        pr_precision, pr_recall, _ = precision_recall_curve(y, probability)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        metrics[split] = dict(
            rows=len(y), prevalence=float(y.mean()),
            roc_auc=float(roc_auc_score(y, probability)),
            average_precision=float(average_precision_score(y, probability)),
            pr_auc_trapezoidal=float(auc(pr_recall, pr_precision)),
            brier_score=float(brier_score_loss(y, probability)),
            log_loss=float(log_loss(y, probability)),
            accuracy=float((prediction == y).mean()),
            sensitivity=float(tp / (tp + fn)), specificity=float(tn / (tn + fp)),
            precision=float(tp / (tp + fp)) if tp + fp else 0.0,
            confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        )
        ids[split].assign(mortality_28d=y, probability=probability,
                          prediction_at_0_5=prediction).to_csv(OUT / f'{split}_predictions.csv', index=False)
    names = model.named_steps['preprocessing'].get_feature_names_out()
    pd.DataFrame({'feature': names,
                  'coefficient': model.named_steps['classifier'].coef_[0]}).to_csv(
                      OUT / 'coefficients.csv', index=False)
    metadata = dict(
        sklearn_version=sklearn.__version__, input_sha256=hashes,
        settings=dict(C=1.0, regularization='L2', solver='lbfgs', max_iter=5000,
                      class_weight=None, threshold=0.5, seed=42),
        fitted_on='train only', hyperparameter_search=False,
        test_evaluated=False, converged=True,
        n_iter=model.named_steps['classifier'].n_iter_.tolist(),
        intercept=model.named_steps['classifier'].intercept_.tolist(),
        limitations=['Late storetime observations retained under existing charttime policy.',
                     'Unknown-label patients excluded by existing cohort policy.',
                     'Brier score and log loss do not establish calibration on their own.'],
        metrics=metrics,
    )
    joblib.dump(model, OUT / 'model.joblib')
    restored = joblib.load(OUT / 'model.joblib')
    np.testing.assert_allclose(restored.predict_proba(frames['validation']),
                               model.predict_proba(frames['validation']))
    (OUT / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# Logistic regression baseline', '',
             'Numeric: train median imputation, then StandardScaler. Gender: binary one-hot encoding.',
             'L2 logistic regression: C=1, lbfgs, no class weights. Threshold=0.5.',
             'All preprocessing and model fitting use train only. No tuning. Test remains reserved.', '',
             '| Split | Rows | ROC-AUC | Average precision | Brier | Sensitivity | Specificity |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for split, m in metrics.items():
        lines.append(f"| {split} | {m['rows']} | {m['roc_auc']:.4f} | {m['average_precision']:.4f} | {m['brier_score']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |")
    lines += ['', f"Converged in {metadata['n_iter'][0]} iterations; saved model roundtrip verified.",
              'Average precision summarizes the precision-recall curve.',
              'Sensitivity and specificity use threshold 0.5; this threshold has not been optimized.',
              'Calibration requires further assessment beyond Brier score/log loss.', '',
              'Limitations: late-storetime observations remain included; existing cohort excludes unknown labels.',
              'Correlated aggregate features limit interpretation of individual coefficients.']
    lines += ['', '## Trapezoidal PR-AUC', '', '| Split | PR-AUC |', '|---|---:|']
    for split, m in metrics.items():
        lines.append(f"| {split} | {m['pr_auc_trapezoidal']:.6f} |")
    lines += ['', 'Computed using auc(recall, precision); distinct from average precision.']
    (OUT / 'training_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
