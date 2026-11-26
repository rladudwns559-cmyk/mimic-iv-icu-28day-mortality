"""Train XGBoost on existing splits; reserve test for final evaluation.

Run: python scripts/train_xgboost_baseline.py
Requires xgboost>=2.0. Optionally loads project-local .codex_deps.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.codex_deps/xgboost').is_dir():
    sys.path.insert(0, str(ROOT / '.codex_deps'))

import numpy as np
import pandas as pd
import sklearn
try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit('XGBoost is required: python -m pip install xgboost') from exc
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             confusion_matrix, log_loss, precision_recall_curve,
                             roc_auc_score)

SOURCE = ROOT / 'data/processed/error_cleaned/splits'
OUT = ROOT / 'results/xgboost_baseline'


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Output already exists; preserve it before rerunning.')
    columns = json.loads((SOURCE / 'model_feature_columns.json').read_text())
    frames, labels, ids, hashes = {}, {}, {}, {}
    baseline = json.loads((ROOT / 'results/logistic_baseline/metrics.json').read_text())
    for split in ['train', 'validation']:
        for name in ['X_features.csv', 'y_target.csv', 'row_ids.csv']:
            key = f'{split}/{name}'
            hashes[key] = hashlib.sha256((SOURCE / key).read_bytes()).hexdigest()
            if hashes[key] != baseline['input_sha256'][key]:
                raise ValueError(f'Input differs from logistic baseline: {key}')
        frame = pd.read_csv(SOURCE / split / 'X_features.csv')
        labels[split] = pd.read_csv(SOURCE / split / 'y_target.csv').mortality_28d
        ids[split] = pd.read_csv(SOURCE / split / 'row_ids.csv')
        assert frame.columns.tolist() == columns
        assert len(frame) == len(labels[split]) == len(ids[split])
        assert set(labels[split]) == {0, 1}
        assert ids[split].subject_id.is_unique
        assert frame.gender.isin(['F', 'M']).all()
        frame['gender'] = frame.gender.map({'F': 0, 'M': 1})
        assert all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns)
        assert not np.isinf(frame.to_numpy()).any()
        frames[split] = frame
    assert not set(ids['train'].subject_id) & set(ids['validation'].subject_id)
    settings = dict(objective='binary:logistic', tree_method='hist',
                    n_estimators=2000, learning_rate=0.03, max_depth=3,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, reg_alpha=0.0, scale_pos_weight=1.0,
                    eval_metric='logloss', early_stopping_rounds=50,
                    random_state=42, n_jobs=4)
    model = xgb.XGBClassifier(**settings)
    model.fit(frames['train'], labels['train'],
              eval_set=[(frames['validation'], labels['validation'])], verbose=100)
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split, frame in frames.items():
        p = model.predict_proba(frame)[:, 1]
        y = labels[split]
        predicted = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        precision, recall, _ = precision_recall_curve(y, p)
        metrics[split] = dict(
            rows=len(y), prevalence=float(y.mean()), roc_auc=float(roc_auc_score(y, p)),
            average_precision=float(average_precision_score(y, p)),
            pr_auc_trapezoidal=float(auc(recall, precision)),
            brier_score=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
            accuracy=float((predicted == y).mean()), sensitivity=float(tp / (tp + fn)),
            specificity=float(tn / (tn + fp)),
            precision=float(tp / (tp + fp)) if tp + fp else 0.0,
            confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
        ids[split].assign(mortality_28d=y, probability=p,
                          prediction_at_0_5=predicted).to_csv(
                              OUT / f'{split}_predictions.csv', index=False)
    model.save_model(OUT / 'model.ubj')
    restored = xgb.XGBClassifier()
    restored.load_model(OUT / 'model.ubj')
    np.testing.assert_allclose(restored.predict_proba(frames['validation']),
                               model.predict_proba(frames['validation']))
    metadata = dict(xgboost_version=xgb.__version__, sklearn_version=sklearn.__version__,
                    input_sha256=hashes, settings=settings, threshold=0.5,
                    gender_encoding={'F': 0, 'M': 1}, numeric_imputation=None,
                    standardization=False, test_evaluated=False,
                    best_iteration=int(model.best_iteration),
                    best_validation_logloss=float(model.best_score),
                    fitted_on='train; validation selects number of trees by early stopping',
                    hyperparameter_search=False, metrics=metrics)
    (OUT / 'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (OUT / 'learning_curve.json').write_text(json.dumps(model.evals_result(), indent=2), encoding='utf-8')
    (OUT / 'feature_columns.json').write_text(json.dumps(columns, indent=2), encoding='utf-8')
    rows = []
    for name, source in [('Logistic regression', baseline), ('XGBoost', metadata)]:
        rows.append({'model': name, **source['metrics']['validation']})
    pd.DataFrame(rows).drop(columns=['confusion_matrix']).to_csv(
        OUT / 'validation_comparison.csv', index=False)
    lines = ['# XGBoost 기준 모델 학습 결과', '',
             '기존 134개 변수와 동일한 환자 분할을 사용했습니다. 성별 F=0/M=1로 인코딩하고 수치 결측값은 그대로 유지했습니다. 표준화·클래스 가중치는 적용하지 않았습니다.', '',
             f'검증 log loss 기준 50회 조기 종료를 적용했습니다. 선택된 트리 수: {model.best_iteration + 1}.',
             '검증 세트는 트리 수 선택에도 사용했으므로 아래 결과는 설정 선택에 사용한 내부 검증 성능입니다. 테스트 세트는 읽거나 평가하지 않았습니다.', '',
             '| 모델 | 검증 ROC-AUC | PR-AUC(사다리꼴) | 민감도 | 특이도 | 정밀도 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc_trapezoidal']:.4f} | {row['sensitivity']:.2%} | {row['specificity']:.2%} | {row['precision']:.2%} |")
    lines += ['', '민감도·특이도·정밀도는 임계값 0.5 기준이며 임계값은 최적화하지 않았습니다.',
              '검증 결과 차이에 대한 통계적 불확실성은 아직 평가하지 않았습니다.',
              '기존 라벨 미확정 환자 제외와 늦은 storetime 관측 포함 문제가 남아 있습니다.',
              'XGBoost에는 로지스틱 회귀와 같은 단일 변수 오즈비가 없습니다.', '',
              '조기 종료 API 참고: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html']
    (OUT / 'training_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
