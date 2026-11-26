"""ALT/AST ablation: native NaN, median+indicators, five-draw MICE+indicators."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / '.codex_deps/xgboost').is_dir():
    sys.path.insert(0, str(ROOT / '.codex_deps'))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, confusion_matrix, precision_recall_curve, auc)
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT / 'data/processed/error_cleaned/splits'
OUT = ROOT / 'results/xgboost_no_alt_ast_imputation'


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def evaluate(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(rows=len(y), prevalence=float(y.mean()), roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                pr_auc_trapezoidal=float(auc(recall, precision)),
                brier_score=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)),
                sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                accuracy=float((y == (p >= .5)).mean()),
                confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving existing outputs: {OUT}')
    baseline = json.loads((ROOT/'results/xgboost_baseline/metrics.json').read_text())
    source_columns = json.loads((SOURCE/'model_feature_columns.json').read_text())
    excluded = [c for c in source_columns if c.startswith(('lab_ALT_', 'lab_AST_'))]
    assert len(excluded) == 12
    columns = [c for c in source_columns if c not in excluded]
    frames, labels, ids, hashes = {}, {}, {}, {}
    for split in ['train', 'validation']:
        for filename in ['X_features.csv', 'y_target.csv', 'row_ids.csv']:
            key = f'{split}/{filename}'
            hashes[key] = hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()
            assert hashes[key] == baseline['input_sha256'][key], key
        frame = pd.read_csv(SOURCE/split/'X_features.csv')
        assert frame.columns.tolist() == source_columns
        frame = frame[columns].copy()
        assert frame.gender.isin(['F', 'M']).all()
        frame['gender'] = frame.gender.map({'F': 0, 'M': 1})
        labels[split] = pd.read_csv(SOURCE/split/'y_target.csv').mortality_28d
        ids[split] = pd.read_csv(SOURCE/split/'row_ids.csv')
        assert len(frame) == len(labels[split]) == len(ids[split])
        assert ids[split].subject_id.is_unique and ids[split].stay_id.is_unique
        assert labels[split].isin([0, 1]).all()
        assert not np.isinf(frame.to_numpy()).any()
        frames[split] = frame
    assert not set(ids['train'].subject_id) & set(ids['validation'].subject_id)
    train, valid = frames['train'], frames['validation']
    assert not train.isna().all().any()
    # Reuse existing lab/vital missing flags. Add only patterns not already represented.
    indicator_sources = []
    patterns = [train[c].to_numpy() for c in columns if c.endswith('_missing')]
    for c in columns:
        mask = train[c].isna().astype(int).to_numpy()
        if mask.any() and not any(np.array_equal(mask, p) for p in patterns):
            indicator_sources.append(c)
            patterns.append(mask)
    indicator_names = [f'{c}__was_missing' for c in indicator_sources]
    def with_indicators(array, original):
        result = pd.DataFrame(array, columns=columns, index=original.index)
        for c, name in zip(indicator_sources, indicator_names):
            result[name] = original[c].isna().astype(np.int8)
        return result
    settings = baseline['settings']
    OUT.mkdir(parents=True)
    metadata = dict(input_sha256=hashes, excluded_features=excluded, retained_features=columns,
                    added_indicator_sources=indicator_sources, settings=settings,
                    versions=dict(xgboost=xgb.__version__, sklearn=sklearn.__version__),
                    test_evaluated=False, threshold=.5,
                    existing_missing_flags_retained_in_all_arms=True,
                    preprocessing_fit_on='train only; no outcome used for imputation',
                    mice=dict(implementation='IterativeImputer with BayesianRidge posterior draws',
                              draws=5, seeds=list(range(42, 47)), max_iter=10,
                              n_nearest_features=20, sample_posterior=True,
                              scaling='train StandardScaler before imputation, inverse afterward',
                              bounds='imputed nonnegative features >=0; AnionGap values unbounded',
                              pooling='mean predicted probability across five fitted XGBoost models'),
                    methods={})
    all_rows, predictions = [], {}
    for method in ['native_nan', 'median_indicator', 'mice_indicator']:
        started = time.perf_counter()
        folder = OUT/method
        folder.mkdir()
        print(f'START {method}', flush=True)
        per_draw, draw_metrics = [], []
        scaler = None
        if method == 'mice_indicator':
            scaler = StandardScaler().fit(train)
            scaled_train, scaled_valid = scaler.transform(train), scaler.transform(valid)
            joblib.dump(scaler, folder/'scaler.joblib')
            minimum = np.full(len(columns), -np.inf)
            for j, c in enumerate(columns):
                if c.endswith(('_mean', '_min', '_max', '_std')):
                    if 'AnionGap' not in c or c.endswith('_std'):
                        minimum[j] = -scaler.mean_[j]/scaler.scale_[j]
        for draw in range(5 if method == 'mice_indicator' else 1):
            if method == 'native_nan':
                xt, xv = train, valid
            elif method == 'median_indicator':
                imputer = SimpleImputer(strategy='median', keep_empty_features=True)
                xt = with_indicators(imputer.fit_transform(train), train)
                xv = with_indicators(imputer.transform(valid), valid)
                joblib.dump(imputer, folder/'imputer.joblib')
            else:
                imputer = IterativeImputer(max_iter=10, n_nearest_features=20,
                                          initial_strategy='median', sample_posterior=True,
                                          random_state=42+draw, skip_complete=True,
                                          min_value=minimum, keep_empty_features=True, verbose=1)
                with threadpool_limits(limits=1):
                    filled_train = imputer.fit_transform(scaled_train)
                    # Save pre-transform RNG state for exact reproducibility at inference.
                    joblib.dump(imputer, folder/f'imputer_{draw+1}.joblib', compress=3)
                    filled_valid = imputer.transform(scaled_valid)
                xt = with_indicators(scaler.inverse_transform(filled_train), train)
                xv = with_indicators(scaler.inverse_transform(filled_valid), valid)
            if method != 'native_nan':
                assert np.isfinite(xt.to_numpy()).all() and np.isfinite(xv.to_numpy()).all()
                # Observed values must be preserved by imputation.
                for original, transformed in [(train, xt), (valid, xv)]:
                    observed = original.notna().to_numpy()
                    np.testing.assert_allclose(transformed[columns].to_numpy()[observed],
                                               original.to_numpy()[observed], rtol=1e-10, atol=1e-10)
            assert not any(c.startswith(('lab_ALT_', 'lab_AST_')) for c in xt)
            model = xgb.XGBClassifier(**settings)
            model.fit(xt, labels['train'], eval_set=[(xv, labels['validation'])], verbose=200)
            model.save_model(folder/f'model_{draw+1}.ubj')
            p = dict(train=model.predict_proba(xt)[:, 1], validation=model.predict_proba(xv)[:, 1])
            restored = xgb.XGBClassifier()
            restored.load_model(folder/f'model_{draw+1}.ubj')
            np.testing.assert_allclose(restored.predict_proba(xv)[:, 1], p['validation'])
            if method == 'mice_indicator':
                with threadpool_limits(limits=1):
                    loaded = joblib.load(folder/f'imputer_{draw+1}.joblib')
                    replay = scaler.inverse_transform(loaded.transform(scaled_valid))
                np.testing.assert_allclose(replay, xv[columns], rtol=1e-10, atol=1e-10)
            per_draw.append(p)
            draw_metrics.append(dict(draw=draw+1, best_iteration=int(model.best_iteration),
                                     best_validation_logloss=float(model.best_score),
                                     validation=evaluate(labels['validation'], p['validation'])))
            write_json(folder/f'learning_curve_{draw+1}.json', model.evals_result())
            write_json(folder/'feature_columns.json', xt.columns.tolist())
            print(f'FINISHED {method} draw={draw+1} trees={model.best_iteration+1}', flush=True)
        metrics = {}
        for split in frames:
            p = np.mean([d[split] for d in per_draw], axis=0)
            metrics[split] = evaluate(labels[split], p)
            ids[split].assign(mortality_28d=labels[split], probability=p,
                              prediction_at_0_5=(p >= .5).astype(int)).to_csv(folder/f'{split}_predictions.csv', index=False)
            if split == 'validation':
                predictions[method] = p
            all_rows.append(dict(method=method, split=split, **{k:v for k,v in metrics[split].items() if k != 'confusion_matrix'}))
        metadata['methods'][method] = dict(metrics=metrics, draws=draw_metrics,
                                          elapsed_seconds=time.perf_counter()-started,
                                          feature_count=len(xt.columns))
        write_json(folder/'metrics.json', metadata['methods'][method])
        write_json(OUT/'experiment.json', metadata)
        pd.DataFrame(all_rows).to_csv(OUT/'comparison.csv', index=False)
        print(json.dumps(dict(method=method, validation=metrics['validation'])), flush=True)
    # Paired patient bootstrap of performance differences, with fixed trained models.
    rng = np.random.default_rng(42)
    y = labels['validation'].to_numpy()
    boot = []
    for method in ['median_indicator', 'mice_indicator']:
        delta = []
        for _ in range(1000):
            ix = rng.integers(0, len(y), len(y))
            if np.unique(y[ix]).size < 2:
                continue
            a, b = predictions[method][ix], predictions['native_nan'][ix]
            delta.append([roc_auc_score(y[ix], a)-roc_auc_score(y[ix], b),
                          average_precision_score(y[ix], a)-average_precision_score(y[ix], b),
                          brier_score_loss(y[ix], a)-brier_score_loss(y[ix], b)])
        for j, metric in enumerate(['roc_auc', 'average_precision', 'brier_score']):
            low, high = np.quantile(np.asarray(delta)[:, j], [.025, .975])
            point = metadata['methods'][method]['metrics']['validation'][metric]-metadata['methods']['native_nan']['metrics']['validation'][metric]
            boot.append(dict(method=method, reference='native_nan', metric=metric,
                             difference=point, ci95_low=low, ci95_high=high))
    pd.DataFrame(boot).to_csv(OUT/'paired_bootstrap.csv', index=False)
    val = pd.DataFrame(all_rows).query("split == 'validation'")
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = ['NaN 그대로', '중앙값 + 지표', 'MICE(5회) + 지표']
    for ax, metric, title in zip(axes, ['roc_auc', 'average_precision', 'brier_score'],
                                ['ROC-AUC (높을수록 좋음)', 'Average precision (높을수록 좋음)', 'Brier score (낮을수록 좋음)']):
        values = val[metric].to_numpy()
        ax.bar(names, values, color=['#187f97', '#b87545', '#6855a0'])
        ax.set_ylim(0, max(values)*1.2)
        ax.set_title(title)
        for j, value in enumerate(values):
            ax.text(j, value, f'{value:.4f}', ha='center', va='bottom')
    fig.suptitle('ALT·AST 제외: 결측 처리별 XGBoost 검증 성능')
    fig.tight_layout()
    fig.savefig(OUT/'validation_comparison.png', dpi=180)
    plt.close(fig)
    lines = ['# ALT·AST 제외 후 XGBoost 결측 처리 비교', '',
             '동일 환자 분할(train 21,078명 / validation 4,517명), 동일 XGBoost 설정을 사용했습니다. 테스트는 읽거나 평가하지 않았습니다.',
             'ALT·AST의 mean/min/max/std/count/missing 총 12개 특징을 제외했습니다. 기존 134개에서 122개가 남았습니다.', '',
             '| 방법 | ROC-AUC | Average precision | PR-AUC(사다리꼴) | Brier | Log loss | 민감도 | 특이도 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in val.itertuples():
        lines.append(f'| {r.method} | {r.roc_auc:.4f} | {r.average_precision:.4f} | {r.pr_auc_trapezoidal:.4f} | {r.brier_score:.4f} | {r.log_loss:.4f} | {r.sensitivity:.2%} | {r.specificity:.2%} |')
    best_logloss = val.loc[val.log_loss.idxmin(), 'method']
    all_cover_zero = all(r['ci95_low'] <= 0 <= r['ci95_high'] for r in boot)
    lines += ['', '## 결과 해석', '',
              f'- 검증 log loss가 가장 낮은 방법은 {best_logloss}입니다.',
              f'- native_nan 대비 ROC-AUC/AP/Brier 차이의 95% 구간이 모두 0을 포함하는가: {all_cover_zero}. 구간이 0을 포함하는 것은 동등성의 증명은 아닙니다.',
              '- MICE 결과는 5개 모델의 앙상블이므로 단일 모델 두 방법과 비교할 때 대체 효과와 앙상블 효과가 함께 포함됩니다.']
    lines += ['', '![검증 성능 비교](validation_comparison.png)', '',
              '## 구현 및 검증', '',
              '- native_nan: 수치 NaN을 그대로 전달합니다. 원래 존재하던 측정 횟수와 결측 지표는 유지합니다.',
              f'- 중앙값/MICE: 기존 결측 지표를 유지하고 훈련 데이터에서 기존 지표와 중복되지 않는 결측 패턴 {len(indicator_sources)}개에 지표를 추가했습니다. 단일 측정으로 생긴 std 결측도 포함합니다. 추가 지표 목록은 experiment.json에 있습니다.',
              '- 모든 대체 및 스케일 통계는 훈련 데이터에만 적합합니다. 사망 라벨은 대체기의 입력으로 사용하지 않습니다.',
              '- MICE: BayesianRidge 기반 IterativeImputer, posterior sampling, 10회 순환, 이웃 특징 20개, seed 42~46의 5개 데이터셋입니다. 각 데이터셋에서 모델을 학습하고 예측확률을 평균합니다. 단일 IterativeImputer 대체와 구별되며 Rubin 규칙으로 회귀계수를 합치는 분석은 아닙니다.',
              '- MICE 수치 안정성을 위해 훈련 평균/표준편차로 스케일한 뒤 대체하고 원래 단위로 복원합니다. 음수가 허용되지 않는 연속 특징의 대체값에 0 하한을 적용합니다(AnionGap 값 제외). 집계 특징 mean/min/max의 상호 일관성은 강제하지 않습니다.',
              '- 원래 관측된 값 보존, 환자 중복/훈련-검증 중첩 부재, ALT·AST 전부 제외, 입력 해시 일치, 대체 후 유한값, 저장 모델 예측 재현 및 MICE 저장 대체기 검증 변환 재현을 확인했습니다.',
              '- XGBoost는 기존 설정(max_depth=3, learning_rate=0.03, 최대 2,000개 트리, 검증 log loss 조기 종료 50회, seed=42)을 그대로 사용합니다. 세부 설정/버전/파일 해시는 experiment.json에 저장했습니다.',
              '- 민감도·특이도는 임계값 0.5입니다. 검증은 조기 종료와 방법 비교에 사용했으므로 독립 최종 성능이 아닙니다.',
              '- paired_bootstrap.csv: 검증 환자를 짝지어 1,000회 재표집한 native_nan 대비 지표 차이의 95% 백분위 구간. 모델 재학습·모델 선택의 불확실성은 반영하지 않습니다.',
              '- 기존 unknown 라벨 제외, 늦은 storetime 관측 포함, 내부 무작위 분할의 한계가 유지됩니다. 성능 차이는 ALT·AST 포함 여부에 대한 직접 결론이 아닙니다.', '',
              '재현: `python scripts/compare_xgboost_imputation.py` (기존 산출물이 있으면 덮어쓰지 않습니다).',
              '각 방법 폴더에 모델·대체기·특징 목록·예측확률·학습 곡선·지표를 저장했습니다.']
    (OUT/'training_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Final source-preservation check.
    for key, digest in hashes.items():
        assert hashlib.sha256((SOURCE/key).read_bytes()).hexdigest() == digest
    print(val.to_string(index=False), flush=True)
    print(f'COMPLETE: {OUT}', flush=True)


if __name__ == '__main__':
    main()
