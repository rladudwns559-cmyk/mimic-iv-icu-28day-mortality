"""Compare tuned XGBoost missing-data strategies on the saved development folds."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from tune_three_models_nested_cv import (
    ROOT, SOURCE, PRIOR, OUT as TUNING, XGB_SETTINGS,
    fit, metrics, prepare, write_json,
)

OUT = ROOT / 'results/xgboost_missingness_ablation'
ARMS = {'mice_indicator': 5, 'native_indicator': 1, 'mice_without_lactate': 5,
        'native_without_lactate': 1, 'native_lactate_measurement_only': 1}
NAMES = {
    'mice_indicator': 'MICE + 결측 지시변수',
    'native_indicator': '대치 없음 + 결측 지시변수',
    'mice_without_lactate': '젖산 제거 + MICE + 결측 지시변수',
    'native_without_lactate': '젖산 제거 + 대치 없음 + 결측 지시변수',
    'native_lactate_measurement_only': '젖산 측정 정보만 + 대치 없음 + 결측 지시변수',
}
METRICS = ['roc_auc', 'average_precision', 'pr_auc_trapezoidal', 'brier_score']


def main(only_fold=None):
    prior = json.loads((PRIOR / 'experiment.json').read_text(encoding='utf-8'))
    tuning = json.loads((TUNING / 'final_parameters.json').read_text(encoding='utf-8'))
    params = tuning['xgboost']['params']
    originals = prior['input_features']
    removed = [c for c in originals if c.startswith('lab_Lactate_')]
    assert len(removed) == 6
    without_lactate = [c for c in originals if c not in removed]
    measure_only = [c for c in originals if c not in removed or
                    c in ('lab_Lactate_count', 'lab_Lactate_missing')]
    frames, targets, ids = [], [], []
    for split in ('train', 'validation'):
        for filename in ('X_features.csv', 'y_target.csv', 'row_ids.csv'):
            key = f'{split}/{filename}'
            assert hashlib.sha256((SOURCE / key).read_bytes()).hexdigest() == prior['input_sha256'][key]
        frame = pd.read_csv(SOURCE / split / 'X_features.csv', usecols=originals)[originals]
        assert frame.gender.isin(['F', 'M']).all()
        frame['gender'] = frame.gender.map({'F': 0, 'M': 1})
        frames.append(frame)
        targets.append(pd.read_csv(SOURCE / split / 'y_target.csv').mortality_28d)
        ids.append(pd.read_csv(SOURCE / split / 'row_ids.csv'))
    x = pd.concat(frames, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    ids = pd.concat(ids, ignore_index=True)
    saved = pd.read_csv(PRIOR / 'train_folds.csv')
    assert len(x) == 25595 and ids.subject_id.is_unique
    pd.testing.assert_frame_equal(saved[ids.columns], ids)
    pd.testing.assert_series_equal(saved.mortality_28d, y)
    assert set(saved.fold) == set(range(1, 6))

    OUT.mkdir(exist_ok=True)
    config = dict(
        parameter_source=str((TUNING / 'final_parameters.json').relative_to(ROOT)),
        xgboost={**XGB_SETTINGS, **params}, draws=ARMS, lactate_features_removed=removed,
        removed_indicators=['lab_Lactate_std__was_missing'],
        standard_deviation_indicators='computed before imputation',
        test_evaluated=False,
    )
    config_path = OUT / 'experiment.json'
    if config_path.exists():
        recorded = json.loads(config_path.read_text(encoding='utf-8'))
        four_arms = {arm: count for arm, count in ARMS.items()
                     if arm != 'native_lactate_measurement_only'}
        three_arms = {arm: count for arm, count in four_arms.items()
                      if arm != 'native_without_lactate'}
        assert recorded in (config, {**config, 'draws': four_arms},
                            {**config, 'draws': three_arms})
        if only_fold is None and recorded != config:
            write_json(config_path, config)
    else:
        write_json(config_path, config)

    folds, draws = [], []
    oof = {arm: np.full(len(y), np.nan) for arm in ARMS}
    for fold in range(1, 6) if only_fold is None else (only_fold,):
        tr = saved.fold.ne(fold)
        va = saved.fold.eq(fold)
        for arm, count in ARMS.items():
            columns = without_lactate if arm.endswith('_without_lactate') else originals
            if arm == 'native_lactate_measurement_only':
                columns = measure_only
            method = 'native' if arm.startswith('native_') else 'mice'
            predictions = []
            for draw in range(count):
                path = OUT / f'fold_{fold}_{arm}_draw_{draw + 1}.npy'
                if path.exists():
                    p = np.load(path, allow_pickle=False)
                else:
                    print(f'fold={fold}/5 arm={arm} draw={draw + 1}/{count}', flush=True)
                    a, b, _ = prepare(x.loc[tr, columns], x.loc[va, columns], columns, method, draw)
                    p, _ = fit('xgboost', params, a, y[tr], b, columns)
                    pending = path.with_suffix(f'.{os.getpid()}.tmp')
                    with pending.open('wb') as handle:
                        np.save(handle, p, allow_pickle=False)
                    pending.replace(path)
                assert p.shape == (va.sum(),) and np.isfinite(p).all()
                predictions.append(p)
                draws.append(dict(fold=fold, arm=arm, draw=draw + 1, **metrics(y[va], p)))
            pooled = np.mean(predictions, axis=0)
            folds.append(dict(fold=fold, arm=arm, **metrics(y[va], pooled)))
            oof[arm][va] = pooled
        print(f'fold={fold}/5 complete', flush=True)

    if only_fold is not None:
        return
    pd.DataFrame(folds).to_csv(OUT / 'cv_fold_metrics.csv', index=False)
    pd.DataFrame(draws).to_csv(OUT / 'cv_draw_metrics.csv', index=False)
    fold_scores = pd.DataFrame(folds).set_index(['arm', 'fold'])[METRICS]
    contrasts = {
        'mice_lactate_included_minus_excluded': ('mice_indicator', 'mice_without_lactate'),
        'native_lactate_included_minus_excluded': ('native_indicator', 'native_without_lactate'),
        'with_lactate_native_minus_mice': ('native_indicator', 'mice_indicator'),
        'without_lactate_native_minus_mice': ('native_without_lactate', 'mice_without_lactate'),
        'native_full_minus_measurement_only': ('native_indicator', 'native_lactate_measurement_only'),
        'native_measurement_only_minus_excluded': ('native_lactate_measurement_only', 'native_without_lactate'),
    }
    paired = pd.concat({name: fold_scores.loc[a] - fold_scores.loc[b]
                        for name, (a, b) in contrasts.items()}, names=['comparison'])
    paired.to_csv(OUT / 'paired_fold_differences.csv')
    summary = fold_scores.groupby('arm')[METRICS].agg(['mean', 'std'])
    summary.columns = [f'{name}_{stat}' for name, stat in summary.columns]
    summary.to_csv(OUT / 'cv_summary.csv')
    final = saved.assign(**oof)
    assert final[list(ARMS)].notna().all().all()
    final.to_csv(OUT / 'oof_predictions.csv', index=False)

    lines = [
        '# 튜닝된 XGBoost 결측 처리 비교',
        '',
        '개발 데이터 25,595명의 기존 층화 5-fold를 재사용했다. 다섯 조건 모두 '
        '트리 1,000개, 깊이 4, 학습률 0.03, min_child_weight 10, reg_lambda 10을 적용했다. '
        '조기 종료와 테스트 평가는 수행하지 않았다.',
        '',
        '| 조건 | ROC-AUC | PR-AUC (AP) | PR-AUC (사다리꼴) | Brier |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm, name in NAMES.items():
        scores = summary.loc[arm]
        values = [f'{scores[m + "_mean"]:.4f} ± {scores[m + "_std"]:.4f}' for m in METRICS]
        lines.append('| ' + ' | '.join([name, *values]) + ' |')
    lines.extend(['', '수치는 5개 폴드의 평균 ± 표준편차다.', '',
                  '### 같은 폴드에서 계산한 평균 차이', '',
                  '| 비교 | ROC-AUC | AP | Brier | 첫 조건의 AP가 높은 폴드 |',
                  '|---|---:|---:|---:|---:|'])
    labels = [
        'MICE에서 젖산 포함 − 제외',
        '대치 없음에서 젖산 포함 − 제외',
        '젖산 포함 시 대치 없음 − MICE',
        '젖산 제외 시 대치 없음 − MICE',
        '대치 없음에서 젖산 전체 사용 − 측정 정보만 사용',
        '대치 없음에서 젖산 측정 정보만 사용 − 전체 제외',
    ]
    for name, label in zip(contrasts, labels):
        diff = paired.loc[name]
        values = [f'{diff[m].mean():+.4f}' for m in
                  ('roc_auc', 'average_precision', 'brier_score')]
        ap_wins = f'{diff.average_precision.gt(0).sum()}/5'
        lines.append('| ' + ' | '.join([label, *values, ap_wins]) + ' |')
    lines.extend([
        '',
        '대치 없음은 관측값과 대치 전 결측 지시변수를 유지하고 NaN을 XGBoost가 직접 처리한다.',
        '젖산 제거 조건은 수치 집계, 측정 횟수, 결측 지시변수를 대치 전에 모두 제거한다.',
        '젖산 측정 정보만 쓰는 조건은 젖산 측정 횟수와 측정 여부만 유지한다. '
        '젖산 평균·최솟값·최댓값·표준편차와 표준편차 결측 지시변수는 제거한다.',
        '두 MICE 조건은 각 학습 폴드에서만 대치기를 적합하고 5개 모델의 확률을 평균한다. '
        '대치 없음은 모델 1개를 사용하므로 MICE와의 차이에는 앙상블 효과도 포함된다. '
        'MICE 개별 실행 지표는 cv_draw_metrics.csv에 있다.',
        '최종 설정을 선택할 때 사용한 개발 데이터의 비교이므로 독립 테스트 결과와 구분해 해석한다.',
    ])
    (OUT / 'training_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
