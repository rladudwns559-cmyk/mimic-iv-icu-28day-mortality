"""Compare missing flags and first ICU type on existing development folds."""
import hashlib
import json

from tune_three_models_nested_cv import ROOT, SOURCE, PRIOR, OUT as TUNING, fit, metrics, prepare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ABLATION = ROOT / 'results/xgboost_missingness_ablation'
OUT = ROOT / 'results/xgboost_indicator_icu_comparison'
IDS = ['subject_id', 'hadm_id', 'stay_id']
ARMS = {
    'baseline': '기준 모델',
    'without_indicators': '결측 지시변수 제거',
    'with_icu': '첫 ICU 유형 추가',
}
KEY_METRICS = ['average_precision', 'roc_auc', 'brier_score']


def evaluate_subgroups(oof):
    rows, bins = [], []
    groups = list(oof.groupby('first_careunit', sort=True))
    plotted_groups = [name for name, group in groups if len(group) >= 100]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True,
                             layout='constrained')
    plots = dict(zip(plotted_groups, axes.flat))
    assert len(plots) == len(plotted_groups)
    for unit, g in groups:
        ax = plots.get(unit)
        y = g.mortality_28d
        if ax is not None:
            ax.plot([0, 1], [0, 1], '--', color='0.6', linewidth=1, label='Perfect calibration')
        for arm, label in [('baseline', 'Baseline'), ('with_icu', 'With first ICU type')]:
            p = g[arm]
            scores = metrics(y, p) if y.nunique() == 2 else dict(
                average_precision=np.nan, roc_auc=np.nan, brier_score=brier_score_loss(y, p))
            rows.append(dict(first_careunit=unit, arm=arm, n=len(g), deaths=int(y.sum()),
                             observed_rate=y.mean(), mean_prediction=p.mean(),
                             predicted_minus_observed=p.mean()-y.mean(), **scores))
            if ax is None:
                continue
            binned = g.assign(bin=pd.qcut(p, q=5, labels=False, duplicates='drop')).groupby('bin')
            points = binned.agg(n=('mortality_28d', 'size'),
                                observed_rate=('mortality_28d', 'mean'),
                                mean_prediction=(arm, 'mean')).reset_index()
            assert points.n.sum() == len(g)
            bins.append(points.assign(first_careunit=unit, arm=arm))
            ax.plot(points.mean_prediction, points.observed_rate, 'o-', markersize=3.8,
                    linewidth=1.3, label=label)
        if ax is not None:
            ax.set_title(f'{unit}\nn={len(g):,}', fontsize=8)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.grid(alpha=.18)
    for ax in list(axes.flat)[len(plotted_groups):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7, loc='upper left')
    fig.supxlabel('Mean predicted 28-day risk')
    fig.supylabel('Observed 28-day mortality')
    fig.suptitle('Calibration by first ICU type · development cross-validation', fontsize=14)
    fig.savefig(OUT / 'icu_calibration.pdf')
    plt.close(fig)
    pd.concat(bins, ignore_index=True).to_csv(OUT / 'icu_calibration_bins.csv', index=False)
    return pd.DataFrame(rows)


def main():
    prior = json.loads((PRIOR / 'experiment.json').read_text(encoding='utf-8'))
    params = json.loads((TUNING / 'final_parameters.json').read_text(encoding='utf-8'))['xgboost']['params']
    columns = prior['input_features']
    frames, targets, identifiers = [], [], []
    for split in ('train', 'validation'):
        folder = SOURCE / split
        for name in ('X_features.csv', 'y_target.csv', 'row_ids.csv'):
            digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()
            assert digest == prior['input_sha256'][f'{split}/{name}']
        x = pd.read_csv(folder / 'X_features.csv', usecols=columns)[columns].copy()
        assert x.gender.isin(['F', 'M']).all()
        x['gender'] = x.gender.map({'F': 0, 'M': 1})
        frames.append(x)
        targets.append(pd.read_csv(folder / 'y_target.csv').mortality_28d)
        identifiers.append(pd.read_csv(folder / 'row_ids.csv'))
    x = pd.concat(frames, ignore_index=True)
    y = pd.concat(targets, ignore_index=True)
    ids = pd.concat(identifiers, ignore_index=True)
    saved = pd.read_csv(ABLATION / 'oof_predictions.csv')
    pd.testing.assert_frame_equal(saved[IDS], ids)
    pd.testing.assert_series_equal(saved.mortality_28d, y)
    assert len(ids) == 25595 and ids.subject_id.is_unique
    icu = pd.read_csv(ROOT / 'data/raw/icustays.csv', usecols=[*IDS, 'first_careunit'])
    unit = ids.merge(icu, on=IDS, how='left', validate='one_to_one').first_careunit
    assert unit.notna().all()
    without = [col for col in columns if not col.endswith('_missing')]
    assert len(columns) - len(without) == 19
    assert sorted(c for c in columns if c.endswith('_count')) == sorted(c for c in without if c.endswith('_count'))
    OUT.mkdir(exist_ok=True)
    rows = []
    oof = ids.assign(mortality_28d=y, fold=saved.fold, first_careunit=unit,
                     baseline=saved.native_indicator, without_indicators=np.nan, with_icu=np.nan)
    for fold in range(1, 6):
        tr, va = saved.fold.ne(fold), saved.fold.eq(fold)
        a, b, prep = prepare(x.loc[tr, columns], x.loc[va, columns], columns, 'native')
        assert len(prep['std_columns']) == 19
        known = sorted(unit[tr].unique())
        a_icu = unit[tr].to_numpy()[:, None] == np.asarray(known)
        b_icu = unit[va].to_numpy()[:, None] == np.asarray(known)
        for arm, ta, tb, names in [
            ('without_indicators', x.loc[tr, without].to_numpy(), x.loc[va, without].to_numpy(), without),
            ('with_icu', np.column_stack([a, a_icu]), np.column_stack([b, b_icu]), columns),
        ]:
            print(f'fold={fold}/5 {arm}', flush=True)
            p, _ = fit('xgboost', params, ta, y[tr], tb, names)
            oof.loc[va, arm] = p
        for arm in ARMS:
            rows.append(dict(fold=fold, arm=arm, **metrics(y[va], oof.loc[va, arm])))
    assert oof[list(ARMS)].notna().all().all()
    oof.to_csv(OUT / 'oof_predictions.csv', index=False)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / 'fold_metrics.csv', index=False)
    summarize(oof, folds)


def summarize(oof, folds):
    wide = folds.set_index(['arm','fold'])[KEY_METRICS]
    differences = pd.concat({arm: wide.loc[arm] - wide.loc['baseline'] for arm in list(ARMS)[1:]},
                            names=['arm'])
    differences.to_csv(OUT / 'paired_fold_differences.csv')
    grouped = folds.groupby('arm')[KEY_METRICS].agg(['mean','std'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped.to_csv(OUT / 'cv_summary.csv')
    subgroups = evaluate_subgroups(oof)
    subgroups.to_csv(OUT / 'icu_subgroup_metrics.csv', index=False)

    report = [
        '# 결측 지시변수와 첫 ICU 유형 실험', '',
        '개발 환자 25,595명의 기존 5-fold와 튜닝 설정을 사용했다. 기준 모델은 대치 없음과 결측 지시변수를 사용한다.',
        '지시변수 제거 조건은 원래 결측 지시변수 19개와 표준편차 결측 지시변수 19개를 제거했다. NaN과 측정 횟수는 유지했다.',
        'ICU 추가 조건은 기준 입력에 입실 시점 첫 ICU 유형을 원-핫 인코딩해 추가했다. 범주는 각 훈련 폴드에서 정했다.',
        '추가 테스트 평가는 수행하지 않았다.', '',
        '| 조건 | AP | ROC-AUC | Brier |',
        '|---|---:|---:|---:|',
    ]
    for arm, label in ARMS.items():
        v = grouped.loc[arm]
        values = [f'{v[k+"_mean"]:.4f} ± {v[k+"_std"]:.4f}' for k in KEY_METRICS]
        report.append('| ' + ' | '.join([label, *values]) + ' |')
    report.extend(['', '## 기준 대비 차이', '',
                   '| 조건 | Δ AP | Δ ROC-AUC | Δ Brier | AP가 오른 폴드 |',
                   '|---|---:|---:|---:|---:|'])
    for arm in list(ARMS)[1:]:
        diff = differences.loc[arm]
        values = [f'{diff[k].mean():+.4f}' for k in KEY_METRICS]
        report.append('| ' + ' | '.join([ARMS[arm], *values, f'{diff.average_precision.gt(0).sum()}/5']) + ' |')
    report.extend(['', '## ICU별 검증 성능과 확률 보정', '',
                   '환자별 교차검증 예측값을 모아 평가했다. 그래프에는 100명 이상인 유형을 표시했고, 모델별 예측 위험을 다섯 분위 구간으로 나눴다. 한 종류의 결과만 있는 유형의 AP와 ROC-AUC는 표시하지 않는다.', '',
                   '| ICU | 환자 수 | 실제 사망률 | 기준 평균 예측 | ICU 추가 평균 예측 | 기준 AP | ICU 추가 AP | 기준 Brier | ICU 추가 Brier |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for unit, group in subgroups.groupby('first_careunit'):
        v = group.set_index('arm')
        a, b = v.loc['baseline'], v.loc['with_icu']
        a_ap, b_ap = [f'{v:.4f}' if pd.notna(v) else '—'
                      for v in (a.average_precision, b.average_precision)]
        report.append(f'| {unit} | {a.n:,} | {a.observed_rate:.1%} | {a.mean_prediction:.1%} | {b.mean_prediction:.1%} | '
                      f'{a_ap} | {b_ap} | {a.brier_score:.4f} | {b.brier_score:.4f} |')
    (OUT / 'training_report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(grouped.to_string())
    print(differences.groupby('arm').mean().to_string())


if __name__ == '__main__':
    main()
