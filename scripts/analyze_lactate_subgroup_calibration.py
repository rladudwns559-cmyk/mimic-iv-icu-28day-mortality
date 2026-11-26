"""Summarize held-out development predictions by first-day lactate measurement."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/error_cleaned/splits'
OUT = ROOT / 'results/xgboost_missingness_ablation'
GROUPS = {0: 'Measured', 1: 'Unmeasured'}
MODELS = {
    'native_indicator': 'Full lactate',
    'native_lactate_measurement_only': 'Measurement information only',
    'native_without_lactate': 'Lactate excluded',
}
IDS = ['subject_id', 'hadm_id', 'stay_id']


def main():
    inputs = []
    for split in ('train', 'validation'):
        folder = SOURCE / split
        ids = pd.read_csv(folder / 'row_ids.csv')
        feature = pd.read_csv(folder / 'X_features.csv',
                              usecols=['lab_Lactate_count', 'lab_Lactate_missing'])
        assert len(ids) == len(feature)
        measured = feature.lab_Lactate_count.gt(0)
        assert feature.lab_Lactate_missing.eq(~measured).all()
        inputs.append(ids.assign(lactate_missing=feature.lab_Lactate_missing))

    predictions = pd.read_csv(OUT / 'oof_predictions.csv')
    data = predictions.merge(pd.concat(inputs), on=IDS, validate='one_to_one')
    assert len(data) == len(predictions) == 25595
    edges = np.linspace(0, 1, 11)
    overview, bins = [], []
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=True,
                             layout='constrained')
    for ax, (flag, name) in zip(axes, GROUPS.items()):
        group = data.loc[data.lactate_missing.eq(flag)].copy()
        y = group.mortality_28d
        prevalence = y.mean()
        ax.plot([0, 1], [0, 1], '--', color='0.55', linewidth=1, label='Perfect calibration')
        for arm, label in MODELS.items():
            p = group[arm]
            assert p.between(0, 1).all()
            overview.append(dict(group=name, arm=arm, n=len(y), deaths=int(y.sum()),
                                 prevalence=prevalence, mean_prediction=p.mean(),
                                 average_precision=average_precision_score(y, p),
                                 roc_auc=roc_auc_score(y, p), brier_score=brier_score_loss(y, p)))
            binned = (
                pd.DataFrame({'bin': pd.cut(p, edges, include_lowest=True),
                              'probability': p, 'death': y})
                .groupby('bin', observed=True)
                .agg(n=('death', 'size'), deaths=('death', 'sum'),
                     mean_prediction=('probability', 'mean'), observed_rate=('death', 'mean'))
                .reset_index()
            )
            assert binned.n.sum() == len(y)
            bins.append(binned.assign(group=name, arm=arm))
            ax.plot(binned.mean_prediction, binned.observed_rate, marker='o',
                    markersize=4, linewidth=1.4, label=label)
        ax.set(title=f'{name} lactate (n={len(y):,})', xlabel='Mean predicted risk',
               xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Observed 28-day mortality')
    axes[1].legend(loc='upper left', fontsize=7)
    fig.suptitle('Calibration by lactate measurement in development cross-validation')
    fig.savefig(OUT / 'lactate_subgroup_calibration.pdf', bbox_inches='tight')
    plt.close(fig)
    summary = pd.DataFrame(overview)
    summary.to_csv(OUT / 'lactate_subgroup_metrics.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(OUT / 'lactate_calibration_bins.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
