"""Describe observed first-24h means of four high-missingness lab variables."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/presentation_eda/high_missing_labs'
FEATURES = [
    ('BilirubinTotal', '총빌리루빈', 'mg/dL'),
    ('ALT', 'ALT', 'IU/L'),
    ('AST', 'AST', 'IU/L'),
    ('Lactate', '젖산', 'mmol/L'),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/processed/error_cleaned/modeling_dataset.csv')
    assert data.subject_id.is_unique
    summaries = []
    values = {}
    for feature, label, unit in FEATURES:
        x = data[f'lab_{feature}_mean'].dropna()
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f'{feature}: nonfinite or negative observations')
        values[feature] = x
        row = dict(feature=feature, label=label, unit=unit, total=len(data),
                   observed=len(x), missing=len(data)-len(x),
                   missing_pct=100*(1-len(x)/len(data)), mean=x.mean(), std=x.std(),
                   min=x.min(), max=x.max(), skew=x.skew(), zeros=int(x.eq(0).sum()))
        row.update({name: x.quantile(q) for name, q in
                    [('p01', .01), ('q25', .25), ('median', .5), ('q75', .75),
                     ('p95', .95), ('p99', .99)]})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'distribution_summary.csv', index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for mode in ('full', 'central99'):
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        subtitle = '전체 관측값 · 가로축 log(1+x)' if mode == 'full' else '원래 단위 · 변수별 99백분위 이하 표시'
        fig.suptitle(f'결측률이 높은 검사 4개 | 첫 24시간 환자별 평균\n{subtitle}', fontsize=19)
        for ax, (feature, label, unit), row in zip(axes.flat, FEATURES, summaries):
            x = values[feature]
            if mode == 'full':
                plot_x = np.log1p(x)
                ax.hist(plot_x, bins=55, color='#5081a8', alpha=.85)
                # Tick labels show raw units; transformation is explicit in axis label.
                raw_ticks = np.unique(np.r_[0, np.geomspace(max(x.max()/10000, .1), x.max(), 6)])
                ax.set_xticks(np.log1p(raw_ticks), [f'{v:,.2g}' for v in raw_ticks])
                ax.axvline(np.log1p(row['median']), color='#b95245', ls='--', label='중앙값')
                ax.set_xlabel(f'{label} ({unit}; log(1+x) 간격)')
            else:
                shown = x[x.le(row['p99'])]
                ax.hist(shown, bins=50, color='#5081a8', alpha=.85)
                ax.axvline(row['median'], color='#b95245', ls='--', label='중앙값')
                ax.set_xlabel(f"{label} ({unit}) | 표시 생략 {len(x)-len(shown):,}명 (>P99 {row['p99']:.2f})")
            ax.set_title(f"{label} | 관측 n={len(x):,}, 결측 {row['missing_pct']:.2f}%\n"
                         f"중앙값 {row['median']:.2f} (IQR {row['q25']:.2f}–{row['q75']:.2f})", fontsize=12)
            ax.set_ylabel('환자 수')
            ax.grid(axis='y', alpha=.15)
            ax.legend(fontsize=9)
        fig.text(.04, .02, '정제 후 코호트 30,112명. 결측 대체 및 추가 이상치 제거 없음. '
                 '99백분위 제한은 그림에만 적용하며 요약통계는 전체 관측값 기준.', fontsize=11)
        fig.tight_layout(rect=(.01, .06, .99, .91), h_pad=2)
        fig.savefig(OUT / f'distribution_{mode}.png', dpi=180, facecolor='white')
        fig.savefig(OUT / f'distribution_{mode}.pdf', facecolor='white')
        plt.close(fig)
    lines = ['# 총빌리루빈·ALT·AST·젖산 값의 분포', '',
             '입력: `data/processed/error_cleaned/modeling_dataset.csv`. '
             '환자별 첫 24시간 평균값을 분석했으며, 원본 이벤트의 개별 측정값 분포와는 다릅니다.', '',
             '| 변수 | 단위 | 관측 n | 결측률 | 중앙값 [Q1–Q3] | 평균 | P99 | 최댓값 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['label']} | {r['unit']} | {r['observed']:,} | {r['missing_pct']:.2f}% | "
                     f"{r['median']:.2f} [{r['q25']:.2f}–{r['q75']:.2f}] | {r['mean']:.2f} | "
                     f"{r['p99']:.2f} | {r['max']:.2f} |")
    lines.extend(['', '## 그림 읽는 법', '',
                  '- `distribution_central99.png/pdf`: 원래 단위에서 주요 분포를 확인하는 발표용 그림. 변수별 P99 초과 값의 표시만 생략하고 생략 인원을 적었습니다.',
                  '- `distribution_full.png/pdf`: 극단값을 포함한 전체 분포. 가로축은 log(1+x)이며 눈금은 원래 단위입니다. 동일한 가로 간격이 원래 단위의 동일한 차이를 뜻하지 않습니다.',
                  '- 요약통계는 결측을 제외한 전체 관측값으로 계산했습니다. 추가 이상치 제거와 결측 대체는 하지 않았습니다.',
                  '- 분포는 검사가 관측된 환자에 한정됩니다. 결측 환자도 같은 분포를 가진다고 가정할 수 없습니다.',
                  '', '재실행: `python scripts/plot_high_missing_labs.py`'])
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary.drop(columns=['label']).to_string(index=False))
    print(f'Output: {OUT}')


if __name__ == '__main__':
    main()
