"""Compare 28-day mortality by availability of four first-24h lab tests."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [('BilirubinTotal', '총빌리루빈'), ('ALT', 'ALT'),
            ('AST', 'AST'), ('Lactate', '젖산')]


def wilson(deaths, n):
    if n == 0:
        return np.nan, np.nan
    p, z = deaths / n, norm.ppf(.975)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return center-half, center+half


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/error_cleaned/modeling_dataset.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/eda_lab_measurement_mortality')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if not data.subject_id.is_unique or not data.stay_id.is_unique:
        raise ValueError('Expected one first ICU stay per patient')
    if data.empty or not data.mortality_28d.isin([0, 1]).all():
        raise ValueError('Expected nonempty cohort with confirmed binary outcomes')
    groups, comparisons = [], []
    for feature, label in FEATURES:
        prefix = f'lab_{feature}'
        count = data[f'{prefix}_count']
        observed = count.gt(0)
        if (count.isna().any() or count.lt(0).any() or count.mod(1).ne(0).any()
                or not observed.equals(data[f'{prefix}_mean'].notna())
                or not data[f'{prefix}_missing'].eq((~observed).astype(int)).all()):
            raise ValueError(f'{feature}: inconsistent measurement indicators')
        rows = []
        for measured, name in [(True, '측정'), (False, '미측정')]:
            y = data.loc[observed.eq(measured), 'mortality_28d']
            n, deaths = len(y), int(y.sum())
            if not n:
                raise ValueError(f'{feature}: empty comparison group')
            low, high = wilson(deaths, n)
            row = dict(feature=feature, label=label, group=name, n=n, deaths=deaths,
                       survivors=n-deaths, cohort_pct=100*n/len(data),
                       mortality_pct=100*deaths/n, ci95_low_pct=100*low, ci95_high_pct=100*high)
            rows.append(row)
            groups.append(row)
        a, b = rows
        diff = a['mortality_pct']-b['mortality_pct']
        # Newcombe hybrid score interval for two independent proportions.
        low = diff - np.hypot(a['mortality_pct']-a['ci95_low_pct'], b['ci95_high_pct']-b['mortality_pct'])
        high = diff + np.hypot(a['ci95_high_pct']-a['mortality_pct'], b['mortality_pct']-b['ci95_low_pct'])
        _, p = fisher_exact([[a['deaths'], a['survivors']], [b['deaths'], b['survivors']]])
        comparisons.append(dict(feature=feature, label=label, missing_pct=b['cohort_pct'],
                                difference_pp=diff, difference_ci95_low_pp=low,
                                difference_ci95_high_pp=high, p_value=p))
    summary, comparison = pd.DataFrame(groups), pd.DataFrame(comparisons)
    order = np.argsort(comparison.p_value.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(comparison.p_value.to_numpy()[order]*np.arange(4, 0, -1)))
    comparison['p_holm'] = np.nan
    comparison.loc[order, 'p_holm'] = adjusted
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out/'mortality_by_measurement.csv', index=False, encoding='utf-8-sig')
    comparison.to_csv(out/'mortality_difference.csv', index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.4, 1]})
    colors = ['#187f97', '#b87545']
    for i, (_, label) in enumerate(FEATURES):
        s = summary[summary.label.eq(label)]
        for j, r in enumerate(s.itertuples()):
            y = i + (j-.5)*.28
            axes[0].errorbar(r.mortality_pct, y,
                            xerr=[[r.mortality_pct-r.ci95_low_pct], [r.ci95_high_pct-r.mortality_pct]],
                            fmt='o', color=colors[j], capsize=4, label=r.group if i == 0 else None)
            axes[0].annotate(f'{r.mortality_pct:.2f}%  ({r.deaths:,}/{r.n:,}명)',
                             (r.mortality_pct, y), xytext=(9, -4), textcoords='offset points', fontsize=10)
    axes[0].set_xlim(0, max(summary.ci95_high_pct)+19)
    axes[0].set_title('측정 여부별 28일 사망률 · 95% 신뢰구간')
    axes[0].set_xlabel('28일 사망률 (%) · 괄호: 사망자/전체 환자')
    axes[0].legend(loc='lower right')
    c = comparison
    axes[1].errorbar(c.difference_pp, np.arange(4),
                     xerr=np.vstack([c.difference_pp-c.difference_ci95_low_pp,
                                     c.difference_ci95_high_pp-c.difference_pp]),
                     fmt='o', color='#6855a0', capsize=5)
    axes[1].axvline(0, color='#888888', linestyle='--', linewidth=1)
    for i, r in enumerate(c.itertuples()):
        axes[1].annotate(f'{r.difference_pp:+.2f}%p [{r.difference_ci95_low_pp:.2f}, {r.difference_ci95_high_pp:.2f}]',
                         (r.difference_pp, i), xytext=(0, -23), textcoords='offset points', ha='center', fontsize=10)
    axes[1].set_title('사망률 차이: 측정군 - 미측정군')
    axes[1].set_xlabel('사망률 차이 (%p) · Newcombe 95% 신뢰구간')
    axes[1].margins(x=.28)
    for ax in axes:
        ax.set_yticks(np.arange(4), [label for _, label in FEATURES])
        ax.set_ylim(3.6, -.6)
        ax.grid(axis='x', alpha=.18)
    fig.suptitle('결측률이 높은 검사 4개의 측정 여부와 28일 사망률', fontsize=20, weight='bold')
    fig.text(.5, .915, f'첫 ICU 입실 · 첫 24시간 유효 검사값 기준 · {len(data):,}명 · 전체 사망률 {data.mortality_28d.mean():.2%}', ha='center')
    fig.text(.04, .025, '미측정: 첫 24시간 내 유효값 없음. 비보정 관찰 비교이며 중증도·검사 선택에 따른 차이가 포함될 수 있습니다.\n각 변수는 같은 코호트를 독립적으로 분류합니다. 신뢰구간은 다중비교 보정 전입니다.', fontsize=10, color='#495666')
    fig.tight_layout(rect=(0, .09, 1, .88), w_pad=3)
    for extension in ['png', 'pdf']:
        fig.savefig(out/f'mortality_comparison.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    lines = ['# 검사 측정 여부별 28일 사망률 EDA', '',
             f'입력: `{args.input.resolve()}`. 분석 대상 {len(data):,}명, 사망 {int(data.mortality_28d.sum()):,}명, 전체 사망률 {data.mortality_28d.mean():.2%}.', '',
             '| 변수 | 미측정률 | 측정군 사망/전체 (사망률) | 미측정군 사망/전체 (사망률) | 차이 (%p), 95% CI | Holm 보정 p |',
             '|---|---:|---:|---:|---:|---:|']
    for r in comparison.itertuples():
        a, b = summary[summary.feature.eq(r.feature)].to_dict('records')
        lines.append(f"| {r.label} | {r.missing_pct:.2f}% | {a['deaths']:,}/{a['n']:,} ({a['mortality_pct']:.2f}%) | {b['deaths']:,}/{b['n']:,} ({b['mortality_pct']:.2f}%) | {r.difference_pp:+.2f} [{r.difference_ci95_low_pp:.2f}, {r.difference_ci95_high_pp:.2f}] | {r.p_holm:.3g} |")
    lines += ['', '![측정 여부별 사망률 비교](mortality_comparison.png)', '',
              '## 분석 정의 및 해석', '',
              '- 환자당 첫 ICU 입실 1건, 기존 코호트의 mortality_28d 라벨을 그대로 사용했습니다. 24시간 관찰 조건을 충족하고 사망 라벨이 확정된 환자만 포함합니다.',
              '- 측정군: ICU 입실 후 [0, 24시간)에 유효 검사값이 1개 이상 있는 환자(count > 0). 미측정군: 같은 시간창 내 유효 검사값이 없는 환자. 원시 검사 자체가 전혀 시행되지 않았다는 의미는 아니며, 정제에서 제외된 값이나 시간창 밖 검사만 있을 수 있습니다.',
              '- count, mean의 결측 여부, missing 지표의 일치 여부와 환자 중복 및 사망 라벨을 검증했습니다. 결측값 대체나 추가 환자 제외는 하지 않았습니다.',
              '- 군별 신뢰구간은 Wilson 95% CI, 차이는 측정군−미측정군이며 Newcombe 95% CI를 제시했습니다. 양측 Fisher 정확검정을 시행하고 4개 검정에 Holm 보정을 적용했습니다. 신뢰구간 자체는 다중비교 보정 전입니다.',
              '- 각 변수에서 동일한 환자들을 측정/미측정으로 다시 나누므로 네 비교는 서로 독립적인 환자 표본이 아닙니다.',
              '- 이 비교는 연령·중증도·진단·ICU 유형 등을 보정하지 않은 연관성 분석입니다. 검사 선택이나 중증도 차이가 결과에 반영될 수 있어 검사의 인과효과로 해석할 수 없습니다. 관찰된 사망률 차이만으로 결측 발생 기전을 확정할 수 없습니다.',
              '- 사망 라벨 미확정 환자의 제외 및 24시간 관찰 조건에 따른 선택 편향이 있어 전체 ICU 환자로 일반화하는 데 제한이 있습니다. 기존 charttime 기준을 따르므로 예측 시점 이후 저장된 검사값이 포함될 수 있습니다.', '',
              '## 산출물', '',
              '- mortality_comparison.png/pdf: 발표용 비교 그림.',
              '- mortality_by_measurement.csv: 군별 환자 수·사망자 수·사망률·Wilson 신뢰구간.',
              '- mortality_difference.csv: 결측률·사망률 차이·Newcombe 신뢰구간·원 p값 및 Holm 보정 p값.',
              '- report.html: 그림과 결과표를 포함한 보고서.', '',
              '재실행: `python scripts/eda_lab_measurement_mortality.py`']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    notes = ''.join(f'<p>{line[2:]}</p>' for line in lines if line.startswith('- '))
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>검사 측정 여부별 사망률</title>'
            '<style>body{font-family:Malgun Gothic,sans-serif;max-width:1400px;margin:40px auto;padding:20px;color:#24354a}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd}p{line-height:1.7}</style>'
            '<h1>검사 측정 여부별 28일 사망률</h1>'
            f'<p>분석 대상 {len(data):,}명 · 사망 {int(data.mortality_28d.sum()):,}명 · 전체 사망률 {data.mortality_28d.mean():.2%}</p>'
            '<img src="mortality_comparison.png" alt="측정군과 미측정군의 사망률과 차이">'
            '<h2>군별 결과</h2>'+summary.to_html(index=False, float_format=lambda x: f'{x:.3f}')+
            '<h2>사망률 차이 및 검정</h2>'+comparison.to_html(index=False, float_format=lambda x: f'{x:.5g}')+
            '<h2>분석 정의 및 해석</h2>'+notes+'</html>')
    (out/'report.html').write_text(html, encoding='utf-8')
    print(summary.drop(columns=['label', 'group']).to_string(index=False))
    print(comparison.drop(columns='label').to_string(index=False))
    print('Output:', out)


if __name__ == '__main__':
    main()

