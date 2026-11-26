"""Compare patient-level continuous measurements by known 28-day outcome."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/survival_comparison'


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/processed/error_cleaned/modeling_dataset.csv')
    assert data.subject_id.is_unique and data.mortality_28d.isin([0, 1]).all()
    config = json.loads((ROOT / 'scripts/feature_config.json').read_text(encoding='utf-8'))
    units = {'age': '세'}
    for kind in ['vital', 'lab']:
        units.update({f'{kind}_{r[0]}_mean': r[3] for r in config[kind]})
    columns = ['age']+[c for c in data if c.endswith('_mean')]
    rows = []
    for col in columns:
        groups = [data.loc[data.mortality_28d.eq(k), col].dropna() for k in [0, 1]]
        a, b = groups
        assert min(len(a), len(b)) > 1
        result = ttest_ind(b, a, equal_var=False)
        va, vb = a.var()/len(a), b.var()/len(b)
        df = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
        margin = t.ppf(.975, df)*np.sqrt(va+vb)
        diff = b.mean()-a.mean()
        row = dict(feature=col, unit=units[col], mean_difference_death_minus_survival=diff,
                   difference_ci95_low=diff-margin, difference_ci95_high=diff+margin,
                   welch_t=result.statistic, welch_df=df, p_welch=result.pvalue,
                   p_mann_whitney=mannwhitneyu(b, a, alternative='two-sided', method='asymptotic').pvalue)
        for k, g in enumerate(groups):
            prefix = ['survival', 'death'][k]
            row.update({f'{prefix}_n': len(g), f'{prefix}_missing': int(data.mortality_28d.eq(k).sum())-len(g),
                        f'{prefix}_mean': g.mean(), f'{prefix}_sd': g.std(),
                        f'{prefix}_median': g.median(), f'{prefix}_q25': g.quantile(.25), f'{prefix}_q75': g.quantile(.75)})
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats['q_welch_bh'] = bh(stats.p_welch)
    stats['q_mann_whitney_bh'] = bh(stats.p_mann_whitney)
    stats.to_csv(OUT / 'continuous_group_tests.csv', index=False, encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    plots = {'vital': ['HR', 'RR', 'MAP', 'SpO2'], 'lab': ['Lactate', 'Creatinine', 'BilirubinTotal', 'Platelet']}
    for kind, names in plots.items():
        fig, axes = plt.subplots(2, 2, figsize=(17, 13))
        for ax, name in zip(axes.flat, names):
            col = f'{kind}_{name}_mean'
            r = stats.set_index('feature').loc[col]
            groups = [data.loc[data.mortality_28d.eq(k), col].dropna() for k in [0, 1]]
            boxes = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=.45,
                               medianprops={'color': '#24354a', 'linewidth': 2})
            for box, color in zip(boxes['boxes'], ['#147D92', '#C56850']):
                box.set_facecolor(color)
                box.set_alpha(.65)
            ax.set_xticks([1, 2], [f'생존군\nn={len(groups[0]):,}', f'사망군\nn={len(groups[1]):,}'])
            ax.set_ylabel(f'{name} ({units[col]})')
            ax.set_title(
                f'{name} · 양측 Welch t-검정\n'
                f't = {r.welch_t:.3f}, 자유도(df) = {r.welch_df:.1f}\n'
                f'p-value = {r.p_welch:.3e}  |  보정 q-value = {r.q_welch_bh:.3e}',
                fontsize=12, pad=16, linespacing=1.5)
            ax.text(.98, .97,
                    f'생존군 평균 ± SD: {r.survival_mean:.2f} ± {r.survival_sd:.2f}\n'
                    f'사망군 평균 ± SD: {r.death_mean:.2f} ± {r.death_sd:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': .9, 'pad': 6})
            ax.grid(axis='y', alpha=.15)
        fig.suptitle(f'{kind.upper()} 주요 변수: 생존군과 사망군 비교', fontsize=21, weight='bold', y=.98)
        fig.text(.5, .93,
                 '귀무가설 H0: 생존군과 사망군의 평균이 같다.\n'
                 '대립가설 H1: 생존군과 사망군의 평균이 다르다. (양측검정)',
                 ha='center', va='top', fontsize=13, linespacing=1.6)
        fig.text(.04, .025, '첫 24시간 환자별 평균값 · t값 방향: 사망군 - 생존군 · 유의수준 α=0.05 · BH 보정: 연령·Vital·Lab 전체 23개 검정\n표시된 모든 변수는 보정 q-value<0.05로 귀무가설 기각 (평균 차이 있음). 인과관계를 뜻하지 않음.\n상자: 중앙값·IQR, 수염: 1.5 IQR 내 관측값. 이상치는 그림에서만 생략하며 검정에 포함. 변수별 결측 제외·대체 없음.', fontsize=10, linespacing=1.6)
        fig.subplots_adjust(left=.09, right=.97, top=.76, bottom=.16, hspace=.95, wspace=.3)
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{kind}_survival_tests.{ext}', dpi=180, facecolor='white')
        plt.close(fig)
    table = pd.DataFrame({'변수': stats.feature, '단위': stats.unit})
    for prefix, label in [('survival', '생존군'), ('death', '사망군')]:
        table[f'{label} n (결측)'] = stats.apply(lambda r: f'{r[prefix+"_n"]:,} ({r[prefix+"_missing"]:,})', axis=1)
        table[f'{label} 평균 ± SD'] = stats.apply(lambda r: f'{r[prefix+"_mean"]:.2f} ± {r[prefix+"_sd"]:.2f}', axis=1)
        table[f'{label} 중앙값 [IQR]'] = stats.apply(lambda r: f'{r[prefix+"_median"]:.2f} [{r[prefix+"_q25"]:.2f}, {r[prefix+"_q75"]:.2f}]', axis=1)
    table['평균차: 사망−생존 [95% CI]'] = stats.apply(lambda r: f'{r.mean_difference_death_minus_survival:.2f} [{r.difference_ci95_low:.2f}, {r.difference_ci95_high:.2f}]', axis=1)
    for col, label in [('p_welch', 'Welch p'), ('q_welch_bh', 'BH q'), ('p_mann_whitney', 'Mann–Whitney p')]:
        table[label] = stats[col].map(lambda p: f'{p:.3e}')
    table.to_csv(OUT / 'comparison_table.csv', index=False, encoding='utf-8-sig')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>생존군·사망군 가설검정</title>
<style>body{font-family:Malgun Gothic,sans-serif;margin:35px;color:#24354a}p{line-height:1.8}table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#edf3f8}img{max-width:1400px;width:100%}</style>
<h1>28일 생존군·사망군 비교 및 가설검정</h1>'''
    html += f'<p>전체 {len(data):,}명: 생존군 {(data.mortality_28d==0).sum():,}명, 사망군 {(data.mortality_28d==1).sum():,}명. 정제된 모델링 데이터의 환자별 첫 24시간 평균값과 연령을 분석했습니다.</p>'
    html += '<p>주 분석은 등분산을 가정하지 않는 양측 Welch t-검정입니다(귀무가설: 두 군의 평균이 같음). 연령·Vital·Lab 전체 23개 변수에 BH 다중검정 보정을 적용했습니다. 왜도·극단값 영향을 살펴보기 위해 양측 Mann–Whitney U 검정도 함께 제시합니다. 이 검정은 분포·순위에 관한 검정이며 단순한 평균 또는 중앙값 차이 검정과 동일하지 않습니다. 결측은 변수별 제외하고, 유효한 극단값은 검정에 포함했습니다.</p>'
    html += table.to_html(index=False, border=0)
    html += '<h2>주요 변수 시각화</h2><img src="vital_survival_tests.png"><img src="lab_survival_tests.png">'
    html += '<p>범주형 성별·연령대·ICU 유형의 카이제곱검정은 <a href="../group_mortality/report.html">집단별 사망률 보고서</a>에서 확인할 수 있습니다. 이 분석은 보정 전 탐색적 비교입니다. 통계적 유의성이 임상적 중요성이나 인과효과를 뜻하지 않습니다. 24시간 관찰 조건을 충족한 라벨 확정 환자만 포함하며, 미확정 21,529명 제외에 따른 선택 편향 가능성이 있습니다.</p>'
    (OUT / 'report.html').write_text(html+'</html>', encoding='utf-8')
    print(f'Completed {len(stats)} continuous-variable comparisons; plots and report saved.')
    print(stats[['feature', 'p_welch', 'q_welch_bh']].to_string(index=False))


if __name__ == '__main__':
    main()
