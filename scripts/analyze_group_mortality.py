"""Reproducible descriptive 28-day mortality analysis of the labeled cohort."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/group_mortality'


def save_csv(frame, filename):
    path = OUT / filename
    content = frame.to_csv(index=False, lineterminator='\n')
    if path.exists() and path.read_text(encoding='utf-8-sig') == content:
        return
    path.write_text(content, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/processed/cohort_base.csv')
    icu = pd.read_csv(ROOT / 'data/raw/icustays.csv', usecols=['stay_id', 'first_careunit'])
    assert data.subject_id.is_unique and data.stay_id.is_unique
    assert data.mortality_28d.isin([0, 1]).all()
    data = data.merge(icu, on='stay_id', how='left', validate='one_to_one')
    assert data.first_careunit.notna().all()
    data['성별'] = data.gender.map({'F': '여성', 'M': '남성'})
    data['연령대'] = pd.cut(data.age, [18, 40, 50, 60, 70, 80, np.inf], right=False,
                          labels=['18–39세', '40–49세', '50–59세', '60–69세', '70–79세', '80세 이상'])
    names = {
        'Medical Intensive Care Unit (MICU)': 'MICU',
        'Surgical Intensive Care Unit (SICU)': 'SICU',
        'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'MICU/SICU',
        'Cardiac Vascular Intensive Care Unit (CVICU)': 'CVICU',
        'Coronary Care Unit (CCU)': 'CCU',
        'Trauma SICU (TSICU)': 'TSICU',
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'Neuro SICU',
    }
    data['ICU 유형'] = data.first_careunit.replace(names)
    raw_units = data.groupby('ICU 유형').mortality_28d.agg(n='size', deaths='sum').reset_index()
    raw_units['mortality_pct'] = 100*raw_units.deaths/raw_units.n
    save_csv(raw_units, 'original_careunit_counts.csv')
    rare = raw_units.loc[raw_units.n.lt(100), 'ICU 유형']
    data.loc[data['ICU 유형'].isin(rare), 'ICU 유형'] = '기타 (각 병동 n<100)'
    assert data[['성별', '연령대', 'ICU 유형']].notna().all().all()
    summaries, tests, expected_rows = [], [], []
    z = norm.ppf(.975)
    for variable in ['성별', '연령대', 'ICU 유형']:
        table = pd.crosstab(data[variable], data.mortality_28d).reindex(columns=[0, 1], fill_value=0)
        chi2, p, dof, expected = chi2_contingency(table, correction=False)
        tests.append(dict(variable=variable, n=int(table.values.sum()), chi2=chi2, df=dof,
                          p_value=p, cramers_v=np.sqrt(chi2/table.values.sum()),
                          minimum_expected_count=expected.min(), cells_expected_lt5=int((expected < 5).sum())))
        for group, row in table.iterrows():
            n, deaths = int(row.sum()), int(row[1])
            rate = deaths/n
            center = (rate + z*z/(2*n))/(1+z*z/n)
            half = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
            summaries.append(dict(variable=variable, group=str(group), n=n, deaths=deaths,
                                  survivors=int(row[0]), mortality_pct=100*rate,
                                  ci95_low_pct=100*(center-half), ci95_high_pct=100*(center+half)))
        for i, group in enumerate(table.index):
            expected_rows.append(dict(variable=variable, group=str(group), expected_survivors=expected[i, 0], expected_deaths=expected[i, 1]))
    summary, test = pd.DataFrame(summaries), pd.DataFrame(tests)
    # Holm correction across the three omnibus tests.
    order = np.argsort(test.p_value.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(test.p_value.to_numpy()[order] * np.arange(3, 0, -1)))
    test['p_holm'] = np.nan
    test.loc[order, 'p_holm'] = adjusted
    for frame, filename in [(summary, 'mortality_summary.csv'), (test, 'chi_square_results.csv'),
                            (pd.DataFrame(expected_rows), 'expected_counts.csv')]:
        save_csv(frame, filename)
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.25, 1.65]})
    colors = ['#147D92', '#5968AD', '#B15E36']
    axis_settings = {'성별': (20, 25, 1), '연령대': (10, 37, 5), 'ICU 유형': (0, 62, 10)}
    for ax, variable, color in zip(axes, ['성별', '연령대', 'ICU 유형'], colors):
        s = summary[summary.variable.eq(variable)].reset_index(drop=True)
        y = np.arange(len(s))
        ax.errorbar(s.mortality_pct, y,
                    xerr=np.vstack([s.mortality_pct-s.ci95_low_pct, s.ci95_high_pct-s.mortality_pct]),
                    fmt='o', color=color, capsize=5, markersize=8, linewidth=2)
        ax.set_yticks(y, [f'{r.group}\n(n={r.n:,})' for r in s.itertuples()])
        ax.set_ylim(len(s)-.35, -.65)
        left, right, step = axis_settings[variable]
        ax.set_xlim(left, right)
        ax.set_xticks(np.arange(left, right+.001, step))
        for i, r in enumerate(s.itertuples()):
            ax.annotate(f'{r.mortality_pct:.1f}% ({r.ci95_low_pct:.1f}–{r.ci95_high_pct:.1f})',
                        (r.mortality_pct, i), xytext=(0, 14), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color=color)
        t = test[test.variable.eq(variable)].iloc[0]
        ax.set_title(f'{variable}\nχ²({t.df})={t.chi2:.2f}, p={t.p_value:.2e}', fontsize=14, pad=20)
        ax.set_xlabel(f'28일 사망률 (%) · 눈금 {step}%p 간격', labelpad=12)
        ax.grid(axis='x', alpha=.18)
    fig.suptitle('성별·연령대·ICU 유형별 28일 사망률과 95% 신뢰구간', fontsize=22, weight='bold', y=.97)
    fig.text(.5, .9, f'성인 첫 ICU 입실 · 24시간 관찰 조건 충족 · 라벨 확정 {len(data):,}명 · 전체 사망률 {data.mortality_28d.mean():.1%}', ha='center', color='#465569')
    fig.text(.035, .035, '점: 사망률 / 선·괄호: Wilson 95% 신뢰구간 · 패널마다 가로축 범위가 다름 (성별·연령대 축은 0에서 시작하지 않음)\nPearson 카이제곱검정(연속성 보정 없음) · 미확정 21,529명 제외: 선택 편향 가능. 보정 전 비교이며 인과효과를 뜻하지 않음.', fontsize=11, color='#465569')
    fig.subplots_adjust(left=.065, right=.98, bottom=.17, top=.76, wspace=.72)
    fig.savefig(OUT / 'mortality_ci95.png', dpi=190, facecolor='white')
    fig.savefig(OUT / 'mortality_ci95.pdf', facecolor='white')
    plt.close(fig)
    display = summary.copy()
    display['사망률 (95% CI)'] = display.apply(lambda r: f'{r.mortality_pct:.2f}% ({r.ci95_low_pct:.2f}–{r.ci95_high_pct:.2f})', axis=1)
    def html_table(frame):
        return frame.to_html(index=False, border=0, float_format=lambda x: f'{x:.5g}')
    report = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>집단별 사망률 분석</title>
<style>body{font-family:Malgun Gothic,sans-serif;max-width:1450px;margin:40px auto;color:#24354a;padding:20px}img{width:100%}table{border-collapse:collapse;width:100%;margin:25px 0}th,td{padding:11px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f8}p{line-height:1.8}</style>
<h1>성별·연령대·ICU 유형별 28일 사망률</h1>'''
    report += f'<p>대상 {len(data):,}명, 사망 {int(data.mortality_28d.sum()):,}명, 전체 사망률 {data.mortality_28d.mean():.2%}. 환자당 첫 ICU 입실 1건을 분석했습니다.</p>'
    report += '<img src="mortality_ci95.png" alt="집단별 사망률과 Wilson 95% 신뢰구간"><h2>집계표</h2>'
    report += html_table(display[['variable', 'group', 'n', 'deaths', 'survivors', '사망률 (95% CI)']].rename(columns={'variable':'분류','group':'집단','n':'환자 수','deaths':'사망','survivors':'생존'}))
    report += '<p>첫 입실 병동에는 중간치료·회복 병동도 포함됩니다. 기대빈도 확보를 위해 원래 병동별 n&lt;100인 범주를 기타로 통합했습니다. 원래 병동별 인원과 사망 수는 original_careunit_counts.csv에 보존했습니다.</p>'
    report += '<h2>카이제곱검정</h2>' + html_table(test)
    report += '<p>각 분류별 집단×사망 여부 교차표에 Pearson 카이제곱검정을 시행했습니다(연속성 보정 없음). p_holm은 세 검정에 대한 Holm 다중검정 보정값입니다. Cramér’s V는 연관성의 크기입니다. 기대빈도 5 미만 셀 수와 최소 기대빈도를 함께 제공합니다. 개별 집단 간 쌍별 비교는 수행하지 않았습니다.</p>'
    report += '<h2>분석 정의 및 제한</h2><p>입력: data/processed/cohort_base.csv, data/raw/icustays.csv. ICU 유형은 first_careunit입니다. 연령 구간은 [18,40), [40,50), [50,60), [60,70), [70,80), [80,∞)입니다. 28일은 첫 ICU 입실 시점부터의 기존 mortality_28d 정의를 사용하며, 24시간 관찰 조건을 충족한 환자만 포함합니다. 사망 여부 미확정 21,529명은 생존으로 처리하지 않고 제외했습니다. 이 제외 및 24시간 조건으로 선택 편향이 있을 수 있어 전체 ICU 환자 사망률로 일반화할 수 없습니다. 비보정 관찰 비교이므로 중증도·연령 등 교란을 통제한 효과나 인과효과로 해석할 수 없습니다. 80세 이상 구간에는 비식별화된 고령 연령값이 포함될 수 있습니다.</p>'
    report += '<p>재실행: python scripts/analyze_group_mortality.py</p></html>'
    (OUT / 'report.html').write_text(report, encoding='utf-8')
    print(summary.to_string(index=False).encode('ascii', errors='backslashreplace').decode())
    print(test.to_string(index=False).encode('ascii', errors='backslashreplace').decode())
    print('Outputs:', OUT)


if __name__ == '__main__':
    main()
