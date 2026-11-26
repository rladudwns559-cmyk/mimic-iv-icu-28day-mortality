"""Selected presentation EDA; patient-level first-24h means, no imputation."""
from pathlib import Path
import json
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/eda_selected'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/processed/error_cleaned/modeling_dataset.csv')
    config = json.loads((ROOT / 'scripts/feature_config.json').read_text(encoding='utf-8'))
    font = FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False})
    summaries = []
    for kind, label in [('lab', 'Lab'), ('vital', 'Vital')]:
        selected = {'lab': ['Lactate', 'Creatinine', 'BilirubinTotal', 'Platelet'],
                    'vital': ['HR', 'RR', 'MAP', 'SpO2']}
        cols = [f'{kind}_{name}_mean' for name in selected[kind]]
        units = {entry[0]: entry[3] for entry in config[kind]}
        ncols = 2
        nrows = math.ceil(len(cols) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11.25), squeeze=False)
        fig.suptitle(f'{label} 대표 4개 항목 | 첫 24시간 환자별 평균',
                     fontsize=24, fontweight='bold', y=.975)
        fig.text(.5, .92, '생존군(청록)과 사망군(주황) 비교 · 정제 후 전체 코호트 · 결측 대체 없음',
                 ha='center', fontsize=13, color='#52647A')
        for ax, col in zip(axes.flat, cols):
            name = col[len(kind)+1:-len('_mean')]
            values = [data.loc[data.mortality_28d.eq(k), col].dropna() for k in (0, 1)]
            artists = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.48,
                                medianprops={'color': '#152B46', 'linewidth': 2})
            for box, color in zip(artists['boxes'], ['#008C95', '#D96950']):
                box.set_facecolor(color)
                box.set_alpha(.65)
            ax.set_title(f'{name} ({units.get(name, "")})', fontsize=13, pad=12)
            ax.set_xticks([1, 2], [f'생존\nn={len(values[0]):,}', f'사망\nn={len(values[1]):,}'])
            ax.tick_params(labelsize=10)
            ax.grid(axis='y', alpha=.16)
            for k, v in enumerate(values):
                total = int(data.mortality_28d.eq(k).sum())
                q = v.quantile([.25, .5, .75])
                summaries.append(dict(group=kind, feature=col, unit=units.get(name, ''),
                                      mortality_28d=k, total=total, observed=len(v),
                                      missing_rate=1-len(v)/total, q25=q.loc[.25],
                                      median=q.loc[.5], q75=q.loc[.75]))
            ax.set_xlabel(f'중앙값 {values[0].median():.2f} / {values[1].median():.2f}',
                          fontsize=10, color='#52647A')
        for ax in list(axes.flat)[len(cols):]:
            ax.axis('off')
        fig.text(.035, .025,
                 '상자: 25–75백분위 · 선: 중앙값 · 수염: 1.5 IQR 범위 내 관측값. 수염 밖 점은 표시만 생략(통계 계산에는 포함).\n'
                 '패널별 세로축이 다름. 관측 n은 변수별 결측에 따라 다름. 보정 전 기술통계이며 유의성·인과관계를 의미하지 않음.',
                 fontsize=11, color='#52647A', linespacing=1.6)
        fig.subplots_adjust(left=.055, right=.98, top=.86, bottom=.14,
                            hspace=.85 if kind == 'lab' else .6, wspace=.48)
        fig.savefig(OUT / f'{kind}_selected_mean.png', dpi=200, facecolor='white')
        fig.savefig(OUT / f'{kind}_selected_mean.pdf', facecolor='white')
        plt.close(fig)
        print(f'{label}: {len(cols)} variables -> {OUT / (kind + "_selected_mean.png")}')
    pd.DataFrame(summaries).to_csv(OUT / 'summary_by_outcome.csv', index=False, encoding='utf-8-sig')
    (OUT / 'README.md').write_text(
        '# 발표용 대표 Lab / Vital EDA\n\n'
        'Lab 4개, Vital 4개를 각각 16:9 한 페이지 PNG와 PDF로 저장했습니다. '
        '각 항목은 첫 24시간 환자별 평균(mean)으로 통일했습니다. '
        'mean/min/max/std/count/missing 6개 파생 특징 전체를 그린 것은 아닙니다.\n\n'
        '입력: data/processed/error_cleaned/modeling_dataset.csv. '
        '전체 코호트 기술통계이며 테스트 환자도 포함하므로 모델 선택에는 사용하지 않습니다. '
        '결측은 변수별 제외하고 n을 표시했습니다. '
        '극단값은 추가 제거하지 않았고, 수염 밖 점은 그림에서만 생략했습니다.\n\n'
        'summary_by_outcome.csv에 관측 수, 결측률, 사분위수와 중앙값을 저장했습니다.\n\n'
        '선정: Vital은 심박수·호흡수·평균동맥압·산소포화도로 심박·호흡·혈압·산소화를 보여줍니다. '
        'Lab은 젖산·크레아티닌·총빌리루빈·혈소판으로 서로 다른 검사 영역을 보여줍니다. '
        '발표 설명을 위한 탐색적 선정이며 중요도 순위·유의성·인과효과에 따른 선정이 아닙니다. '
        '검사 시행 여부의 집단 차이도 고려해야 합니다.\n\n'
        '재실행: `python scripts/plot_selected_eda.py`\n', encoding='utf-8')


if __name__ == '__main__':
    main()
