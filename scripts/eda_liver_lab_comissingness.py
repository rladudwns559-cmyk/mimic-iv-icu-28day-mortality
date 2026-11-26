"""Patient-level joint missingness of total bilirubin, ALT and AST in first 24h."""
from pathlib import Path
from itertools import product, combinations
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['BilirubinTotal', 'ALT', 'AST']
LABELS = ['총빌리루빈', 'ALT', 'AST']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/processed/error_cleaned/modeling_dataset.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/eda_liver_lab_comissingness')
    args = parser.parse_args()
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    columns = ['subject_id', 'stay_id'] + [f'lab_{f}_{s}' for f in FEATURES for s in ['mean', 'count', 'missing']]
    data = pd.read_csv(args.input, usecols=columns)
    if data.empty or data[['subject_id', 'stay_id']].isna().any().any():
        raise ValueError('Expected nonempty cohort with patient/stay IDs')
    if not data.subject_id.is_unique or not data.stay_id.is_unique:
        raise ValueError('Expected one first ICU stay per patient')
    missing = pd.DataFrame(index=data.index)
    for f in FEATURES:
        count = data[f'lab_{f}_count']
        mask = data[f'lab_{f}_mean'].isna()
        if (count.isna().any() or count.lt(0).any() or count.mod(1).ne(0).any()
                or not mask.equals(count.eq(0))
                or not data[f'lab_{f}_missing'].eq(mask.astype(int)).all()):
            raise ValueError(f'{f}: inconsistent missingness indicators')
        missing[f] = mask
    n = len(data)
    rows = []
    for bits in product([0, 1], repeat=3):
        size = int(missing.eq(bits).all(axis=1).sum())
        rows.append(dict(pattern=''.join(map(str, bits)), **dict(zip(FEATURES, bits)),
                         missing_count=sum(bits), n=size, cohort_pct=100*size/n,
                         description=' / '.join(f'{label} {"결측" if bit else "관측"}' for label, bit in zip(LABELS, bits))))
    patterns = pd.DataFrame(rows)
    assert patterns.n.sum() == n
    for f in FEATURES:
        assert patterns.loc[patterns[f].eq(1), 'n'].sum() == missing[f].sum()
    summaries = pd.DataFrame([dict(feature=f, label=label, missing_n=int(missing[f].sum()),
                                   observed_n=int((~missing[f]).sum()), missing_pct=100*missing[f].mean())
                              for f, label in zip(FEATURES, LABELS)])
    pair_rows = []
    for a, b in combinations(FEATURES, 2):
        ma, mb = missing[a], missing[b]
        both = int((ma & mb).sum())
        union = int((ma | mb).sum())
        pair_rows.append(dict(feature_a=a, feature_b=b, both_missing_n=both,
                              both_observed_n=int((~ma & ~mb).sum()),
                              only_a_missing_n=int((ma & ~mb).sum()),
                              only_b_missing_n=int((~ma & mb).sum()),
                              both_missing_cohort_pct=100*both/n,
                              agreement_pct=100*ma.eq(mb).mean(),
                              jaccard=both/union if union else np.nan,
                              phi=ma.astype(int).corr(mb.astype(int)),
                              p_b_missing_given_a_missing_pct=100*both/ma.sum() if ma.any() else np.nan,
                              p_a_missing_given_b_missing_pct=100*both/mb.sum() if mb.any() else np.nan))
    pairs = pd.DataFrame(pair_rows)
    conditional = np.array([[100*(missing[a] & missing[b]).sum()/missing[a].sum()
                             if missing[a].any() else np.nan for b in FEATURES] for a in FEATURES])
    all_missing = int(missing.all(axis=1).sum())
    any_missing = int(missing.any(axis=1).sum())
    all_observed = int((~missing).all(axis=1).sum())
    mixed = n-all_missing-all_observed
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for frame, name in [(patterns, 'missingness_patterns'), (summaries, 'variable_missingness'),
                        (pairs, 'pairwise_overlap')]:
        frame.to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(conditional, index=FEATURES, columns=FEATURES).rename_axis('given_missing').to_csv(
        out/'conditional_missingness_pct.csv', encoding='utf-8-sig')
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 11})
    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[3, 1.25], hspace=.15, wspace=.32)
    bar = fig.add_subplot(grid[0, 0])
    matrix = fig.add_subplot(grid[1, 0], sharex=bar)
    heat = fig.add_subplot(grid[:, 1])
    sorted_patterns = patterns.sort_values(['n', 'pattern'], ascending=[False, True]).reset_index(drop=True)
    xs = np.arange(8)
    colors = ['#187f97' if r.missing_count == 0 else '#b87545' if r.missing_count == 3 else '#8795ab'
              for r in sorted_patterns.itertuples()]
    bar.bar(xs, sorted_patterns.n, color=colors)
    for i, r in enumerate(sorted_patterns.itertuples()):
        bar.text(i, r.n+max(sorted_patterns.n)*.018, f'{r.n:,}\n({r.cohort_pct:.2f}%)', ha='center', fontsize=10)
    bar.set_ylim(0, max(sorted_patterns.n)*1.23)
    bar.set_ylabel('환자 수 (명)')
    bar.set_title('8가지 배타적 결측 조합 · 환자 수 순서')
    bar.tick_params(axis='x', bottom=False, labelbottom=False)
    bar.grid(axis='y', alpha=.15)
    for i, r in sorted_patterns.iterrows():
        ys = [j for j, f in enumerate(FEATURES) if r[f] == 1]
        matrix.scatter([i]*3, range(3), s=55, color='#dedede', zorder=2)
        if ys:
            matrix.plot([i, i], [min(ys), max(ys)], color=colors[i], lw=2, zorder=3)
            matrix.scatter([i]*len(ys), ys, s=65, color=colors[i], zorder=4)
    matrix.set_yticks(range(3), LABELS)
    matrix.set_ylim(2.65, -.65)
    matrix.set_xticks(xs, sorted_patterns.pattern)
    matrix.set_xlabel('코드 순서: 총빌리루빈 / ALT / AST · 1=결측, 0=관측\n진한 점=결측, 회색 점=관측')
    matrix.spines[['left', 'bottom']].set_visible(False)
    matrix.tick_params(axis='both', length=0)
    im = heat.imshow(conditional, vmin=0, vmax=100, cmap='Blues')
    heat.set_xticks(range(3), LABELS)
    heat.set_yticks(range(3), LABELS)
    heat.set_ylabel('이 변수가 결측인 환자 중')
    heat.set_xlabel('이 변수도 결측인 비율 (%)')
    heat.set_title('조건부 동시 결측률\n분모는 각 행 변수의 결측 환자 수')
    for i in range(3):
        for j in range(3):
            heat.text(j, i, f'{conditional[i, j]:.2f}%', ha='center', va='center',
                      color='white' if conditional[i, j]>55 else '#24354a', fontsize=14)
    fig.colorbar(im, ax=heat, fraction=.046, pad=.04)
    fig.suptitle('총빌리루빈·ALT·AST: 같은 환자에서 결측이 함께 나타나는가?', fontsize=20, weight='bold', y=.97)
    fig.text(.5, .915, f'환자 {n:,}명 | 세 검사 모두 결측 {all_missing:,}명 ({100*all_missing/n:.2f}%) | 모두 관측 {all_observed:,}명 ({100*all_observed/n:.2f}%)', ha='center')
    fig.text(.04, .025, '첫 ICU 입실 후 24시간 내 유효 검사값 기준. 함께 결측이라는 것은 같은 환자·같은 관찰창 기준이며, 동일 시각 채혈 여부를 뜻하지 않습니다.', fontsize=10)
    fig.subplots_adjust(top=.83, bottom=.19, left=.075, right=.96)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'joint_missingness.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    lines = ['# 총빌리루빈·ALT·AST 동시 결측 EDA', '',
             f'입력: `{args.input.resolve()}`. 환자당 첫 ICU 입실 1건, 총 {n:,}명.', '',
             '## 핵심 결과', '',
             f'- 세 검사 모두 결측: **{all_missing:,}명 ({100*all_missing/n:.2f}%)**.',
             f'- 세 검사 모두 관측: **{all_observed:,}명 ({100*all_observed/n:.2f}%)**.',
             f'- 관측/결측이 섞인 환자: **{mixed:,}명 ({100*mixed/n:.2f}%)**.',
             f'- 하나 이상 결측인 {any_missing:,}명 중 세 검사 모두 결측인 비율: **{100*all_missing/any_missing:.2f}%**.' if any_missing else '- 하나 이상 결측인 환자 없음.', '',
             '## 변수별 결측', '', '| 변수 | 결측 환자 | 결측률 |', '|---|---:|---:|']
    for r in summaries.itertuples():
        lines.append(f'| {r.label} | {r.missing_n:,} | {r.missing_pct:.2f}% |')
    lines += ['', '## 결측 조합 8가지', '', '각 환자는 아래 한 행에만 포함됩니다. 코드 순서: 총빌리루빈/ALT/AST, 1=결측, 0=관측.', '',
              '| 코드 | 조합 | 환자 수 | 전체 대비 |', '|---|---|---:|---:|']
    for r in sorted_patterns.itertuples():
        lines.append(f'| {r.pattern} | {r.description} | {r.n:,} | {r.cohort_pct:.2f}% |')
    lines += ['', '![동시 결측 조합 및 조건부 비율](joint_missingness.png)', '',
              '## 두 변수씩 결측 겹침', '',
              '| A | B | 모두 결측 | 결측 Jaccard | 결측 지표 상관계수(phi) | A 결측 중 B 결측 | B 결측 중 A 결측 |',
              '|---|---|---:|---:|---:|---:|---:|']
    labels = dict(zip(FEATURES, LABELS))
    for r in pairs.itertuples():
        lines.append(f'| {labels[r.feature_a]} | {labels[r.feature_b]} | {r.both_missing_n:,} | {r.jaccard:.4f} | {r.phi:.4f} | {r.p_b_missing_given_a_missing_pct:.2f}% | {r.p_a_missing_given_b_missing_pct:.2f}% |')
    lines += ['', '## 정의 및 해석', '',
              '- 결측: 첫 ICU 입실 후 [0,24시간)에 유효한 정제 검사값이 없음. mean의 NaN, count=0, missing=1의 일치를 검증했습니다. std는 한 번만 측정해도 결측이므로 이번 결측 판정에는 사용하지 않습니다.',
              '- 이전 모델 학습에서 ALT·AST를 제외했어도 이 EDA는 세 변수가 보존된 전체 정제 데이터셋을 사용합니다. 원본 및 학습 산출물은 수정하지 않습니다.',
              '- Jaccard = 두 변수 모두 결측인 수 / 둘 중 하나 이상 결측인 수. 모두 관측인 환자를 제외하고 결측 집합의 겹침을 평가합니다. 1에 가까울수록 결측 환자가 같습니다.',
              '- phi = 두 이진 결측 지표의 Pearson 상관계수. 조건부 비율의 분모는 해당 조건 변수의 결측 환자 수이므로 방향에 따라 값이 다릅니다.',
              '- 동시 결측은 같은 환자의 24시간 관찰창 기준입니다. 같은 시각에 채혈했거나 같은 검사 패널로 처방했다는 사실을 확인한 분석은 아닙니다. 공통 처방 관행 등을 가설로 제시할 수 있지만 결측 발생 기전이나 원인을 확정할 수 없습니다.',
              '- 기존 24시간 관찰 조건 및 사망 라벨 확정 코호트를 사용하므로 전체 ICU 환자로의 일반화에는 제한이 있습니다. 늦은 storetime을 포함하는 기존 charttime 정책을 유지합니다.', '',
              '재실행: `python scripts/eda_liver_lab_comissingness.py`', '',
              '산출물: joint_missingness.png/pdf, missingness_patterns.csv, variable_missingness.csv, pairwise_overlap.csv, conditional_missingness_pct.csv, report.html, analysis_metadata.json.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    pattern_display = sorted_patterns[['pattern', 'description', 'n', 'cohort_pct']].rename(
        columns={'pattern':'코드', 'description':'결측 조합', 'n':'환자 수', 'cohort_pct':'전체 대비 (%)'})
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>간 검사 동시 결측 EDA</title>'
            '<style>body{font-family:Malgun Gothic,sans-serif;max-width:1500px;margin:36px auto;padding:20px;color:#24354a}img{width:100%}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd}p{line-height:1.8}</style>'
            '<h1>총빌리루빈·ALT·AST 동시 결측 EDA</h1>'
            f'<p>총 {n:,}명 · 세 검사 모두 결측 {all_missing:,}명 ({100*all_missing/n:.2f}%) · 모두 관측 {all_observed:,}명 ({100*all_observed/n:.2f}%) · 혼합 {mixed:,}명 ({100*mixed/n:.2f}%)</p>'
            '<img src="joint_missingness.png" alt="환자별 결측 조합과 조건부 동시 결측률">'
            '<h2>결측 조합</h2>'+pattern_display.to_html(index=False, float_format=lambda x:f'{x:.2f}')+
            '<h2>조건부 동시 결측률 (%)</h2><p>행 변수가 결측인 환자 중 열 변수도 결측인 비율입니다.</p>'+
            pd.DataFrame(conditional, index=LABELS, columns=LABELS).to_html(float_format=lambda x:f'{x:.2f}')+
            '<h2>정의 및 해석</h2>'+''.join(f'<p>{line[2:]}</p>' for line in lines if line.startswith('- '))+'</html>')
    (out/'report.html').write_text(html, encoding='utf-8')
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == digest
    (out/'analysis_metadata.json').write_text(json.dumps(dict(input=str(args.input.resolve()), input_sha256=digest,
        n=n, features=FEATURES, all_missing=all_missing, all_observed=all_observed, mixed=mixed,
        definition='valid first-24h lab mean missing; checked against count and missing flags',
        checks=['unique patient/stay', 'no missing identifiers', 'consistent mean/count/missing',
                '8 disjoint patterns sum to cohort', 'pattern marginals match missing counts', 'source hash unchanged']),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(patterns.drop(columns='description').to_string(index=False))
    print(pairs.to_string(index=False))
    print(f'All missing={all_missing}; all observed={all_observed}; mixed={mixed}; n={n}')
    print('Output:', out)


if __name__ == '__main__':
    main()
