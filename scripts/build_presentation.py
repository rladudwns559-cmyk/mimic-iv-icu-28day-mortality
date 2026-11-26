"""Build the eight-slide research deck from local analysis artifacts."""
import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.codex_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, roc_auc_score
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUT = ROOT / 'results/presentation'
OUT.mkdir(parents=True, exist_ok=True)
FONT = '맑은 고딕'
fp = FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': fp.get_name(), 'axes.unicode_minus': False,
                     'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
NAVY = '152B46'
TEAL = '008C95'
CORAL = 'D96950'
GRAY = '52647A'
LIGHT = 'EDF3F7'
BG = 'F8FAFC'
WHITE = 'FFFFFF'
lr = json.loads((ROOT / 'results/logistic_baseline/metrics.json').read_text())
xgb = json.loads((ROOT / 'results/xgboost_baseline/metrics.json').read_text())
assert lr['input_sha256'] == xgb['input_sha256']
assert not lr['test_evaluated'] and not xgb['test_evaluated']
data = pd.read_csv(ROOT / 'data/processed/error_cleaned/modeling_dataset.csv')


def boxplots(cols, titles, filename):
    fig, axes = plt.subplots(1, len(cols), figsize=(12.4, 3.65))
    for ax, col, title in zip(axes, cols, titles):
        values = [data.loc[data.mortality_28d.eq(k), col].dropna() for k in (0, 1)]
        b = ax.boxplot(values, patch_artist=True, showfliers=False, widths=.48,
                       medianprops={'color': '#' + NAVY, 'linewidth': 2})
        for patch, color in zip(b['boxes'], [TEAL, CORAL]):
            patch.set_facecolor('#' + color)
            patch.set_alpha(.65)
        ax.set_xticks([1, 2], [f'생존\nn={len(values[0]):,}', f'사망\nn={len(values[1]):,}'])
        ax.set_title(title, fontsize=14, pad=18)
        ax.grid(axis='y', alpha=.15)
        for i, v in enumerate(values, 1):
            ax.annotate(f'{v.median():.2f}', (i, v.median()), xytext=(26, 4),
                        textcoords='offset points', color='#' + NAVY, fontsize=11)
    fig.tight_layout(pad=1.5)
    fig.savefig(OUT / filename, dpi=190, facecolor='white')
    plt.close(fig)


boxplots(['vital_HR_mean', 'vital_RR_mean', 'vital_MAP_mean'],
         ['평균 심박수 (bpm)', '평균 호흡수 (회/분)', '평균동맥압 (mmHg)'], 'eda_vitals.png')
boxplots(['lab_Lactate_max', 'lab_Creatinine_mean'],
         ['젖산 최댓값 (mmol/L)', '크레아티닌 평균 (mg/dL)'], 'eda_labs.png')
fig, ax = plt.subplots(figsize=(5, 3.75))
for folder, label, color, meta in [('logistic_baseline', 'Logistic regression', GRAY, lr),
                                    ('xgboost_baseline', 'XGBoost', TEAL, xgb)]:
    p = pd.read_csv(ROOT / f'results/{folder}/validation_predictions.csv')
    np.testing.assert_allclose(roc_auc_score(p.mortality_28d, p.probability),
                               meta['metrics']['validation']['roc_auc'])
    fpr, tpr, _ = roc_curve(p.mortality_28d, p.probability)
    ax.plot(fpr, tpr, color='#' + color, linewidth=2.2,
            label=f"{label}  {meta['metrics']['validation']['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], '--', color='#B6C2CE', linewidth=1)
ax.set(xlabel='위양성률 (1 - 특이도)', ylabel='민감도', xlim=(0, 1), ylim=(0, 1.02))
ax.legend(loc='lower right', fontsize=10, frameon=False)
ax.grid(alpha=.12)
fig.tight_layout()
fig.savefig(OUT / 'validation_roc.png', dpi=190, facecolor='white')
plt.close(fig)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def rect(s, x, y, w, h, fill, radius=False):
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                               Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.fill.background()
    return shape


def text(s, x, y, w, h, content, size=20, color=NAVY, bold=False):
    shape = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.025)
    tf.margin_top = tf.margin_bottom = Inches(.025)
    for i, line in enumerate(content.split('\n')):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = FONT
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(9)
    return shape


def slide(n, label, title, subtitle='', source=''):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    rect(s, 0, 0, .14, 7.5, TEAL)
    text(s, .55, .28, 11.6, .3, label.upper(), 11, TEAL, True)
    text(s, .55, .82, 12.2, 1.65 if n == 1 else .75, title, 32 if n == 1 else 24, NAVY, True)
    if subtitle:
        text(s, .58, 1.55, 12.1, .5, subtitle, 14, GRAY)
    rect(s, .58, 7.05, 12.1, .012, 'DAE3EC')
    text(s, .58, 7.15, 11.7, .22, source or 'MIMIC-IV · ICU 28일 사망 예측 · 내부 검증 결과', 9, GRAY)
    text(s, 12.25, 7.12, .5, .28, f'{n:02d}', 11, TEAL, True)
    return s


def note(s, value):
    s.notes_slide.notes_text_frame.text = value


def table(s, x, y, widths, rows, height=.5, fontsize=16):
    t = s.shapes.add_table(len(rows), len(widths), Inches(x), Inches(y),
                           Inches(sum(widths)), Inches(height * len(rows))).table
    for j, width in enumerate(widths):
        t.columns[j].width = Inches(width)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(NAVY if i == 0 else (WHITE if i % 2 else LIGHT))
            cell.margin_left = Inches(.14)
            cell.margin_top = Inches(.08)
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(fontsize)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor.from_string(WHITE if i == 0 else NAVY)
    return t


s = slide(1, 'Research question', 'ICU 첫 24시간 정보로\n28일 사망을 예측할 수 있을까?', source='분석 자료: MIMIC-IV | 로지스틱 회귀 · XGBoost')
# Cover uses extra vertical room for the two-line title.
rect(s, .6, 2.7, 12.0, 1.5, NAVY, True)
text(s, .9, 2.95, 11.4, 1.05,
     '성인 첫 ICU 입실 환자의 초기 정보로\n입실 24시간 이후부터 입실 28일까지의 사망 위험을 예측한다.', 24, WHITE, True)
text(s, .65, 4.65, 11.8, .6, '팀  [팀명 · 팀원]     /     트랙  [트랙명]', 19, GRAY)
text(s, .65, 5.55, 11.8, .75, '30,112명  ·  134개 입력 변수  ·  동일 환자 분할로 두 모델 비교', 20, TEAL, True)
note(s, '첫날 관찰을 마친 시점에 이후 사망 위험을 얼마나 구분할 수 있는지 확인했습니다. 28일은 예측 시점으로부터가 아니라 ICU 입실 시점으로부터 계산했습니다. 팀과 트랙은 발표 전 입력하세요.')

s = slide(2, 'Clinical motivation', '첫날 정보를 이후 위험 평가로 연결한다', '연구 목적: 추가 관찰과 재평가를 지원할 예측 모델의 가능성 탐색')
for x, num, title, body in [(.6, '01', '관찰', '첫 24시간의 생체징후·검사와\n나이·성별을 수집'), (4.7, '02', '예측', '이후 28일 이내 사망 위험을\n확률로 추정'), (8.8, '03', '평가', '위험 구분 성능과\n사망자 탐지 비율을 확인')]:
    rect(s, x, 2.5, 3.9, 2.65, WHITE, True)
    text(s, x+.23, 2.7, 3.35, .5, num, 26, TEAL, True)
    text(s, x+.23, 3.37, 3.35, .5, title, 24, NAVY, True)
    text(s, x+.23, 4.05, 3.35, .9, body, 17, GRAY)
text(s, .75, 5.65, 11.8, .75, '핵심 질문  “실제 사망자를 얼마나 찾아낼 수 있는가?”', 24, TEAL, True)
note(s, '예측 결과가 추가 관찰이나 재평가를 위한 보조 정보가 될 가능성을 탐색합니다. 임상적 효용이나 진료 효과를 입증한 연구는 아닙니다. 정확도보다 AUC와 민감도를 함께 살펴봅니다.')

s = slide(3, 'Cohort', '성인 후보 65,366명 → 최종 분석 30,112명', '환자당 첫 ICU 입실 1건 · 입실+24시간에 예측 · 입실+28일에 사망 판정 종료', '출처: cohort_audit.csv · cohort_base.csv · splits/split_report.md')
for y, title, body in [(2.15, '65,366명', '첫 ICU 입실이 성인인 후보'), (3.4, '51,641명', '24시간 관찰 조건 충족'), (4.65, '30,112명', '생존 23,415명 / 사망 6,697명 (22.24%)')]:
    rect(s, .65, y, 6.0, .95, WHITE, True)
    text(s, .9, y+.12, 2.1, .55, title, 26, TEAL, True)
    text(s, 3.0, y+.2, 3.4, .55, body, 16)
text(s, 3.2, 3.08, .4, .3, '↓', 20, TEAL)
text(s, 3.2, 4.33, .4, .3, '↓', 20, TEAL)
text(s, 7.0, 2.18, 5.55, 1.65, '제외 13,725명\n24시간 이내 사망 1,379명 · ICU 퇴실 12,314명\n경계 모호 13명 · 퇴실 시각 누락 12명 · 데이터 문제 7명', 16, GRAY)
rect(s, 6.95, 4.05, 5.65, 1.55, 'FFF0E9', True)
text(s, 7.15, 4.22, 5.2, 1.22, '28일 상태 확인 근거 부족: 21,529명 제외\n24시간 조건 충족자의 41.7%\n사망 기록이 없다는 이유만으로 생존 처리하지 않음', 16, CORAL)
text(s, .8, 6.05, 11.9, .6, '훈련 21,078명     /     검증 4,517명     /     테스트 4,517명', 22, NAVY, True)
note(s, '제외 사유는 순차 적용되어 중복되지 않습니다. 사망·퇴실은 입실 24시간 이전 또는 동일 시점입니다. 24시간 관찰 이후에도 28일 상태를 확정할 근거가 부족하면 unknown으로 제외했습니다. 분할은 환자 단위 층화 무작위 분할이며 시드는 42입니다.')

s = slide(4, 'Exploratory analysis / 01', '사망군에서 심박수·호흡수가 높고 평균동맥압은 낮았다', '첫 24시간 환자별 평균 생체징후 · 생존군과 사망군의 분포 비교', '출처: 정제 후 modeling_dataset.csv | 전체 코호트 기술통계, 결측값은 변수별 제외')
s.shapes.add_picture(str(OUT / 'eda_vitals.png'), Inches(.55), Inches(2.05), width=Inches(12.2))
text(s, .75, 5.88, 11.8, .85, '중앙값: 심박수 83.15 → 88.17 bpm · 호흡수 18.46 → 20.32회/분\n평균동맥압 77.00 → 74.88 mmHg. 보정 전 비교이며 인과관계를 의미하지 않는다.', 17, GRAY)
note(s, '그림의 상자는 사분위 범위와 중앙값, 수염은 1.5 IQR 규칙입니다. 가독성을 위해 수염 밖 점은 표시하지 않았으며 통계 계산에서는 관측값을 제거하지 않았습니다. 그림 숫자는 중앙값입니다. 전체 코호트 기술통계로 테스트 환자도 포함하며 이 그림을 모델 선택에 사용하지 않습니다. 출처: data/processed/error_cleaned/modeling_dataset.csv.')

s = slide(5, 'Exploratory analysis / 02', '사망군에서 젖산·크레아티닌 중앙값이 높았다', '첫 24시간 젖산 최댓값과 크레아티닌 평균 · 관측값이 있는 환자 비교', '출처: 정제 후 modeling_dataset.csv | 수염 밖 점은 표시 생략, 검정·인과효과 추정 미수행')
s.shapes.add_picture(str(OUT / 'eda_labs.png'), Inches(.55), Inches(2.05), width=Inches(12.2))
text(s, .75, 5.88, 11.8, .85, '젖산 중앙값 2.00 → 2.50 mmol/L · 크레아티닌 중앙값 0.95 → 1.25 mg/dL\n젖산 관측 비율: 생존군 58.4% / 사망군 71.8% — 검사 시행 차이도 고려해야 한다.', 17, GRAY)
note(s, '환자마다 검사 시행 여부가 다릅니다. 관측된 검사값만 비교하므로 집단 차이에 측정 선택이 영향을 줄 수 있습니다. 박스플롯은 1.5 IQR 수염을 사용하고 수염 밖 점은 표시하지 않았습니다. EDA는 전체 코호트 기술통계이며 후속 모델 선택은 훈련 자료에서 해야 합니다.')

s = slide(6, 'Models & odds ratios', '동일한 입력·분할로 로지스틱 회귀와 XGBoost 비교', '공통 입력: 나이·성별 + Vital/Lab의 mean · min · max · std · count · missing = 134개', '출처: 두 모델 metrics.json · logistic_baseline/odds_ratios.csv')
table(s, .65, 2.15, [2.2, 4.1, 5.55], [
    ['설정', '로지스틱 회귀', 'XGBoost'],
    ['전처리', '훈련 중앙값 대체 · 표준화', '결측값 자체 처리 · 표준화 없음'],
    ['모델', 'L2 규제, C=1.0', '깊이 3 · 학습률 0.03 · 선택 트리 1,235개'],
    ['학습 기준', '훈련 자료에 적합', '훈련 학습 · 검증 log loss로 조기 종료'],
], height=.52, fontsize=16)
text(s, .8, 4.48, 11.8, .4, '로지스틱 회귀의 설명용 오즈비', 20, TEAL, True)
table(s, .65, 5.02, [3.0, 2.95, 2.95, 2.95], [
    ['변수', '총빌리루빈 최솟값', '나이', '젖산 최댓값'],
    ['오즈비', '2.67', '1.81', '1.78'],
], height=.47, fontsize=16)
text(s, .75, 6.14, 11.8, .65, '다른 입력이 같을 때 수치형 변수 1 훈련 표준편차 증가 기준. 신뢰구간 미산출.\n상관된 통계 특징으로 개별 계수 해석에 주의가 필요하며, XGBoost에는 단일 오즈비가 없다.', 13, GRAY)
note(s, '양 모델 모두 성별 F=0/M=1 인코딩, 클래스 가중치 미적용입니다. XGBoost는 최대 2,000개 트리, 50회 개선이 없으면 조기 종료하며 검증 log loss로 1,235개가 선택되었습니다. 검증 세트는 모델 선택에도 사용했습니다. 로지스틱 회귀 오즈비는 확률의 배수나 인과효과가 아니며, 설명용 변수 선택입니다. 식별자와 미래 사망·생존 근거는 학습에서 제외했습니다.')

s = slide(7, 'Validation results', 'XGBoost의 AUC·민감도가 높았지만 미탐지는 남았다', '동일 검증 환자 4,517명 · 분류 임계값 0.5 · 통계적 유의성은 아직 평가하지 않음', '출처: 두 모델 metrics.json · validation_predictions.csv | PR-AUC는 사다리꼴 적분')
s.shapes.add_picture(str(OUT / 'validation_roc.png'), Inches(.6), Inches(2.12), width=Inches(5.0))
rows = [['검증 지표', '로지스틱', 'XGBoost']]
for label, key, percentage in [('ROC-AUC', 'roc_auc', False), ('PR-AUC', 'pr_auc_trapezoidal', False), ('민감도', 'sensitivity', True), ('특이도', 'specificity', True), ('정밀도', 'precision', True)]:
    vals = [m['metrics']['validation'][key] for m in (lr, xgb)]
    rows.append([label] + [f'{v:.2%}' if percentage else f'{v:.4f}' for v in vals])
table(s, 5.9, 2.17, [2.2, 2.15, 2.25], rows, height=.49, fontsize=16)
rect(s, 5.9, 5.35, 6.6, 1.15, 'E3F2F1', True)
text(s, 6.1, 5.48, 6.2, .92, '사망자 탐지 279 → 385명 (+106명)\n오경보 149 → 188명 (+39명) · 사망자 619명은 미탐지', 17, TEAL, True)
text(s, .8, 6.55, 11.7, .3, '검증 자료는 XGBoost 조기 종료에도 사용함. 테스트 세트는 아직 평가하지 않음.', 12, GRAY)
note(s, '검증 사망자 1,004명 중 로지스틱 회귀는 279명, XGBoost는 385명을 탐지했습니다. 민감도는 약 10.56%p 증가했고 특이도는 약 1.11%p 감소했습니다. 임계값 0.5에서의 비교이며 같은 민감도·특이도에 맞춘 비교는 아닙니다. 검증 ROC-AUC는 0.7878 대 0.8245이고, 차이의 신뢰구간은 아직 계산하지 않았습니다. XGBoost 훈련 AUC 0.9024와 검증 0.8245 사이 차이가 있어 일반화 성능을 추가 확인해야 합니다.')

s = slide(8, 'Limitations & next steps', '정보 가용성과 일반화를 확인하고 탐지 기준을 개선한다', '검증 결과: XGBoost의 성능 개선 가능성 확인 · 임상 적용 전 추가 평가 필요')
table(s, .65, 2.15, [5.75, 6.1], [
    ['현재 한계', '다음 계획'],
    ['28일 상태 확인 근거 부족으로 41.7% 제외', '포함·제외 환자 비교 → 선택 편향 점검'],
    ['예측 시점 이후 저장된 관측도 포함', 'storetime도 제한하는 민감도 분석'],
    ['XGBoost 훈련/검증 AUC: 0.902 / 0.824', '규제·모델 복잡도와 일반화 성능 점검'],
    ['민감도 38.3%: 사망자의 61.7% 미탐지', '임계값별 민감도·특이도·정밀도 비교'],
    ['확률 보정·성능 차이의 불확실성 미평가', '보정 곡선·신뢰구간 평가 후 최종 테스트'],
], height=.57, fontsize=16)
text(s, .8, 6.0, 11.7, .65, '설정 확정 → 보존한 테스트 세트 최종 평가 → 시간적·외부 검증', 23, TEAL, True)
note(s, '41.7%는 전체 후보가 아니라 24시간 관찰 조건을 충족한 51,641명 중 21,529명 비율입니다. 예를 들어 조기 생존 퇴원 후 추적 근거가 없으면 28일 생존을 확정하지 않습니다. 포함 환자가 장기 입원·재입원 환자에 치우칠 수 있으므로 집단 비교가 필요합니다. storetime 제한 분석은 지금 별도 수행한 것이 아니라 다음 계획입니다. 현재 테스트는 읽거나 평가하지 않았고 모델과 임계값을 확정한 뒤 최종 평가합니다.')

path = OUT / 'ICU_28day_mortality_LR_XGBoost.pptx'
prs.save(path)
loaded = Presentation(path)
assert len(loaded.slides) == 8
for i, sl in enumerate(loaded.slides, 1):
    for shape in sl.shapes:
        assert shape.left >= 0 and shape.top >= 0, (i, shape.name)
        assert shape.left + shape.width <= loaded.slide_width + 1000, (i, shape.name)
        assert shape.top + shape.height <= loaded.slide_height + 1000, (i, shape.name)
    assert sl.notes_slide.notes_text_frame.text.strip()
(OUT / 'README.md').write_text(
    '# ICU 28일 사망 예측 발표자료\n\n'
    '8장, 16:9 PPTX. 제목·본문·표·흐름도는 편집 가능하며 EDA와 ROC는 PNG 그림입니다. '
    '슬라이드 노트에 발표 멘트와 해석 주의를 넣었습니다. 팀명·트랙명은 첫 장에서 입력하세요.\n\n'
    'EDA는 정제 후 전체 코호트 기술통계로, 수염 밖 점을 표시하지 않았습니다. '
    '모델 결과는 저장된 산출물과 예측으로 확인했으며 재학습하지 않았습니다. 테스트 평가는 하지 않았습니다.\n\n'
    '생성: `python scripts/build_presentation.py` (python-pptx, matplotlib 필요).\n'
    '검증: 8장 재열기, 도형 경계, 노트 존재, 예측 ROC-AUC 및 모델 입력 해시 일치 확인. '
    'PowerPoint 렌더링에 의한 화면 검수는 수행하지 않았습니다.\n', encoding='utf-8')
print(path)
print('Verified: 8 slides, shape bounds, speaker notes, ROC-AUC, matching input hashes.')
