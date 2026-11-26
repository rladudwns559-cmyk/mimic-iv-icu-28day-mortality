"""보고서 그림 1: 연구 모식도(한국어). 자료 → 연구 대상 → 입력 변수 → 모델 비교 → 최종 모델 → 평가.
팀 요청으로 보고서 그림 1·2는 한국어로 쓴다. 숫자는 팀 기록과 같다."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "outputs/report_extras/study_schematic.png"
steps = [
    ("자료", "MIMIC-IV 3.1\nBIDMC 중환자실\n2008–2019년"),
    ("연구 대상", "성인 첫 ICU 입실\n24시간 시점 ICU 재실\n28일 결과 확인\n30,112명 (사망 22.2%)"),
    ("입력 변수 150개", "첫 24시간 활력징후 7종\n혈액검사 12종\n평균·최소·최대·표준편차·횟수\n결측 지시변수·첫 ICU 유형"),
    ("모델 비교", "로지스틱 회귀\n랜덤 포레스트 · XGBoost\n5×3 중첩 교차검증\n개발 자료 25,595명"),
    ("최종 모델", "XGBoost\n결측값 대체 없음\n+ 결측 지시변수 + 젖산\n+ 첫 ICU 유형"),
    ("평가", "보류한 테스트 4,517명\nROC-AUC · AP · Brier\n경보 기준 0.5 / 0.3\n부트스트랩 95% CI"),
]
fig, ax = plt.subplots(figsize=(13, 2.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 2.4)
ax.axis("off")
w, gap = 1.9, 0.26
for i, (head, body) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.06), w, 2.27, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#F2F4F7" if i != 4 else "#E4EEFA", ec="#225CAA" if i == 4 else "#BECADC", lw=1.4))
    ax.text(x + w / 2, 2.05, head, ha="center", va="center", fontsize=12.5, fontweight="bold", color="#172D4E")
    ax.text(x + w / 2, 1.0, body, ha="center", va="center", fontsize=9.6, color="#232A32", linespacing=1.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.02, 1.1), xytext=(x + w + 0.02, 1.1),
                    arrowprops=dict(arrowstyle="-|>", color="#69717D", lw=1.6))
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=200)
print(OUT)
