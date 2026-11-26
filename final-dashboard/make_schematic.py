"""보고서 그림 1: 연구 모식도 (데이터 → 코호트 → 입력 → 모델 → 평가 → 대시보드). 숫자는 팀 기록과 같다."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "outputs/report_extras/study_schematic.png"
steps = [
    ("Data", "MIMIC-IV 3.1\nBIDMC ICU, 2008–2019"),
    ("Cohort", "First ICU stay, adults\nalive in ICU at 24 h\n28-day outcome known\nn = 30,112 (death 22.2%)"),
    ("Inputs (150)", "First 24 h: 7 vitals, 12 labs\nmean·min·max·SD·count\nmissing indicators\nfirst ICU type"),
    ("Models", "Logistic · Random forest\n· XGBoost\n5×3 nested CV\ndevelopment n = 25,595"),
    ("Final model", "XGBoost\nno imputation (NaN)\n+ indicators + lactate\n+ first ICU type"),
    ("Evaluation", "Held-out test n = 4,517\nROC-AUC · AP · Brier\nthresholds 0.5 / 0.3\nbootstrap 95% CI"),
]
fig, ax = plt.subplots(figsize=(13, 2.15))
ax.set_xlim(0, 13)
ax.set_ylim(0, 2.15)
ax.axis("off")
w, gap = 1.9, 0.26
for i, (head, body) in enumerate(steps):
    x = 0.1 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0.06), w, 2.02, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#F2F4F7" if i != 4 else "#E4EEFA", ec="#225CAA" if i == 4 else "#BECADC", lw=1.4))
    ax.text(x + w / 2, 1.8, head, ha="center", va="center", fontsize=12, fontweight="bold", color="#172D4E")
    ax.text(x + w / 2, 0.88, body, ha="center", va="center", fontsize=9.5, color="#232A32", linespacing=1.45)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + w + gap - 0.02, 1.0), xytext=(x + w + 0.02, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color="#69717D", lw=1.6))
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=200)
print(OUT)
