"""보고서 그림 2: 코호트 선정 흐름도(한국어). 보고서 그림 1·2는 한국어로 쓴다.
인원은 모두 원자료·코호트 감사표·분할 파일에서 세며, 보고서 숫자와 다르면 멈춘다."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs/report_extras/cohort_flow.png"

stays = pd.read_csv(ROOT / "data/raw/icustays.csv", usecols=["subject_id", "stay_id"])
audit = pd.read_csv(ROOT / "data/processed/cohort_audit.csv", usecols=["exclusion_reason", "mortality_28d"])
reason = audit.exclusion_reason.value_counts()
kept = audit[audit.exclusion_reason.isna()]
split = {}
for part in ("train", "validation", "test"):
    y = pd.read_csv(ROOT / f"data/processed/error_cleaned/splits/{part}/y_target.csv").mortality_28d
    split[part] = (len(y), int(y.sum()))

n_stays, n_first = len(stays), stays.subject_id.nunique()
n_left, n_died = int(reason["icu_exit_at_or_before_prediction"]), int(reason["death_at_or_before_prediction"])
n_other = int(reason[["ambiguous_death_at_prediction", "missing_outtime", "data_issue"]].sum())
n_24 = len(kept)
n_unknown = int((kept.mortality_28d == "unknown").sum())
n_final = n_24 - n_unknown
deaths = int((kept.mortality_28d == "1").sum())
dev = split["train"][0] + split["validation"][0]
assert (n_stays, n_first, len(audit)) == (94458, 65366, 65366)
assert (n_left, n_died, n_other, n_24, n_unknown, n_final, deaths) == (12314, 1379, 32, 51641, 21529, 30112, 6697)
assert (dev, split["test"]) == (25595, (4517, 1005))

fig, ax = plt.subplots(figsize=(10, 7.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
MAIN, EXC, FINAL = ("#F2F4F7", "#BECADC"), ("#FBEFEF", "#D9A3A3"), ("#E4EEFA", "#225CAA")


def box(x, y, w, h, title, body, style, size=13):
    fc, ec = style
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + 0.15, y + h - 0.2, title, ha="left", va="top", fontsize=size, fontweight="bold", color="#172D4E")
    if body:
        ax.text(x + 0.15, y + h - 0.56, body, ha="left", va="top", fontsize=11.5, color="#232A32", linespacing=1.4)


def down(y0, y1):
    ax.annotate("", xy=(2.2, y1), xytext=(2.2, y0), arrowprops=dict(arrowstyle="-|>", color="#69717D", lw=1.5))


def side(y):
    ax.plot([2.2, 4.85], [y, y], color="#B05A5A", lw=1.2)
    ax.annotate("", xy=(5.0, y), xytext=(4.85, y), arrowprops=dict(arrowstyle="-|>", color="#B05A5A", lw=1.2))


box(0.2, 6.55, 4.0, 0.75, f"MIMIC-IV 3.1 ICU 입실 {n_stays:,}건", f"환자 {n_first:,}명", MAIN)
side(6.0)
box(5.0, 5.65, 4.8, 0.7, f"제외: 환자별 첫 입실이 아님 ({n_stays - n_first:,}건)", None, EXC, 12.5)
down(6.55, 5.1)
box(0.2, 4.35, 4.0, 0.75, f"성인 환자별 첫 ICU 입실 {n_first:,}명", "18세 미만으로 제외된 환자 0명", MAIN)
side(3.85)
box(5.0, 3.25, 4.8, 1.3, f"제외: 24시간 시점에 ICU에 없음 ({n_left + n_died + n_other:,}명)",
    f"24시간 이전·동시 퇴실 {n_left:,}명\n24시간 이전·동시 사망 {n_died:,}명\n"
    f"사망 시각 불명확·퇴실 시각 누락·기록 오류 {n_other}명", EXC, 12.5)
down(4.35, 2.95)
box(0.2, 2.2, 4.0, 0.75, f"24시간 시점 ICU 재실 {n_24:,}명", None, MAIN)
side(1.75)
box(5.0, 1.15, 4.8, 1.0, f"제외: 28일 결과 미확정 ({n_unknown:,}명)",
    "28일 시점 생존·사망 기록 없음\n(생존으로 간주하지 않음)", EXC, 12.5)
down(2.2, 1.35)
box(0.2, 0.1, 4.0, 1.25, f"분석 대상 {n_final:,}명",
    f"28일 사망 {deaths:,}명 ({deaths / n_final * 100:.1f}%)\n"
    f"개발 자료 {dev:,}명 (5겹 교차검증)\n테스트 {split['test'][0]:,}명 (사망 {split['test'][1]:,}명)", FINAL)
ax.text(5.0, 0.38, "분할: 28일 사망 여부로 층화, 학습 70% · 검증 15% · 테스트 15%\nrandom_state 42 (학습 + 검증 = 개발 자료)",
        fontsize=10.5, color="#555E6A", va="center", linespacing=1.5)
fig.tight_layout(pad=0.3)
fig.savefig(OUT, dpi=200)
print(OUT)
