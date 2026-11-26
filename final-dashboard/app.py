"""ICU 28일 사망 위험 대시보드 (Gradio).

outputs/model.ubj(학습된 최종 XGBoost)를 불러 쓰기만 한다. 이 파일에는 모델을 학습하는 코드가 없다.
실행: ..\\.venv\\Scripts\\python app.py  →  브라우저에서 http://127.0.0.1:7860
"""
import html
import tempfile

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dashboard_core as core

ASSETS, BOOSTER = core.load_assets()
M = ASSETS["metrics"]
TH = {round(t["threshold"], 2): t for t in ASSETS["thresholds"]}
MEAN_RISK = ASSETS["development_mortality"]
EN = {"ICU": "First ICU type", "age": "Age", "gender": "Sex"}
FIELDS = ("min", "mean", "max", "count")
FIELD_KO = {"min": "최저", "mean": "평균", "max": "최고", "count": "측정 횟수"}

# ---------------------------------------------------------------- 디자인
# 참고: NHS 디자인 시스템(색·대비·"색만으로 정보 전달 금지"), 국내 공공·병원 화면에서 쓰는 Pretendard 글꼴,
# EMR식 촘촘한 표 배치. 대비는 WCAG 2.2 AA(본문 4.5:1) 이상.
C = dict(bg="#f0f4f5", text="#212b32", sub="#4c6272", border="#d8dde0", input_border="#768692",
         blue="#005eb8", navy="#0b2545", red="#d5281b", amber="#a35200", green="#007f3b",
         pale_red="#fbe9e7", pale_amber="#fdf1e2", pale_green="#e6f2ec", focus="#ffeb3b")
FONT = '"Pretendard Variable", Pretendard, "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif'
HEAD = ('<link rel="stylesheet" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/'
        'dist/web/variable/pretendardvariable-dynamic-subset.min.css">'
        # 컴퓨터가 다크 모드여도 항상 밝은 화면으로 연다(다크 모드에서 일부 글자가 배경과 같은 색이 됨).
        # Gradio는 주소의 __theme=light 값을 보고 밝은 테마를 고정한다.
        '<meta name="color-scheme" content="light">'
        '<script>(function(){var u=new URL(window.location.href);'
        'if(u.searchParams.get("__theme")!=="light"){u.searchParams.set("__theme","light");'
        'window.location.replace(u.toString());}})();</script>')

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.Font(f) for f in ("Pretendard Variable", "Pretendard", "Malgun Gothic", "sans-serif")],
    radius_size=gr.themes.sizes.radius_lg, spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="linear-gradient(180deg, #e4ebf3 0%, #eef3f8 320px, #f3f6fa 100%)",
    body_text_color=C["text"], body_text_color_subdued=C["sub"],
    block_background_fill="#ffffff", block_border_color="rgba(15,40,70,0.06)", block_border_width="1px",
    block_radius="16px", block_padding="14px",
    block_shadow="0 1px 2px rgba(16,32,56,.06), 0 10px 28px -8px rgba(16,32,56,.18)",
    block_label_background_fill="transparent", block_label_text_color=C["sub"], block_label_border_width="0px",
    block_title_text_color=C["sub"], block_title_text_weight="600",
    input_background_fill="#f6f8fb", input_border_color="#cfd8e3", input_border_color_focus=C["blue"],
    input_radius="10px", input_shadow="inset 0 2px 4px rgba(16,32,56,.08)",
    input_shadow_focus="inset 0 2px 4px rgba(16,32,56,.08), 0 0 0 3px rgba(0,94,184,.18)",
    button_primary_background_fill="linear-gradient(180deg, #1a74d1 0%, #005eb8 55%, #004c96 100%)",
    button_primary_background_fill_hover="linear-gradient(180deg, #2280de 0%, #0066c7 55%, #00539f 100%)",
    button_primary_text_color="#ffffff", button_primary_shadow="0 6px 16px -4px rgba(0,94,184,.55), inset 0 1px 0 rgba(255,255,255,.25)",
    button_large_radius="12px", button_small_radius="10px",
    button_secondary_background_fill="linear-gradient(180deg,#ffffff,#eef3f8)",
    button_secondary_border_color="#cfd8e3", button_secondary_text_color="#0b3a6e",
    button_secondary_shadow="0 2px 6px -2px rgba(16,32,56,.18)",
    color_accent=C["blue"], border_color_primary="#dbe3ec",
    checkbox_background_color_selected=C["blue"], table_border_color=C["border"], layout_gap="14px",
)

# 헤더 장식용 심전도(ECG) 선
ECG = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='420' height='60'>"
       "<polyline fill='none' stroke='rgba(255,255,255,0.18)' stroke-width='2' points='0,30 120,30 135,30 145,8 "
       "158,52 170,22 180,30 260,30 272,30 280,14 290,44 300,30 420,30'/></svg>\")")

CSS = f"""
:root {{ --font: {FONT}; --lift: 0 1px 2px rgba(16,32,56,.06), 0 12px 32px -10px rgba(16,32,56,.22); }}
body, .gradio-container, button, input, textarea, select {{ font-family: {FONT} !important; }}
.gradio-container {{max-width: 1440px !important; margin: auto; padding-top: 14px !important;}}
.num, .big, .kv b, td {{font-variant-numeric: tabular-nums;}}
*:focus-visible {{outline: 3px solid {C['focus']} !important; outline-offset: 1px;}}
button {{transition: transform .15s ease, box-shadow .15s ease, filter .15s ease !important;}}
button:hover {{transform: translateY(-1px);}}
button:active {{transform: translateY(1px);}}
@media (prefers-reduced-motion: reduce) {{ * {{transition:none !important;}} button:hover, button:active {{transform:none;}} }}

/* 헤더: 그라데이션 + 심전도 선 + 그림자 */
.topbar {{position:relative; overflow:hidden; color:#fff; padding:20px 26px; border-radius:20px;
  background: {ECG} right 24px center / 420px 60px no-repeat,
              radial-gradient(1200px 300px at 85% -40%, rgba(80,170,255,.35), transparent 60%),
              linear-gradient(135deg, #0a2342 0%, #0b3a6e 55%, #005eb8 100%);
  box-shadow: 0 18px 40px -16px rgba(5,30,70,.65), inset 0 1px 0 rgba(255,255,255,.12);
  display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;}}
.topbar .brand {{display:flex; align-items:center; gap:14px;}}
.topbar .logo {{width:46px; height:46px; border-radius:14px; display:grid; place-items:center;
  background:linear-gradient(145deg, rgba(255,255,255,.28), rgba(255,255,255,.06));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.35), 0 6px 14px -4px rgba(0,0,0,.4);}}
.topbar h1 {{font-size:24px; font-weight:800; margin:0; color:#fff; letter-spacing:-0.4px;}}
.topbar .sys {{font-size:11px; color:#bcd3ea; letter-spacing:1px; text-transform:uppercase; font-weight:600;}}
.topbar .perf {{font-size:13px; color:#dbe8f5; text-align:right; line-height:1.6;}}

/* 핵심 지표 타일 */
.tiles {{display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin:16px 0 4px;}}
.tile {{background:linear-gradient(180deg,#ffffff,#f7f9fc); border-radius:16px; padding:14px 16px;
  box-shadow: var(--lift); border:1px solid rgba(15,40,70,.06); display:flex; gap:12px; align-items:center;}}
.tile .ic {{width:40px; height:40px; border-radius:12px; display:grid; place-items:center; flex:none;
  background:linear-gradient(145deg,#e6f0fb,#cfe2f6); color:{C['blue']};
  box-shadow: inset 0 1px 0 #fff, 0 4px 10px -4px rgba(0,94,184,.45);}}
.tile .v {{font-size:22px; font-weight:800; color:{C['text']}; line-height:1.1;}}
.tile .l {{font-size:12px; color:{C['sub']}; font-weight:600;}}
@media (max-width: 900px) {{ .tiles {{grid-template-columns: repeat(2, minmax(0,1fr));}} }}

.banner {{background:linear-gradient(90deg,#fdecea,#fff6f5); border-radius:14px; color:{C['text']};
  padding:11px 16px; font-size:14px; margin:12px 0 6px; box-shadow: inset 4px 0 0 {C['red']}, 0 4px 14px -8px rgba(213,40,27,.4);}}

/* 탭: 둥근 알약형 */
.tab-wrapper, [role=tablist] {{gap:6px;}}
button[role=tab] {{border-radius:999px !important; padding:6px 16px !important; font-weight:600;}}
button[role=tab][aria-selected=true] {{background:#ffffff !important; color:{C['blue']} !important;
  box-shadow: 0 4px 12px -4px rgba(16,32,56,.25) !important;}}

/* 섹션 제목 */
.sec {{display:flex; align-items:center; gap:10px; margin:4px 0 0; flex-wrap:wrap;}}
.sec .t {{white-space:nowrap;}}
.sec .hint {{flex-basis:100%; margin-left:38px; margin-top:-4px;}}
.sec .no {{width:28px; height:28px; border-radius:9px; display:grid; place-items:center; color:#fff; font-weight:800;
  font-size:14px; background:linear-gradient(145deg,#2a86e0,#004c96); box-shadow:0 4px 10px -3px rgba(0,94,184,.6);}}
.sec .t {{font-size:17px; font-weight:800; color:{C['text']};}}
.hint {{font-size:13px; color:{C['sub']}; font-weight:400;}}

/* 결과 카드 */
.card {{border-radius:18px; padding:18px 20px; background:linear-gradient(180deg,#ffffff,#fafcfe);
  box-shadow: var(--lift); border:1px solid rgba(15,40,70,.06);}}
.hero {{display:flex; gap:20px; align-items:center;}}
.ring {{position:relative; width:168px; height:168px; flex:none; filter: drop-shadow(0 10px 18px rgba(16,32,56,.22));}}
.ring svg {{width:168px; height:168px; transform: rotate(-90deg);}}
.ring .center {{position:absolute; inset:0; display:grid; place-items:center; text-align:center;}}
.ring .center .pct {{font-size:36px; font-weight:800; letter-spacing:-1px; line-height:1;}}
.ring .center .cap {{font-size:11px; color:{C['sub']}; font-weight:600; margin-top:4px;}}
.risk-label {{font-size:13px; color:{C['sub']}; font-weight:700;}}
.badge {{display:inline-flex; align-items:center; gap:8px; padding:8px 14px; border-radius:999px;
  font-weight:800; font-size:17px; color:#fff; box-shadow: 0 8px 18px -6px currentColor, inset 0 1px 0 rgba(255,255,255,.3);}}
.badge-wrap {{margin:8px 0 10px;}}
.scale {{display:flex; gap:6px; flex-wrap:wrap;}}
.scale span {{font-size:12px; padding:3px 9px; border-radius:999px; background:#eef2f7; color:{C['sub']}; font-weight:600;}}
.kv {{display:grid; grid-template-columns: 1fr auto; gap:0; font-size:14px; margin-top:14px;
  background:#f5f8fb; border-radius:12px; padding:4px 14px; box-shadow: inset 0 1px 3px rgba(16,32,56,.08);}}
.kv > div {{padding:8px 0; border-bottom:1px solid #e3e9f0;}}
.kv > div:nth-last-child(-n+2) {{border-bottom:none;}}
.kv > div:nth-child(even) {{text-align:right; padding-left:12px;}}
.alerts {{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:14px;}}
.alert {{border-radius:14px; padding:10px 12px; font-size:14px; background:#fff;
  box-shadow: 0 6px 16px -8px rgba(16,32,56,.3); border:1px solid rgba(15,40,70,.06);}}
.alert.on {{background:linear-gradient(180deg,#fff,#fdeeee); box-shadow: inset 4px 0 0 {C['red']}, 0 8px 18px -8px rgba(213,40,27,.45);}}
.alert.off {{background:linear-gradient(180deg,#fff,#ecf6f0); box-shadow: inset 4px 0 0 {C['green']}, 0 8px 18px -8px rgba(0,127,59,.35);}}
.alert b {{display:block; margin-bottom:2px;}}
.muted {{color:{C['sub']}; font-size:13px;}}
.factor-up {{color:{C['red']};}} .factor-down {{color:{C['blue']};}}
ul.tight {{margin:4px 0 0 0; padding:0; list-style:none;}}
ul.tight li {{margin:5px 0; font-size:14px; padding:6px 10px; border-radius:10px; background:#f5f8fb;}}
.err {{border-radius:14px; background:linear-gradient(180deg,#fff,#fdeeee); padding:14px 16px;
  box-shadow: inset 4px 0 0 {C['red']}, 0 10px 22px -10px rgba(213,40,27,.5);}}

/* 입력 */
.varname {{padding-top:22px; font-weight:800; font-size:15px; color:{C['text']};}}
.varname .unit {{display:block; font-weight:500; font-size:12px; color:{C['sub']};}}
.ctx {{font-size:14px; color:{C['text']}; background:#fff; border-radius:12px; padding:10px 14px; box-shadow: var(--lift);}}
.empty {{text-align:center; padding:34px 18px;}}
.empty .big-ic {{width:64px; height:64px; margin:0 auto 10px; border-radius:20px; display:grid; place-items:center;
  background:linear-gradient(145deg,#e6f0fb,#cfe2f6); color:{C['blue']}; box-shadow: inset 0 1px 0 #fff, 0 10px 20px -8px rgba(0,94,184,.5);}}
"""

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": C["sub"],
                     "axes.labelcolor": C["text"], "xtick.color": C["sub"], "ytick.color": C["sub"],
                     "axes.grid": True, "grid.color": "#e5e9eb", "grid.linewidth": 0.8, "axes.axisbelow": True})


# ---------------------------------------------------------------- 입력 → 예측
def parse(text, label, errors):
    """텍스트 칸 → float 또는 None. 숫자가 아니면 오류 목록에 남긴다."""
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        errors.append(f"{label}: '{text}'는 숫자가 아닙니다.")
        return None


def collect(values):
    """화면 입력 리스트 → (build_row용 dict, 입력 형식 오류)."""
    age, gender, icu, *nums = values
    errors = []
    patient = dict(age=parse(age, "나이", errors), gender={"남": "M", "여": "F"}.get(gender), icu=icu, vars={})
    for i, (key, _, name, *_) in enumerate(core.VARIABLES):
        block = nums[i * 4:(i + 1) * 4]
        patient["vars"][key] = {f: parse(v, f"{name} {FIELD_KO[f]}", errors) for f, v in zip(FIELDS, block)}
    return patient, errors


def calibration_row(prob):
    for r in ASSETS["calibration"]:
        if prob <= r["upper"] or r is ASSETS["calibration"][-1]:
            return r


def result_card(prob, summary):
    label, color = core.risk_band(prob)
    pct = core.percentile(prob, ASSETS)
    cal = calibration_row(prob)
    ratio = prob / MEAN_RISK

    def alert(t):
        on = prob >= t
        s = TH[t]
        head = '▲ 경보 대상' if on else '● 경보 아님'
        return (f"<div class='alert {'on' if on else 'off'}'><span class='muted'>경보 기준 {t:.0%}</span><b>{head}</b>"
                f"<span class='muted'>테스트: 사망자 {s['sensitivity']:.0%} 포착 · 오경보 {1 - s['specificity']:.0%}</span></div>")

    unmeasured = [core.VAR[k][2] for k, v in summary.items() if v is None]
    miss = ("미측정으로 계산한 검사: " + ", ".join(unmeasured)) if unmeasured else "19개 변수 모두 입력됨"
    shape = core.RISK_SHAPE[label]
    frac = min(max(prob, 0.0), 1.0)
    circ = 2 * np.pi * 66
    light = {"높음": "#ff6b5e", "보통": "#e08a2e", "낮음": "#2fbf71"}[label]
    return f"""
<div class="card" role="region" aria-label="28일 사망 위험 결과">
  <div class="hero">
    <div class="ring" role="img" aria-label="28일 사망 예측 확률 {prob:.1%}">
      <svg viewBox="0 0 168 168">
        <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="{light}"/><stop offset="100%" stop-color="{color}"/></linearGradient></defs>
        <circle cx="84" cy="84" r="66" fill="none" stroke="#e6edf5" stroke-width="16"/>
        <circle cx="84" cy="84" r="66" fill="none" stroke="url(#rg)" stroke-width="16" stroke-linecap="round"
          stroke-dasharray="{circ * frac:.1f} {circ:.1f}"/>
      </svg>
      <div class="center"><div><div class="pct num" style="color:{color}">{prob:.1%}</div>
        <div class="cap">28일 사망 예측</div></div></div>
    </div>
    <div style="flex:1;min-width:0">
      <div class="risk-label">28일 내 사망 예측 확률</div>
      <div class="badge-wrap"><span class="badge" style="background:linear-gradient(145deg,{light},{color});color:#fff"
        aria-label="위험 {label}">{shape} 위험 {label}</span></div>
      <div class="scale"><span>● 낮음 &lt;15%</span><span>◆ 보통 15~30%</span><span>▲ 높음 ≥30%</span></div>
    </div>
  </div>
  <div class="kv">
    <div>학습 환자 평균 사망률({MEAN_RISK:.1%}) 대비</div><div><b>{ratio:.1f}배</b></div>
    <div>테스트 환자 4,517명 중 이 환자보다 예측이 낮은 환자</div><div><b>{pct:.0f}%</b></div>
    <div>비슷한 예측({cal['lower']:.0%}~{cal['upper']:.0%})을 받은 테스트 환자의 실제 사망률</div>
    <div><b>{cal['observed']:.0%}</b> <span class="muted">({cal['deaths']}/{cal['n']}명)</span></div>
  </div>
  <div class="alerts">{alert(0.3)}{alert(0.5)}</div>
  <div class="muted" style="margin-top:10px">{html.escape(miss)}</div>
</div>"""


def factor_figure(groups):
    items = sorted(groups.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10][::-1]
    names = [EN.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh(names, vals, color=[C["red"] if v > 0 else C["blue"] for v in vals],
            hatch=["" if v > 0 else "//" for v in vals], edgecolor="white", height=0.62)
    ax.axvline(0, color=C["text"], lw=1)
    ax.set_xlabel("Contribution to predicted risk (log-odds)\n<- lowers risk (blue, hatched)    raises risk (red) ->")
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def position_figure(prob):
    dist = np.asarray(ASSETS["test_predictions_sorted"])
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.hist(dist, bins=np.linspace(0, 1, 41), color="#aeb7bd", edgecolor="white")
    ax.axvspan(0.3, 1, color=C["pale_red"], zorder=0, label="Alert zone (>= 30%)")
    ax.axvline(prob, color=C["text"], lw=3, label=f"This patient ({prob:.1%})")
    ax.axvline(MEAN_RISK, color=C["sub"], ls="--", lw=1.2, label=f"Cohort mortality ({MEAN_RISK:.1%})")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0, 1); ax.set_xlabel("Predicted 28-day mortality probability")
    ax.set_ylabel("Test patients")
    ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def factor_text(groups, summary):
    def desc(k):
        if k == "ICU" or k in ("age", "gender"):
            return core.group_label(k)
        s = summary.get(k)
        if s is None:
            return f"{core.VAR[k][2]} (미측정)"
        u = core.VAR[k][3]
        return f"{core.VAR[k][2]} (최저 {s['min']:g} · 최고 {s['max']:g} {u})"
    up = [k for k, v in sorted(groups.items(), key=lambda kv: -kv[1]) if v > 0.02][:4]
    down = [k for k, v in sorted(groups.items(), key=lambda kv: kv[1]) if v < -0.02][:4]
    li = lambda ks, cls, sym: "".join(f"<li class='{cls}'>{sym} {html.escape(desc(k))}</li>" for k in ks) or "<li>없음</li>"
    return (f"<div class='card'><b>위험을 높인 요인</b><ul class='tight'>{li(up, 'factor-up', '▲')}</ul>"
            f"<b style='display:block;margin-top:8px'>위험을 낮춘 요인</b><ul class='tight'>{li(down, 'factor-down', '▼')}</ul>"
            "<div class='muted' style='margin-top:8px'>요인은 이 환자의 예측에 각 변수가 기여한 정도(SHAP)이며, "
            "원인이나 치료 효과를 뜻하지 않습니다.</div></div>")


def run(*values):
    patient, input_errors = collect(values)
    row, errors, notes, summary = core.build_row(patient, ASSETS)
    if any(e.startswith("나이:") for e in input_errors):
        errors = [e for e in errors if e != "나이를 입력하세요."]
    errors = input_errors + errors
    if errors:
        msg = "".join(f"<li>{html.escape(e)}</li>" for e in errors)
        card = (f"<div class='err' role='alert'><b>▲ 입력을 확인하세요 ({len(errors)}건)</b>"
                f"<ul class='tight' style='list-style:disc;margin-left:18px'>{msg}</ul></div>")
        return card, None, None, "", card
    prob, contrib = core.predict(BOOSTER, row, ASSETS)
    groups, _ = core.group_contributions(contrib, ASSETS)
    note_html = ""
    if notes:
        note_html = ("<div class='card'><b>입력 안내</b><ul class='tight'>"
                     + "".join(f"<li class='muted'>{html.escape(n)}</li>" for n in notes) + "</ul></div>")
    label, color = core.risk_band(prob)
    status = (f"<div class='ctx' role='status'>계산 완료 · 28일 사망 예측 "
              f"<b style='color:{color}'>{prob:.1%} ({core.RISK_SHAPE[label]} 위험 {label})</b> · 결과는 오른쪽</div>")
    return (result_card(prob, summary), factor_figure(groups), position_figure(prob),
            factor_text(groups, summary) + note_html, status)


def example_values(name):
    ex = ASSETS["examples"][name]
    out = [f"{ex['age']:g}", "남" if ex["gender"] == "M" else "여", ex["icu"]]
    for key, *_ in core.VARIABLES:
        v = ex["vars"].get(key, {})
        out += ["" if v.get(f) is None else f"{v[f]:g}" for f in FIELDS]
    return out


def clear_values():
    return ["", None, None] + [""] * (4 * len(core.VARIABLES))


# ---------------------------------------------------------------- 일괄 예측
TEMPLATE_COLS = ["patient_id", "age", "sex(M/F)", "first_careunit"] + [
    f"{k}_{f}" for k, *_ in core.VARIABLES for f in FIELDS]


def template_file():
    path = tempfile.NamedTemporaryFile(delete=False, suffix="_template.csv").name
    ex = ASSETS["examples"]
    rows = []
    for i, (name, prof) in enumerate(ex.items()):
        r = {"patient_id": f"example_{i + 1}", "age": prof["age"], "sex(M/F)": prof["gender"],
             "first_careunit": prof["icu"]}
        for k, *_ in core.VARIABLES:
            for f in FIELDS:
                r[f"{k}_{f}"] = prof["vars"].get(k, {}).get(f)
        rows.append(r)
    pd.DataFrame(rows, columns=TEMPLATE_COLS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def batch(file):
    if file is None:
        return None, None
    df = pd.read_csv(file if isinstance(file, str) else file.name)
    missing = [c for c in ("age", "sex(M/F)", "first_careunit") if c not in df.columns]
    if missing:
        raise gr.Error(f"필수 열이 없습니다: {missing}. 양식 파일을 내려받아 쓰세요.")
    results = []
    for _, r in df.iterrows():
        input_errors = []
        cell = lambda c: None if pd.isna(r.get(c)) else r.get(c)
        patient = dict(age=parse(cell("age"), "age", input_errors),
                       gender=str(r.get("sex(M/F)")).strip().upper(),
                       icu=str(r.get("first_careunit")).strip(), vars={})
        for k, *_ in core.VARIABLES:
            patient["vars"][k] = {f: parse(cell(f"{k}_{f}"), f"{k}_{f}", input_errors) for f in FIELDS}
        row, errors, _, _ = core.build_row(patient, ASSETS)
        if input_errors:
            errors = input_errors + [e for e in errors if e != "나이를 입력하세요."]
        if errors:
            results.append(dict(patient_id=r.get("patient_id"), probability=None, risk=None,
                                alert_30=None, problem=" / ".join(errors)))
            continue
        prob, contrib = core.predict(BOOSTER, row, ASSETS)
        groups, _ = core.group_contributions(contrib, ASSETS)
        top = [core.group_label(k) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])[:3] if v > 0]
        results.append(dict(patient_id=r.get("patient_id"), probability=round(prob, 4),
                            risk=core.risk_band(prob)[0], alert_30="예" if prob >= 0.3 else "아니오",
                            top_factors=", ".join(top), problem=""))
    out = pd.DataFrame(results)
    path = tempfile.NamedTemporaryFile(delete=False, suffix="_predictions.csv").name
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, path


# ---------------------------------------------------------------- 모델 정보
def calibration_figure():
    cal = ASSETS["calibration"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot([c["mean_predicted"] for c in cal], [c["observed"] for c in cal], "o-", color=C["blue"],
            label="XGBoost (10 groups)")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed 28-day mortality")
    ax.set_title("Calibration (test set, n=4,517)"); ax.legend(loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); fig.tight_layout()
    return fig


def threshold_table():
    rows = []
    for t in ASSETS["thresholds"]:
        rows.append({"경보 기준": f"{t['threshold']:.0%}", "사망자 포착(민감도)": f"{t['sensitivity']:.1%}",
                     "생존자 오경보": f"{1 - t['specificity']:.1%}", "경보 중 실제 사망(PPV)": f"{t['ppv']:.1%}",
                     "전체 중 경보 비율": f"{t['alert_rate']:.1%}",
                     "놓친 사망자": f"{t['fn']} / {t['fn'] + t['tp']}명"})
    return pd.DataFrame(rows)


MODEL_MD = f"""
### 이 대시보드가 쓰는 모델
- **XGBoost** · 입력 150개: 나이·성별, 활력징후 7종과 혈액검사 12종의 첫 24시간 최저·평균·최고·표준편차·측정 횟수·측정 여부, 표준편차 결측 여부 19개, 첫 ICU 유형 15개
- 빈 값은 채우지 않고 그대로 두고(XGBoost가 직접 처리), 측정하지 않았다는 사실 자체를 입력으로 씁니다.
- 학습: MIMIC-IV 첫 ICU 입실 성인 **{ASSETS['n_development']:,}명** (28일 사망률 {ASSETS['development_mortality']:.1%})
- 평가: 학습에 쓰지 않은 **{ASSETS['n_test']:,}명** (사망 {ASSETS['test_deaths']:,}명)

| 지표 | 값 | 의미 |
|---|---|---|
| ROC-AUC | **{M['roc_auc']:.3f}** | 사망·생존 환자를 한 명씩 뽑으면 약 {M['roc_auc']:.0%}에서 사망 환자에게 더 높은 위험을 줌 |
| PR-AUC (AP) | **{M['average_precision']:.3f}** | 무작위 예측 기준 {ASSETS['test_mortality']:.3f}의 약 {M['average_precision'] / ASSETS['test_mortality']:.1f}배 |
| Brier | **{M['brier_score']:.3f}** | 모두에게 평균 사망률을 준 예측({ASSETS['brier_reference']:.3f})보다 오차 {1 - M['brier_score'] / ASSETS['brier_reference']:.0%} 감소 |

### 입력 방법
- 각 변수에 **입실 후 첫 24시간**의 최저·최고값을 넣습니다. 평균을 비우면 최저와 최고의 중간값, 측정 횟수를 비우면 학습 환자의 중앙 횟수를 씁니다.
- 검사를 하지 않았다면 칸을 모두 비워 두세요. "미측정"으로 계산되며, 학습 데이터에서도 미측정 여부가 위험과 관련 있었습니다.
- 표준편차는 최저·최고 범위로 추정합니다. 테스트 환자에서 실제 표준편차 대신 추정값을 넣었을 때 ROC-AUC {ASSETS['std_estimation_check']['roc_auc']:.3f}, 예측 차이 중앙값 {ASSETS['std_estimation_check']['median_abs_diff'] * 100:.1f}%p였습니다.

### 한계 — 반드시 읽어 주세요
1. 한 병원(Beth Israel Deaconess Medical Center)의 2008~2019년 자료로 만든 모델입니다. 다른 병원·다른 시기에는 성능이 다를 수 있고 외부 검증을 하지 않았습니다.
2. 같은 테스트 환자를 이전 모델 평가에도 썼기 때문에 완전히 처음 보는 데이터의 성능은 아닙니다.
3. 입실 24시간 전에 퇴실·사망한 환자와 28일 결과를 확인할 수 없는 환자는 학습에서 빠졌습니다.
4. APACHE II·SAPS II 같은 기존 중증도 점수와 직접 비교하지 못했습니다.
5. 요인 그래프는 예측에 대한 기여이며 원인이나 치료 효과가 아닙니다.
"""

DISCLAIMER = ("<b>주의 · 교육용 연구 모델입니다. 실제 진료·치료 결정에 사용할 수 없습니다.</b> "
              "임상 판단을 대신하지 않으며, 의료기기로 허가받지 않았습니다.")


def sec(no, title, hint=""):
    return gr.HTML(f"<div class='sec'><span class='no'>{no}</span><span class='t'>{title}</span>"
                   f"{f'<span class=hint>{hint}</span>' if hint else ''}</div>")


def icon(path, size=22):
    """Lucide 아이콘(선 그림) — 장식용이므로 스크린리더에서 숨긴다."""
    return (f"<svg aria-hidden='true' width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
            f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{path}</svg>")


IC_HEART = ("<path d='M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5"
            "c0 2.3 1.5 4.05 3 5.5l7 7Z'/><path d='M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27'/>")
IC_USERS = ("<path d='M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2'/><circle cx='9' cy='7' r='4'/>"
            "<path d='M22 21v-2a4 4 0 0 0-3-3.87'/><path d='M16 3.13a4 4 0 0 1 0 7.75'/>")
IC_ACT = "<polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/>"
IC_TARGET = "<circle cx='12' cy='12' r='10'/><circle cx='12' cy='12' r='6'/><circle cx='12' cy='12' r='2'/>"
IC_GAUGE = "<path d='m12 14 4-4'/><path d='M3.34 19a10 10 0 1 1 17.32 0'/>"


# ---------------------------------------------------------------- 화면
with gr.Blocks(title="ICU 28일 사망 위험 대시보드") as demo:
    gr.HTML(f"""
<header class="topbar">
  <div class="brand"><div class="logo">{icon(IC_HEART, 26)}</div>
    <div><div class="sys">ICU Clinical Decision Support · Research Prototype</div>
    <h1>ICU 28일 사망 위험 평가</h1></div></div>
  <div class="perf">MIMIC-IV · 첫 ICU 입실 성인<br>입실 후 첫 24시간 활력징후·혈액검사</div>
</header>
<div class="tiles">
  <div class="tile"><div class="ic">{icon(IC_USERS)}</div><div><div class="v num">{ASSETS['n_development']:,}명</div><div class="l">학습 환자 (사망률 {MEAN_RISK:.1%})</div></div></div>
  <div class="tile"><div class="ic">{icon(IC_TARGET)}</div><div><div class="v num">{ASSETS['n_test']:,}명</div><div class="l">테스트 환자 (사망 {ASSETS['test_deaths']:,}명)</div></div></div>
  <div class="tile"><div class="ic">{icon(IC_ACT)}</div><div><div class="v num">{M['roc_auc']:.3f}</div><div class="l">ROC-AUC · 순위 구분력</div></div></div>
  <div class="tile"><div class="ic">{icon(IC_GAUGE)}</div><div><div class="v num">{M['average_precision']:.3f}</div><div class="l">PR-AUC · 무작위의 {M['average_precision'] / ASSETS['test_mortality']:.1f}배</div></div></div>
</div>
<div class="banner" role="note">{DISCLAIMER}</div>""")

    with gr.Tabs():
        with gr.Tab("환자 위험 평가"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=11):
                    sec(1, "환자 정보")
                    with gr.Row():
                        age = gr.Textbox(label="나이 (세)", placeholder="예: 65", max_lines=1)
                        sex = gr.Radio(["남", "여"], label="성별")
                        icu = gr.Dropdown([(core.ICU_KO.get(c, c), c) for c in ASSETS["icu_categories"]],
                                          label="첫 ICU 유형", value=None)
                    with gr.Row():
                        ex_buttons = [gr.Button(f"예시 {n.replace('가상 환자 ', '')}", size="sm") for n in ASSETS["examples"]]
                        clear_btn = gr.Button("모두 지우기", size="sm", variant="secondary")
                    sec(2, "입실 후 첫 24시간 측정값", "검사하지 않았으면 비워 두세요 · 평균·횟수는 비워도 됩니다")
                    num_inputs = []
                    with gr.Tabs():
                        for kind, title in (("vital", "활력징후 7종"), ("lab", "혈액검사 12종")):
                            with gr.Tab(title):
                                for key, k2, name, unit, _ in core.VARIABLES:
                                    if k2 != kind:
                                        continue
                                    st = ASSETS["variables"][key]
                                    q = st["quantiles"]
                                    hint = {"min": f"예: {q['min']['p50']:.4g}", "mean": "비우면 중간값",
                                            "max": f"예: {q['max']['p50']:.4g}", "count": f"비우면 {st['count_median']}회"}
                                    with gr.Row(equal_height=True):
                                        gr.HTML(f"<div class='varname'>{name}<span class='unit'>{unit}</span></div>",
                                                min_width=110)
                                        for f in FIELDS:
                                            num_inputs.append(gr.Textbox(label=FIELD_KO[f], placeholder=hint[f],
                                                                         max_lines=1, min_width=90))
                    run_btn = gr.Button("28일 사망 위험 계산", variant="primary", size="lg")
                    status = gr.HTML()
                with gr.Column(scale=9):
                    sec(3, "결과")
                    out_card = gr.HTML(f"<div class='card empty'><div class='big-ic'>{icon(IC_HEART, 32)}</div>"
                                       "<b>환자 정보를 넣고 ‘28일 사망 위험 계산’을 누르세요</b>"
                                       "<div class='muted' style='margin-top:4px'>예시 버튼을 누르면 바로 결과가 나옵니다.</div></div>")
                    out_factor = gr.Plot(label="이 환자의 위험 요인")
                    out_text = gr.HTML()
                    out_pos = gr.Plot(label="테스트 환자 분포에서의 위치")

            all_inputs = [age, sex, icu] + num_inputs
            outputs = [out_card, out_factor, out_pos, out_text, status]
            run_btn.click(run, all_inputs, outputs)
            for b, name in zip(ex_buttons, ASSETS["examples"]):
                b.click(lambda n=name: example_values(n), None, all_inputs).then(run, all_inputs, outputs)
            clear_btn.click(clear_values, None, all_inputs)

        with gr.Tab("여러 환자 한 번에 (CSV)"):
            gr.Markdown("양식 파일을 내려받아 한 줄에 환자 한 명씩 채운 뒤 올리면, 환자별 예측 확률·위험 수준·주요 요인을 표로 돌려줍니다. "
                        "열 이름은 `변수_min`, `변수_mean`, `변수_max`, `변수_count` 형식이며 미측정 검사는 비워 둡니다. "
                        "파일은 이 컴퓨터에서만 처리되고 저장되지 않습니다.")
            with gr.Row():
                tpl_btn = gr.Button("양식 파일 만들기 (예시 2명 포함)")
                tpl_file = gr.File(label="양식")
            up = gr.File(label="환자 CSV 올리기", file_types=[".csv"])
            table = gr.Dataframe(label="예측 결과", wrap=True)
            result_file = gr.File(label="결과 내려받기")
            tpl_btn.click(template_file, None, tpl_file)
            up.change(batch, up, [table, result_file])

        with gr.Tab("모델 정보·한계"):
            gr.Markdown(MODEL_MD)
            with gr.Row():
                gr.Image(str(core.OUT / "roc.png"), show_label=False)
                gr.Image(str(core.OUT / "pr.png"), show_label=False)
                gr.Plot(calibration_figure(), show_label=False)
            gr.Markdown("#### 경보 기준별 테스트 성능 (사망 1,005명 · 생존 3,512명)")
            gr.Dataframe(threshold_table(), interactive=False)

    gr.HTML(f"<div class='muted' style='text-align:center;margin-top:8px'>{DISCLAIMER}</div>")


if __name__ == "__main__":
    import argparse
    import os
    import secrets

    ap = argparse.ArgumentParser(description="ICU 28일 사망 위험 대시보드")
    ap.add_argument("--share", action="store_true",
                    help="팀원 확인용 임시 인터넷 링크(비밀번호 필요)를 만든다. 확인이 끝나면 바로 끈다.")
    args = ap.parse_args()
    options = dict(css=CSS, theme=THEME, head=HEAD, server_name="127.0.0.1", server_port=7860)
    if args.share:
        # 제한 데이터로 학습한 모델이므로 공개하지 않는다: 팀원만 아는 아이디·비밀번호로 막고, 짧게만 연다.
        user = os.environ.get("DASH_USER", "team2")
        password = os.environ.get("DASH_PASS") or secrets.token_urlsafe(9)
        print(f"\n[팀원 전용 접속] 아이디: {user}  비밀번호: {password}")
        print("링크와 비밀번호는 팀원에게만 보내고, 확인이 끝나면 이 창에서 Ctrl+C로 끄세요.\n")
        options.update(share=True, auth=(user, password),
                       auth_message="1트랙 2팀 팀원 전용 · 교육용 연구 모델, 실제 진료에 사용할 수 없음")
    demo.launch(**options)
