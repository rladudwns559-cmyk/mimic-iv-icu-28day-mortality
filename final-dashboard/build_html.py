"""팀원 전용 단일 HTML 대시보드 만들기 (서버·설치 없이 브라우저에서 계산).

- outputs/model.ubj와 outputs/assets.json을 그대로 넣는다(다시 학습하지 않음).
- 계산은 dashboard_core.js(= dashboard_core.py와 같은 규칙)가 한다.
- 결과: outputs/ICU_28일_사망위험_대시보드_팀원전용.html
  MIMIC-IV 제한 데이터로 학습한 모델이 들어 있으므로 PhysioNet 권한이 있는 팀원에게만 보낸다.
"""
import base64
import json

import xgboost as xgb

import dashboard_core as core

OUT = core.OUT
HTML_PATH = OUT / "ICU_28일_사망위험_대시보드_팀원전용.html"


def export_model():
    b = xgb.Booster()
    b.load_model(str(OUT / "model.ubj"))
    j = json.loads(b.save_raw("json"))
    L = j["learner"]
    assert L["objective"]["name"] == "binary:logistic"
    base = float(L["learner_model_param"]["base_score"].strip("[]"))
    import math
    trees = []
    for t in L["gradient_booster"]["model"]["trees"]:
        assert all(s == 0 for s in t["split_type"]) and not t["categories"]
        trees.append([t["left_children"], t["right_children"], t["split_indices"], t["split_conditions"],
                      t["default_left"], t["sum_hessian"]])
    return dict(base_margin=math.log(base / (1 - base)), trees=trees)


def export_data():
    a = json.loads((OUT / "assets.json").read_text(encoding="utf-8"))
    var = {k: dict(kind=kind, name=name, unit=unit, lo=lim[0], hi=lim[1]) for k, kind, name, unit, lim in core.VARIABLES}
    keep = ["clinical_features", "std_columns", "icu_categories", "feature_names", "n_development", "n_test",
            "test_deaths", "development_mortality", "test_mortality", "metrics", "brier_reference", "thresholds",
            "calibration", "test_predictions_sorted", "examples", "example_probabilities", "std_estimation_check"]
    d = {k: a[k] for k in keep}
    d["variables"] = {k: dict(count_median=v["count_median"], std_range_ratio=v["std_range_ratio"],
                              quantiles={c: dict(p1=v["quantiles"][c]["p1"], p99=v["quantiles"][c]["p99"])
                                         for c in ("min", "max")})
                      for k, v in a["variables"].items()}
    d["VAR"] = var
    d["VAR_KEYS"] = [k for k, *_ in core.VARIABLES]
    d["ICU_KO"] = core.ICU_KO
    d["RISK_BANDS"] = core.RISK_BANDS
    d["RISK_SHAPE"] = core.RISK_SHAPE
    return d


def img64(name):
    return "data:image/png;base64," + base64.b64encode((OUT / name).read_bytes()).decode()


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>ICU 28일 사망 위험 대시보드 (팀원 전용)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<style>
:root{--text:#212b32;--sub:#4c6272;--blue:#005eb8;--red:#d5281b;--amber:#a35200;--green:#007f3b;--border:#d8dde0;
  --lift:0 1px 2px rgba(16,32,56,.06),0 12px 32px -10px rgba(16,32,56,.22)}
*{box-sizing:border-box}
body{margin:0;font-family:"Pretendard Variable",Pretendard,"Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",Arial,sans-serif;
  color:var(--text);background:linear-gradient(180deg,#e4ebf3 0%,#eef3f8 320px,#f3f6fa 100%);font-size:15px;line-height:1.5}
.wrap{max-width:1440px;margin:auto;padding:14px 16px 40px}
.num,td,.kv b{font-variant-numeric:tabular-nums}
:focus-visible{outline:3px solid #ffeb3b;outline-offset:1px}
.topbar{color:#fff;padding:20px 26px;border-radius:20px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;
  background:radial-gradient(1200px 300px at 85% -40%,rgba(80,170,255,.35),transparent 60%),linear-gradient(135deg,#0a2342 0%,#0b3a6e 55%,#005eb8 100%);
  box-shadow:0 18px 40px -16px rgba(5,30,70,.65)}
.topbar h1{font-size:24px;margin:0;font-weight:800;letter-spacing:-.4px}
.topbar .sys{font-size:11px;color:#bcd3ea;letter-spacing:1px;font-weight:600}
.topbar .perf{font-size:13px;color:#dbe8f5;text-align:right}
.tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:16px 0 4px}
.tile{background:linear-gradient(180deg,#fff,#f7f9fc);border-radius:16px;padding:14px 16px;box-shadow:var(--lift)}
.tile .v{font-size:22px;font-weight:800}.tile .l{font-size:12px;color:var(--sub);font-weight:600}
.banner{background:linear-gradient(90deg,#fdecea,#fff6f5);border-radius:14px;padding:11px 16px;font-size:14px;margin:12px 0 6px;
  box-shadow:inset 4px 0 0 var(--red)}
.team{background:#fff8e1;border-radius:12px;padding:8px 14px;font-size:13px;box-shadow:inset 4px 0 0 #a35200;margin-bottom:6px}
.tabs{display:flex;gap:6px;margin:14px 0;flex-wrap:wrap}
.tabs button{border:0;border-radius:999px;padding:7px 16px;font:inherit;font-weight:600;background:transparent;color:var(--sub);cursor:pointer}
.tabs button[aria-selected=true]{background:#fff;color:var(--blue);box-shadow:0 4px 12px -4px rgba(16,32,56,.25)}
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:18px}
@media (max-width:1000px){.grid{grid-template-columns:1fr}.tiles{grid-template-columns:repeat(2,1fr)}}
.card{background:linear-gradient(180deg,#fff,#fafcfe);border-radius:18px;padding:18px 20px;box-shadow:var(--lift);margin-bottom:14px}
.sec{display:flex;align-items:center;gap:10px;margin:4px 0 10px;flex-wrap:wrap}
.sec .no{width:28px;height:28px;border-radius:9px;display:grid;place-items:center;color:#fff;font-weight:800;font-size:14px;
  background:linear-gradient(145deg,#2a86e0,#004c96)}
.sec .t{font-size:17px;font-weight:800;white-space:nowrap}.hint{font-size:13px;color:var(--sub)}
label.f{display:block;font-size:13px;color:var(--sub);font-weight:600;margin-bottom:4px}
input[type=text],select{width:100%;font:inherit;padding:8px 10px;border:1px solid #cfd8e3;border-radius:10px;background:#f6f8fb;
  box-shadow:inset 0 2px 4px rgba(16,32,56,.08)}
input[type=text]:focus,select:focus{border-color:var(--blue);outline:none;box-shadow:0 0 0 3px rgba(0,94,184,.18)}
.row3{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:12px}
.btn{border:1px solid #cfd8e3;border-radius:10px;padding:7px 12px;font:inherit;font-size:13px;cursor:pointer;
  background:linear-gradient(180deg,#fff,#eef3f8);color:#0b3a6e}
.btn.primary{width:100%;border:0;color:#fff;font-size:16px;font-weight:700;padding:12px;margin-top:12px;
  background:linear-gradient(180deg,#1a74d1 0%,#005eb8 55%,#004c96 100%);box-shadow:0 6px 16px -4px rgba(0,94,184,.55)}
.btns{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.vtabs{display:flex;gap:6px;margin-bottom:8px}
.vrow{display:grid;grid-template-columns:120px repeat(4,1fr);gap:8px;align-items:end;padding:8px 0;border-bottom:1px solid #edf1f5}
.vrow .nm{font-weight:800;font-size:14px}.vrow .nm span{display:block;font-weight:500;font-size:12px;color:var(--sub)}
.vrow label{font-size:12px;color:var(--sub)}
.hero{display:flex;gap:20px;align-items:center;flex-wrap:wrap}
.ring{position:relative;width:168px;height:168px;flex:none}
.ring svg{width:168px;height:168px;transform:rotate(-90deg)}
.ring .center{position:absolute;inset:0;display:grid;place-items:center;text-align:center}
.pct{font-size:36px;font-weight:800;letter-spacing:-1px;line-height:1}.cap{font-size:11px;color:var(--sub);font-weight:600}
.badge{display:inline-block;padding:8px 14px;border-radius:999px;font-weight:800;font-size:17px;color:#fff;margin:8px 0 10px}
.scale span{font-size:12px;padding:3px 9px;border-radius:999px;background:#eef2f7;color:var(--sub);font-weight:600;margin-right:4px;display:inline-block;margin-bottom:4px}
.kv{display:grid;grid-template-columns:1fr auto;font-size:14px;margin-top:14px;background:#f5f8fb;border-radius:12px;padding:4px 14px}
.kv>div{padding:8px 0;border-bottom:1px solid #e3e9f0}.kv>div:nth-last-child(-n+2){border-bottom:0}
.kv>div:nth-child(even){text-align:right;padding-left:12px}
.alerts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
.alert{border-radius:14px;padding:10px 12px;font-size:14px;background:#fff}
.alert b{display:block}.alert.on{background:linear-gradient(180deg,#fff,#fdeeee);box-shadow:inset 4px 0 0 var(--red)}
.alert.off{background:linear-gradient(180deg,#fff,#ecf6f0);box-shadow:inset 4px 0 0 var(--green)}
.muted{color:var(--sub);font-size:13px}
ul.tight{list-style:none;margin:4px 0 0;padding:0}ul.tight li{margin:5px 0;font-size:14px;padding:6px 10px;border-radius:10px;background:#f5f8fb}
.up{color:var(--red)}.down{color:var(--blue)}
.err{border-radius:14px;background:linear-gradient(180deg,#fff,#fdeeee);padding:14px 16px;box-shadow:inset 4px 0 0 var(--red);margin-bottom:14px}
.err ul{list-style:disc;margin:6px 0 0 18px;padding:0}
.empty{text-align:center;padding:34px 18px;color:var(--sub)}
.status{font-size:14px;background:#fff;border-radius:12px;padding:10px 14px;margin-top:10px;box-shadow:var(--lift)}
table.t{border-collapse:collapse;width:100%;font-size:14px;background:#fff}
table.t th,table.t td{border:1px solid var(--border);padding:6px 8px;text-align:left}table.t th{background:#e9edf2}
.scroll{overflow-x:auto}
.imgs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.imgs img,.imgs svg{width:100%;height:auto;background:#fff;border-radius:12px}
@media (max-width:1000px){.imgs{grid-template-columns:1fr}}
.foot{text-align:center;font-size:13px;margin-top:20px}
[hidden]{display:none!important}
</style></head>
<body><div class="wrap">
<header class="topbar"><div><div class="sys">ICU CLINICAL DECISION SUPPORT · RESEARCH PROTOTYPE</div>
<h1>ICU 28일 사망 위험 평가</h1></div>
<div class="perf">MIMIC-IV · 첫 ICU 입실 성인<br>입실 후 첫 24시간 활력징후·혈액검사</div></header>
<div class="tiles" id="tiles"></div>
<div class="banner" id="disc1"></div>
<div class="team"><b>1트랙 2팀 팀원 전용</b> · MIMIC-IV 제한 데이터로 학습한 모델이 들어 있는 파일입니다. 팀원 외에는 보내거나 올리지 마세요.</div>

<div class="tabs" role="tablist">
  <button role="tab" aria-selected="true" data-tab="p1">환자 위험 평가</button>
  <button role="tab" aria-selected="false" data-tab="p2">여러 환자 한 번에 (CSV)</button>
  <button role="tab" aria-selected="false" data-tab="p3">모델 정보·한계</button>
</div>

<section id="p1"><div class="grid">
 <div>
  <div class="card">
   <div class="sec"><span class="no">1</span><span class="t">환자 정보</span></div>
   <div class="row3">
    <div><label class="f" for="age">나이 (세)</label><input type="text" id="age" placeholder="예: 65" inputmode="decimal"></div>
    <div><span class="f" style="display:block;font-size:13px;color:#4c6272;font-weight:600;margin-bottom:4px">성별</span>
      <label><input type="radio" name="sex" value="M"> 남</label> &nbsp; <label><input type="radio" name="sex" value="F"> 여</label></div>
    <div><label class="f" for="icu">첫 ICU 유형</label><select id="icu"><option value="">선택하세요</option></select></div>
   </div>
   <div class="btns" id="exbtns"></div>
  </div>
  <div class="card">
   <div class="sec"><span class="no">2</span><span class="t">입실 후 첫 24시간 측정값</span>
     <span class="hint">검사하지 않았으면 비워 두세요 · 평균·횟수는 비워도 됩니다</span></div>
   <div class="vtabs tabs" style="margin:0 0 6px">
     <button aria-selected="true" data-v="vital">활력징후 7종</button><button aria-selected="false" data-v="lab">혈액검사 12종</button></div>
   <div id="vars"></div>
   <button class="btn primary" id="run">28일 사망 위험 계산</button>
   <div id="status"></div>
  </div>
 </div>
 <div>
  <div class="sec"><span class="no">3</span><span class="t">결과</span></div>
  <div id="result"><div class="card empty">환자 정보를 넣고 ‘28일 사망 위험 계산’을 누르세요.<br>예시 버튼을 누르면 바로 결과를 볼 수 있습니다.</div></div>
 </div>
</div></section>

<section id="p2" hidden><div class="card">
 <div class="sec"><span class="no">1</span><span class="t">양식 내려받기</span><span class="hint">예시 2명이 들어 있습니다. 같은 열 이름으로 채워 주세요.</span></div>
 <button class="btn" id="tpl">양식 파일 내려받기 (CSV)</button>
 <div class="sec" style="margin-top:18px"><span class="no">2</span><span class="t">채운 파일 올리기</span></div>
 <input type="file" id="csv" accept=".csv">
 <div id="batchout" style="margin-top:14px"></div>
</div></section>

<section id="p3" hidden>
 <div class="card" id="modelinfo"></div>
 <div class="card"><b>테스트 환자 4,517명 성능 그림</b><div class="imgs" style="margin-top:10px">
   <img alt="ROC curve" src="__ROC__"><img alt="Precision-recall curve" src="__PR__"><div id="calsvg"></div></div></div>
 <div class="card"><b>경보 기준별 성능 (테스트 환자)</b><div class="scroll" id="thtable" style="margin-top:8px"></div></div>
</section>

<div class="banner foot" id="disc2"></div>
</div>
<script>const DATA = __DATA__;</script>
<script>const MODEL = __MODEL__;</script>
<script>__CORE__</script>
<script>
"use strict";
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct1 = v => (v * 100).toFixed(1) + "%", pct0 = v => Math.round(v * 100) + "%";
const FIELDS = ["min", "mean", "max", "count"], FK = {min: "최저", mean: "평균", max: "최고", count: "측정 횟수"};
const EN = {ICU: "First ICU type", age: "Age", gender: "Sex"};
const M = DATA.metrics, MEAN = DATA.development_mortality;
const TH = {}; DATA.thresholds.forEach(t => TH[Math.round(t.threshold * 100) / 100] = t);
const DISC = "<b>주의 · 교육용 연구 모델입니다. 실제 진료·치료 결정에 사용할 수 없습니다.</b> 임상 판단을 대신하지 않으며, 의료기기로 허가받지 않았습니다.";
$("#disc1").innerHTML = DISC; $("#disc2").innerHTML = DISC;
$("#tiles").innerHTML = [
  [DATA.n_development.toLocaleString() + "명", `학습 환자 (사망률 ${pct1(MEAN)})`],
  [DATA.n_test.toLocaleString() + "명", `테스트 환자 (사망 ${DATA.test_deaths.toLocaleString()}명)`],
  [M.roc_auc.toFixed(3), "ROC-AUC · 순위 구분력"],
  [M.average_precision.toFixed(3), `PR-AUC · 무작위의 ${(M.average_precision / DATA.test_mortality).toFixed(1)}배`]
].map(([v, l]) => `<div class="tile"><div class="v num">${v}</div><div class="l">${l}</div></div>`).join("");

// 탭
document.querySelectorAll(".tabs[role=tablist] button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs[role=tablist] button").forEach(x => x.setAttribute("aria-selected", x === b));
  ["p1", "p2", "p3"].forEach(id => $("#" + id).hidden = id !== b.dataset.tab);
});
// ICU 목록
DATA.icu_categories.forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = DATA.ICU_KO[c] || c; $("#icu").append(o); });
// 측정값 입력칸
let html = "";
for (const k of DATA.VAR_KEYS) {
  const v = DATA.VAR[k];
  html += `<div class="vrow" data-kind="${v.kind}"><div class="nm">${esc(v.name)}<span>${esc(v.unit)}</span></div>` +
    FIELDS.map(f => `<div><label for="${k}_${f}">${FK[f]}</label><input type="text" id="${k}_${f}" inputmode="decimal" aria-label="${esc(v.name)} ${FK[f]}"></div>`).join("") + "</div>";
}
$("#vars").innerHTML = html;
function showKind(kind) {
  document.querySelectorAll(".vrow").forEach(r => r.hidden = r.dataset.kind !== kind);
  document.querySelectorAll(".vtabs button").forEach(b => b.setAttribute("aria-selected", b.dataset.v === kind));
}
document.querySelectorAll(".vtabs button").forEach(b => b.onclick = () => showKind(b.dataset.v));
showKind("vital");
// 예시 버튼
Object.entries(DATA.examples).forEach(([name, ex]) => {
  const b = document.createElement("button"); b.className = "btn"; b.textContent = "예시 " + name.replace("가상 환자 ", "");
  b.onclick = () => fill(ex); $("#exbtns").append(b);
});
const clr = document.createElement("button"); clr.className = "btn"; clr.textContent = "모두 지우기";
clr.onclick = () => { fill(null); $("#result").innerHTML = '<div class="card empty">환자 정보를 넣고 ‘28일 사망 위험 계산’을 누르세요.</div>'; $("#status").innerHTML = ""; };
$("#exbtns").append(clr);
function fill(ex) {
  $("#age").value = ex ? fmtG(ex.age) : "";
  document.querySelectorAll("input[name=sex]").forEach(r => r.checked = ex ? r.value === ex.gender : false);
  $("#icu").value = ex ? ex.icu : "";
  for (const k of DATA.VAR_KEYS) for (const f of FIELDS) {
    const v = ex && ex.vars[k] ? ex.vars[k][f] : null;
    $(`#${k}_${f}`).value = v === null || v === undefined ? "" : fmtG(v);
  }
}
function parse(text, label, errors) {
  if (text === null || text === undefined) return null;
  const s = String(text).trim().replace(/,/g, "");
  if (!s) return null;
  const v = Number(s);
  if (Number.isNaN(v)) { errors.push(`${label}: '${text}'는 숫자가 아닙니다.`); return null; }
  return v;
}
function readForm() {
  const errors = [];
  const sex = document.querySelector("input[name=sex]:checked");
  const p = {age: parse($("#age").value, "나이", errors), gender: sex ? sex.value : null, icu: $("#icu").value || null, vars: {}};
  for (const k of DATA.VAR_KEYS) {
    p.vars[k] = {};
    for (const f of FIELDS) p.vars[k][f] = parse($(`#${k}_${f}`).value, `${DATA.VAR[k].name} ${FK[f]}`, errors);
  }
  return [p, errors];
}
function evaluate(p, inputErrors, batch) {
  const r = buildRow(p);
  let errors = r.errors;
  // app.py와 같은 규칙: 화면은 나이 형식 오류일 때, CSV는 형식 오류가 하나라도 있으면 '나이를 입력하세요'를 뺀다
  if (batch ? inputErrors.length : inputErrors.some(e => e.startsWith("나이:"))) errors = errors.filter(e => e !== "나이를 입력하세요.");
  errors = inputErrors.concat(errors);
  if (errors.length) return {errors};
  const prob = predictProb(r.row);
  const groups = groupContributions(contributions(r.row));
  return {prob, groups, notes: r.notes, summary: r.summary, errors: []};
}

// ---------- 결과 화면
function resultCard(prob, summary) {
  const [label, color] = riskBand(prob), shape = DATA.RISK_SHAPE[label];
  const light = {"높음": "#ff6b5e", "보통": "#e08a2e", "낮음": "#2fbf71"}[label];
  const cal = calibrationRow(prob), circ = 2 * Math.PI * 66, frac = Math.min(Math.max(prob, 0), 1);
  const alert = t => { const on = prob >= t, s = TH[t];
    return `<div class="alert ${on ? "on" : "off"}"><span class="muted">경보 기준 ${pct0(t)}</span><b>${on ? "▲ 경보 대상" : "● 경보 아님"}</b>` +
      `<span class="muted">테스트: 사망자 ${pct0(s.sensitivity)} 포착 · 오경보 ${pct0(1 - s.specificity)}</span></div>`; };
  const un = DATA.VAR_KEYS.filter(k => summary[k] === null).map(k => DATA.VAR[k].name);
  return `<div class="card" role="region" aria-label="28일 사망 위험 결과"><div class="hero">
   <div class="ring" role="img" aria-label="28일 사망 예측 확률 ${pct1(prob)}"><svg viewBox="0 0 168 168">
    <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="${light}"/><stop offset="100%" stop-color="${color}"/></linearGradient></defs>
    <circle cx="84" cy="84" r="66" fill="none" stroke="#e6edf5" stroke-width="16"/>
    <circle cx="84" cy="84" r="66" fill="none" stroke="url(#rg)" stroke-width="16" stroke-linecap="round" stroke-dasharray="${(circ * frac).toFixed(1)} ${circ.toFixed(1)}"/></svg>
    <div class="center"><div><div class="pct num" style="color:${color}">${pct1(prob)}</div><div class="cap">28일 사망 예측</div></div></div></div>
   <div style="flex:1;min-width:0"><div class="muted" style="font-weight:700">28일 내 사망 예측 확률</div>
    <span class="badge" style="background:linear-gradient(145deg,${light},${color})">${shape} 위험 ${label}</span>
    <div class="scale"><span>● 낮음 &lt;15%</span><span>◆ 보통 15~30%</span><span>▲ 높음 ≥30%</span></div></div></div>
   <div class="kv"><div>학습 환자 평균 사망률(${pct1(MEAN)}) 대비</div><div><b>${(prob / MEAN).toFixed(1)}배</b></div>
    <div>테스트 환자 4,517명 중 이 환자보다 예측이 낮은 환자</div><div><b>${Math.round(percentile(prob))}%</b></div>
    <div>비슷한 예측(${pct0(cal.lower)}~${pct0(cal.upper)})을 받은 테스트 환자의 실제 사망률</div>
    <div><b>${pct0(cal.observed)}</b> <span class="muted">(${cal.deaths}/${cal.n}명)</span></div></div>
   <div class="alerts">${alert(0.3)}${alert(0.5)}</div>
   <div class="muted" style="margin-top:10px">${esc(un.length ? "미측정으로 계산한 검사: " + un.join(", ") : "19개 변수 모두 입력됨")}</div></div>`;
}
function factorSvg(groups) {
  const items = Object.entries(groups).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10);
  const mx = Math.max(...items.map(i => Math.abs(i[1])), 1e-9), W = 620, rowH = 26, left = 150, half = (W - left - 110) / 2, mid = left + 55 + half;
  let s = `<svg viewBox="0 0 ${W} ${items.length * rowH + 50}" role="img" aria-label="Contribution chart"><line x1="${mid}" y1="4" x2="${mid}" y2="${items.length * rowH + 6}" stroke="#212b32"/>`;
  s += `<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#005eb8"/><line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2"/></pattern></defs>`;
  items.forEach(([k, v], i) => {
    const y = 8 + i * rowH, w = Math.abs(v) / mx * half, x = v >= 0 ? mid : mid - w;
    s += `<text x="${left - 8}" y="${y + 14}" text-anchor="end" font-size="13" fill="#212b32">${esc(EN[k] || k)}</text>`;
    s += `<rect x="${x}" y="${y}" width="${Math.max(w, 1)}" height="18" fill="${v >= 0 ? "#d5281b" : "url(#hatch)"}"/>`;
    s += `<text x="${v >= 0 ? x + w + 4 : x - 4}" y="${y + 14}" font-size="11" fill="#4c6272" text-anchor="${v >= 0 ? "start" : "end"}">${(v >= 0 ? "+" : "") + v.toFixed(2)}</text>`;
  });
  const yb = items.length * rowH + 26;
  s += `<text x="${mid}" y="${yb}" text-anchor="middle" font-size="12" fill="#212b32">Contribution to predicted risk (log-odds)</text>`;
  s += `<text x="${mid}" y="${yb + 16}" text-anchor="middle" font-size="11" fill="#4c6272">&lt;- lowers risk (blue, hatched)    raises risk (red) -&gt;</text></svg>`;
  return s;
}
function positionSvg(prob) {
  const bins = new Array(40).fill(0);
  DATA.test_predictions_sorted.forEach(v => bins[Math.min(39, Math.floor(v * 40))]++);
  const mx = Math.max(...bins), W = 620, H = 220, l = 50, r = 16, t = 34, b = 36, pw = W - l - r, ph = H - t - b;
  const X = v => l + v * pw;
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Position among test patients">`;
  s += `<rect x="${X(0.3)}" y="${t}" width="${X(1) - X(0.3)}" height="${ph}" fill="#fdecea"/>`;
  bins.forEach((c, i) => { const h = c / mx * ph; s += `<rect x="${X(i / 40) + 0.5}" y="${t + ph - h}" width="${pw / 40 - 1}" height="${h}" fill="#aeb7bd"/>`; });
  s += `<line x1="${X(MEAN)}" y1="${t}" x2="${X(MEAN)}" y2="${t + ph}" stroke="#4c6272" stroke-dasharray="4 3"/>`;
  s += `<line x1="${X(prob)}" y1="${t - 4}" x2="${X(prob)}" y2="${t + ph}" stroke="#212b32" stroke-width="3"/>`;
  s += `<line x1="${l}" y1="${t + ph}" x2="${W - r}" y2="${t + ph}" stroke="#212b32"/>`;
  for (let v = 0; v <= 1.001; v += 0.2) s += `<text x="${X(v)}" y="${t + ph + 16}" font-size="11" text-anchor="middle" fill="#212b32">${v.toFixed(1)}</text>`;
  s += `<text x="${l + pw / 2}" y="${H - 4}" font-size="12" text-anchor="middle" fill="#212b32">Predicted 28-day mortality probability</text>`;
  s += `<text x="14" y="${t + ph / 2}" font-size="12" text-anchor="middle" fill="#212b32" transform="rotate(-90 14 ${t + ph / 2})">Test patients</text>`;
  s += `<text x="${l}" y="14" font-size="11" fill="#212b32">■ This patient (${pct1(prob)})   ┄ Cohort mortality (${pct1(MEAN)})   <tspan fill="#d5281b">■ Alert zone (&gt;= 30%)</tspan></text></svg>`;
  return s;
}
function factorList(groups, summary) {
  const desc = k => {
    if (k === "ICU" || k === "age" || k === "gender") return groupLabel(k);
    const s = summary[k];
    if (!s) return `${DATA.VAR[k].name} (미측정)`;
    return `${DATA.VAR[k].name} (최저 ${fmtG(s.min)} · 최고 ${fmtG(s.max)} ${DATA.VAR[k].unit})`;
  };
  const e = Object.entries(groups);
  const up = e.filter(x => x[1] > 0.02).sort((a, b) => b[1] - a[1]).slice(0, 4).map(x => x[0]);
  const dn = e.filter(x => x[1] < -0.02).sort((a, b) => a[1] - b[1]).slice(0, 4).map(x => x[0]);
  const li = (ks, cls, sym) => ks.length ? ks.map(k => `<li class="${cls}">${sym} ${esc(desc(k))}</li>`).join("") : "<li>없음</li>";
  return `<div class="card"><b>위험을 높인 요인</b><ul class="tight">${li(up, "up", "▲")}</ul>` +
    `<b style="display:block;margin-top:8px">위험을 낮춘 요인</b><ul class="tight">${li(dn, "down", "▼")}</ul>` +
    `<div class="muted" style="margin-top:8px">요인은 이 환자의 예측에 각 변수가 기여한 정도(SHAP)이며, 원인이나 치료 효과를 뜻하지 않습니다.</div></div>`;
}
$("#run").onclick = () => {
  const [p, inErr] = readForm();
  const r = evaluate(p, inErr);
  if (r.errors.length) {
    const card = `<div class="err" role="alert"><b>▲ 입력을 확인하세요 (${r.errors.length}건)</b><ul>${r.errors.map(e => `<li>${esc(e)}</li>`).join("")}</ul></div>`;
    $("#result").innerHTML = card; $("#status").innerHTML = card; return;
  }
  const [label, color] = riskBand(r.prob);
  const notes = r.notes.length ? `<div class="card"><b>입력 안내</b><ul class="tight">${r.notes.map(n => `<li class="muted">${esc(n)}</li>`).join("")}</ul></div>` : "";
  $("#result").innerHTML = resultCard(r.prob, r.summary) +
    `<div class="card"><b>이 환자의 위험 요인</b>${factorSvg(r.groups)}</div>` + factorList(r.groups, r.summary) + notes +
    `<div class="card"><b>테스트 환자 분포에서의 위치</b>${positionSvg(r.prob)}</div>`;
  $("#status").innerHTML = `<div class="status" role="status">계산 완료 · 28일 사망 예측 <b style="color:${color}">${pct1(r.prob)} (${DATA.RISK_SHAPE[label]} 위험 ${label})</b> · 결과는 오른쪽</div>`;
};

// ---------- CSV
const COLS = ["patient_id", "age", "sex(M/F)", "first_careunit"].concat(DATA.VAR_KEYS.flatMap(k => FIELDS.map(f => `${k}_${f}`)));
function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + text], {type: "text/csv;charset=utf-8"}));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const csvCell = v => { if (v === null || v === undefined) return ""; const s = String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
$("#tpl").onclick = () => {
  const rows = Object.values(DATA.examples).map((ex, i) => [`example_${i + 1}`, ex.age, ex.gender, ex.icu]
    .concat(DATA.VAR_KEYS.flatMap(k => FIELDS.map(f => (ex.vars[k] || {})[f] ?? "")) ));
  download("icu_28day_template.csv", [COLS].concat(rows).map(r => r.map(csvCell).join(",")).join("\n"));
};
function parseCsv(text) {
  const rows = []; let row = [], cell = "", q = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.some(x => x.trim() !== ""));
}
$("#csv").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const rows = parseCsv(await f.text());
  const head = rows[0].map(h => h.trim()), body = rows.slice(1);
  const miss = ["age", "sex(M/F)", "first_careunit"].filter(c => !head.includes(c));
  if (miss.length) { $("#batchout").innerHTML = `<div class="err">필수 열이 없습니다: ${esc(miss.join(", "))}. 양식 파일을 내려받아 쓰세요.</div>`; return; }
  const out = [];
  for (const r of body) {
    const get = c => { const i = head.indexOf(c); return i < 0 ? null : (r[i] ?? "").trim() || null; };
    const errs = [];
    const p = {age: parse(get("age"), "age", errs), gender: (get("sex(M/F)") || "").toUpperCase(), icu: get("first_careunit") || "", vars: {}};
    for (const k of DATA.VAR_KEYS) { p.vars[k] = {}; for (const fd of FIELDS) p.vars[k][fd] = parse(get(`${k}_${fd}`), `${k}_${fd}`, errs); }
    const res = evaluate(p, errs, true);
    if (res.errors.length) { out.push([get("patient_id"), "", "", "", "", res.errors.join(" / ")]); continue; }
    const top = Object.entries(res.groups).sort((a, b) => b[1] - a[1]).slice(0, 3).filter(x => x[1] > 0).map(x => groupLabel(x[0]));
    out.push([get("patient_id"), res.prob.toFixed(4), riskBand(res.prob)[0], res.prob >= 0.3 ? "예" : "아니오", top.join(", "), ""]);
  }
  const H = ["patient_id", "probability", "risk", "alert_30", "top_factors", "problem"];
  $("#batchout").innerHTML = `<div class="scroll"><table class="t"><tr>${H.map(h => `<th>${h}</th>`).join("")}</tr>` +
    out.map(r => `<tr>${r.map(c => `<td>${esc(c ?? "")}</td>`).join("")}</tr>`).join("") + "</table></div>" +
    `<button class="btn" id="dl" style="margin-top:10px">결과 내려받기 (CSV)</button>`;
  $("#dl").onclick = () => download("icu_28day_predictions.csv", [H].concat(out).map(r => r.map(csvCell).join(",")).join("\n"));
};

// ---------- 모델 정보
(function () {
  const S = DATA.std_estimation_check;
  $("#modelinfo").innerHTML = `<h3 style="margin-top:0">이 대시보드가 쓰는 모델</h3><ul>
   <li><b>XGBoost</b> · 입력 150개: 나이·성별, 활력징후 7종과 혈액검사 12종의 첫 24시간 최저·평균·최고·표준편차·측정 횟수·측정 여부, 표준편차 결측 여부 19개, 첫 ICU 유형 15개</li>
   <li>빈 값은 채우지 않고 그대로 두고(XGBoost가 직접 처리), 측정하지 않았다는 사실 자체를 입력으로 씁니다.</li>
   <li>학습: MIMIC-IV 첫 ICU 입실 성인 <b>${DATA.n_development.toLocaleString()}명</b> (28일 사망률 ${pct1(MEAN)})</li>
   <li>평가: 학습에 쓰지 않은 <b>${DATA.n_test.toLocaleString()}명</b> (사망 ${DATA.test_deaths.toLocaleString()}명)</li>
   <li>이 파일은 저장된 모델을 그대로 넣어 브라우저에서 계산합니다. 확인 숫자: 가상 환자 A ${pct1(Object.values(DATA.example_probabilities)[0])}, B ${pct1(Object.values(DATA.example_probabilities)[1])}</li></ul>
   <table class="t"><tr><th>지표</th><th>값</th><th>의미</th></tr>
   <tr><td>ROC-AUC</td><td><b>${M.roc_auc.toFixed(3)}</b></td><td>사망·생존 환자를 한 명씩 뽑으면 약 ${pct0(M.roc_auc)}에서 사망 환자에게 더 높은 위험을 줌</td></tr>
   <tr><td>PR-AUC (AP)</td><td><b>${M.average_precision.toFixed(3)}</b></td><td>무작위 예측 기준 ${DATA.test_mortality.toFixed(3)}의 약 ${(M.average_precision / DATA.test_mortality).toFixed(1)}배</td></tr>
   <tr><td>Brier</td><td><b>${M.brier_score.toFixed(3)}</b></td><td>모두에게 평균 사망률을 준 예측(${DATA.brier_reference.toFixed(3)})보다 오차 ${pct0(1 - M.brier_score / DATA.brier_reference)} 감소</td></tr></table>
   <h3>입력 방법</h3><ul>
   <li>각 변수에 <b>입실 후 첫 24시간</b>의 최저·최고값을 넣습니다. 평균을 비우면 최저와 최고의 중간값, 측정 횟수를 비우면 학습 환자의 중앙 횟수를 씁니다.</li>
   <li>검사를 하지 않았다면 칸을 모두 비워 두세요. "미측정"으로 계산되며, 학습 데이터에서도 미측정 여부가 위험과 관련 있었습니다.</li>
   <li>표준편차는 최저·최고 범위로 추정합니다. 테스트 환자에서 실제 표준편차 대신 추정값을 넣었을 때 ROC-AUC ${S.roc_auc.toFixed(3)}, 예측 차이 중앙값 ${(S.median_abs_diff * 100).toFixed(1)}%p였습니다.</li></ul>
   <h3>한계 — 반드시 읽어 주세요</h3><ol>
   <li>한 병원(Beth Israel Deaconess Medical Center)의 2008~2019년 자료로 만든 모델입니다. 다른 병원·다른 시기에는 성능이 다를 수 있고 외부 검증을 하지 않았습니다.</li>
   <li>같은 테스트 환자를 이전 모델 평가에도 썼기 때문에 완전히 처음 보는 데이터의 성능은 아닙니다.</li>
   <li>입실 24시간 전에 퇴실·사망한 환자와 28일 결과를 확인할 수 없는 환자는 학습에서 빠졌습니다.</li>
   <li>APACHE II·SAPS II 같은 기존 중증도 점수와 직접 비교하지 못했습니다.</li>
   <li>요인 그래프는 예측에 대한 기여이며 원인이나 치료 효과가 아닙니다.</li></ol>`;
  $("#thtable").innerHTML = `<table class="t"><tr><th>경보 기준</th><th>사망자 포착(민감도)</th><th>생존자 오경보</th><th>경보 중 실제 사망(PPV)</th><th>전체 중 경보 비율</th><th>놓친 사망자</th></tr>` +
    DATA.thresholds.map(t => `<tr><td>${pct0(t.threshold)}</td><td>${pct1(t.sensitivity)}</td><td>${pct1(1 - t.specificity)}</td><td>${pct1(t.ppv)}</td><td>${pct1(t.alert_rate)}</td><td>${t.fn} / ${t.fn + t.tp}명</td></tr>`).join("") + "</table>";
  const W = 420, P = 50, X = v => P + v * (W - P - 16), Y = v => W - P - v * (W - P - 16);
  let s = `<svg viewBox="0 0 ${W} ${W}" role="img" aria-label="Calibration plot"><text x="${W / 2}" y="16" text-anchor="middle" font-size="13">Calibration (test set, n=4,517)</text>`;
  s += `<rect x="${P}" y="16" width="${W - P - 16}" height="${W - P - 16}" fill="none" stroke="#212b32"/>`;
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="gray" stroke-dasharray="5 4"/>`;
  const pts = DATA.calibration.map(c => `${X(c.mean_predicted)},${Y(c.observed)}`).join(" ");
  s += `<polyline points="${pts}" fill="none" stroke="#005eb8" stroke-width="2"/>`;
  DATA.calibration.forEach(c => s += `<circle cx="${X(c.mean_predicted)}" cy="${Y(c.observed)}" r="4" fill="#005eb8"/>`);
  for (let v = 0; v <= 1.001; v += 0.2) { s += `<text x="${X(v)}" y="${Y(0) + 16}" font-size="11" text-anchor="middle">${v.toFixed(1)}</text><text x="${P - 6}" y="${Y(v) + 4}" font-size="11" text-anchor="end">${v.toFixed(1)}</text>`; }
  s += `<text x="${X(0.5)}" y="${W - 8}" font-size="12" text-anchor="middle">Mean predicted probability</text>`;
  s += `<text x="14" y="${Y(0.5)}" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${Y(0.5)})">Observed 28-day mortality</text>`;
  s += `<text x="${P + 10}" y="36" font-size="11" fill="gray">--- Perfect calibration</text><text x="${P + 10}" y="52" font-size="11" fill="#005eb8">●— XGBoost (10 groups)</text></svg>`;
  $("#calsvg").innerHTML = s;
})();
</script></body></html>
"""


def main():
    model = export_model()
    data = export_data()
    core_js = (core.HERE / "dashboard_core.js").read_text(encoding="utf-8")
    page = (PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
                .replace("__MODEL__", json.dumps(model, separators=(",", ":")))
                .replace("__CORE__", core_js)
                .replace("__ROC__", img64("roc.png")).replace("__PR__", img64("pr.png")))
    HTML_PATH.write_text(page, encoding="utf-8")
    print(HTML_PATH, f"{HTML_PATH.stat().st_size / 1e6:.2f} MB")
    return data, model, core_js


if __name__ == "__main__":
    main()
