// 대시보드 계산 핵심(브라우저용). dashboard_core.py와 같은 규칙으로 입력 → 모델 입력 150개 → 예측 → SHAP.
// build_html.py가 DATA(집계값)와 MODEL(XGBoost 트리)을 함께 넣는다.
"use strict";

function fmtG(v) {               // 파이썬 {v:g}와 같은 모양(유효숫자 6자리, 끝자리 0 제거)
  if (!isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-4)) {
    let s = v.toExponential(5).replace(/\.?0+e/, "e");
    return s.replace(/e([+-])(\d)$/, "e$10$2");
  }
  return String(parseFloat(v.toPrecision(6)));
}

function toNum(x) {               // 빈 칸·null·NaN → null
  if (x === null || x === undefined) return null;
  if (typeof x === "string") { x = x.trim(); if (!x) return null; x = Number(x); }
  return Number.isNaN(x) ? null : x;
}

function prefix(key) { return (DATA.VAR[key].kind === "vital" ? "vital_" : "lab_") + key; }

function variableBlock(key, values) {
  const V = DATA.VAR[key], name = V.name, unit = V.unit, loLim = V.lo, hiLim = V.hi;
  const st = DATA.variables[key];
  const errors = [], notes = [];
  let lo = toNum(values.min), mean = toNum(values.mean), hi = toNum(values.max);
  let count = toNum(values.count), std = toNum(values.std);
  const p = prefix(key), feats = {};
  if (lo === null && mean === null && hi === null) {
    if (count !== null && count !== 0) errors.push(`${name}: 값 없이 측정 횟수(${fmtG(count)})만 입력됐습니다.`);
    feats[p + "_mean"] = NaN; feats[p + "_min"] = NaN; feats[p + "_max"] = NaN; feats[p + "_std"] = NaN;
    feats[p + "_count"] = 0; feats[p + "_missing"] = 1;
    return { feats, errors, notes, summary: null };
  }
  const given = [lo, mean, hi].filter(v => v !== null);
  if (lo === null) lo = Math.min(...given);
  if (hi === null) hi = Math.max(...given);
  if (mean === null) mean = (lo + hi) / 2;
  for (const [label, v] of [["최솟값", lo], ["평균", mean], ["최댓값", hi]]) {
    if (!(loLim <= v && v <= hiLim))
      errors.push(`${name} ${label} ${fmtG(v)}${unit}는 입력 가능 범위(${fmtG(loLim)}~${fmtG(hiLim)})를 벗어납니다.`);
  }
  if (!(lo <= mean && mean <= hi))
    errors.push(`${name}: 최솟값 ≤ 평균 ≤ 최댓값 순서가 아닙니다 (${fmtG(lo)}, ${fmtG(mean)}, ${fmtG(hi)}).`);
  const countDefault = count === null;
  if (countDefault) count = st.count_median;
  if (count < 1 || count !== Math.trunc(count)) errors.push(`${name}: 측정 횟수는 1 이상의 정수여야 합니다.`);
  if (count === 1 && !(lo === mean && mean === hi)) errors.push(`${name}: 측정 1회인데 최솟값·평균·최댓값이 다릅니다.`);
  if (count >= 2) {
    if (std === null) std = hi > lo ? (hi - lo) / st.std_range_ratio : 0;
    else if (std < 0) errors.push(`${name}: 표준편차는 0 이상이어야 합니다.`);
  } else std = NaN;
  const q = st.quantiles;
  for (const [label, v, col] of [["최솟값", lo, "min"], ["최댓값", hi, "max"]]) {
    if (v < q[col].p1 || v > q[col].p99)
      notes.push(`${name} ${label} ${fmtG(v)}${unit}는 학습 환자의 1~99% 범위(${fmtG(q[col].p1)}~${fmtG(q[col].p99)}) 밖이라 예측이 덜 정확할 수 있습니다.`);
  }
  feats[p + "_mean"] = mean; feats[p + "_min"] = lo; feats[p + "_max"] = hi; feats[p + "_std"] = std;
  feats[p + "_count"] = count; feats[p + "_missing"] = 0;
  return { feats, errors, notes, summary: { min: lo, mean, max: hi, count, count_default: countDefault } };
}

function buildRow(patient) {
  let errors = [], notes = [];
  const summary = {}, feats = {};
  let age = toNum(patient.age);
  if (age === null) { errors.push("나이를 입력하세요."); age = NaN; }
  else if (!(18 <= age && age <= 120)) errors.push("나이는 18~120세만 입력할 수 있습니다 (성인 코호트로 학습).");
  const gender = patient.gender;
  if (gender !== "M" && gender !== "F") errors.push("성별을 선택하세요.");
  feats.age = age; feats.gender = gender === "M" ? 1 : 0;
  for (const key of DATA.VAR_KEYS) {
    const r = variableBlock(key, (patient.vars || {})[key] || {});
    Object.assign(feats, r.feats); errors = errors.concat(r.errors); notes = notes.concat(r.notes); summary[key] = r.summary;
  }
  const defaulted = DATA.VAR_KEYS.filter(k => summary[k] && summary[k].count_default)
    .map(k => `${DATA.VAR[k].name} ${Math.trunc(summary[k].count)}회`);
  if (defaulted.length) notes.unshift("측정 횟수를 비운 변수는 학습 환자의 중앙 횟수로 계산했습니다: " + defaulted.join(", "));
  const row = [];
  for (const c of DATA.clinical_features) row.push(feats[c]);
  for (const c of DATA.std_columns) row.push(Number.isNaN(feats[c]) ? 1 : 0);
  if (!DATA.icu_categories.includes(patient.icu)) errors.push("첫 ICU 유형을 선택하세요.");
  for (const c of DATA.icu_categories) row.push(patient.icu === c ? 1 : 0);
  if (row.length !== DATA.feature_names.length) throw new Error("입력 개수 오류");
  return { row, errors, notes, summary };
}

// ---------------- XGBoost 예측 (트리: [왼쪽, 오른쪽, 특징, 기준값/잎값, 결측 시 왼쪽, cover])
function goLeft(t, j, x) {
  const v = x[t[2][j]];
  if (Number.isNaN(v)) return t[4][j] === 1;
  // XGBoost는 입력값과 기준값을 모두 float32로 비교한다(예: 2.4 < 2.4가 되지 않도록 양쪽을 같은 정밀도로).
  return Math.fround(v) < Math.fround(t[3][j]);
}

function predictMargin(x) {
  let s = MODEL.base_margin;
  for (const t of MODEL.trees) {
    let j = 0;
    while (t[0][j] !== -1) j = goLeft(t, j, x) ? t[0][j] : t[1][j];
    s += t[3][j];
  }
  return s;
}

function predictProb(x) { return 1 / (1 + Math.exp(-predictMargin(x))); }

// ---------------- TreeSHAP (Lundberg 등, xgboost pred_contribs와 같은 정확한 알고리즘)
function treeShap(t, x, phi) {
  const L = t[0], R = t[1], F = t[2], T = t[3], C = t[5];
  function unwoundSum(m, d, i) {
    const o = m[i].o, z = m[i].z;
    let n = m[d].w, total = 0;
    for (let j = d - 1; j >= 0; j--) {
      if (o !== 0) { const tt = n * (d + 1) / ((j + 1) * o); total += tt; n = m[j].w - tt * z * ((d - j) / (d + 1)); }
      else total += (m[j].w / z) / ((d - j) / (d + 1));
    }
    return total;
  }
  function unwind(m, d, i) {
    const o = m[i].o, z = m[i].z;
    let n = m[d].w;
    for (let j = d - 1; j >= 0; j--) {
      if (o !== 0) { const tt = m[j].w; m[j].w = n * (d + 1) / ((j + 1) * o); n = tt - m[j].w * z * (d - j) / (d + 1); }
      else m[j].w = m[j].w * (d + 1) / (z * (d - j));
    }
    for (let j = i; j < d; j++) { m[j].d = m[j + 1].d; m[j].z = m[j + 1].z; m[j].o = m[j + 1].o; }
  }
  function recurse(j, parent, d, pz, po, pi) {
    const m = parent.slice(0, d).map(e => ({ d: e.d, z: e.z, o: e.o, w: e.w }));
    m[d] = { d: pi, z: pz, o: po, w: d === 0 ? 1 : 0 };
    for (let i = d - 1; i >= 0; i--) {
      m[i + 1].w += po * m[i].w * (i + 1) / (d + 1);
      m[i].w = pz * m[i].w * (d - i) / (d + 1);
    }
    if (L[j] === -1) {
      for (let i = 1; i <= d; i++) phi[m[i].d] += unwoundSum(m, d, i) * (m[i].o - m[i].z) * T[j];
      return;
    }
    const f = F[j];
    const hot = goLeft(t, j, x) ? L[j] : R[j];
    const cold = hot === L[j] ? R[j] : L[j];
    let iz = 1, io = 1, k = 1;
    for (; k <= d; k++) if (m[k].d === f) break;
    if (k <= d) { iz = m[k].z; io = m[k].o; unwind(m, d, k); d -= 1; }
    recurse(hot, m, d + 1, iz * C[hot] / C[j], io, f);
    recurse(cold, m, d + 1, iz * C[cold] / C[j], 0, f);
  }
  recurse(0, [], 0, 1, 1, -1);
}

function contributions(x) {
  const phi = new Array(x.length).fill(0);
  for (const t of MODEL.trees) treeShap(t, x, phi);
  return phi;
}

function groupContributions(phi) {
  const g = {};
  DATA.feature_names.forEach((n, i) => {
    let k;
    if (n.startsWith("first_careunit=")) k = "ICU";
    else if (n === "age" || n === "gender") k = n;
    else k = n.split("_")[1];
    g[k] = (g[k] || 0) + phi[i];
  });
  return g;
}

function groupLabel(g) {
  if (g === "ICU") return "첫 ICU 유형";
  if (g === "age") return "나이";
  if (g === "gender") return "성별";
  return DATA.VAR[g].name;
}

function riskBand(p) {
  for (const [cut, label, color] of DATA.RISK_BANDS) if (p >= cut) return [label, color];
  const last = DATA.RISK_BANDS[DATA.RISK_BANDS.length - 1];
  return [last[1], last[2]];
}

function percentile(p) {             // np.searchsorted(side="right") / n * 100
  const a = DATA.test_predictions_sorted;
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] <= p) lo = mid + 1; else hi = mid; }
  return lo / a.length * 100;
}

function calibrationRow(p) {
  const cal = DATA.calibration;
  for (let i = 0; i < cal.length; i++) if (p <= cal[i].upper || i === cal.length - 1) return cal[i];
}
