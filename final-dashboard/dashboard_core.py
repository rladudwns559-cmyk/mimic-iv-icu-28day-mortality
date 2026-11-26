"""대시보드 공통 코드: 입력값 → 최종 XGBoost 입력 150개 변환, 예측, 요인 분석.

train.py(검증)와 app.py(화면)가 같은 함수를 쓰므로 두 곳의 계산이 달라질 수 없다.
"""
import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# (변수 키, 종류, 화면 이름, 단위, 입력 허용 범위[파이프라인 정제 기준과 같은 물리적 한계])
VARIABLES = [
    ("HR", "vital", "심박수", "회/분", (0, 500)),
    ("RR", "vital", "호흡수", "회/분", (0, 300)),
    ("SpO2", "vital", "산소포화도", "%", (0.1, 100)),
    ("SBP", "vital", "수축기 혈압", "mmHg", (0, 1000)),
    ("DBP", "vital", "이완기 혈압", "mmHg", (0, 1000)),
    ("MAP", "vital", "평균동맥압", "mmHg", (0, 1000)),
    ("Temperature", "vital", "체온", "°C", (0, 60)),
    ("Creatinine", "lab", "크레아티닌", "mg/dL", (0, 100)),
    ("Glucose", "lab", "혈당", "mg/dL", (0, 5000)),
    ("Sodium", "lab", "나트륨", "mEq/L", (0, 300)),
    ("Potassium", "lab", "칼륨", "mEq/L", (0, 30)),
    ("Bicarbonate", "lab", "중탄산염", "mEq/L", (0, 100)),
    ("AnionGap", "lab", "음이온차", "mEq/L", (-50, 100)),
    ("Hemoglobin", "lab", "헤모글로빈", "g/dL", (0, 40)),
    ("Hematocrit", "lab", "헤마토크릿", "%", (0, 100)),
    ("WBC", "lab", "백혈구", "K/uL", (0, 1000)),
    ("Platelet", "lab", "혈소판", "K/uL", (0, 5000)),
    ("INR", "lab", "INR", "비율", (0, 50)),
    ("Lactate", "lab", "젖산", "mmol/L", (0, 50)),
]
VAR = {v[0]: v for v in VARIABLES}

ICU_KO = {
    "Cardiac Vascular Intensive Care Unit (CVICU)": "심혈관 중환자실 (CVICU)",
    "Coronary Care Unit (CCU)": "관상동맥 중환자실 (CCU)",
    "Intensive Care Unit (ICU)": "중환자실 (ICU)",
    "Medical Intensive Care Unit (MICU)": "내과 중환자실 (MICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": "내과·외과 중환자실 (MICU/SICU)",
    "Medicine": "내과 (Medicine)",
    "Neuro Intermediate": "신경계 중등도 병동 (Neuro Intermediate)",
    "Neuro Stepdown": "신경계 스텝다운 (Neuro Stepdown)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": "신경외과 중환자실 (Neuro SICU)",
    "Neurology": "신경과 (Neurology)",
    "PACU": "마취회복실 (PACU)",
    "Surgery/Trauma": "외과·외상 (Surgery/Trauma)",
    "Surgery/Vascular/Intermediate": "외과·혈관 중등도 (Surgery/Vascular/Intermediate)",
    "Surgical Intensive Care Unit (SICU)": "외과 중환자실 (SICU)",
    "Trauma SICU (TSICU)": "외상 외과 중환자실 (TSICU)",
}

# 위험 수준: 색과 함께 모양 기호·글자로도 구분한다(색만으로 정보 전달 금지)
RISK_BANDS = [(0.30, "높음", "#d5281b"), (0.15, "보통", "#a35200"), (0.0, "낮음", "#007f3b")]
RISK_SHAPE = {"높음": "▲", "보통": "◆", "낮음": "●"}


def prefix(key):
    return ("vital_" if VAR[key][1] == "vital" else "lab_") + key


def load_assets(out=OUT):
    assets = json.loads((out / "assets.json").read_text(encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(str(out / "model.ubj"))
    return assets, booster


def _num(x):
    """빈 칸·None·NaN → None, 그 외 float."""
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
    x = float(x)
    return None if math.isnan(x) else x


def variable_block(key, values, assets):
    """한 변수의 입력(min/mean/max/count/std)을 6개 특징으로 바꾼다.

    반환: (dict 특징값, 오류 목록, 안내 목록, 화면용 요약)
    규칙
      - 값이 하나도 없으면 미측정: count=0, missing=1, 나머지 NaN (학습 데이터와 같은 표기)
      - 값이 하나만 있으면 최솟값=평균=최댓값
      - 최솟값·최댓값만 있으면 평균은 중간값
      - 측정 횟수를 비우면 학습 데이터에서 그 변수를 잰 환자의 중앙 횟수
      - 표준편차를 비우면 (최댓값−최솟값)/k 로 추정. k는 학습 데이터의 중앙 비율
    """
    name, unit = VAR[key][2], VAR[key][3]
    lo_lim, hi_lim = VAR[key][4]
    st = assets["variables"][key]
    errors, notes = [], []
    lo, mean, hi = _num(values.get("min")), _num(values.get("mean")), _num(values.get("max"))
    count, std = _num(values.get("count")), _num(values.get("std"))
    p = prefix(key)
    nan = float("nan")
    if lo is None and mean is None and hi is None:
        if count not in (None, 0):
            errors.append(f"{name}: 값 없이 측정 횟수({count:g})만 입력됐습니다.")
        feats = {f"{p}_mean": nan, f"{p}_min": nan, f"{p}_max": nan, f"{p}_std": nan,
                 f"{p}_count": 0.0, f"{p}_missing": 1.0}
        return feats, errors, notes, None

    given = [v for v in (lo, mean, hi) if v is not None]
    if lo is None:
        lo = min(given)
    if hi is None:
        hi = max(given)
    if mean is None:
        mean = (lo + hi) / 2
    for label, v in (("최솟값", lo), ("평균", mean), ("최댓값", hi)):
        if not lo_lim <= v <= hi_lim:
            errors.append(f"{name} {label} {v:g}{unit}는 입력 가능 범위({lo_lim:g}~{hi_lim:g})를 벗어납니다.")
    if not lo <= mean <= hi:
        errors.append(f"{name}: 최솟값 ≤ 평균 ≤ 최댓값 순서가 아닙니다 ({lo:g}, {mean:g}, {hi:g}).")

    count_default = count is None
    if count_default:
        count = float(st["count_median"])
    if count < 1 or count != int(count):
        errors.append(f"{name}: 측정 횟수는 1 이상의 정수여야 합니다.")
    if count == 1 and not (lo == mean == hi):
        errors.append(f"{name}: 측정 1회인데 최솟값·평균·최댓값이 다릅니다.")

    if count >= 2:
        if std is None:
            std = (hi - lo) / st["std_range_ratio"] if hi > lo else 0.0
        elif std < 0:
            errors.append(f"{name}: 표준편차는 0 이상이어야 합니다.")
    else:
        std = nan  # 1회 측정은 학습 데이터에서도 표준편차가 비어 있다

    q = st["quantiles"]
    for label, v, col in (("최솟값", lo, "min"), ("최댓값", hi, "max")):
        if v < q[col]["p1"] or v > q[col]["p99"]:
            notes.append(f"{name} {label} {v:g}{unit}는 학습 환자의 1~99% 범위"
                         f"({q[col]['p1']:g}~{q[col]['p99']:g}) 밖이라 예측이 덜 정확할 수 있습니다.")
    feats = {f"{p}_mean": mean, f"{p}_min": lo, f"{p}_max": hi, f"{p}_std": std,
             f"{p}_count": float(count), f"{p}_missing": 0.0}
    return feats, errors, notes, dict(min=lo, mean=mean, max=hi, count=count, count_default=count_default)


def build_row(patient, assets):
    """patient: {age, gender('M'/'F'), icu(영문 범주), vars: {키: {min,mean,max,count,std}}}

    반환: (np.ndarray[1,150], 오류 목록, 안내 목록, 변수별 요약)
    """
    errors, notes, summary, feats = [], [], {}, {}
    age = _num(patient.get("age"))
    if age is None:
        errors.append("나이를 입력하세요.")
        age = float("nan")
    elif not 18 <= age <= 120:
        errors.append("나이는 18~120세만 입력할 수 있습니다 (성인 코호트로 학습).")
    gender = patient.get("gender")
    if gender not in ("M", "F"):
        errors.append("성별을 선택하세요.")
    feats["age"] = age
    feats["gender"] = 1.0 if gender == "M" else 0.0
    for key, *_ in VARIABLES:
        f, e, n, s = variable_block(key, patient.get("vars", {}).get(key, {}), assets)
        feats.update(f)
        errors += e
        notes += n
        summary[key] = s
    defaulted = [f"{VAR[k][2]} {int(s['count'])}회" for k, s in summary.items() if s and s["count_default"]]
    if defaulted:
        notes.insert(0, "측정 횟수를 비운 변수는 학습 환자의 중앙 횟수로 계산했습니다: " + ", ".join(defaulted))
    cols = assets["clinical_features"]
    base = np.array([feats[c] for c in cols], dtype=float)
    std_missing = np.array([float(np.isnan(feats[c])) for c in assets["std_columns"]])
    icu = patient.get("icu")
    cats = assets["icu_categories"]
    if icu not in cats:
        errors.append("첫 ICU 유형을 선택하세요.")
    onehot = np.array([float(icu == c) for c in cats])
    row = np.concatenate([base, std_missing, onehot])[None, :]
    assert row.shape[1] == len(assets["feature_names"])
    return row, errors, notes, summary


def predict(booster, row, assets):
    d = xgb.DMatrix(row, feature_names=assets["feature_names"], missing=np.nan)
    prob = float(booster.predict(d)[0])
    contrib = booster.predict(d, pred_contribs=True)[0]
    return prob, contrib


def group_contributions(contrib, assets):
    """특징별 기여(로그오즈)를 임상 변수 단위로 합친다. 마지막 원소는 기준값(bias)."""
    names = assets["feature_names"]
    groups = {}
    for n, c in zip(names, contrib[:-1]):
        if n.startswith("first_careunit="):
            g = "ICU"
        elif n in ("age", "gender"):
            g = n
        else:
            g = n.split("_")[1]
        groups[g] = groups.get(g, 0.0) + float(c)
    return groups, float(contrib[-1])


def group_label(g):
    if g == "ICU":
        return "첫 ICU 유형"
    if g == "age":
        return "나이"
    if g == "gender":
        return "성별"
    return VAR[g][2]


def risk_band(prob):
    for cut, label, color in RISK_BANDS:
        if prob >= cut:
            return label, color
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def percentile(prob, assets):
    dist = np.asarray(assets["test_predictions_sorted"])
    return float(np.searchsorted(dist, prob, side="right") / len(dist) * 100)
