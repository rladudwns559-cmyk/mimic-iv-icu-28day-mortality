"""최종 XGBoost(대치 없음 + 결측 지시변수 + 젖산 + 첫 ICU 유형)를 재학습하고 대시보드 자료를 만든다.

- 입력·설정은 팀 재현 스크립트(scripts/reproduce_final_icu_xgboost.py)와 같다.
- 테스트 성능이 팀 기록(final_icu_test_metrics.json)과 같은지 확인한다. 다르면 멈춘다.
- dashboard_core.build_row(화면 입력 변환)가 실제 특징 행렬을 그대로 재현하는지 테스트 환자 전원으로 확인한다.
- 결과는 모두 outputs/에 저장한다. 환자 개인 단위 자료는 저장하지 않는다(집계값만).

실행: ..\\.venv\\Scripts\\python train.py
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score,
                             roc_curve, precision_recall_curve)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from train_three_models_cv import XGB_SETTINGS  # noqa: E402  팀 학습 코드와 같은 기본 설정

import dashboard_core as core  # noqa: E402

SPLITS = ROOT / "data/processed/error_cleaned/splits"
OUT = HERE / "outputs"
IDS = ["subject_id", "hadm_id", "stay_id"]
REFERENCE = ROOT / "reports/xgboost_indicator_icu_comparison/final_icu_test_metrics.json"


def load_split(split, columns, icu):
    folder = SPLITS / split
    x = pd.read_csv(folder / "X_features.csv", usecols=columns)[columns].copy()
    y = pd.read_csv(folder / "y_target.csv").mortality_28d.astype(int)
    ids = pd.read_csv(folder / "row_ids.csv")
    linked = ids.merge(icu, on=IDS, how="left", validate="one_to_one")
    assert len(x) == len(y) == len(ids) and linked.first_careunit.notna().all()
    x["gender"] = x.gender.map({"F": 0, "M": 1})
    assert x.gender.notna().all()
    return x, y, linked.first_careunit.reset_index(drop=True)


def design(x, units, std_cols, cats):
    std_missing = x[std_cols].isna().to_numpy(dtype=float)
    onehot = (units.to_numpy()[:, None] == np.array(cats)[None, :]).astype(float)
    return np.column_stack([x.to_numpy(dtype=float), std_missing, onehot])


def metrics_at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return dict(threshold=t, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                ppv=tp / (tp + fp) if tp + fp else float("nan"), alert_rate=float(pred.mean()),
                tp=tp, fp=fp, fn=fn, tn=tn)


def variable_stats(dev):
    stats = {}
    for key, *_ in core.VARIABLES:
        p = core.prefix(key)
        measured = dev[dev[f"{p}_count"] > 0]
        multi = measured[(measured[f"{p}_count"] >= 2) & (measured[f"{p}_std"] > 0)]
        ratio = float(np.median((multi[f"{p}_max"] - multi[f"{p}_min"]) / multi[f"{p}_std"]))
        q = {}
        for col in ("min", "mean", "max"):
            v = measured[f"{p}_{col}"]
            q[col] = {f"p{k}": round(float(np.percentile(v, k)), 3) for k in (1, 5, 25, 50, 75, 95, 99)}
        stats[key] = dict(
            measured_rate=round(float(len(measured) / len(dev)), 4),
            count_median=int(np.median(measured[f"{p}_count"])),
            std_range_ratio=round(ratio, 4),
            quantiles=q,
        )
    return stats


def figures(y, p, ref):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1d4ed8", label=f"XGBoost (AUC = {ref['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("1 - Specificity (False positive rate)")
    ax.set_ylabel("Sensitivity (True positive rate)")
    ax.set_title("ROC curve: 28-day mortality (test set, n=4,517)")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "roc.png", dpi=150); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#1d4ed8", label=f"XGBoost (AP = {ref['average_precision']:.3f})")
    ax.axhline(y.mean(), ls="--", color="gray", label=f"Random (prevalence {y.mean():.3f})")
    ax.set_xlabel("Recall (Sensitivity)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(OUT / "pr.png", dpi=150); plt.close(fig)


def calibration(y, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append(dict(bin=b + 1, lower=float(edges[b]), upper=float(edges[b + 1]), n=int(m.sum()),
                         deaths=int(y[m].sum()), mean_predicted=float(p[m].mean()),
                         observed=float(y[m].mean())))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    ref = json.loads(REFERENCE.read_text(encoding="utf-8-sig"))
    columns = json.loads((ROOT / "schemas/final_model_inputs.json").read_text(encoding="utf-8-sig"))["input_features"]
    icu = pd.read_csv(ROOT / "data/raw/icustays.csv", usecols=[*IDS, "first_careunit"])

    for name in ("X_features.csv", "y_target.csv", "row_ids.csv"):
        digest = hashlib.sha256((SPLITS / "test" / name).read_bytes()).hexdigest()
        print(f"test/{name} SHA-256 {'일치' if digest == ref['test_sha256'][name] else '불일치'}")

    parts = [load_split(s, columns, icu) for s in ("train", "validation")]
    dev = pd.concat([p[0] for p in parts], ignore_index=True)
    y = pd.concat([p[1] for p in parts], ignore_index=True)
    units = pd.concat([p[2] for p in parts], ignore_index=True)
    test, z, test_units = load_split("test", columns, icu)

    std_cols = [c for c in columns if c.endswith("_std")]
    cats = sorted(units.unique())
    assert cats == ref["icu_categories"], "ICU 범주가 팀 기록과 다릅니다"
    feature_names = columns + [f"{c}__was_missing" for c in std_cols] + [f"first_careunit={c}" for c in cats]
    assert feature_names == ref["feature_columns"], "입력 순서가 팀 기록과 다릅니다"

    params = {**XGB_SETTINGS, **ref["xgboost"]}
    model = xgb.XGBClassifier(**params)
    model.fit(design(dev, units, std_cols, cats), y)
    booster = model.get_booster()
    booster.feature_names = feature_names
    booster.save_model(str(OUT / "model.ubj"))

    xt = design(test, test_units, std_cols, cats)
    p = model.predict_proba(xt)[:, 1]
    yt = z.to_numpy()
    m = dict(roc_auc=roc_auc_score(yt, p), average_precision=average_precision_score(yt, p),
             brier_score=brier_score_loss(yt, p))
    print("테스트 성능:", {k: round(v, 4) for k, v in m.items()})
    for k, v in m.items():
        if abs(v - ref["metrics"][k]) > 1e-6:
            raise SystemExit(f"{k}가 팀 기록({ref['metrics'][k]:.6f})과 다릅니다: {v:.6f}")
    print("팀 기록(final_icu_test_metrics.json)과 일치")

    brier_ref = float(np.mean((yt - y.mean()) ** 2))
    thresholds = [metrics_at(yt, p, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]
    pd.DataFrame([dict(metric=k, value=v) for k, v in m.items()]
                 + [dict(metric="brier_reference", value=brier_ref)]
                 + [dict(metric=f"sensitivity@{t['threshold']}", value=t["sensitivity"]) for t in thresholds]
                 + [dict(metric=f"specificity@{t['threshold']}", value=t["specificity"]) for t in thresholds]
                 ).to_csv(OUT / "metrics.csv", index=False)
    figures(yt, p, m)

    assets = dict(
        model="XGBoost (no imputation + missing indicators + lactate + first ICU type)",
        clinical_features=columns, std_columns=std_cols, icu_categories=cats, feature_names=feature_names,
        n_development=int(len(y)), n_test=int(len(yt)), test_deaths=int(yt.sum()),
        development_mortality=float(y.mean()), test_mortality=float(yt.mean()),
        metrics=m, brier_reference=brier_ref, thresholds=thresholds, calibration=calibration(yt, p),
        test_predictions_sorted=[round(float(v), 4) for v in np.sort(p)],
        variables=variable_stats(dev),
    )
    # 예시 환자: 실제 환자가 아닌 설명용 가상 사례. 측정 횟수는 학습 환자의 중앙 횟수.
    def case(age, gender, icu_name, **values):
        vars_ = {k: dict(min=v[0], mean=v[1], max=v[2], count=assets["variables"][k]["count_median"])
                 for k, v in values.items()}
        return dict(age=age, gender=gender, icu=icu_name, vars=vars_)
    assets["examples"] = {
        "가상 환자 A · 패혈성 쇼크 의심": case(
            78, "M", "Medical Intensive Care Unit (MICU)",
            HR=(92, 112, 134), RR=(18, 26, 36), SpO2=(86, 93, 98), SBP=(74, 98, 128), DBP=(38, 52, 70),
            MAP=(52, 66, 84), Temperature=(36.1, 37.4, 38.7), Creatinine=(2.1, 2.4, 2.7), Glucose=(98, 165, 240),
            Sodium=(133, 135, 137), Potassium=(4.2, 4.9, 5.6), Bicarbonate=(14, 16, 18), AnionGap=(18, 21, 24),
            Hemoglobin=(8.6, 9.3, 10.1), Hematocrit=(26, 28.5, 31), WBC=(17, 19.5, 22), Platelet=(92, 105, 118),
            INR=(1.6, 1.8, 2.0), Lactate=(3.4, 4.6, 5.9)),
        "가상 환자 B · 심장수술 후 안정": case(
            58, "F", "Cardiac Vascular Intensive Care Unit (CVICU)",
            HR=(68, 80, 96), RR=(12, 16, 22), SpO2=(95, 98, 100), SBP=(102, 118, 138), DBP=(52, 62, 76),
            MAP=(68, 80, 94), Temperature=(36.4, 36.9, 37.5), Creatinine=(0.7, 0.75, 0.8), Glucose=(105, 122, 145),
            Sodium=(138, 139, 140), Potassium=(4.0, 4.2, 4.5), Bicarbonate=(23, 24, 25), AnionGap=(11, 12, 13),
            Hemoglobin=(10.4, 10.9, 11.5), Hematocrit=(31, 32.5, 34), WBC=(8.5, 10, 11.5), Platelet=(170, 185, 200),
            INR=(1.2, 1.25, 1.3), Lactate=(1.2, 1.6, 2.1)),
    }
    (OUT / "assets.json").write_text(json.dumps(assets, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 검증 1: 화면 입력 변환이 실제 특징 행렬을 그대로 만드는가 (테스트 환자 전원) ----
    assets_loaded, booster2 = core.load_assets(OUT)
    rows = []
    for i in range(len(test)):
        r = test.iloc[i]
        patient = dict(age=r.age, gender="M" if r.gender == 1 else "F", icu=test_units.iloc[i], vars={})
        for key, *_ in core.VARIABLES:
            pre = core.prefix(key)
            if r[f"{pre}_count"] > 0:
                patient["vars"][key] = dict(min=r[f"{pre}_min"], mean=r[f"{pre}_mean"], max=r[f"{pre}_max"],
                                            count=r[f"{pre}_count"],
                                            std=None if np.isnan(r[f"{pre}_std"]) else r[f"{pre}_std"])
        row, errors, _, _ = core.build_row(patient, assets_loaded)
        rows.append(row[0])
    built = np.vstack(rows)
    same = np.array_equal(np.isnan(built), np.isnan(xt)) and np.allclose(np.nan_to_num(built), np.nan_to_num(xt))
    p2 = booster2.predict(xgb.DMatrix(built, feature_names=feature_names, missing=np.nan))
    print(f"검증 1: 화면 변환 행렬 = 실제 행렬 {same}, 예측 최대 차이 {np.abs(p2 - p).max():.2e}")
    if not same or np.abs(p2 - p).max() > 1e-6:
        raise SystemExit("화면 입력 변환이 실제 특징과 다릅니다")

    # ---- 검증 2: 표준편차를 비워 두고 추정할 때의 성능 손실 ----
    est = built.copy()
    for key, *_ in core.VARIABLES:
        pre = core.prefix(key)
        j = feature_names.index(f"{pre}_std")
        lo, hi = built[:, feature_names.index(f"{pre}_min")], built[:, feature_names.index(f"{pre}_max")]
        cnt = built[:, feature_names.index(f"{pre}_count")]
        ratio = assets["variables"][key]["std_range_ratio"]
        est[:, j] = np.where(cnt >= 2, np.where(hi > lo, (hi - lo) / ratio, 0.0), np.nan)
    p3 = booster2.predict(xgb.DMatrix(est, feature_names=feature_names, missing=np.nan))
    m3 = dict(roc_auc=roc_auc_score(yt, p3), average_precision=average_precision_score(yt, p3),
              brier_score=brier_score_loss(yt, p3), median_abs_diff=float(np.median(np.abs(p3 - p))))
    print("검증 2: 표준편차를 추정값으로 바꾼 테스트 성능", {k: round(v, 4) for k, v in m3.items()})

    # ---- 확인 숫자: 예시 환자 두 명 ----
    checks = {}
    for name, prof in assets["examples"].items():
        row, errors, _, _ = core.build_row(prof, assets_loaded)
        assert not errors, errors
        checks[name] = round(core.predict(booster2, row, assets_loaded)[0], 4)
    assets["std_estimation_check"] = m3
    assets["example_probabilities"] = checks
    (OUT / "assets.json").write_text(json.dumps(assets, ensure_ascii=False, indent=1), encoding="utf-8")
    print("확인 숫자(예시 환자):", checks)


if __name__ == "__main__":
    main()
