"""보고서 보강 자료: 테스트 성능 신뢰구간, 임계값별 혼동행렬, 대상자 기본 특성표, ROC·PR·보정 그림.

- 최종 모델은 outputs/model.ubj를 불러 쓰고 다시 학습하지 않는다.
- 비교 모델(첫 ICU 유형 제외)은 팀 기록과 같은 설정으로 학습해 팀 기록(final_native_test_metrics.json)과
  소수 여섯째 자리까지 같은지 확인한 뒤에만 사용한다.
- 저장하는 것은 집계값뿐이다(환자 단위 자료 없음). random_state=42.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

import train as T

OUT = T.OUT / "report_extras"
NATIVE_REF = T.ROOT / "reports/xgboost_missingness_ablation/final_native_test_metrics.json"
B = 2000
SEED = 42


def metrics(y, p):
    return dict(roc_auc=roc_auc_score(y, p), average_precision=average_precision_score(y, p),
                brier_score=brier_score_loss(y, p))


def main():
    OUT.mkdir(exist_ok=True)
    ref = json.loads(T.REFERENCE.read_text(encoding="utf-8-sig"))
    nref = json.loads(NATIVE_REF.read_text(encoding="utf-8-sig"))
    columns = json.loads((T.ROOT / "schemas/final_model_inputs.json").read_text(encoding="utf-8-sig"))["input_features"]
    icu = pd.read_csv(T.ROOT / "data/raw/icustays.csv", usecols=[*T.IDS, "first_careunit"])
    parts = [T.load_split(s, columns, icu) for s in ("train", "validation")]
    dev = pd.concat([p[0] for p in parts], ignore_index=True)
    ydev = pd.concat([p[1] for p in parts], ignore_index=True)
    test, yt, test_units = T.load_split("test", columns, icu)
    yt = yt.to_numpy()
    std_cols = [c for c in columns if c.endswith("_std")]
    cats = ref["icu_categories"]

    # 최종 모델: 저장된 모델 불러 쓰기
    booster = xgb.Booster(); booster.load_model(str(T.OUT / "model.ubj"))
    xt = T.design(test, test_units, std_cols, cats)
    p_final = booster.predict(xgb.DMatrix(xt, feature_names=ref["feature_columns"], missing=np.nan))
    for k, v in metrics(yt, p_final).items():
        assert abs(v - ref["metrics"][k]) < 1e-6, k

    # 비교 모델: 첫 ICU 유형 제외(같은 설정) — 팀 기록과 같을 때만 사용
    base_cols = columns + [f"{c}__was_missing" for c in std_cols]
    assert base_cols == nref["feature_columns"], "비교 모델 입력 순서가 팀 기록과 다릅니다"
    xd = T.design(dev, pd.Series(["_"] * len(dev)), std_cols, [])
    xtb = T.design(test, pd.Series(["_"] * len(test)), std_cols, [])
    m = xgb.XGBClassifier(**nref["xgboost"])
    m.fit(xd, ydev)
    p_base = m.predict_proba(xtb)[:, 1]
    for k, v in metrics(yt, p_base).items():
        assert abs(v - nref["metrics"][k]) < 1e-6, (k, v, nref["metrics"][k])
    print("최종 모델·비교 모델 모두 팀 기록과 일치")

    # 1) 부트스트랩 95% 신뢰구간 (같은 표본으로 두 모델을 함께 계산 → 차이도 짝지어 계산)
    rng = np.random.default_rng(SEED)
    n = len(yt)
    keys = ["roc_auc", "average_precision", "brier_score", "sens@0.5", "fpr@0.5", "sens@0.3", "fpr@0.3"]

    def all_metrics(y, p):
        r = metrics(y, p)
        for t in (0.5, 0.3):
            pred = p >= t
            r[f"sens@{t}"] = pred[y == 1].mean()
            r[f"fpr@{t}"] = pred[y == 0].mean()
        return r

    point = {"final": all_metrics(yt, p_final), "base": all_metrics(yt, p_base)}
    boot = {"final": {k: [] for k in keys}, "base": {k: [] for k in keys}, "diff": {k: [] for k in keys}}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        y = yt[idx]
        if y.min() == y.max():
            continue
        a, b = all_metrics(y, p_final[idx]), all_metrics(y, p_base[idx])
        for k in keys:
            boot["final"][k].append(a[k]); boot["base"][k].append(b[k]); boot["diff"][k].append(a[k] - b[k])
    rows = []
    for k in keys:
        row = dict(metric=k, final=point["final"][k], base=point["base"][k],
                   diff=point["final"][k] - point["base"][k])
        for g in ("final", "base", "diff"):
            lo, hi = np.percentile(boot[g][k], [2.5, 97.5])
            row[f"{g}_lo"], row[f"{g}_hi"] = lo, hi
        rows.append(row)
    ci = pd.DataFrame(rows)
    ci.to_csv(OUT / "test_bootstrap_ci.csv", index=False)
    print(ci.round(4).to_string())

    # 2) 임계값별 혼동행렬
    cm = pd.DataFrame([T.metrics_at(yt, p_final, t) for t in (0.5, 0.3)])
    cm.to_csv(OUT / "test_confusion.csv", index=False)
    print(cm.to_string())

    # 3) 대상자 기본 특성 (분석 코호트 30,112명, 생존·사망별)
    P = T.ROOT / "data/processed/error_cleaned"
    X = pd.read_csv(P / "X_features.csv")
    yy = pd.read_csv(P / "y_target.csv").mortality_28d.astype(int)
    ids = pd.read_csv(P / "row_ids.csv")
    unit = ids.merge(icu, on=T.IDS, how="left", validate="one_to_one").first_careunit
    sz = unit.value_counts()
    unit = unit.where(unit.map(sz) >= 100, "Other (n<100 each)")
    lab_keys = [k for k, kind, *_ in T.core.VARIABLES if kind == "lab"]
    n_lab_missing = sum((X[f"lab_{k}_missing"] == 1).astype(int) for k in lab_keys)
    groups = {"전체": np.ones(len(X), bool), "생존": (yy == 0).values, "사망": (yy == 1).values}
    out = []

    def add(label, f):
        out.append([label] + [f(m) for m in groups.values()])

    add("환자 수", lambda m: f"{m.sum():,}")
    add("나이, 세 (중앙값 [IQR])", lambda m: "{:.0f} [{:.0f}–{:.0f}]".format(*np.percentile(X.age[m], [50, 25, 75])))
    add("남성, n (%)", lambda m: f"{(X.gender[m] == 'M').sum():,} ({(X.gender[m] == 'M').mean() * 100:.1f})")
    for u in unit.value_counts().index:
        add(f"첫 ICU: {u}", lambda m, u=u: f"{(unit[m] == u).sum():,} ({(unit[m] == u).mean() * 100:.1f})")
    for k, name in (("Lactate", "젖산 측정"), ("INR", "INR 측정")):
        add(f"{name}, n (%)", lambda m, k=k: f"{(X[f'lab_{k}_missing'][m] == 0).sum():,} ({(X[f'lab_{k}_missing'][m] == 0).mean() * 100:.1f})")
    add("혈액검사 12종 모두 미측정, n (%)",
        lambda m: f"{(n_lab_missing[m] == 12).sum():,} ({(n_lab_missing[m] == 12).mean() * 100:.1f})")
    base_tbl = pd.DataFrame(out, columns=["항목", *groups])
    base_tbl.to_csv(OUT / "baseline_characteristics.csv", index=False, encoding="utf-8-sig")
    print(base_tbl.to_string())

    # 4) 그림: ROC · PR · 보정 (그림 안 글자는 영어)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for p, lab, col, ls in ((p_final, "Final (with first ICU type)", "#005eb8", "-"),
                            (p_base, "Without first ICU type", "#768692", "--")):
        fpr, tpr, _ = roc_curve(yt, p)
        ax[0].plot(fpr, tpr, ls, color=col, lw=2, label=f"{lab}: AUC {roc_auc_score(yt, p):.3f}")
        pr, rc, _ = precision_recall_curve(yt, p)
        ax[1].plot(rc, pr, ls, color=col, lw=2, label=f"{lab}: AP {average_precision_score(yt, p):.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax[0].set(xlabel="1 - Specificity (false positive rate)", ylabel="Sensitivity", title="A. ROC curve")
    ax[1].axhline(yt.mean(), ls=":", color="gray", lw=1, label=f"Random (prevalence {yt.mean():.3f})")
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title="B. Precision-recall curve")
    for t, mk in ((0.5, "o"), (0.3, "s")):
        pred = p_final >= t
        ax[0].plot(pred[yt == 0].mean(), pred[yt == 1].mean(), mk, color="#d5281b", ms=7,
                   label=f"Threshold {t}")
    cal = T.calibration(yt, p_final)
    mp = [r["mean_predicted"] for r in cal]; ob = [r["observed"] for r in cal]
    se = [np.sqrt(r["observed"] * (1 - r["observed"]) / r["n"]) for r in cal]
    ax[2].plot([0, 1], [0, 1], ":", color="gray", lw=1, label="Perfect calibration")
    ax[2].errorbar(mp, ob, yerr=1.96 * np.array(se), fmt="o-", color="#005eb8", lw=1.5, capsize=3,
                   label="Final model (10 risk groups)")
    ax[2].set(xlabel="Mean predicted probability", ylabel="Observed 28-day mortality",
              title="C. Calibration (test, deciles)", xlim=(0, 1), ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=8, loc="lower right" if a is not ax[1] else "upper right", frameon=False)
        a.grid(alpha=.25); a.set_aspect("equal") if a is not ax[1] else None
    ax[1].set(xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUT / "test_curves.png", dpi=200)
    pd.DataFrame(cal).to_csv(OUT / "test_calibration_deciles.csv", index=False)
    print(pd.DataFrame(cal).round(3).to_string())


if __name__ == "__main__":
    main()
