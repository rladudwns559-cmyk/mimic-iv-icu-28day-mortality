"""발표자료 9쪽 그림: 테스트 ROC · 정밀도–재현율 · 확률 보정 (한국어판).
report_extras.py의 test_curves.png(보고서 그림 3, 영어)와 같은 모델·같은 계산이며 글자만 한국어다.
최종 모델은 outputs/model.ubj를 불러 쓰고, 비교 모델(첫 ICU 유형 제외)은 팀 기록과 소수 여섯째 자리까지 같을 때만 쓴다.
저장하는 것은 그림 한 장뿐이다(환자 단위 자료 없음)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import train as T
from report_extras import NATIVE_REF, OUT, metrics

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
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

    booster = xgb.Booster(); booster.load_model(str(T.OUT / "model.ubj"))
    xt = T.design(test, test_units, std_cols, ref["icu_categories"])
    p_final = booster.predict(xgb.DMatrix(xt, feature_names=ref["feature_columns"], missing=np.nan))
    for k, v in metrics(yt, p_final).items():
        assert abs(v - ref["metrics"][k]) < 1e-6, k

    assert columns + [f"{c}__was_missing" for c in std_cols] == nref["feature_columns"]
    m = xgb.XGBClassifier(**nref["xgboost"])
    m.fit(T.design(dev, pd.Series(["_"] * len(dev)), std_cols, []), ydev)
    p_base = m.predict_proba(T.design(test, pd.Series(["_"] * len(test)), std_cols, []))[:, 1]
    for k, v in metrics(yt, p_base).items():
        assert abs(v - nref["metrics"][k]) < 1e-6, k

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for p, lab, col, ls in ((p_final, "최종 모델 (첫 ICU 유형 포함)", "#005eb8", "-"),
                            (p_base, "첫 ICU 유형 제외", "#768692", "--")):
        fpr, tpr, _ = roc_curve(yt, p)
        ax[0].plot(fpr, tpr, ls, color=col, lw=2, label=f"{lab}: AUC {roc_auc_score(yt, p):.3f}")
        pr, rc, _ = precision_recall_curve(yt, p)
        ax[1].plot(rc, pr, ls, color=col, lw=2, label=f"{lab}: AP {average_precision_score(yt, p):.3f}")
    ax[0].plot([0, 1], [0, 1], ":", color="gray", lw=1)
    ax[0].set(xlabel="1 - 특이도 (위양성률)", ylabel="민감도", title="A. ROC 곡선")
    ax[1].axhline(yt.mean(), ls=":", color="gray", lw=1, label=f"무작위 기준 (사망 비율 {yt.mean():.3f})")
    ax[1].set(xlabel="재현율 (민감도)", ylabel="정밀도", title="B. 정밀도–재현율 곡선")
    for t, mk in ((0.5, "o"), (0.3, "s")):
        pred = p_final >= t
        ax[0].plot(pred[yt == 0].mean(), pred[yt == 1].mean(), mk, color="#d5281b", ms=7, label=f"경보 기준 {t}")
    cal = T.calibration(yt, p_final)
    mp = [r["mean_predicted"] for r in cal]; ob = [r["observed"] for r in cal]
    se = [np.sqrt(r["observed"] * (1 - r["observed"]) / r["n"]) for r in cal]
    ax[2].plot([0, 1], [0, 1], ":", color="gray", lw=1, label="완전 보정")
    ax[2].errorbar(mp, ob, yerr=1.96 * np.array(se), fmt="o-", color="#005eb8", lw=1.5, capsize=3,
                   label="최종 모델 (위험 10분위)")
    ax[2].set(xlabel="평균 예측 확률", ylabel="실제 28일 사망률", title="C. 확률 보정 (테스트, 10분위)",
              xlim=(0, 1), ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=8.5, loc="lower right" if a is not ax[1] else "upper right", frameon=False)
        a.grid(alpha=.25); a.set_aspect("equal") if a is not ax[1] else None
    ax[1].set(xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(OUT / "test_curves_ko.png", dpi=200)
    print(OUT / "test_curves_ko.png")


if __name__ == "__main__":
    main()
