"""보고서 4.2: 젖산 수치 포함 vs 측정 여부만 — 개발 교차검증 예측의 차이와 95% 신뢰구간.

팀 스크립트(compare_tuned_xgboost_missingness.py)와 같은 폴드(StratifiedKFold 5, seed 42)·같은 함수로
교차검증 예측을 다시 만들고, 폴드별 AP 차이가 팀 기록(paired_fold_differences.csv)과 같을 때만 결과를 쓴다.
신뢰구간은 개발 환자 25,595명 단위 짝지은 부트스트랩 2,000회(seed 42). 환자 단위 예측은 저장하지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tune_three_models_nested_cv import fit, prepare  # noqa: E402

OUT = HERE / "outputs/report_extras"
SPLITS = ROOT / "data/processed/error_cleaned/splits"
TEAM = ROOT / "reports/xgboost_missingness_ablation"


def main():
    cols = json.loads((ROOT / "schemas/final_model_inputs.json").read_text(encoding="utf-8-sig"))["input_features"]
    params = json.loads((ROOT / "reports/three_models_nested_tuning/final_parameters.json").read_text())["xgboost"]["params"]
    xs, ys = [], []
    for split in ("train", "validation"):
        x = pd.read_csv(SPLITS / split / "X_features.csv", usecols=cols)[cols].copy()
        x["gender"] = x.gender.map({"F": 0, "M": 1})
        xs.append(x)
        ys.append(pd.read_csv(SPLITS / split / "y_target.csv").mortality_28d)
    x = pd.concat(xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    assert len(y) == 25595

    lactate = [c for c in cols if c.startswith("lab_Lactate_")]
    arms = {"full": cols,
            "measurement_only": [c for c in cols if c not in lactate or c in ("lab_Lactate_count", "lab_Lactate_missing")]}
    fold = np.zeros(len(y), int)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(x, y), 1):
        fold[va] = k
    oof = {arm: np.full(len(y), np.nan) for arm in arms}
    fold_ap = {arm: [] for arm in arms}
    for k in range(1, 6):
        tr, va = fold != k, fold == k
        for arm, cc in arms.items():
            a, b, _ = prepare(x.loc[tr, cc], x.loc[va, cc], cc, "native")
            oof[arm][va], _ = fit("xgboost", params, a, y[tr], b, cc)
            fold_ap[arm].append(average_precision_score(y[va], oof[arm][va]))
        print(f"fold {k}/5", flush=True)

    team = pd.read_csv(TEAM / "paired_fold_differences.csv")
    team_d = team[team.comparison == "native_full_minus_measurement_only"].sort_values("fold").average_precision.to_numpy()
    mine_d = np.array(fold_ap["full"]) - np.array(fold_ap["measurement_only"])
    assert np.abs(team_d - mine_d).max() < 1e-9, "팀 기록과 폴드별 차이가 다릅니다"

    yv = y.to_numpy()
    score = {"roc_auc": roc_auc_score, "average_precision": average_precision_score, "brier_score": brier_score_loss}
    rng = np.random.default_rng(42)
    boot = {m: [] for m in score}
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        for m, f in score.items():
            boot[m].append(f(yv[i], oof["full"][i]) - f(yv[i], oof["measurement_only"][i]))
    rows = []
    for m, f in score.items():
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        rows.append(dict(metric=m, full=f(yv, oof["full"]), measurement_only=f(yv, oof["measurement_only"]),
                         diff=f(yv, oof["full"]) - f(yv, oof["measurement_only"]), diff_lo=lo, diff_hi=hi))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "lactate_diff_ci.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
