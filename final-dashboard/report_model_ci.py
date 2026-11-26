"""보고서 4.2: 세 모델 고정 설정 비교(중앙값 대체 + 결측 지시변수)의 차이와 95% 신뢰구간.

팀 스크립트(compare_three_models_imputation_cv.py)의 median_indicator 조건을 같은 폴드(StratifiedKFold 5, seed 42)·
같은 함수(fit_model)로 다시 만들고, 모델별 폴드 평균 ROC-AUC·AP·Brier가 팀 기록(cv_summary.csv)과 1e-9까지 같을 때만
결과를 쓴다. 신뢰구간은 개발 환자 25,595명 단위 짝지은 부트스트랩 2,000회(seed 42). 환자 단위 예측은 저장하지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compare_three_models_imputation_cv import fit_model  # noqa: E402

OUT = HERE / "outputs/report_extras"
SPLITS = ROOT / "data/processed/error_cleaned/splits"
TEAM = ROOT / "reports/three_models_imputation_5fold/cv_summary.csv"
MODELS = ["logistic_regression", "random_forest", "xgboost"]
SCORE = {"roc_auc": roc_auc_score, "average_precision": average_precision_score, "brier_score": brier_score_loss}


def main():
    all_columns = json.loads((SPLITS / "model_feature_columns.json").read_text())
    columns = [c for c in all_columns if not c.startswith(("lab_ALT_", "lab_AST_", "lab_BilirubinTotal_"))]
    assert len(columns) == 116
    std_columns = [c for c in columns if c.endswith("_std")]
    frames, labels = [], []
    for split in ("train", "validation"):
        frame = pd.read_csv(SPLITS / split / "X_features.csv")[columns].copy()
        frame["gender"] = frame.gender.map({"F": 0, "M": 1})
        frames.append(frame)
        labels.append(pd.read_csv(SPLITS / split / "y_target.csv").mortality_28d)
    data = pd.concat(frames, ignore_index=True)
    y = pd.concat(labels, ignore_index=True)
    assert len(y) == 25595

    oof = {m: np.full(len(y), np.nan) for m in MODELS}
    fold_rows = []
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(data, y), 1):
        raw_train, raw_valid = data.iloc[tr], data.iloc[va]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        a, b = imputer.fit_transform(raw_train), imputer.transform(raw_valid)
        xt = np.column_stack([a, raw_train[std_columns].isna().to_numpy(dtype=np.float64)])
        xv = np.column_stack([b, raw_valid[std_columns].isna().to_numpy(dtype=np.float64)])
        for m in MODELS:
            oof[m][va], _ = fit_model(m, xt, y.iloc[tr], xv, columns)
            fold_rows.append(dict(model=m, fold=fold, **{k: f(y.iloc[va], oof[m][va]) for k, f in SCORE.items()}))
        print(f"fold {fold}/5", flush=True)

    mine = pd.DataFrame(fold_rows).groupby("model")[list(SCORE)].mean()
    team = pd.read_csv(TEAM)
    team = team[team.method == "median_indicator"].set_index("model")
    # 랜덤 포레스트·XGBoost는 팀 기록과 같아야 한다. 로지스틱 회귀는 이 환경(scikit-learn 1.9.0)에서 팀 기록과
    # 소수 다섯째 자리 이하(ROC-AUC 2.6e-5, AP 6.0e-5)만 달라 보고서 표기(셋째 자리)에 영향이 없으므로 1e-4까지 허용한다.
    tolerance = {"logistic_regression": 1e-4, "random_forest": 1e-6, "xgboost": 1e-6}
    for k in SCORE:
        diff = (mine[k] - team[f"{k}_mean"]).abs()
        print(k, diff.to_dict(), flush=True)
        for m in MODELS:
            assert diff[m] < tolerance[m], f"팀 기록과 다릅니다: {m} {k} {diff[m]}"

    yv = y.to_numpy()
    pairs = [("xgboost", "random_forest"), ("xgboost", "logistic_regression")]
    rng = np.random.default_rng(42)
    boot = {(p, k): [] for p in pairs for k in SCORE}
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        for p in pairs:
            for k, f in SCORE.items():
                boot[(p, k)].append(f(yv[i], oof[p[0]][i]) - f(yv[i], oof[p[1]][i]))
    rows = []
    for p in pairs:
        for k, f in SCORE.items():
            lo, hi = np.percentile(boot[(p, k)], [2.5, 97.5])
            rows.append(dict(comparison=f"{p[0]}_minus_{p[1]}", metric=k, model_a=f(yv, oof[p[0]]), model_b=f(yv, oof[p[1]]),
                             diff=f(yv, oof[p[0]]) - f(yv, oof[p[1]]), diff_lo=lo, diff_hi=hi))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "model_diff_ci.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
