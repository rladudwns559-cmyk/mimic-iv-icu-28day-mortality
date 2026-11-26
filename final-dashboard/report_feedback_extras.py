"""13주차 보고서 피드백 반영용 집계 (환자 단위 자료는 저장하지 않음).

만드는 것 (outputs/report_extras/)
  exclusion_comparison.csv   제외군(24시간 관찰 미충족, 28일 결과 미확정)과 분석군 비교 + 검정
  table1_tests.csv           분석군 생존·사망 비교(중앙값[IQR], n(%)) + Mann–Whitney U / χ² 검정
  cv_with_sd.csv             팀 reports/의 교차검증 평균 ± 표준편차 모음
  proposal_numbers.csv       하루 입실 20명 가정 시 경보·실제 사망·오경보 인원
  check_numbers.json          보고서에 이미 적힌 숫자(간기능 결측, 카이제곱, 효과 크기)를 다시 계산해 대조
보고서에 이미 있는 숫자와 다르면 멈춘다(assert). random 요소 없음.
실행: ..\\.venv\\Scripts\\python report_feedback_extras.py  (MIMIC-IV 3.1 원자료와 팀 전처리 결과 필요)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P = ROOT / "data/processed"
EC = P / "error_cleaned"
REP = ROOT / "reports"
OUT = HERE / "outputs/report_extras"
RAW_ADMISSIONS = [ROOT / "data/raw/admissions.csv.gz", ROOT.parent / "admissions.csv.gz"]
ICU_KO = {"Medical Intensive Care Unit (MICU)": "MICU", "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
          "Surgical Intensive Care Unit (SICU)": "SICU", "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
          "Coronary Care Unit (CCU)": "CCU", "Trauma SICU (TSICU)": "TSICU", "Neuro Intermediate": "Neuro Intermediate",
          "Neuro Surgical Intensive Care Unit (Neuro SICU)": "Neuro SICU", "Neuro Stepdown": "Neuro Stepdown"}


def med_iqr(x):
    q = np.percentile(x.dropna(), [50, 25, 75])
    return f"{q[0]:.1f} [{q[1]:.1f}–{q[2]:.1f}]"


def n_pct(mask):
    return f"{int(mask.sum()):,} ({mask.mean() * 100:.1f})"


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def mwu(a, b):
    return stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided").pvalue


def chi(a_mask, b_mask):
    t = np.array([[a_mask.sum(), (~a_mask).sum()], [b_mask.sum(), (~b_mask).sum()]])
    return stats.chi2_contingency(t)[1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    icu = pd.read_csv(ROOT / "data/raw/icustays.csv", usecols=["stay_id", "first_careunit"])
    adm_path = next(p for p in RAW_ADMISSIONS if p.exists())
    adm = pd.read_csv(adm_path, usecols=["hadm_id", "hospital_expire_flag"])

    # ---------- 1. 제외군 vs 분석군 ----------
    c = pd.read_csv(P / "cohort_audit.csv", usecols=["hadm_id", "stay_id", "intime", "outtime", "gender", "age",
                                                      "mortality_28d", "exclusion_reason"], dtype={"mortality_28d": str})
    c = c.merge(icu, on="stay_id", how="left").merge(adm, on="hadm_id", how="left")
    c["los"] = (pd.to_datetime(c.outtime) - pd.to_datetime(c.intime)).dt.total_seconds() / 86400
    grp = np.where(c.exclusion_reason.notna(), "excl_24h", np.where(c.mortality_28d == "unknown", "excl_label", "analysis"))
    c["grp"] = grp
    n = c.grp.value_counts().to_dict()
    assert n == {"analysis": 30112, "excl_label": 21529, "excl_24h": 13725}, n
    A, L, E = (c[c.grp == g] for g in ("analysis", "excl_label", "excl_24h"))
    rows = [["환자 수", f"{len(A):,}", f"{len(L):,}", "", f"{len(E):,}", ""]]
    rows.append(["나이, 세 (중앙값 [IQR])", med_iqr(A.age), med_iqr(L.age), pfmt(mwu(A.age, L.age)), med_iqr(E.age), pfmt(mwu(A.age, E.age))])
    rows.append(["남성, n (%)", n_pct(A.gender == "M"), n_pct(L.gender == "M"), pfmt(chi(A.gender == "M", L.gender == "M")),
                 n_pct(E.gender == "M"), pfmt(chi(A.gender == "M", E.gender == "M"))])
    rows.append(["ICU 재원, 일 (중앙값 [IQR])", med_iqr(A.los), med_iqr(L.los), pfmt(mwu(A.los, L.los)), med_iqr(E.los), pfmt(mwu(A.los, E.los))])
    hd = lambda d: d.hospital_expire_flag == 1
    rows.append(["병원 내 사망, n (%)", n_pct(hd(A)), n_pct(hd(L)), pfmt(chi(hd(A), hd(L))), n_pct(hd(E)), pfmt(chi(hd(A), hd(E)))])
    for u in ["Medical Intensive Care Unit (MICU)", "Cardiac Vascular Intensive Care Unit (CVICU)", "Neuro Surgical Intensive Care Unit (Neuro SICU)"]:
        m = lambda d: d.first_careunit == u
        rows.append([f"첫 ICU: {ICU_KO[u]}, n (%)", n_pct(m(A)), n_pct(m(L)), pfmt(chi(m(A), m(L))), n_pct(m(E)), pfmt(chi(m(A), m(E)))])
    ex = pd.DataFrame(rows, columns=["항목", "분석군", "28일 결과 미확정", "p (분석군 vs 미확정)", "24시간 관찰 미충족", "p (분석군 vs 24시간)"])
    ex.to_csv(OUT / "exclusion_comparison.csv", index=False, encoding="utf-8-sig")
    print(ex.to_string())

    # ---------- 2. Table 1: 분석군 생존 vs 사망 + 검정 ----------
    X = pd.read_csv(EC / "X_features.csv")
    y = pd.read_csv(EC / "y_target.csv").iloc[:, 0].astype(int)
    ids = pd.read_csv(EC / "row_ids.csv")
    unit = ids.merge(icu, on="stay_id", how="left").first_careunit
    S, D = (y == 0).values, (y == 1).values
    t1 = [["환자 수", f"{S.sum():,}", f"{D.sum():,}", ""]]
    t1.append(["나이, 세 (중앙값 [IQR])", med_iqr(X.age[S]), med_iqr(X.age[D]), pfmt(mwu(X.age[S], X.age[D]))])
    male = X.gender == "M"
    t1.append(["남성, n (%)", n_pct(male[S]), n_pct(male[D]), pfmt(chi(male[S].values, male[D].values))])
    ct = pd.crosstab(unit.where(unit.map(unit.value_counts()) >= 100, "Other"), y)
    chi2_icu, p_icu, dof_icu, _ = stats.chi2_contingency(ct)
    t1.append([f"첫 ICU 유형 (10개 범주)", "", "", pfmt(p_icu)])
    for u in ["Medical Intensive Care Unit (MICU)", "Cardiac Vascular Intensive Care Unit (CVICU)", "Neuro Surgical Intensive Care Unit (Neuro SICU)"]:
        t1.append([f"   {ICU_KO[u]}, n (%)", n_pct((unit == u)[S]), n_pct((unit == u)[D]), ""])
    for col, name in [("vital_HR_mean", "심박수, 회/분"), ("vital_MAP_mean", "평균동맥압, mmHg"), ("vital_RR_mean", "호흡수, 회/분"),
                      ("lab_Lactate_mean", "젖산, mmol/L"), ("lab_Creatinine_mean", "크레아티닌, mg/dL"),
                      ("lab_Bicarbonate_mean", "중탄산염, mEq/L"), ("lab_AnionGap_mean", "음이온차, mEq/L")]:
        t1.append([f"{name} (중앙값 [IQR])", med_iqr(X[col][S]), med_iqr(X[col][D]), pfmt(mwu(X[col][S], X[col][D]))])
    for col, name in [("lab_Lactate_missing", "젖산 측정"), ("lab_INR_missing", "INR 측정")]:
        meas = X[col] == 0
        t1.append([f"{name}, n (%)", n_pct(meas[S]), n_pct(meas[D]), pfmt(chi(meas[S].values, meas[D].values))])
    T1 = pd.DataFrame(t1, columns=["항목", "생존", "28일 사망", "p"])
    T1.to_csv(OUT / "table1_tests.csv", index=False, encoding="utf-8-sig")
    print(T1.to_string())

    # ---------- 3. 교차검증 평균 ± SD ----------
    cv = []
    a = pd.read_csv(REP / "three_models_imputation_5fold/cv_summary.csv")
    for r in a.itertuples():
        cv.append(["기본 비교", f"{r.model} · {r.method}", r.roc_auc_mean, r.roc_auc_std, r.average_precision_mean, r.average_precision_std, r.brier_score_mean, r.brier_score_std])
    a = pd.read_csv(REP / "three_models_nested_tuning/nested_cv_summary.csv")
    for r in a.itertuples():
        cv.append(["중첩 교차검증", r.model, r.roc_auc_mean, r.roc_auc_std, r.average_precision_mean, r.average_precision_std, r.brier_score_mean, r.brier_score_std])
    a = pd.read_csv(REP / "xgboost_missingness_ablation/cv_summary.csv")
    for r in a.itertuples():
        cv.append(["결측·젖산", r.arm, r.roc_auc_mean, r.roc_auc_std, r.average_precision_mean, r.average_precision_std, r.brier_score_mean, r.brier_score_std])
    a = pd.read_csv(REP / "xgboost_indicator_icu_comparison/cv_summary.csv")
    for r in a.itertuples():
        cv.append(["지시변수·ICU", r.arm, r.roc_auc_mean, r.roc_auc_std, r.average_precision_mean, r.average_precision_std, r.brier_score_mean, r.brier_score_std])
    CV = pd.DataFrame(cv, columns=["비교", "조건", "roc_auc", "roc_auc_sd", "ap", "ap_sd", "brier", "brier_sd"])
    CV.to_csv(OUT / "cv_with_sd.csv", index=False, encoding="utf-8-sig")
    print(CV.round(4).to_string())

    # ---------- 4. 제안의 숫자 조건 ----------
    cm = pd.read_csv(OUT / "test_confusion.csv")
    prop = []
    for r in cm.itertuples():
        nt = r.tp + r.fp + r.fn + r.tn
        per = 20 / nt
        prop.append(dict(threshold=r.threshold, alert_rate=(r.tp + r.fp) / nt, alerts_per_20=(r.tp + r.fp) * per,
                         true_deaths_per_20=r.tp * per, false_alarms_per_20=r.fp * per, missed_deaths_per_20=r.fn * per))
    PR = pd.DataFrame(prop)
    PR.to_csv(OUT / "proposal_numbers.csv", index=False)
    print(PR.round(2).to_string())

    # ---------- 5. 보고서에 이미 적힌 숫자 대조 ----------
    lf = pd.read_csv(P / "lab_features.csv", usecols=["lab_BilirubinTotal_missing", "lab_ALT_missing", "lab_AST_missing"]) == 1
    rates = (lf.mean() * 100).round(2).to_dict()
    assert rates == {"lab_BilirubinTotal_missing": 50.58, "lab_ALT_missing": 50.31, "lab_AST_missing": 49.94}, rates
    all_m, any_m = lf.all(axis=1), lf.any(axis=1)
    assert (int(all_m.sum()), round(all_m.mean() * 100, 2), round(all_m.sum() / any_m.sum() * 100, 2)) == (14806, 49.17, 95.01)
    assert (round(chi2_icu, 2), dof_icu) == (1285.67, 9)
    cramers_v = float(np.sqrt(chi2_icu / ct.values.sum()))
    excl = pd.read_csv(EC / "exclusion_summary.csv")
    assert (int(excl.observations.sum()), int(pd.read_csv(EC / "excluded_observations.csv", usecols=["stay_id"]).stay_id.nunique())) == (176, 146)
    chk = dict(liver_missing_pct=rates, liver_all_missing=int(all_m.sum()), chi2_icu=round(chi2_icu, 2), dof=int(dof_icu),
               cramers_v=round(cramers_v, 3), cleaned_observations=176, cleaned_stays=146)
    (OUT / "check_numbers.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
    print("보고서 숫자와 일치:", chk)


if __name__ == "__main__":
    main()
