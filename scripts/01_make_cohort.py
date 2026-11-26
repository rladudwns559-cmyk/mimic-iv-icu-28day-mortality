"""첫 ICU 입실의 24시간 landmark cohort를 생성합니다.

실행: python scripts/01_make_cohort.py
필요 패키지: pandas

공식 자료:
https://mimic.mit.edu/docs/iv/modules/hosp/patients.html
https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html

주의: dod 결측은 생존 증거가 아닙니다. 버전별 사망 추적 범위를
검증하지 않았으므로 '퇴원 후 1년간 사망 기록 없음' 가정도 사용하지 않습니다.
생존/사망 확인에 쓰는 미래 기록은 라벨용이며 모델 feature가 아닙니다.
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "subject_id", "hadm_id", "stay_id", "intime", "prediction_time",
    "outtime", "age", "gender", "death_time", "mortality_28d",
]


def parse_times(frame, columns):
    """빈 값은 허용하지만, 잘못된 날짜 문자열은 조용히 버리지 않습니다."""
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], errors="raise")


def make_cohort(icu, patients, admissions):
    """파일 입출력 없이 코호트를 계산하므로 작은 예제로도 검증할 수 있습니다."""
    icu, patients, admissions = icu.copy(), patients.copy(), admissions.copy()
    parse_times(icu, ["intime", "outtime"])
    parse_times(patients, ["dod"])
    parse_times(admissions, ["admittime", "dischtime", "deathtime"])
    assert icu["stay_id"].is_unique, "원본 stay_id 중복을 확인하세요."
    assert patients["subject_id"].is_unique, "patients의 환자 ID가 중복됩니다."
    assert admissions["hadm_id"].is_unique, "admissions의 입원 ID가 중복됩니다."
    assert icu[["subject_id", "hadm_id", "stay_id"]].notna().all().all()
    assert admissions[["subject_id", "hadm_id"]].notna().all().all()

    stats = {"원본 ICU stay 수": len(icu)}
    # 입실시각이 없는 환자는 어느 stay가 첫 번째인지 확정할 수 없습니다.
    # 이 환자의 나머지 stay도 제외하여 나중 입실을 첫 입실로 오인하지 않습니다.
    invalid_subjects = icu.loc[icu.intime.isna(), "subject_id"]
    valid = icu.loc[~icu.subject_id.isin(invalid_subjects)].copy()
    stats["입실시각 누락 환자 관련 제외 stay 수"] = len(icu) - len(valid)
    valid = valid.merge(patients, on="subject_id", how="left", validate="many_to_one")
    valid["age"] = valid.anchor_age + valid.intime.dt.year - valid.anchor_year
    # anchor_age=91인 고령 환자는 익명화된 나이이므로 정확한 실제 나이가 아닙니다.
    stats["나이 계산 불가 stay 수"] = int(valid.age.isna().sum())
    stats["성인 필터 후 수"] = int(valid.age.ge(18).sum())

    # 반드시 전체 입실에서 첫 stay를 먼저 정합니다. 첫 stay가 소아/조기퇴실이면
    # 해당 환자의 나중 성인 stay나 긴 stay로 대체하지 않습니다.
    first = valid.sort_values(["subject_id", "intime", "stay_id"]).drop_duplicates("subject_id")
    first = first.loc[first.age.ge(18)].copy()
    stats["첫 ICU stay 선택 후 환자 수"] = len(first)
    first = first.merge(
        admissions[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"],
        how="left", validate="many_to_one", indicator=True,
    )
    first["prediction_time"] = first.intime + pd.Timedelta(hours=24)
    first["window_end"] = first.prediction_time  # 이후 24시간 feature 추출용 별칭
    first["horizon_time"] = first.intime + pd.Timedelta(days=28)

    # 첫 입원의 deathtime만 보면 재입원 중 사망을 놓칩니다.
    # 환자의 모든 admissions에서 정확한 사망시각과 생존 관찰시각을 모읍니다.
    histories = {sid: group for sid, group in admissions.groupby("subject_id", sort=False)}
    rows = []
    for item in first.to_dict("records"):
        history = histories.get(item["subject_id"])
        prediction, horizon = item["prediction_time"], item["horizon_time"]
        dod = item["dod"]
        exacts = history.deathtime.dropna().unique() if history is not None else []
        exact = pd.Timestamp(exacts[0]) if len(exacts) == 1 else pd.NaT
        # 날짜만 있는 dod를 자정 사망으로 바꾸지 않습니다.
        # [death_date_lower, death_date_upper) 범위 안에서 사망한 것으로만 압니다.
        lower = dod.normalize() if pd.notna(dod) else pd.NaT
        upper = lower + pd.Timedelta(days=1) if pd.notna(lower) else pd.NaT
        item.update(
            death_time=exact, death_source="admissions.deathtime" if pd.notna(exact) else
            ("patients.dod_date_only" if pd.notna(lower) else "none"),
            death_date_lower=lower, death_date_upper=upper,
            mortality_28d="unknown", label_reason="insufficient_followup",
            survival_evidence="none", survival_confirmed_at=pd.NaT,
            exclusion_reason="", data_issue="",
        )

        # 서로 다른 사망 기록, 사망 이후의 입원 기록 등은 자동 판정하지 않습니다.
        issues = []
        if item["_merge"] != "both":
            issues.append("missing_index_admission")
        if len(exacts) > 1:
            issues.append("multiple_deathtimes")
        if pd.notna(exact) and pd.notna(lower) and exact.normalize() != lower:
            issues.append("deathtime_dod_date_conflict")
        if pd.notna(item["outtime"]) and item["outtime"] < item["intime"]:
            issues.append("outtime_before_intime")
        if history is not None:
            # 입실/입원 시각은 생존 관찰, 생존 퇴원(flag=0)은 그 시각까지 생존 근거입니다.
            # 사망 퇴원이나 flag 불명인 dischtime은 생존 근거로 사용하지 않습니다.
            observations = [(item["intime"], "icustays.intime")]
            observations += [(t, "admissions.admittime") for t in history.admittime.dropna()]
            observations += [(t, "admissions.dischtime_alive") for t in
                             history.loc[history.hospital_expire_flag.eq(0), "dischtime"].dropna()]
            # ICU outtime은 생존 퇴실인지 별도 표시가 없어 단독 증거로 쓰지 않습니다.
            for t, _ in observations:
                if (pd.notna(exact) and t > exact) or (
                    pd.isna(exact) and pd.notna(upper) and t >= upper
                ):
                    issues.append("observation_after_death")
                    break
            # 사망 퇴원인데 사망 날짜조차 없으면 생존 0 판정을 막습니다.
            if history.hospital_expire_flag.eq(1).any() and pd.isna(exact) and pd.isna(lower):
                issues.append("death_flag_without_death_date")
        else:
            observations = []
        item["data_issue"] = ";".join(issues)

        # 제외 수를 중복 집계하지 않도록 사망 → 퇴실 → 데이터 문제 순으로 적용합니다.
        if pd.notna(exact) and exact <= prediction:
            item["exclusion_reason"] = "death_at_or_before_prediction"
        elif pd.isna(exact) and pd.notna(upper) and upper <= prediction:
            item["exclusion_reason"] = "death_at_or_before_prediction"
        elif pd.notna(item["outtime"]) and item["outtime"] <= prediction:
            item["exclusion_reason"] = "icu_exit_at_or_before_prediction"
        elif issues:
            item["exclusion_reason"] = "data_issue"
        elif pd.isna(item["outtime"]):
            item["exclusion_reason"] = "missing_outtime"
        elif pd.isna(exact) and pd.notna(lower) and lower <= prediction < upper:
            # 사망일이 prediction_time과 같은 날이면 생존 24시간을 확정할 수 없습니다.
            item["exclusion_reason"] = "ambiguous_death_at_prediction"

        if not item["exclusion_reason"]:
            if pd.notna(exact):
                item["mortality_28d"] = int(exact <= horizon)
                item["label_reason"] = "exact_death_within_window" if exact <= horizon else "exact_death_after_horizon"
                if exact > horizon:
                    item["survival_evidence"] = "admissions.deathtime_after_horizon"
                    item["survival_confirmed_at"] = exact
            elif pd.notna(lower):
                if lower > prediction and upper <= horizon:
                    item["mortality_28d"] = 1
                    item["label_reason"] = "entire_death_date_within_window"
                elif lower > horizon:
                    item["mortality_28d"] = 0
                    item["label_reason"] = "death_date_after_horizon"
                    item["survival_evidence"] = "patients.dod_after_horizon"
                    item["survival_confirmed_at"] = lower
                else:
                    item["label_reason"] = "death_date_overlaps_horizon"
            else:
                evidence = sorted((t, source) for t, source in observations if t >= horizon)
                if evidence:
                    item["mortality_28d"] = 0
                    item["label_reason"] = "documented_alive_at_or_after_horizon"
                    item["survival_confirmed_at"], item["survival_evidence"] = evidence[0]
        rows.append(item)

    # 대상이 0명인 경우에도 같은 열 구조를 갖도록 열을 명시합니다.
    extra = ["death_time", "death_source", "death_date_lower", "death_date_upper",
             "mortality_28d", "label_reason", "survival_evidence", "survival_confirmed_at",
             "exclusion_reason", "data_issue"]
    audit = pd.DataFrame(rows, columns=list(first.columns) + extra).drop(columns="_merge")
    eligible = audit.loc[audit.exclusion_reason.eq("")].copy()
    unknown = eligible.loc[eligible.mortality_28d.eq("unknown")].copy()
    cohort = eligible.loc[~eligible.mortality_28d.eq("unknown")].copy()
    cohort["mortality_28d"] = cohort.mortality_28d.astype("int64")
    for frame in (cohort, unknown, audit):
        frame.drop(columns=["anchor_age", "anchor_year"], inplace=True, errors="ignore")
    cohort = cohort[REQUIRED + [c for c in cohort if c not in REQUIRED]]

    counts = audit.exclusion_reason.value_counts()
    stats["24시간 이전 또는 동일 시점 사망 제외 수"] = int(counts.get("death_at_or_before_prediction", 0))
    stats["24시간 이전 또는 동일 시점 ICU 퇴실 제외 수 (사망 제외 후)"] = int(counts.get("icu_exit_at_or_before_prediction", 0))
    for reason in ["data_issue", "missing_outtime", "ambiguous_death_at_prediction"]:
        stats[f"추가 제외: {reason}"] = int(counts.get(reason, 0))
    stats["24시간 조건 충족 수 (unknown 포함)"] = len(eligible)
    stats["28일 라벨 unknown 수"] = len(unknown)
    stats["최종 cohort 크기 (0/1 확정)"] = len(cohort)
    assert len(first) == len(audit)
    assert len(audit) == int(audit.exclusion_reason.ne("").sum()) + len(unknown) + len(cohort)
    validate(cohort)
    return cohort, unknown, audit, stats


def validate(cohort):
    """최종 학습용 코호트는 0/1 확정 환자만 포함합니다."""
    assert len(cohort) == cohort.subject_id.nunique(), "한 환자당 한 행이어야 합니다."
    assert cohort.subject_id.is_unique
    assert cohort.stay_id.is_unique
    assert cohort.age.ge(18).all()
    assert cohort.prediction_time.eq(cohort.intime + pd.Timedelta(hours=24)).all()
    assert cohort.outtime.gt(cohort.prediction_time).all()
    assert cohort.mortality_28d.isin([0, 1]).all()
    assert cohort.loc[cohort.mortality_28d.eq(0), "survival_evidence"].ne("none").all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    icu = pd.read_csv(args.raw_dir / "icustays.csv", usecols=[
        "subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    patients = pd.read_csv(args.raw_dir / "patients.csv.gz", usecols=[
        "subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    admissions = pd.read_csv(args.raw_dir / "admissions.csv.gz", usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag"])
    cohort, unknown, audit, stats = make_cohort(icu, patients, admissions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_dir / "cohort_base.csv", index=False)
    unknown.to_csv(args.output_dir / "cohort_unknown.csv", index=False)
    audit.to_csv(args.output_dir / "cohort_audit.csv", index=False)
    print("\n검증 결과")
    for name, count in stats.items():
        print(f"{name}: {count:,}")
    print("subject_id 중복 개수:", int(cohort.subject_id.duplicated().sum()))
    print("stay_id 중복 개수:", int(cohort.stay_id.duplicated().sum()))
    print("\n라벨별 개수와 비율 (분모: 최종 0/1 cohort)")
    for label in (0, 1):
        count = int(cohort.mortality_28d.eq(label).sum())
        ratio = count / len(cohort) if len(cohort) else 0
        print(f"mortality_28d={label}: {count:,} ({ratio:.2%})")
    print("\nunknown 사유별 집계")
    print(unknown.label_reason.value_counts().to_string())
    print("\n0 라벨의 생존 확인 근거")
    print(cohort.loc[cohort.mortality_28d.eq(0), "survival_evidence"].value_counts().to_string())
    print("\n필수 변수 결측값 수")
    print(cohort[["age", "gender", "intime", "prediction_time"]].isna().sum().to_string())
    print(f"\n저장 위치: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
