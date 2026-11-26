"""사망/퇴실 경계와 생존 근거를 가상 환자로 검증합니다."""
import importlib.util
from pathlib import Path
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location("cohort", Path(__file__).with_name("01_make_cohort.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CohortTests(unittest.TestCase):
    def build(self, death=None, dod=None, out="2200-01-05 12:00:00", discharge="2200-01-06 12:00:00", flag=0, later=None):
        icu = pd.DataFrame([dict(subject_id=1, hadm_id=10, stay_id=100,
                                 intime="2200-01-01 12:00:00", outtime=out)])
        patients = pd.DataFrame([dict(subject_id=1, anchor_age=40, anchor_year=2200, gender="F", dod=dod)])
        admissions = pd.DataFrame([dict(subject_id=1, hadm_id=10, admittime="2200-01-01 10:00:00",
                                        dischtime=discharge, deathtime=death, hospital_expire_flag=flag)])
        if later:
            admissions = pd.concat([admissions, pd.DataFrame([dict(subject_id=1, hadm_id=11,
                admittime=later, dischtime=None, deathtime=None, hospital_expire_flag=0)])], ignore_index=True)
        return module.make_cohort(icu, patients, admissions)

    def test_no_death_is_unknown(self):
        cohort, unknown, _, _ = self.build()
        self.assertEqual(len(cohort), 0)
        self.assertEqual(unknown.mortality_28d.tolist(), ["unknown"])

    def test_death_at_prediction_excluded(self):
        cohort, unknown, audit, _ = self.build(death="2200-01-02 12:00:00", flag=1,
                                              discharge="2200-01-02 12:00:00")
        self.assertEqual(len(cohort) + len(unknown), 0)
        self.assertEqual(audit.exclusion_reason.iloc[0], "death_at_or_before_prediction")

    def test_exit_at_prediction_excluded(self):
        _, _, audit, _ = self.build(out="2200-01-02 12:00:00")
        self.assertEqual(audit.exclusion_reason.iloc[0], "icu_exit_at_or_before_prediction")

    def test_exact_horizon_death_positive(self):
        cohort, _, _, _ = self.build(death="2200-01-29 12:00:00", flag=1,
                                     discharge="2200-01-29 12:00:00")
        self.assertEqual(cohort.mortality_28d.tolist(), [1])

    def test_date_at_prediction_ambiguous(self):
        _, _, audit, _ = self.build(dod="2200-01-02", discharge="2200-01-02 12:00:00", flag=1)
        self.assertEqual(audit.exclusion_reason.iloc[0], "ambiguous_death_at_prediction")

    def test_date_at_horizon_unknown(self):
        cohort, unknown, _, _ = self.build(dod="2200-01-29")
        self.assertEqual(len(cohort), 0)
        self.assertEqual(unknown.label_reason.tolist(), ["death_date_overlaps_horizon"])

    def test_date_inside_window_positive_without_fake_time(self):
        cohort, _, _, _ = self.build(dod="2200-01-10")
        self.assertEqual(cohort.mortality_28d.tolist(), [1])
        self.assertTrue(cohort.death_time.isna().all())

    def test_later_alive_admission_proves_zero(self):
        cohort, _, _, _ = self.build(later="2200-02-01 00:00:00")
        self.assertEqual(cohort.mortality_28d.tolist(), [0])
        self.assertEqual(cohort.survival_evidence.tolist(), ["admissions.admittime"])

    def test_alive_discharge_at_horizon_proves_zero(self):
        cohort, _, _, _ = self.build(discharge="2200-01-29 12:00:00")
        self.assertEqual(cohort.mortality_28d.tolist(), [0])

    def test_conflicting_dates_excluded(self):
        _, _, audit, _ = self.build(death="2200-01-10 12:00:00", dod="2200-01-11", flag=1)
        self.assertEqual(audit.exclusion_reason.iloc[0], "data_issue")

    def test_first_stay_not_replaced(self):
        icu = pd.DataFrame([
            dict(subject_id=1, hadm_id=10, stay_id=100, intime="2200-01-01", outtime="2200-01-02"),
            dict(subject_id=1, hadm_id=10, stay_id=101, intime="2200-01-03", outtime="2200-01-10"),
        ])
        patients = pd.DataFrame([dict(subject_id=1, anchor_age=40, anchor_year=2200, gender="F", dod=None)])
        admissions = pd.DataFrame([dict(subject_id=1, hadm_id=10, admittime="2200-01-01",
            dischtime="2200-02-10", deathtime=None, hospital_expire_flag=0)])
        cohort, unknown, audit, _ = module.make_cohort(icu, patients, admissions)
        self.assertEqual(len(cohort) + len(unknown), 0)
        self.assertEqual(audit.stay_id.tolist(), [100])


if __name__ == "__main__":
    unittest.main()
