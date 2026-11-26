"""작은 CSV에서 INR만 복원하고 다른 관측·특징을 보존하는지 검증한다."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import pipeline_common as common
import repair_inr


class InrRepairTest(unittest.TestCase):
    """원본 재읽기 금지·수치 오류 유지·재실행 중복 방지를 확인한다."""
    def test_incremental_restore(self):
        original_config = copy.deepcopy(common.CONFIG)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw,out = root/'raw',root/'out'
                raw.mkdir()
                out.mkdir()
                # 현재 정책과 무관하게 수정 전의 INR 전량 제외 상태를 재현한다.
                fixture_config = copy.deepcopy(original_config)
                next(r for r in fixture_config['lab'] if r[1]==51237)[4] = ['_','ratio']
                common.CONFIG.clear()
                common.CONFIG.update(fixture_config)
                config_path = root/'config.json'
                config_path.write_text(json.dumps(fixture_config),encoding='utf-8')
                for kind,name in [('vital','d_items'),('lab','d_labitems')]:
                    pd.DataFrame([dict(itemid=r[1],label=r[2],unitname=r[4][0]) for r in original_config[kind]]).to_csv(raw/f'{name}.csv',index=False)
                cohort = pd.DataFrame(dict(subject_id=[1,2],hadm_id=[10,20],stay_id=[100,200],
                            intime=['2020-01-01']*2,prediction_time=['2020-01-02']*2,
                            outtime=['2020-01-03']*2,mortality_28d=[0,1]))
                cp = root/'cohort.csv'
                cohort.to_csv(cp,index=False)
                pd.DataFrame(dict(subject_id=[1,1],hadm_id=[10,10],stay_id=[100,100],
                     charttime=['2020-01-01','2020-01-01 01:00:00'],storetime=['2020-01-01 02:00:00']*2,
                     itemid=[220045]*2,valuenum=[60,80],valueuom=['bpm']*2)).to_csv(raw/'chartevents.csv',index=False)
                # 1.9/1.9/2.0의 평균은 CSV 재파싱 시 1 ULP 차이를 재현한다.
                pd.DataFrame(dict(subject_id=[1]*7,hadm_id=[10]*7,
                     charttime=['2020-01-01 01:00:00']*7,storetime=['2020-01-01 02:00:00']*7,
                     itemid=[51237]*5+[50931]*2,valuenum=[1.9,1.9,2.0,np.nan,-1,100,200],
                     valueuom=['','','','','','mg/dL',''])).to_csv(raw/'labevents.csv',index=False)
                args = SimpleNamespace(raw_dir=raw,cohort=cp,output_dir=out,chunksize=2,max_chunks=None)
                with patch.object(common,'CONFIG_PATH',config_path),patch.object(repair_inr,'CONFIG_PATH',config_path):
                    for kind in ['vital','lab']:
                        common.extract(args,kind)
                        common.aggregate(args,kind)
                    common.merge(args)
                    common.validate(args)
                    initial = pd.read_csv(out/'lab_features.csv')
                    read_csv = pd.read_csv
                    def guarded(path,*a,**kw):
                        if Path(path).name in ['chartevents.csv','labevents.csv']:
                            raise AssertionError('복원 중 원본 CSV를 읽었습니다.')
                        return read_csv(path,*a,**kw)
                    with patch.object(pd,'read_csv',side_effect=guarded):
                        repair_inr.repair(args)
                        repair_inr.repair(args)
                    new = pd.read_csv(out/'lab_features.csv')
                    self.assertEqual(new.loc[0,'lab_INR_count'],3)
                    self.assertAlmostEqual(new.loc[0,'lab_INR_mean'],np.mean([1.9,1.9,2.0]))
                    self.assertAlmostEqual(new.loc[0,'lab_INR_std'],np.std([1.9,1.9,2.0],ddof=1))
                    self.assertEqual(new.loc[1,'lab_INR_missing'],1)
                    pd.testing.assert_frame_equal(initial.filter(regex='^(?!lab_INR_)'),new.filter(regex='^(?!lab_INR_)'),check_exact=True)
                    result = json.loads((out/'inr_repair_report.json').read_text(encoding='utf-8'))
                    self.assertEqual(result['restored_INR_observations'],3)
                    self.assertEqual(result['remaining_invalid_INR'],2)
                    self.assertEqual(result['after_INR_missing_rate'],0.5)
                    # 설정 지문과 파티션·진단도 일반 검증기와 일치해야 한다.
                    common.validate(args)
        finally:
            common.CONFIG.clear()
            common.CONFIG.update(original_config)


if __name__ == '__main__':
    unittest.main()
