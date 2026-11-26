"""작은 합성 CSV로 시간 경계·단위·연결·재시작을 검증한다."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline_common import CONFIG, aggregate, clean, cohort, extract, merge, validate


class FeaturePipelineTest(unittest.TestCase):
    """실제 환자 데이터 없이 누락·오연결·청크 경계 사례를 재현한다."""

    def test_numeric_and_units(self):
        """C/F 반전, SpO2 경계, 음수 anion gap, 잘못된 단위를 구분한다."""
        f = pd.DataFrame({'itemid':[223761,223762,220277,220277,220277,220045,220045],
                          'valuenum':['98.6','37','0','100','101','inf','90'],
                          'valueuom':['°F','°C','%','%','%','bpm','mmHg'],
                          'storetime':pd.to_datetime(['2020-01-01']*7),
                          'prediction_time':pd.to_datetime(['2020-01-02']*7)})
        q = clean(f, 'vital')
        self.assertTrue(np.allclose(q.value.iloc[:2],[37,37]))
        self.assertEqual(q.reason.tolist(),['','','impossible_value','','impossible_value',
                                         'invalid_numeric','unknown_or_mismatched_unit'])
        lab = pd.DataFrame({'itemid':[50868,50912,50931], 'valuenum':['-5','-1','9000'],
                            'valueuom':['mEq/L','mg/dL','mg/dL'],
                            'storetime':pd.to_datetime(['2020-01-01']*3),
                            'prediction_time':pd.to_datetime(['2020-01-02']*3)})
        q = clean(lab,'lab')
        self.assertEqual(q.reason.tolist(),['','impossible_value',''])
        self.assertTrue(q.extreme.iloc[2])

    def test_end_to_end_and_resume(self):
        """복수 stay·중첩 구간·경계 배제·결측 stay 보존·재실행을 검증한다."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, out = root/'raw', root/'out'
            raw.mkdir()
            out.mkdir()
            for kind, name in [('vital','d_items'),('lab','d_labitems')]:
                pd.DataFrame([{'itemid':r[1],'label':r[2], 'unitname':r[4][0]} for r in CONFIG[kind]]).to_csv(raw/f'{name}.csv',index=False)
            c = pd.DataFrame({'subject_id':[1,1,2], 'hadm_id':[10,10,20], 'stay_id':[100,101,200],
                              'intime':['2020-01-01','2020-01-01 12:00:00','2020-01-03'],
                              'prediction_time':['2020-01-02','2020-01-02 12:00:00','2020-01-04'],
                              'outtime':['2020-01-04']*3, 'mortality_28d':[0,0,1]})
            cp = root/'cohort.csv'
            c.to_csv(cp,index=False)
            original = cp.read_bytes()
            pd.DataFrame({'subject_id':[1]*5,'hadm_id':[10]*5,'stay_id':[100]*5,
                          'charttime':['2020-01-01','2020-01-01 01:00:00','2020-01-02',
                                       '2019-12-31 23:59:59','2020-01-01 02:00:00'],
                          'storetime':['2020-01-02']*5,
                          'itemid':[220045]*5,'valuenum':[60,80,999,999,-1],
                          'valueuom':['bpm']*5}).to_csv(raw/'chartevents.csv',index=False)
            pd.DataFrame({'subject_id':[1,1,1], 'hadm_id':[10,10,10],
                          'charttime':['2020-01-01 12:00:00','2020-01-02','2020-01-01 02:00:00'],
                          'storetime':['2020-01-02']*3,'itemid':[50931,50931,50931],
                          'valuenum':[100,200,np.nan], 'valueuom':['mg/dL']*3}).to_csv(raw/'labevents.csv',index=False)
            args = SimpleNamespace(raw_dir=raw, cohort=cp, output_dir=out, chunksize=1, max_chunks=None)
            for kind in ['vital','lab']:
                extract(args,kind)
                # 중단 직전의 마지막 청크를 다시 처리해도 같은 이름으로 교체한다.
                mp = out/f'{kind}_audit/manifest.json'
                manifest = json.loads(mp.read_text(encoding='utf-8'))
                manifest['complete'] = False
                manifest['completed_chunks'] -= 1
                manifest['chunks'] = manifest['chunks'][:-1]
                mp.write_text(json.dumps(manifest),encoding='utf-8')
                extract(args,kind)
                aggregate(args,kind)
            merge(args)
            validate(args)
            f = pd.read_csv(out/'modeling_dataset.csv').set_index('stay_id')
            self.assertEqual(f.loc[100,'vital_HR_count'],2)
            self.assertAlmostEqual(f.loc[100,'vital_HR_mean'],70)
            self.assertAlmostEqual(f.loc[100,'vital_HR_std'],np.sqrt(200))
            self.assertEqual(f.loc[100,'lab_Glucose_count'],1)
            self.assertEqual(f.loc[101,'lab_Glucose_count'],2)
            self.assertEqual(f.loc[200,'lab_Glucose_missing'],1)
            self.assertTrue(pd.isna(f.loc[200,'lab_Glucose_mean']))
            self.assertEqual(cp.read_bytes(),original)
            c.loc[0,'prediction_time'] = '2020-01-03'
            c.to_csv(cp,index=False)
            with self.assertRaisesRegex(ValueError,'코호트 검증 실패'):
                cohort(args)


if __name__ == '__main__':
    unittest.main()
