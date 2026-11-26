"""Preserve existing outputs; remove only gross suspect vital observations; no training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed'
OUT = SOURCE / 'error_cleaned'
RULES = {'HR': (None, 500), 'RR': (None, 300), 'SBP': (None, 1000),
         'DBP': (None, 1000), 'MAP': (None, 1000), 'Temperature': (0, 60)}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    protected = [SOURCE / n for n in ['cohort_base.csv', 'vital_features.csv',
                 'lab_features.csv', 'modeling_dataset.csv', 'model_feature_columns.json']]
    protected += sorted((SOURCE / 'vital_24h.parquet').glob('*.parquet'))
    protected += sorted((SOURCE / 'lab_24h.parquet').glob('*.parquet'))
    before_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    c = pd.read_csv(SOURCE / 'cohort_base.csv')
    columns = json.loads((SOURCE / 'model_feature_columns.json').read_text(encoding='utf-8'))
    assert c.stay_id.is_unique and c.subject_id.is_unique
    assert c.mortality_28d.notna().all() and set(c.mortality_28d) == {0, 1}
    assert len(columns) == 134 and len(set(columns)) == 134
    summaries, comparisons, rejects, pending, affected_data = [], [], [], [], []
    totals = {}
    for kind in ['vital', 'lab']:
        print(f'Cleaning {kind}', flush=True)
        target = OUT / f'{kind}_24h.parquet'
        target.mkdir(exist_ok=True)
        chunks = []
        n_before = n_after = 0
        for p in sorted((SOURCE / f'{kind}_24h.parquet').glob('*.parquet')):
            f = pd.read_parquet(p)
            bad = pd.Series(False, index=f.index)
            if kind == 'vital':
                for name, (low, high) in RULES.items():
                    mask = f.feature.eq(name) & (f.value.gt(high) | (f.value.lt(low) if low is not None else False))
                    bad |= mask
            r = f.loc[bad].copy()
            if len(r):
                r['reason'] = 'gross_suspect_value_policy'
                rejects.append(r)
            clean = f.loc[~bad].copy()
            clean.to_parquet(target / p.name, index=False)
            n_before += len(f)
            n_after += len(clean)
            pending.append(clean.loc[clean.extreme].assign(kind=kind))
            if kind == 'vital':
                affected_data.append(clean)
            g = clean.groupby(['stay_id', 'feature']).value.agg(['count', 'mean', 'var', 'min', 'max']).reset_index()
            chunks.append(g)
        # Combine per-partition sample moments without losing between-partition variance.
        g = pd.concat(chunks, ignore_index=True)
        g['weighted'] = g['count'] * g['mean']
        keys = ['stay_id', 'feature']
        s = g.groupby(keys).agg(count=('count', 'sum'), weighted=('weighted', 'sum'),
                                min=('min', 'min'), max=('max', 'max'))
        s['mean'] = s.weighted / s['count']
        g = g.join(s['mean'].rename('combined_mean'), on=keys)
        g['m2'] = g['var'].fillna(0) * (g['count'] - 1) + g['count'] * (g['mean'] - g.combined_mean)**2
        s['std'] = np.sqrt(g.groupby(keys).m2.sum().clip(lower=0) / (s['count'] - 1).where(s['count'] > 1))
        old = pd.read_csv(SOURCE / f'{kind}_features.csv').set_index('stay_id').reindex(c.stay_id)
        feat = pd.DataFrame(index=pd.Index(c.stay_id, name='stay_id'))
        names = list(dict.fromkeys(col.split('_')[1] for col in old.columns))
        for name in names:
            v = s.xs(name, level='feature').reindex(feat.index)
            v['count'] = v['count'].fillna(0).astype('int64')
            v['missing'] = v['count'].eq(0).astype('int8')
            prefix = f'{kind}_{name}_'
            for stat in ['mean', 'min', 'max', 'std', 'count', 'missing']:
                feat[prefix + stat] = v[stat]
                assert np.isfinite(v[stat].dropna()).all()
            assert v.loc[v['count'] <= 1, 'std'].isna().all()
            comparisons.append(dict(kind=kind, feature=name,
                before_observations=int(old[prefix+'count'].sum()), after_observations=int(v['count'].sum()),
                before_missing_rate=float(old[prefix+'missing'].mean()), after_missing_rate=float(v['missing'].mean())))
        pd.testing.assert_frame_equal(feat, old, check_dtype=False, rtol=1e-10, atol=1e-9) if kind == 'lab' else None
        feat.reset_index().to_csv(OUT / f'{kind}_features.csv', index=False, encoding='utf-8-sig')
        totals[kind] = dict(before=n_before, after=n_after, excluded=n_before-n_after)
        c = c.merge(feat.reset_index(), on='stay_id', validate='one_to_one', sort=False)
    rejected = pd.concat(rejects, ignore_index=True)
    rejected.to_parquet(OUT / 'excluded_observations.parquet', index=False)
    rejected.to_csv(OUT / 'excluded_observations.csv', index=False, encoding='utf-8-sig')
    rejected.groupby(['feature', 'reason']).agg(observations=('value', 'size'),
        affected_stays=('stay_id', 'nunique')).reset_index().to_csv(OUT / 'exclusion_summary.csv', index=False)
    pd.concat(pending, ignore_index=True).to_parquet(OUT / 'retained_extreme_observations.parquet', index=False)
    pd.DataFrame(comparisons).to_csv(OUT / 'cleaning_comparison.csv', index=False)
    # Independent direct recomputation for every affected stay/feature pair.
    affected = rejected[['stay_id', 'feature']].drop_duplicates()
    obs = pd.concat(affected_data, ignore_index=True).merge(affected, on=['stay_id', 'feature'])
    direct = obs.groupby(['stay_id', 'feature']).value.agg(['mean', 'min', 'max', 'std', 'count'])
    indexed = c.set_index('stay_id')
    for row in affected.itertuples(index=False):
        key = (row.stay_id, row.feature)
        for stat in ['mean', 'min', 'max', 'std', 'count', 'missing']:
            expected = direct.loc[key, stat] if key in direct.index and stat != 'missing' else (0 if key in direct.index else (1 if stat == 'missing' else 0 if stat == 'count' else np.nan))
            actual = indexed.loc[row.stay_id, f'vital_{row.feature}_{stat}']
            assert np.isclose(actual, expected, equal_nan=True, rtol=1e-10, atol=1e-9), (key, stat)
    assert c[columns].select_dtypes('number').replace([np.inf, -np.inf], np.nan).isna().sum().sum() == c[columns].select_dtypes('number').isna().sum().sum()
    assert set(c.gender) == {'F', 'M'}
    c.to_csv(OUT / 'modeling_dataset.csv', index=False, encoding='utf-8-sig')
    c[columns].to_csv(OUT / 'X_features.csv', index=False, encoding='utf-8-sig')
    c[['mortality_28d']].astype(int).to_csv(OUT / 'y_target.csv', index=False)
    c[['subject_id', 'hadm_id', 'stay_id']].to_csv(OUT / 'row_ids.csv', index=False)
    save_json('model_feature_columns.json', columns)
    save_json('cleaning_policy.json', {'rules': RULES, 'boundary': 'exclude value < lower or > upper; retain equality',
        'meaning': 'Conservative project-specific gross-error screening; not clinically adjudicated errors or normal ranges.',
        'lab': 'Retain all previously valid lab observations, including extremes.',
        'zero_vitals': 'Preserved; no context-free inference of artifact.',
        'timing': 'Original charttime policy unchanged; late storetime observations remain included.'})
    save_json('training_contract.json', {'X': 'X_features.csv', 'y': 'y_target.csv', 'ids': 'row_ids.csv',
        'row_alignment': 'All three files have identical row order; row_ids maps each row to patient and stay.',
        'features': columns, 'n_features': len(columns), 'target': 'mortality_28d',
        'label_source': '../cohort_base.csv; existing labels preserved, not re-adjudicated',
        'target_values': {'0': 'Existing cohort 28-day survival label', '1': 'Existing cohort 28-day mortality label'},
        'categorical': {'gender': sorted(c.gender.unique().tolist())},
        'not_performed': ['train/test split', 'imputation', 'encoding', 'scaling', 'winsorization', 'model training'],
        'readiness': 'Raw selected inputs; preprocessing required before logistic regression.',
        'timing_limitation': 'Includes measurements stored after prediction time; availability policy unresolved.'})
    after_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before_hashes == after_hashes
    original = pd.read_csv(SOURCE / 'cohort_base.csv')
    pd.testing.assert_frame_equal(c[original.columns], original)
    save_json('input_preservation_check.json', {'all_unchanged': True, 'sha256': before_hashes})
    save_json('validation_report.json', {'passed': True, 'rows': len(c), 'features': len(columns),
        'label_counts': {str(k): int(v) for k, v in c.mortality_28d.value_counts().items()},
        'observations': totals, 'affected_pairs_independently_verified': len(affected),
        'lab_features_unchanged_with_numeric_tolerance': True, 'cohort_and_labels_unchanged': True,
        'source_outputs_unchanged': True})
    summary = rejected.groupby('feature').agg(observations=('value', 'size'), affected_stays=('stay_id', 'nunique'))
    lines = ['# 명백한 오류 의심값 정제 및 학습 입력 확정', '',
        '모델 학습, 데이터 분할, 결측 대체, 인코딩, 표준화는 수행하지 않았습니다.', '',
        '## 정제 정책', '',
        '기존 진단용 극단값 경계를 삭제 기준으로 사용하지 않았습니다. 아래는 정상범위가 아닌 프로젝트의 보수적 운영 기준입니다. 개별 기록의 오류를 임상적으로 확증한 것은 아닙니다.', '',
        '| 항목 | 추가 제외 기준 |', '|---|---|',
        '| HR | >500 bpm |', '| RR | >300 insp/min |',
        '| SBP·DBP·MAP | >1,000 mmHg |', '| 체온 | <0℃ 또는 >60℃ |', '',
        '경계와 같은 값은 보존합니다. Lab은 추가 제거하지 않았습니다. 0인 Vital과 나머지 극단값도 유지하며 retained_extreme_observations.parquet에 검토 대상을 저장했습니다.',
        '위 기준은 MIT-LCP 기준을 그대로 구현한 것이 아닙니다. 더 넓은 범위를 보존하는 별도 정책입니다. 참고: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/vitalsign.sql', '',
        '## 제외 결과', '', '| 항목 | 제외 관측 | 영향받은 stay |', '|---|---:|---:|']
    lines += [f'| {name} | {int(row.observations)} | {int(row.affected_stays)} |' for name, row in summary.iterrows()]
    lines += ['', f'총 {len(rejected):,}건 제외, 고유 stay {rejected.stay_id.nunique():,}건에 영향. 환자는 제외하지 않았습니다.',
        '제외 관측은 원시 값·단위·시각·원본 행 번호·제외 사유와 함께 excluded_observations.csv 및 Parquet에 보존했습니다.',
        '정제 후 관측에서 6개 통계를 재계산했습니다. std는 표본 표준편차(ddof=1)이며 0~1회 측정은 NaN입니다.', '',
        '## 학습 입력과 정답', '',
        f'- X_features.csv: {len(c):,}행 × {len(columns)}열. 나이·성별 및 132개 Vital/Lab 통계.',
        '- y_target.csv: 기존 cohort의 mortality_28d. 라벨을 재판정하지 않고 그대로 확정했습니다.',
        '- row_ids.csv: 같은 행 순서의 subject_id/hadm_id/stay_id. 학습 입력에서 제외합니다.',
        '- training_contract.json: 입력 목록, 정답, 범주형 변수, 미수행 전처리 명세.',
        '- modeling_dataset.csv: 코호트 근거 열을 보존한 전체 테이블. 전체 열을 학습에 사용하면 안 됩니다.', '',
        'X의 NaN 및 문자열 성별은 유지됩니다. 로지스틱 회귀에 바로 입력할 수 없으며, 분할 후 훈련 데이터로 전처리를 적합해야 합니다.', '',
        '## 검증과 한계', '',
        '기존 관측 Parquet·특징·모델링 테이블·코호트·입력 목록의 SHA256 불변성을 확인했습니다. raw CSV는 읽거나 수정하지 않았습니다.',
        '모든 영향받은 stay/항목의 통계를 별도 직접 집계와 비교했습니다. Lab 특징은 수치 허용오차 내 동일하며 코호트 및 정답은 동일합니다.',
        '기존 charttime 시간 기준을 유지합니다. 예측 시점 이후 저장된 관측도 포함되므로 실제 시점 정보 가용성 문제는 해결하지 않았습니다.',
        '미확정 사망 라벨 환자를 포함하지 않는 기존 코호트의 선택 편향도 그대로 남습니다.']
    (OUT / 'cleaning_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
