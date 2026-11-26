"""Reproducible patient-disjoint stratified 70/15/15 split; no preprocessing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/error_cleaned'
OUT = SOURCE / 'splits'
SEED = 42


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = ['X_features.csv', 'y_target.csv', 'row_ids.csv', 'model_feature_columns.json']
    hashes = {name: sha(SOURCE / name) for name in names}
    X, y, ids = [pd.read_csv(SOURCE / name) for name in names[:3]]
    columns = json.loads((SOURCE / names[3]).read_text(encoding='utf-8'))
    assert len(X) == len(y) == len(ids)
    assert X.columns.tolist() == columns and y.columns.tolist() == ['mortality_28d']
    assert ids.subject_id.is_unique and ids.stay_id.is_unique
    assert y.mortality_28d.notna().all() and set(y.mortality_28d) == {0, 1}
    cohort = pd.read_csv(SOURCE / 'modeling_dataset.csv')
    pd.testing.assert_frame_equal(ids, cohort[ids.columns])
    pd.testing.assert_frame_equal(y, cohort[y.columns])
    pd.testing.assert_frame_equal(X, cohort[columns])
    n = len(X)
    # Round total sizes, then allocate each label proportionally using largest remainders.
    targets = [int(np.floor(n * .7)), int(np.ceil(n * .15))]
    targets.append(n - sum(targets))
    rng = np.random.default_rng(SEED)
    remaining = {label: rng.permutation(np.flatnonzero(y.mortality_28d.to_numpy() == label)) for label in [0, 1]}
    split_indices = {}
    for name, size in zip(['train', 'validation', 'test'], targets):
        total = sum(len(v) for v in remaining.values())
        exact = {label: len(v) * size / total for label, v in remaining.items()}
        counts = {label: int(np.floor(v)) for label, v in exact.items()}
        for label in sorted(counts, key=lambda label: (-(exact[label] - counts[label]), label))[:size-sum(counts.values())]:
            counts[label] += 1
        chosen = []
        for label, count in counts.items():
            chosen.extend(remaining[label][:count])
            remaining[label] = remaining[label][count:]
        split_indices[name] = rng.permutation(chosen).astype(int)
    combined = np.concatenate(list(split_indices.values()))
    assert len(combined) == n and np.array_equal(np.sort(combined), np.arange(n))
    OUT.mkdir(exist_ok=True)
    summary, manifest = [], []
    for name, ix in split_indices.items():
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        for filename, frame in zip(names[:3], [X, y, ids]):
            dest = directory / filename
            frame.iloc[ix].to_csv(dest, index=False, encoding='utf-8-sig')
            pd.testing.assert_frame_equal(pd.read_csv(dest), frame.iloc[ix].reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-12)
        labels = y.iloc[ix].mortality_28d
        summary.append(dict(split=name, rows=len(ix), fraction=len(ix)/n,
                            survivors=int(labels.eq(0).sum()), deaths=int(labels.eq(1).sum()), mortality_rate=float(labels.mean())))
        manifest.append(ids.iloc[ix].assign(source_row_index=ix, split=name))
    assignment = pd.concat(manifest).sort_values('source_row_index')
    assert assignment.subject_id.nunique() == n and assignment.source_row_index.is_unique
    assignment.to_csv(OUT / 'split_assignments.csv', index=False, encoding='utf-8-sig')
    stats = pd.DataFrame(summary)
    stats.to_csv(OUT / 'split_summary.csv', index=False)
    assert all(sha(SOURCE / name) == value for name, value in hashes.items())
    report = dict(passed=True, seed=SEED, generator='numpy.default_rng/PCG64',
        method='Patient-disjoint stratified random split; proportional label allocation by largest remainder',
        requested_ratios={'train': .7, 'validation': .15, 'test': .15},
        summary=summary, input_sha256=hashes,
        checks=['all rows assigned exactly once', 'no patient or stay overlap', 'X/y/IDs aligned with modeling dataset',
                'saved CSV roundtrip verified', 'input files unchanged'],
        preprocessing='None; NaNs and original gender categories preserved',
        measurement_window='intime <= charttime < prediction_time; storetime unrestricted as requested',
        training='Not performed')
    (OUT / 'split_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'model_feature_columns.json').write_text(json.dumps(columns, indent=2), encoding='utf-8')
    lines = ['# Train / Validation / Test 분할', '',
        '사망·생존 비율을 유지하는 환자 단위 무작위 분할입니다. 난수 시드는 42입니다.', '',
        '| 세트 | 환자 수 | 생존(0) | 사망(1) | 사망률 |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['split']} | {r['rows']:,} | {r['survivors']:,} | {r['deaths']:,} | {r['mortality_rate']:.2%} |" for r in summary]
    lines += ['', '각 폴더의 X_features.csv, y_target.csv, row_ids.csv는 동일한 행 순서입니다.',
        'split_assignments.csv의 source_row_index는 원본 X/y/row_ids의 0부터 시작하는 데이터 행 번호입니다.',
        '전체 행이 정확히 한 번 배정되었고 환자·stay 중복이 없음을 검증했습니다. 기존 입력 파일은 변경하지 않았습니다.',
        '결측 대체·인코딩·표준화·모델 학습은 수행하지 않았습니다. 전처리 통계는 train에서만 적합하세요.',
        'validation은 설정 선택에, test는 모든 선택을 마친 뒤 최종 평가에 사용하세요.',
        '시간 기준은 입실 후 첫 24시간의 charttime이며, 사용자 결정에 따라 storetime은 제한하지 않습니다.',
        '이 분할은 동일 코호트 내부 평가용이며 시간적·외부 검증은 아닙니다.']
    (OUT / 'split_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(stats.to_string(index=False))


if __name__ == '__main__':
    main()
