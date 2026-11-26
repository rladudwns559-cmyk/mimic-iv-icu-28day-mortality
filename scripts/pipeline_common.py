"""24시간 Vital/Lab 파이프라인의 공통 검증·정제·저장 기능.

원본과 코호트는 읽기 전용으로 사용한다. 파티션은 원본 청크 번호로
결정되어 중단 후 재실행해도 중복 추가되지 않는다.
"""
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name('feature_config.json')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
# 수치 제외 경계는 여기서 변경한다. 진단 전용 경계·단위 목록은 JSON에 있다.
NONNEGATIVE_MIN = 0
SPO2_MIN_EXCLUSIVE = 0
SPO2_MAX_INCLUSIVE = 100
TEMPERATURE_MIN_C = -273.15
IDS = ['subject_id', 'hadm_id', 'stay_id']
OBS_COLUMNS = IDS + ['charttime', 'storetime', 'itemid', 'feature', 'value', 'unit',
                     'raw_value', 'raw_unit', 'source_row', 'extreme', 'storetime_late', 'storetime_missing']
SCHEMA = pa.schema([(x, pa.int64()) for x in IDS] + [
    ('charttime', pa.timestamp('ns')), ('storetime', pa.timestamp('ns')),
    ('itemid', pa.int64()), ('feature', pa.string()), ('value', pa.float64()),
    ('unit', pa.string()), ('raw_value', pa.string()), ('raw_unit', pa.string()),
    ('source_row', pa.int64()), ('extreme', pa.bool_()),
    ('storetime_late', pa.bool_()), ('storetime_missing', pa.bool_())])


def arguments(description):
    """모든 단계에 동일한 경로/청크 옵션을 제공한다."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    p.add_argument('--cohort', type=Path, default=ROOT / 'data/processed/cohort_base.csv')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'data/processed')
    p.add_argument('--chunksize', type=int, default=500000)
    p.add_argument('--max-chunks', type=int, default=None,
                   help='표본 검증용. 최종 산출물은 전체 실행에서만 생성하세요.')
    return p


def setup(args):
    """UTF-8 파일과 콘솔에 진행 경과를 함께 기록한다."""
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            args.output_dir / 'pipeline.log', encoding='utf-8')], force=True)


def save_json(path, value):
    """완료된 JSON만 노출하기 위해 같은 디렉터리에서 원자적으로 교체한다."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def csv(frame, path):
    """CSV도 임시 파일에서 완성한 후 교체한다."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(tmp, index=False, encoding='utf-8-sig')
    os.replace(tmp, path)


def raw_path(directory, name):
    """일반 CSV와 gzip CSV를 모두 지원한다."""
    for p in [directory / (name + '.csv'), directory / (name + '.csv.gz')]:
        if p.is_file():
            return p
    raise FileNotFoundError(f'원본 파일 없음: {name}.csv 또는 .csv.gz')


def cohort(args):
    """코호트를 변경하지 않고 landmark 조건과 기존 사망 근거를 검사한다."""
    c = pd.read_csv(args.cohort)
    required = IDS + ['intime', 'prediction_time', 'outtime', 'mortality_28d']
    absent = set(required) - set(c)
    if absent:
        raise ValueError(f'코호트 필수 열 없음: {sorted(absent)}')
    problems = []
    def check(mask, reason):
        if mask.any():
            q = c.loc[mask, required].copy()
            q['reason'] = reason
            problems.append(q)
    check(c[required].isna().any(axis=1), 'required_null')
    check(c.stay_id.duplicated(keep=False), 'duplicate_stay_id')
    for col in IDS:
        n = pd.to_numeric(c[col], errors='raise')
        check(n.isna() | n.mod(1).ne(0) | n.le(0), f'invalid_{col}')
        c[col] = n.astype('Int64')
    for col in ['intime', 'prediction_time', 'outtime', 'death_time',
                'death_date_lower', 'death_date_upper', 'survival_confirmed_at']:
        if col in c:
            c[col] = pd.to_datetime(c[col], errors='raise', format='mixed')
    check(c.prediction_time.ne(c.intime + pd.Timedelta(hours=24)), 'prediction_not_intime_plus_24h')
    check(c.outtime.lt(c.prediction_time), 'icu_exit_before_prediction')
    check(~c.mortality_28d.isin([0, 1]), 'invalid_label')
    if 'death_time' in c:
        check(c.death_time.le(c.prediction_time), 'death_at_or_before_prediction')
    if 'death_date_lower' in c and 'death_time' in c:
        check(c.death_time.isna() & c.death_date_lower.le(c.prediction_time),
              'date_only_death_before_or_ambiguous_at_prediction')
    for col in ['exclusion_reason', 'data_issue']:
        if col in c:
            check(c[col].fillna('').str.strip().ne(''), f'nonempty_{col}')
    csv(pd.concat(problems, ignore_index=True) if problems else
        pd.DataFrame(columns=required + ['reason']), args.output_dir / 'cohort_validation_issues.csv')
    if problems:
        raise ValueError(f'코호트 검증 실패: {sum(map(len, problems))}건. cohort_validation_issues.csv 확인')
    for col in IDS:
        c[col] = c[col].astype('int64')
    logging.info('코호트: ICU stay %s / 환자 %s', len(c), c.subject_id.nunique())
    return c


def mapping(args):
    """모든 itemid의 label을 로컬 사전과 정확히 대조하고 매핑표를 저장한다."""
    rows = []
    for kind, dictionary in [('vital', 'd_items'), ('lab', 'd_labitems')]:
        d = pd.read_csv(raw_path(args.raw_dir, dictionary)).set_index('itemid')
        for name, item, label, unit, allowed, low, high in CONFIG[kind]:
            if item not in d.index or str(d.loc[item, 'label']).strip() != label:
                raise ValueError(f'로컬 사전 매핑 불일치: {name}, itemid={item}, 예상 label={label}')
            if kind == 'vital' and str(d.loc[item,'unitname']).strip() not in allowed:
                raise ValueError(f'로컬 사전 단위 불일치: {name}, itemid={item}, unit={d.loc[item,"unitname"]}')
            rows.append(dict(kind=kind, feature=name, itemid=item, dictionary_label=label,
                             dictionary_unit=d.loc[item, 'unitname'] if 'unitname' in d else '(사전에 단위 열 없음)',
                             accepted_raw_units=' | '.join(allowed), output_unit=unit,
                             diagnostic_low=low, diagnostic_high=high,
                             exclusion_rule=f'finite; recognized unit; SpO2 ({SPO2_MIN_EXCLUSIVE},{SPO2_MAX_INCLUSIVE}]; Temperature >= {TEMPERATURE_MIN_C} C; AnionGap any finite; others >= {NONNEGATIVE_MIN}'))
    csv(pd.DataFrame(rows), args.output_dir / 'item_mapping.csv')
    csv(pd.DataFrame(rows)[['kind','feature','itemid','exclusion_rule','diagnostic_low','diagnostic_high']],
        args.output_dir / 'cleaning_rules.csv')
    return rows


def clean(frame, kind):
    """단위를 먼저 검사하고 섭씨 변환 후 보수적인 수치 조건을 적용한다.

    극단값 범위는 분포 검토용 표시일 뿐 제외 조건이 아니다.
    Anion gap은 음수가 실제로 가능하므로 음수를 제거하지 않는다.
    """
    f = frame.copy()
    f['raw_value'] = f.valuenum.fillna('').astype(str)
    f['raw_unit'] = f.valueuom.fillna('').astype(str)
    f['value'] = pd.to_numeric(f.valuenum, errors='coerce').astype('float64')
    f['reason'] = ''
    f['feature'] = ''
    f['unit'] = ''
    f['extreme'] = False
    for name, item, label, unit, allowed, low, high in CONFIG[kind]:
        ix = f.itemid.eq(item)
        f.loc[ix, 'feature'] = name
        f.loc[ix, 'unit'] = unit
        # 소실된 degree 문자 표기는 itemid와 F/C 접미사가 모두 일치할 때만 허용한다.
        valid_unit = f.raw_unit.str.strip().isin(allowed)
        f.loc[ix & ~valid_unit, 'reason'] = 'unknown_or_mismatched_unit'
        if item == 223761:
            f.loc[ix, 'value'] = (f.loc[ix, 'value'] - 32) * 5 / 9
        finite = pd.Series(np.isfinite(f.value), index=f.index)
        f.loc[ix & f.reason.eq('') & ~finite, 'reason'] = 'invalid_numeric'
        impossible = f.value.lt(NONNEGATIVE_MIN)
        if name == 'AnionGap':
            impossible = pd.Series(False, index=f.index)
        elif name == 'Temperature':
            impossible = f.value.lt(TEMPERATURE_MIN_C)
        elif name == 'SpO2':
            impossible = f.value.le(SPO2_MIN_EXCLUSIVE) | f.value.gt(SPO2_MAX_INCLUSIVE)
        f.loc[ix & f.reason.eq('') & impossible, 'reason'] = 'impossible_value'
        f.loc[ix, 'extreme'] = f.loc[ix, 'value'].lt(low) | f.loc[ix, 'value'].gt(high)
    f['storetime_late'] = f.storetime.ge(f.prediction_time)
    f['storetime_missing'] = f.storetime.isna()
    return f


def parquet(frame, path, schema=None):
    """일관된 Arrow 스키마와 원자적 저장으로 빈 파티션도 안전하게 기록한다."""
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    tmp = path.with_suffix('.tmp')
    pq.write_table(table, tmp, compression='snappy')
    os.replace(tmp, path)


def fingerprint(args, source, kind):
    """재시작 시 원본·코호트·설정·코드 변경을 감지한다."""
    h = hashlib.sha256()
    for p in [args.cohort, CONFIG_PATH, Path(__file__)]:
        h.update(p.read_bytes())
    for name in ['d_items', 'd_labitems']:
        h.update(raw_path(args.raw_dir, name).read_bytes())
    return dict(version=h.hexdigest(), source=str(source.resolve()), size=source.stat().st_size,
                mtime_ns=source.stat().st_mtime_ns, chunksize=args.chunksize,
                max_chunks=args.max_chunks, kind=kind)


def extract(args, kind):
    """CSV 청크를 cohort에 연결하고 정제 관측·제외 관측·진단을 저장한다."""
    c = cohort(args)
    mapping(args)
    if args.chunksize < 1 or (args.max_chunks is not None and args.max_chunks < 1):
        raise ValueError('chunksize/max-chunks는 양수여야 합니다.')
    source = raw_path(args.raw_dir, 'chartevents' if kind == 'vital' else 'labevents')
    signature = fingerprint(args, source, kind)
    base = args.output_dir / f'{kind}_24h.parquet'
    audit = args.output_dir / f'{kind}_audit'
    base.mkdir(exist_ok=True)
    audit.mkdir(exist_ok=True)
    manifest_path = audit / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['signature'] != signature:
            raise ValueError(f'{kind}: 입력/설정 변경. 기존 결과를 보존하려면 다른 --output-dir을 지정하세요.')
        if manifest.get('complete'):
            logging.info('%s 추출 완료 체크포인트 재사용', kind)
            return
    else:
        manifest = dict(signature=signature, completed_chunks=0, complete=False, chunks=[])
        save_json(manifest_path, manifest)
    header = pd.read_csv(source, nrows=0).columns.tolist()
    needed = ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum', 'valueuom']
    if kind == 'vital':
        needed.append('stay_id')
    if not set(needed).issubset(header):
        raise ValueError(f'{source.name}: 필수 열 누락 {set(needed)-set(header)}')
    use = needed + (['storetime'] if 'storetime' in header else [])
    if kind == 'lab' and 'labevent_id' in header:
        use.append('labevent_id')
    selected_items = {r[1] for r in CONFIG[kind]}
    reader = pd.read_csv(source, usecols=use, dtype={col: 'string' for col in use},
                         chunksize=args.chunksize)
    started = time.monotonic()
    for idx, chunk in enumerate(reader):
        if args.max_chunks is not None and idx >= args.max_chunks:
            break
        if idx < manifest['completed_chunks']:
            if idx % 100 == 0:
                logging.info('%s 체크포인트까지 원본 스캔: 청크 %s', kind, idx)
            continue
        scanned = len(chunk)
        chunk['source_row'] = np.arange(idx * args.chunksize, idx * args.chunksize + scanned, dtype='int64')
        chunk['itemid'] = pd.to_numeric(chunk.itemid, errors='raise').astype('Int64')
        f = chunk.loc[chunk.itemid.isin(selected_items)].copy()
        selected = len(f)
        for col in ['subject_id', 'hadm_id'] + (['stay_id'] if kind == 'vital' else []):
            f[col] = pd.to_numeric(f[col], errors='raise').astype('Int64')
        join_key = 'stay_id' if kind == 'vital' else 'hadm_id'
        missing_join_id = int(f[join_key].isna().sum())
        outside_cohort = int((f[join_key].notna() & ~f[join_key].isin(c[join_key])).sum())
        f = f.loc[f[join_key].isin(c[join_key])]
        f = f.merge(c[IDS + ['intime','prediction_time']], on=join_key, how='inner',
                    suffixes=('_raw', ''), validate='many_to_one' if kind == 'vital' else 'many_to_many')
        bad_id = f.subject_id_raw.ne(f.subject_id).fillna(True)
        if kind == 'vital':
            bad_id |= f.hadm_id_raw.ne(f.hadm_id).fillna(True)
        storetime_present = f['storetime'].notna() if 'storetime' in f else pd.Series(False, index=f.index)
        for col in ['charttime', 'storetime']:
            if col not in f:
                f[col] = pd.NaT
            else:
                f[col] = pd.to_datetime(f[col], errors='coerce', format='mixed')
        invalid_time = f.charttime.isna()
        inside = f.charttime.ge(f.intime) & f.charttime.lt(f.prediction_time)
        linkage = dict(id_mismatch=int(bad_id.sum()), invalid_charttime=int(invalid_time.sum()),
                       invalid_storetime=int((storetime_present & f.storetime.isna()).sum()),
                       outside_window=int((~inside & ~invalid_time).sum()))
        # 잘못 연결된 관측과 잘못된 시각은 별도 진단 파일에 남긴다.
        bad = f.loc[bad_id | invalid_time].copy()
        if len(bad):
            csv(bad, audit / f'linkage-{idx:06d}.csv')
        f = f.loc[~bad_id & inside].copy().reset_index(drop=True)
        multi = int(f.source_row.value_counts().gt(1).sum())
        # 중복 후보를 기록하되 임상 반복 측정을 임의로 제거하지 않는다.
        # source_row로 원본 행을 식별하므로 재시작 중복은 발생하지 않는다.
        duplicate_candidates = int(f.duplicated(IDS + ['charttime','itemid','valuenum','valueuom','storetime']).sum())
        cleaned = clean(f, kind)
        before = cleaned.groupby(['stay_id','feature']).size().rename('before_count').reset_index()
        good = cleaned.loc[cleaned.reason.eq(''), OBS_COLUMNS].copy()
        rejects = cleaned.loc[cleaned.reason.ne('')].copy()
        parquet(good, base / f'part-{idx:06d}.parquet', SCHEMA)
        # 원시 단위/값·사유·stay를 보존하여 기준 변경을 검토할 수 있다.
        parquet(rejects[OBS_COLUMNS + ['reason']], audit / f'rejected-{idx:06d}.parquet')
        csv(before, audit / f'before-{idx:06d}.csv')
        reasons = rejects.groupby(['feature','itemid','reason','raw_unit'], dropna=False).agg(
            observations=('stay_id','size'), affected_stays=('stay_id','nunique')).reset_index()
        csv(reasons, audit / f'reasons-{idx:06d}.csv')
        units = cleaned.groupby(['feature','itemid','raw_unit'], dropna=False).size().rename('observations').reset_index()
        csv(units, audit / f'units-{idx:06d}.csv')
        stats = dict(chunk=idx, source_rows=scanned, selected_item_rows=selected,
                     missing_join_id=missing_join_id, outside_cohort=outside_cohort,
                     invalid_numeric_in_window=int((~np.isfinite(pd.to_numeric(f.valuenum, errors='coerce').astype(float))).sum()),
                     in_window_before_cleaning=len(f), valid_rows=len(good), rejected_rows=len(rejects),
                     overlapping_lab_observations=multi, duplicate_candidates_within_chunk=duplicate_candidates,
                     **linkage)
        manifest['chunks'].append(stats)
        manifest['completed_chunks'] = idx + 1
        save_json(manifest_path, manifest)
        logging.info('%s 청크 %s: 원본 누적 %s, 구간 내 %s → 유효 %s, 경과 %.1f분',
                     kind, idx + 1, (idx * args.chunksize + scanned), len(f), len(good),
                     (time.monotonic()-started)/60)
    manifest['complete'] = True
    manifest['sample_only'] = args.max_chunks is not None
    save_json(manifest_path, manifest)
    csv(pd.DataFrame(manifest['chunks']), audit / 'extraction_counts.csv')


def parts(args, kind):
    """커밋된 청크만 순회해 미완료 파티션의 혼입을 방지한다."""
    p = args.output_dir / f'{kind}_audit/manifest.json'
    m = json.loads(p.read_text(encoding='utf-8'))
    if not m['complete']:
        raise ValueError(f'{kind}: 추출 미완료. 먼저 추출을 재실행하세요.')
    source = raw_path(args.raw_dir, 'chartevents' if kind == 'vital' else 'labevents')
    if m['signature'] != fingerprint(args, source, kind):
        raise ValueError(f'{kind}: 추출 후 입력/설정 변경 감지')
    for idx in range(m['completed_chunks']):
        yield idx, args.output_dir / f'{kind}_24h.parquet/part-{idx:06d}.parquet'


def aggregate(args, kind):
    """관측 전체를 적재하지 않고 청크 평균·M2를 결합하여 표준편차를 계산한다."""
    c = cohort(args)
    index = pd.Index(c.stay_id, name='stay_id')
    names = list(dict.fromkeys(r[0] for r in CONFIG[kind]))
    features = pd.DataFrame(index=index)
    diagnostics = []
    before_all = {}
    state = {}
    reject_stays = {}
    units = []
    reason_counts = []
    histogram = {}
    invalid_numbers = {}
    for idx, path in parts(args, kind):
        f = pd.read_parquet(path)
        b = pd.read_csv(args.output_dir / f'{kind}_audit/before-{idx:06d}.csv')
        for row in b.itertuples():
            key = (row.stay_id, row.feature)
            before_all[key] = before_all.get(key, 0) + row.before_count
        r = pd.read_parquet(args.output_dir / f'{kind}_audit/rejected-{idx:06d}.parquet')
        for name, group in r.groupby('feature'):
            invalid_numbers[name] = invalid_numbers.get(name, 0) + int((~np.isfinite(
                pd.to_numeric(group.raw_value,errors='coerce').astype(float))).sum())
        for key, group in r.groupby(['feature','reason']):
            reject_stays.setdefault(key, set()).update(group.stay_id)
        reason_counts.append(r.groupby(['feature','reason']).size().rename('observations').reset_index())
        units.append(pd.read_csv(args.output_dir / f'{kind}_audit/units-{idx:06d}.csv', keep_default_na=False))
        for name, group in f.groupby('feature'):
            config_row = next(r for r in CONFIG[kind] if r[0] == name)
            edges = np.concatenate(([-np.inf], np.linspace(config_row[5],config_row[6],51), [np.inf]))
            values, _ = np.histogram(group.value.to_numpy(), bins=edges)
            histogram[name] = histogram.get(name,np.zeros(len(edges)-1,dtype='int64')) + values
        g = f.groupby(['stay_id','feature']).agg(n=('value','size'), mean=('value','mean'),
                var=('value','var'), minimum=('value','min'), maximum=('value','max'),
                first=('charttime','min'), last=('charttime','max'), extreme=('extreme','sum'),
                late=('storetime_late','sum'), missing_storetime=('storetime_missing','sum'))
        for key, row in g.iterrows():
            n = int(row['n'])
            m2 = 0.0 if n == 1 else row['var'] * (n-1)
            if key not in state:
                state[key] = [n, row['mean'], m2, row['minimum'], row['maximum'], row['first'], row['last'],
                              int(row['extreme']), int(row['late']), int(row['missing_storetime'])]
            else:
                a = state[key]
                total = a[0] + n
                delta = row['mean'] - a[1]
                a[2] += m2 + delta * delta * a[0] * n / total
                a[1] += delta * n / total
                a[0] = total
                a[3], a[4] = min(a[3],row['minimum']), max(a[4],row['maximum'])
                a[5], a[6] = min(a[5],row['first']), max(a[6],row['last'])
                a[7] += int(row['extreme'])
                a[8] += int(row['late'])
                a[9] += int(row['missing_storetime'])
    times = c.set_index('stay_id')
    missing_rows = []
    for name in names:
        prefix = f'{kind}_{name}'
        s = pd.DataFrame.from_dict({sid: a[:5] for (sid, feat), a in state.items() if feat == name},
                                    orient='index', columns=['count','mean','m2','min','max'], dtype=float).reindex(index)
        s['count'] = s['count'].fillna(0).astype('int64')
        s['std'] = np.sqrt(s.m2.clip(lower=0) / (s['count']-1).where(s['count'].gt(1)))
        s['missing'] = s['count'].eq(0).astype('int8')
        for stat in ['mean','min','max','std','count','missing']:
            features[f'{prefix}_{stat}'] = s[stat]
        before_stays = sum(feat == name for sid, feat in before_all)
        before_obs = sum(n for (sid, feat), n in before_all.items() if feat == name)
        missing_rows.append(dict(kind=kind, feature=name, before_observations=before_obs,
                                 after_observations=int(s['count'].sum()), before_stays=before_stays,
                                 invalid_numeric_observations=invalid_numbers.get(name,0),
                                 after_stays=int(s['count'].gt(0).sum()),
                                 before_patients=c.loc[c.stay_id.isin([sid for sid,feat in before_all if feat==name]),'subject_id'].nunique(),
                                 after_patients=c.loc[c.stay_id.isin(s.index[s['count'].gt(0)]),'subject_id'].nunique(),
                                 before_missing_stays=len(c)-before_stays,
                                 after_missing_stays=int(s.missing.sum()),
                                 before_missing_rate=1-before_stays/len(c), after_missing_rate=s.missing.mean()))
        for sid in index:
            a = state.get((sid,name))
            diagnostics.append(dict(stay_id=sid, feature=name, count=a[0] if a else 0,
                  first_hours=(a[5]-times.loc[sid,'intime']).total_seconds()/3600 if a else np.nan,
                  last_hours=(a[6]-times.loc[sid,'intime']).total_seconds()/3600 if a else np.nan,
                  hours_since_last=(times.loc[sid,'prediction_time']-a[6]).total_seconds()/3600 if a else np.nan,
                  observation_span_hours=(a[6]-a[5]).total_seconds()/3600 if a else np.nan,
                  extreme_count=a[7] if a else 0, storetime_late_count=a[8] if a else 0,
                  storetime_missing_count=a[9] if a else 0))
    csv(features.reset_index(), args.output_dir / f'{kind}_features.csv')
    diag = pd.DataFrame(diagnostics)
    csv(diag, args.output_dir / f'{kind}_timing_diagnostics.csv')
    csv(diag.groupby('feature')[['count','first_hours','last_hours','hours_since_last','observation_span_hours']].describe().stack(0, future_stack=True).reset_index(),
        args.output_dir / f'{kind}_timing_distribution.csv')
    csv(pd.DataFrame(missing_rows), args.output_dir / f'{kind}_missingness_comparison.csv')
    counts = pd.concat(reason_counts, ignore_index=True).groupby(['feature','reason']).observations.sum().reset_index() if reason_counts else pd.DataFrame(columns=['feature','reason','observations'])
    counts['affected_stays'] = [len(reject_stays[(r.feature,r.reason)]) for r in counts.itertuples()]
    csv(counts, args.output_dir / f'{kind}_exclusion_summary.csv')
    if units:
        csv(pd.concat(units).groupby(['feature','itemid','raw_unit'], dropna=False).observations.sum().reset_index(),
            args.output_dir / f'{kind}_unit_inventory.csv')
    # 극단값 개수·실제 min/max와 stay 평균 분포를 모두 보고한다.
    dist = []
    histogram_rows = []
    for name in names:
        a = [v for (sid, feat), v in state.items() if feat == name]
        n = sum(v[0] for v in a)
        mean = sum(v[0]*v[1] for v in a)/n if n else np.nan
        m2 = sum(v[2]+v[0]*(v[1]-mean)**2 for v in a)
        dist.append(dict(feature=name, observations=n, mean=mean,
                         std=np.sqrt(m2/(n-1)) if n>1 else np.nan,
                         extreme_observations=sum(v[7] for v in a),
                         extreme_stays=sum(v[7]>0 for v in a),
                         min=min((v[3] for v in a), default=np.nan),
                         max=max((v[4] for v in a), default=np.nan)))
        config_row = next(r for r in CONFIG[kind] if r[0] == name)
        edges = np.concatenate(([-np.inf],np.linspace(config_row[5],config_row[6],51),[np.inf]))
        for lower, upper, number in zip(edges[:-1],edges[1:],histogram.get(name,np.zeros(len(edges)-1,dtype=int))):
            histogram_rows.append(dict(feature=name, lower=lower, upper=upper, observations=number))
    csv(pd.DataFrame(dist), args.output_dir / f'{kind}_value_distribution.csv')
    csv(pd.DataFrame(histogram_rows), args.output_dir / f'{kind}_value_histogram.csv')
    logging.info('%s 특징: %s행 × %s열', kind, len(features), len(features.columns)+1)
    logging.info('\n%s', pd.DataFrame(missing_rows).to_string(index=False))
    logging.info('\n%s', counts.to_string(index=False))


def merge(args):
    """원본 cohort 열과 라벨을 유지하며 stay_id 기준 left join한다."""
    cohort(args)
    c = pd.read_csv(args.cohort)
    for kind in ['vital','lab']:
        f = pd.read_csv(args.output_dir / f'{kind}_features.csv')
        if not f.stay_id.is_unique or set(f.stay_id) != set(c.stay_id):
            raise ValueError(f'{kind}: 특징 stay_id가 cohort와 다릅니다.')
        c = c.merge(f, on='stay_id', how='left', validate='one_to_one', sort=False)
    csv(c, args.output_dir / 'modeling_dataset.csv')
    # 라벨 판정용 미래 열을 학습에 넣지 않도록 명시적 후보 목록을 제공한다.
    candidate = ['age','gender'] + [x for x in c if x.startswith(('vital_','lab_'))]
    save_json(args.output_dir / 'model_feature_columns.json', candidate)
    logging.info('모델링 데이터셋: %s행 × %s열 (라벨 근거 열은 학습 입력 제외)', *c.shape)


def validate(args):
    """전체 관측을 스트리밍 재검증하고 특징·cohort·라벨 무결성을 검사한다."""
    c = cohort(args)
    original = pd.read_csv(args.cohort)
    model = pd.read_csv(args.output_dir / 'modeling_dataset.csv')
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    require(model.stay_id.is_unique and len(model)==len(c), '최종 행 수/고유성 불일치')
    require(original.equals(model[original.columns]), 'cohort 값 또는 행 순서 변경')
    rows = []
    for kind in ['vital','lab']:
        # 디스크 인덱스로 청크 경계를 넘는 동일 관측 후보도 집계한다.
        db = sqlite3.connect(args.output_dir / f'{kind}_duplicate_audit.sqlite')
        db.execute('DROP TABLE IF EXISTS seen')
        db.execute('CREATE TABLE seen (key BLOB PRIMARY KEY) WITHOUT ROWID')
        duplicate_candidates = 0
        feat = pd.read_csv(args.output_dir / f'{kind}_features.csv')
        require(feat.stay_id.is_unique and set(feat.stay_id)==set(c.stay_id), f'{kind}: 특징 ID 불일치')
        names = list(dict.fromkeys(r[0] for r in CONFIG[kind]))
        counts = {name: pd.Series(0, index=c.stay_id, dtype='int64') for name in names}
        total = 0
        for idx, path in parts(args, kind):
            f = pd.read_parquet(path)
            identity = IDS + ['charttime','storetime','itemid','raw_value','raw_unit']
            # 후보 검사용 128비트 digest. 관측 제거에는 사용하지 않는다.
            digests = [hashlib.blake2b(x.encode('utf-8'),digest_size=16).digest()
                       for x in f[identity].astype(str).agg('|'.join, axis=1)]
            inserted_before = db.total_changes
            db.executemany('INSERT OR IGNORE INTO seen VALUES (?)', ((x,) for x in digests))
            db.commit()
            duplicate_candidates += len(f) - (db.total_changes-inserted_before)
            total += len(f)
            require(f.stay_id.isin(c.stay_id).all(), f'{kind}/{idx}: cohort 외 ID')
            q = f.merge(c[IDS+['intime','prediction_time']], on='stay_id', suffixes=('_obs',''), validate='many_to_one')
            require((q.subject_id_obs.eq(q.subject_id) & q.hadm_id_obs.eq(q.hadm_id)).all(), f'{kind}/{idx}: 환자/입원 ID 불일치')
            require((q.charttime.ge(q.intime)&q.charttime.lt(q.prediction_time)).all(), f'{kind}/{idx}: 시간 범위 위반')
            require(np.isfinite(f.value).all(), f'{kind}/{idx}: 비유한 수치')
            check = q.rename(columns={'raw_value':'valuenum','raw_unit':'valueuom'})
            check = check.drop(columns=['value','feature','unit'])
            checked = clean(check, kind)
            require(checked.reason.eq('').all(), f'{kind}/{idx}: 정제 기준 위반')
            require(np.allclose(checked.value, f.value, equal_nan=False), f'{kind}/{idx}: 변환값 불일치')
            require((checked.feature.to_numpy()==f.feature.to_numpy()).all() and
                    (checked.unit.to_numpy()==f.unit.to_numpy()).all(), f'{kind}/{idx}: 변수/단위 불일치')
            for name, group in f.groupby('feature'):
                require(name in counts, f'{kind}: 알 수 없는 변수 {name}')
                if name in counts:
                    counts[name] = counts[name].add(group.groupby('stay_id').size(), fill_value=0)
        for name in names:
            p = f'{kind}_{name}_'
            indexed = feat.set_index('stay_id').reindex(c.stay_id)
            n = indexed[p+'count']
            require(np.isfinite(n).all() and n.ge(0).all() and n.mod(1).eq(0).all(), f'{p}: 잘못된 count')
            require(np.array_equal(n.to_numpy(),counts[name].reindex(c.stay_id).to_numpy()), f'{p}: 관측 count 불일치')
            require(indexed[p+'missing'].eq(n.eq(0).astype(int)).all(), f'{p}: missing 불일치')
            require(indexed.loc[n.eq(0),[p+s for s in ['mean','min','max','std']]].isna().all().all(), f'{p}: 결측 통계 불일치')
            require(indexed.loc[n.gt(0),[p+s for s in ['mean','min','max']]].notna().all().all(), f'{p}: 유효 관측 통계 NaN')
            require(indexed.loc[n.le(1),p+'std'].isna().all(), f'{p}: 단일 관측 std')
            require(indexed.loc[n.gt(1),p+'std'].notna().all(), f'{p}: 다중 관측 std NaN')
        require(not np.isinf(feat.select_dtypes('number').to_numpy()).any(), f'{kind}: 특징 무한대')
        m = json.loads((args.output_dir / f'{kind}_audit/manifest.json').read_text(encoding='utf-8'))
        totals = pd.DataFrame(m['chunks']).select_dtypes('number').sum().to_dict()
        require(total == totals.get('valid_rows',0), f'{kind}: manifest 관측 수 불일치')
        require(totals.get('id_mismatch',0)==0 and totals.get('invalid_charttime',0)==0, f'{kind}: ID 연결/시각 오류 진단 확인 필요')
        db.close()
        rows.append(dict(kind=kind, cohort_stays=len(c), cohort_patients=c.subject_id.nunique(),
                         duplicate_candidates_all_chunks=duplicate_candidates,
                         feature_rows=len(feat), feature_columns=len(feat.columns), **totals))
    csv(pd.DataFrame(rows), args.output_dir / 'pipeline_summary.csv')
    save_json(args.output_dir / 'validation_report.json', dict(passed=not errors, errors=errors,
              modeling_shape=list(model.shape), label_distribution=original.mortality_28d.value_counts().to_dict(),
              quality_notes=['storetime>=prediction_time 관측은 요청한 charttime 기준 추출에 포함되며 별도 진단됨',
                             '동일 내용 관측은 반복 측정일 수 있어 보존; 전체 청크의 중복 후보 수는 pipeline_summary.csv에 기록',
                             '극단값 진단 경계는 임상 정상범위가 아니며 제외에 사용하지 않음',
                             '원본 사망/생존 근거 열은 라벨용; model_feature_columns.json만 학습 후보로 사용',
                             '표본 실행은 전체 산출물을 대표하지 않음' if args.max_chunks else '전체 원본 스캔 완료']))
    logging.info('\n%s', pd.DataFrame(rows).to_string(index=False))
    logging.info('라벨 분포:\n%s', original.mortality_28d.value_counts().to_string())
    if errors:
        raise ValueError('검증 실패: ' + '; '.join(errors))
    logging.info('전체 자동 검증 통과: 최종 %s행 × %s열', *model.shape)
