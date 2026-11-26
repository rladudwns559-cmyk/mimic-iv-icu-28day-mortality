"""제외 Parquet만으로 INR을 복원한다. 원본 CSV 및 Vital 추출은 실행하지 않는다.

빈 단위가 허용되는 유일한 항목은 무차원 비율인 INR이다. 기존 결과를
백업하고 staging에서 검증한 뒤 교체한다. 중단 후 같은 명령으로 복구 가능하다.
"""
import copy
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pipeline_common import (CONFIG, CONFIG_PATH, IDS, OBS_COLUMNS, SCHEMA, arguments,
                             clean, cohort, csv, fingerprint, mapping, merge, parquet,
                             raw_path, save_json, setup)
from write_report import report


def sha(path):
    """큰 산출물도 고정 메모리로 지문을 계산한다."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(args, kind, config_bytes):
    """INR 정책 변경 외의 입력 변경은 허용하지 않는다. 원본은 stat만 읽는다."""
    h = hashlib.sha256()
    import pipeline_common
    for data in [args.cohort.read_bytes(), config_bytes, Path(pipeline_common.__file__).read_bytes()]:
        h.update(data)
    for name in ['d_items','d_labitems']:
        h.update(raw_path(args.raw_dir,name).read_bytes())
    source = raw_path(args.raw_dir,'chartevents' if kind=='vital' else 'labevents')
    return dict(version=h.hexdigest(), source=str(source.resolve()), size=source.stat().st_size,
                mtime_ns=source.stat().st_mtime_ns, chunksize=args.chunksize,
                max_chunks=args.max_chunks, kind=kind)


def replace_inr(path, frame, stage):
    """진단 CSV의 INR 행만 교체하고 다른 항목 행은 유지한다."""
    old = pd.read_csv(path)
    csv(pd.concat([old.loc[old.feature.ne('INR')],frame[old.columns]],ignore_index=True),stage/path.name)


def repair(args):
    """기존 검증 증거·불변 지문과 새 INR 전수 검사로 증분 검증한다."""
    root = args.output_dir
    backup = root/'inr_repair_backup'
    stage = root/'.inr_repair_stage'
    finished = root/'inr_repair_report.json'
    if finished.exists():
        result = json.loads(finished.read_text(encoding='utf-8'))
        if result.get('passed'):
            if sha(CONFIG_PATH)!=result['config_sha256']:
                raise ValueError('완료 이후 INR 설정 변경 감지')
            for name,digest in result['updated_sha256'].items():
                if sha(root/name) != digest:
                    raise ValueError(f'완료 이후 산출물 변경 감지: {name}')
            logging.info('INR 복원 완료 결과 재사용: 중복 추가 없음')
            return
    backup.mkdir(exist_ok=True)
    checkpoint = backup/'state.json'
    if not checkpoint.exists():
        previous = json.loads((root/'validation_report.json').read_text(encoding='utf-8'))
        if not previous.get('passed'):
            raise ValueError('기존 전체 검증이 통과한 데이터셋만 증분 수정할 수 있습니다.')
        saved = [p for p in root.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md']]
        saved += list((root/'lab_24h.parquet').glob('*.parquet'))
        saved += [p for p in (root/'lab_audit').iterdir() if p.is_file()]
        saved += [root/'vital_audit/manifest.json']
        # 완료된 백업은 덮어쓰지 않아 중단된 교체도 최초 상태에서 재구축할 수 있다.
        for p in saved:
            dest = backup/p.relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
        shutil.copy2(CONFIG_PATH,backup/'feature_config.json')
        protected = [args.cohort, root/'vital_features.csv']
        protected += sorted((root/'vital_24h.parquet').glob('*.parquet'))
        protected += [p for p in (root/'vital_audit').iterdir() if p.is_file() and p.name!='manifest.json']
        save_json(checkpoint,dict(protected={str(p.resolve()):sha(p) for p in protected},previous_validation=previous))
    state = json.loads(checkpoint.read_text(encoding='utf-8'))
    for name,digest in state['protected'].items():
        if sha(Path(name)) != digest:
            raise ValueError(f'보존 대상 변경 감지: {name}')
    stage.mkdir(exist_ok=True)
    (stage/'lab_24h.parquet').mkdir(exist_ok=True)
    (stage/'lab_audit').mkdir(exist_ok=True)
    (stage/'vital_audit').mkdir(exist_ok=True)
    old_config_bytes = (backup/'feature_config.json').read_bytes()
    original_config = json.loads(old_config_bytes)
    new_config = copy.deepcopy(original_config)
    row = next(r for r in new_config['lab'] if r[1]==51237)
    if row[0:4] != ['INR',51237,'INR(PT)','ratio']:
        raise ValueError('INR 매핑 불일치')
    if '' not in row[4]:
        row[4].append('')
    new_config['policy'] = original_config['policy'] + ' INR(51237)은 무차원 비율이므로 빈 단위를 허용하며, 원시 단위는 그대로 보존.'
    new_config_bytes = (json.dumps(new_config,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    # 다른 itemid의 허용 단위는 전혀 변경하지 않는다.
    assert original_config['vital']==new_config['vital']
    assert [r for r in original_config['lab'] if r[1]!=51237]==[r for r in new_config['lab'] if r[1]!=51237]
    CONFIG.clear()
    CONFIG.update(new_config)
    c = cohort(SimpleNamespace(**{**vars(args),'output_dir':stage}))
    lab_manifest = json.loads((backup/'lab_audit/manifest.json').read_text(encoding='utf-8'))
    vital_manifest = json.loads((backup/'vital_audit/manifest.json').read_text(encoding='utf-8'))
    for kind,m in [('lab',lab_manifest),('vital',vital_manifest)]:
        if not m['complete'] or m['signature'] != signature(args,kind,old_config_bytes):
            raise ValueError(f'{kind}: 백업 시점의 입력·설정·코드와 불일치')
    accepted, all_inr, remaining_inr = [], [], []
    for idx in range(lab_manifest['completed_chunks']):
        part = f'part-{idx:06d}.parquet'
        reject = f'rejected-{idx:06d}.parquet'
        old_good = pd.read_parquet(backup/'lab_24h.parquet'/part)
        old_bad = pd.read_parquet(backup/'lab_audit'/reject)
        if old_good.itemid.eq(51237).any():
            raise ValueError('기존 INR 유효 관측이 있어 최초 전량 제외 상태와 다릅니다.')
        candidate = old_bad.loc[old_bad.itemid.eq(51237)].copy()
        if not candidate.reason.eq('unknown_or_mismatched_unit').all() or not candidate.raw_unit.fillna('').str.strip().eq('').all():
            raise ValueError('빈 단위로 제외된 INR 이외의 관측은 자동 복원하지 않습니다.')
        joined = candidate.merge(c[IDS+['intime','prediction_time']],on='stay_id',suffixes=('_obs',''),validate='many_to_one')
        if len(joined)!=len(candidate) or not (joined.subject_id_obs.eq(joined.subject_id)&joined.hadm_id_obs.eq(joined.hadm_id)).all():
            raise ValueError('INR ID 연결 오류')
        if not (joined.charttime.ge(joined.intime)&joined.charttime.lt(joined.prediction_time)).all():
            raise ValueError('INR 관측 시각 오류')
        joined['valuenum'] = joined.raw_value
        joined['valueuom'] = joined.raw_unit
        checked = clean(joined,'lab')
        restored = checked.loc[checked.reason.eq(''),OBS_COLUMNS]
        rejected = checked.loc[checked.reason.ne(''),OBS_COLUMNS+['reason']]
        all_inr.append(checked[OBS_COLUMNS+['reason']])
        accepted.append(restored)
        remaining_inr.append(rejected)
        good = pd.concat([old_good,restored],ignore_index=True)
        bad = pd.concat([old_bad.loc[old_bad.itemid.ne(51237)],rejected],ignore_index=True)
        pd.testing.assert_frame_equal(good.loc[good.itemid.ne(51237)].reset_index(drop=True),old_good.reset_index(drop=True))
        pd.testing.assert_frame_equal(bad.loc[bad.itemid.ne(51237)].reset_index(drop=True),old_bad.loc[old_bad.itemid.ne(51237)].reset_index(drop=True))
        parquet(good,stage/'lab_24h.parquet'/part,SCHEMA)
        parquet(bad,stage/'lab_audit'/reject)
        reasons = bad.groupby(['feature','itemid','reason','raw_unit'],dropna=False).agg(observations=('stay_id','size'),affected_stays=('stay_id','nunique')).reset_index()
        csv(reasons,stage/'lab_audit'/f'reasons-{idx:06d}.csv')
        chunk = lab_manifest['chunks'][idx]
        if chunk['chunk']!=idx:
            raise ValueError('manifest 청크 순서 불일치')
        chunk['valid_rows'] += len(restored)
        chunk['rejected_rows'] -= len(restored)
        if idx%50==0:
            logging.info('INR 복원: 감사 파티션 %s/%s',idx+1,lab_manifest['completed_chunks'])
    f = pd.concat(accepted,ignore_index=True)
    original_inr = pd.concat(all_inr,ignore_index=True)
    rejected = pd.concat(remaining_inr,ignore_index=True)
    if f.duplicated(['stay_id','source_row']).any():
        raise ValueError('같은 원본 INR 관측의 중복 복원 감지')
    if not np.isfinite(f.value).all() or not f.value.ge(0).all() or not f.unit.eq('ratio').all():
        raise ValueError('복원 INR의 수치·단위 검증 실패')
    ids = pd.Index(c.stay_id,name='stay_id')
    stats = f.groupby('stay_id').value.agg(['mean','min','max','std','count']).reindex(ids)
    stats['count'] = stats['count'].fillna(0).astype('int64')
    stats['missing'] = stats['count'].eq(0).astype('int8')
    old_features = pd.read_csv(backup/'lab_features.csv')
    features = old_features.copy()
    inr_cols = ['lab_INR_'+s for s in ['mean','min','max','std','count','missing']]
    for stat in ['mean','min','max','std','count','missing']:
        features['lab_INR_'+stat] = features.stay_id.map(stats[stat])
    pd.testing.assert_frame_equal(features.drop(columns=inr_cols),old_features.drop(columns=inr_cols),check_exact=True)
    assert stats['count'].sum()==len(f)
    assert stats.loc[stats['count'].le(1),'std'].isna().all()
    assert stats.loc[stats['count'].eq(0),['mean','min','max','std']].isna().all().all()
    assert not np.isinf(features.select_dtypes('number').to_numpy()).any()
    # 비INR 열의 CSV 문자열까지 보존하여 불필요한 실수 반올림도 막는다.
    feature_strings = pd.read_csv(backup/'lab_features.csv',dtype=str,keep_default_na=False)
    for col in inr_cols:
        feature_strings[col] = features[col].map(lambda x: '' if pd.isna(x) else str(x))
    csv(feature_strings,stage/'lab_features.csv')
    group = f.groupby('stay_id').agg(count=('value','size'),first=('charttime','min'),last=('charttime','max'),
                 extreme_count=('extreme','sum'),storetime_late_count=('storetime_late','sum'),storetime_missing_count=('storetime_missing','sum')).reindex(ids)
    timebase = c.set_index('stay_id')
    diag = pd.DataFrame(index=ids)
    diag['feature'] = 'INR'
    diag['count'] = stats['count']
    diag['first_hours'] = (group['first']-timebase.intime).dt.total_seconds()/3600
    diag['last_hours'] = (group['last']-timebase.intime).dt.total_seconds()/3600
    diag['hours_since_last'] = (timebase.prediction_time-group['last']).dt.total_seconds()/3600
    diag['observation_span_hours'] = (group['last']-group['first']).dt.total_seconds()/3600
    for col in ['extreme_count','storetime_late_count','storetime_missing_count']:
        diag[col] = group[col].fillna(0).astype('int64')
    replace_inr(backup/'lab_timing_diagnostics.csv',diag.reset_index(),stage)
    timing_dist = diag.reset_index().groupby('feature')[['count','first_hours','last_hours','hours_since_last','observation_span_hours']].describe().stack(0,future_stack=True).reset_index()
    replace_inr(backup/'lab_timing_distribution.csv',timing_dist,stage)
    miss = pd.read_csv(backup/'lab_missingness_comparison.csv')
    ix = miss.feature.eq('INR')
    updates = dict(after_observations=len(f), after_stays=int(stats['count'].gt(0).sum()),
                   after_patients=c.loc[c.stay_id.isin(f.stay_id),'subject_id'].nunique(),
                   after_missing_stays=int(stats.missing.sum()),after_missing_rate=float(stats.missing.mean()),
                   invalid_numeric_observations=int((~np.isfinite(pd.to_numeric(original_inr.raw_value,errors='coerce'))).sum()))
    for col,value in updates.items():
        miss.loc[ix,col] = value
    csv(miss,stage/'lab_missingness_comparison.csv')
    exclusions = rejected.groupby(['feature','reason']).agg(observations=('stay_id','size'),affected_stays=('stay_id','nunique')).reset_index()
    replace_inr(backup/'lab_exclusion_summary.csv',exclusions,stage)
    dist = pd.DataFrame([dict(feature='INR',observations=len(f),mean=f.value.mean(),std=f.value.std(),
                extreme_observations=int(f.extreme.sum()),extreme_stays=f.loc[f.extreme,'stay_id'].nunique(),min=f.value.min(),max=f.value.max())])
    replace_inr(backup/'lab_value_distribution.csv',dist,stage)
    edges = np.concatenate(([-np.inf],np.linspace(row[5],row[6],51),[np.inf]))
    hist,_ = np.histogram(f.value,bins=edges)
    replace_inr(backup/'lab_value_histogram.csv',pd.DataFrame(dict(feature='INR',lower=edges[:-1],upper=edges[1:],observations=hist)),stage)
    # Vital 특징은 복사해서 병합에만 사용한다. 추출·특징 재생성을 호출하지 않는다.
    shutil.copy2(root/'vital_features.csv',stage/'vital_features.csv')
    staged_args = SimpleNamespace(**{**vars(args),'output_dir':stage})
    merge(staged_args)
    # CSV 실수의 재파싱으로 생길 수 있는 1 ULP 차이도 막기 위해 INR 셀을
    # 특징 파일의 문자열과 동일하게 병합한다. 비INR 셀은 그대로 둔다.
    model_strings = pd.read_csv(stage/'modeling_dataset.csv',dtype=str,keep_default_na=False)
    feature_text = feature_strings.set_index('stay_id')
    for col in inr_cols:
        model_strings[col] = model_strings.stay_id.map(feature_text[col])
    csv(model_strings,stage/'modeling_dataset.csv')
    model = pd.read_csv(stage/'modeling_dataset.csv')
    old_model = pd.read_csv(backup/'modeling_dataset.csv')
    pd.testing.assert_frame_equal(model.drop(columns=inr_cols),old_model.drop(columns=inr_cols),check_exact=True)
    pd.testing.assert_frame_equal(model[inr_cols],pd.read_csv(stage/'lab_features.csv')[inr_cols],check_exact=True)
    original_cohort = pd.read_csv(args.cohort)
    pd.testing.assert_frame_equal(model[original_cohort.columns],original_cohort)
    assert model.stay_id.is_unique and len(model)==len(c)
    # 두 지문 변경은 INR 빈 단위 허용으로만 제한되며 Vital 정책·관측은 불변이다.
    for kind,m in [('lab',lab_manifest),('vital',vital_manifest)]:
        old_signature = m['signature']
        m['signature'] = signature(args,kind,new_config_bytes)
        m.setdefault('policy_migrations',[]).append(dict(operation='restore_unitless_INR_from_rejected_parquet',
            previous_signature=old_signature,updated_signature=m['signature'],
            vital_policy_unchanged=True,other_lab_policy_unchanged=True))
        save_json(stage/f'{kind}_audit/manifest.json',m)
    csv(pd.DataFrame(lab_manifest['chunks']),stage/'lab_audit/extraction_counts.csv')
    summary = pd.read_csv(backup/'pipeline_summary.csv')
    ix = summary.kind.eq('lab')
    summary.loc[ix,'valid_rows'] += len(f)
    summary.loc[ix,'rejected_rows'] -= len(f)
    # 기존 중복 후보는 INR 이외의 관측이며 복원 INR의 후보를 추가 확인한다.
    identity = IDS+['charttime','storetime','itemid','raw_value','raw_unit']
    new_duplicates = int(f.duplicated(identity).sum())
    summary.loc[ix,'duplicate_candidates_all_chunks'] += new_duplicates
    csv(summary,stage/'pipeline_summary.csv')
    validation = copy.deepcopy(state['previous_validation'])
    validation['validation_scope'] = 'incremental_INR; previously validated unchanged data protected by SHA256 and exact dataframe comparisons'
    validation['quality_notes'] = [s for s in validation['quality_notes'] if '전체 원본 스캔 완료' not in s]
    validation['quality_notes'].append('INR은 무차원 비율: 빈 원시 단위를 보존한 채 ratio로 해석. 제외 Parquet만으로 복원; 원본 CSV 재추출 없음.')
    validation['quality_notes'].append('기존 전체 검증을 바탕으로 Vital SHA256·비INR 관측/특징 불변성과 복원 INR 전수를 증분 검증함.')
    save_json(stage/'validation_report.json',validation)
    # 매핑표에는 빈 단위를 명시적으로 표시한다.
    mapping(staged_args)
    for name in ['item_mapping.csv']:
        m = pd.read_csv(stage/name,keep_default_na=False)
        m.loc[m.feature.eq('INR'),'accepted_raw_units'] = '_ | ratio | (빈 단위: 무차원 INR만 허용)'
        csv(m,stage/name)
    restored_result = dict(passed=True,source='lab_audit/rejected-*.parquet',original_csv_read=False,
            vital_reextracted=False,excluded_INR_candidates=len(original_inr),restored_INR_observations=len(f),
            remaining_invalid_INR=len(rejected),before_INR_missing_rate=1.0,after_INR_missing_rate=float(stats.missing.mean()),
            after_INR_missing_stays=int(stats.missing.sum()),after_INR_observed_stays=int(stats['count'].gt(0).sum()),
            total_lab_observations=int(summary.loc[ix,'valid_rows'].iloc[0]),modeling_shape=list(model.shape),
            label_distribution=model.mortality_28d.value_counts().to_dict(),new_INR_duplicate_candidates=new_duplicates,
            validation_scope=validation['validation_scope'])
    # 검증 완료한 staging 파일만 원자적으로 교체한다. 백업은 계속 남긴다.
    targets = [p for p in stage.rglob('*') if p.is_file() and p.name!='vital_features.csv']
    for path in targets:
        dest = root/path.relative_to(stage)
        tmp = dest.with_suffix(dest.suffix+'.inr-tmp')
        shutil.copyfile(path,tmp)
        os.replace(tmp,dest)
    tmp = CONFIG_PATH.with_suffix('.json.inr-tmp')
    tmp.write_bytes(new_config_bytes)
    os.replace(tmp,CONFIG_PATH)
    for name,digest in state['protected'].items():
        if sha(Path(name)) != digest:
            raise ValueError(f'수정 후 보존 검증 실패: {name}')
    # 모든 변경 파일을 다시 읽어 staging과 동일함을 검증한다.
    for path in targets:
        if sha(path)!=sha(root/path.relative_to(stage)):
            raise ValueError('교체 파일 지문 불일치')
    restored_result['updated_sha256'] = {str(p.relative_to(stage)):sha(root/p.relative_to(stage)) for p in targets}
    restored_result['config_sha256'] = sha(CONFIG_PATH)
    save_json(finished,restored_result)
    report(args)
    logging.info('INR 증분 복원·검증 완료: %s',json.dumps({k:v for k,v in restored_result.items() if k!='updated_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    args = arguments(__doc__).parse_args()
    setup(args)
    repair(args)
