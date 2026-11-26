"""검증 결과와 항목별 품질 통계를 하나의 한글 보고서로 정리한다."""
import json
import logging

import pandas as pd

from pipeline_common import arguments, setup


def report(args):
    """저장한 통계만 읽으며 원본 관측이나 cohort를 변경하지 않는다."""
    root = args.output_dir
    validation = json.loads((root/'validation_report.json').read_text(encoding='utf-8'))
    sections = ['# Vital/Lab 24시간 추출·전처리 결과',
                f"자동 검증: {'통과' if validation['passed'] else '실패'}",
                f"최종 모델링 테이블: {validation['modeling_shape'][0]:,}행 × {validation['modeling_shape'][1]}열"]
    comparisons = []
    for kind in ['vital','lab']:
        comparison = pd.read_csv(root/f'{kind}_missingness_comparison.csv')
        comparison['excluded_observations'] = comparison.before_observations-comparison.after_observations
        comparisons.append(comparison)
    combined = pd.concat(comparisons,ignore_index=True)
    combined.to_csv(root/'cleaning_comparison.csv',index=False,encoding='utf-8-sig')
    inr = combined.loc[combined.feature.eq('INR')]
    if len(inr) and inr.iloc[0].before_observations > 0 and inr.iloc[0].after_observations == 0:
        sections.append(f"INR은 관찰 구간 내 {int(inr.iloc[0].before_observations):,}건 모두 정제에서 제외되어 최종 결측률이 100%입니다. 단위·제외 사유 표를 확인하세요.")
    repair_path = root/'inr_repair_report.json'
    if repair_path.exists():
        repaired = json.loads(repair_path.read_text(encoding='utf-8'))
        sections.append(f"INR 복원: 제외 Parquet에서 {repaired['restored_INR_observations']:,}건 복원, 수치 오류 {repaired['remaining_invalid_INR']:,}건 제외 유지. 결측률 {repaired['after_INR_missing_rate']:.2%}. 원본 CSV 및 Vital 재추출 없이 증분 검증했습니다.")

    def table(title, frame):
        # tabulate 등 추가 패키지 없이 Markdown 표를 만든다.
        frame = frame.copy().fillna('')
        lines = ['| ' + ' | '.join(map(str,frame.columns)) + ' |',
                 '| ' + ' | '.join(['---']*len(frame.columns)) + ' |']
        lines += ['| ' + ' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row) + ' |'
                  for row in frame.itertuples(index=False,name=None)]
        sections.extend(['## '+title, '\n'.join(lines)])
        logging.info('%s\n%s', title, frame.to_string(index=False))

    table('코호트·추출 전후·오류·특징 크기',pd.read_csv(root/'pipeline_summary.csv'))
    table('사망·생존 라벨 분포',pd.DataFrame([
        dict(mortality_28d=k, count=v) for k,v in validation['label_distribution'].items()]))
    preserved = root/'input_preservation_check.json'
    if preserved.exists():
        table('원본·기존 코호트 보존 확인',pd.DataFrame(json.loads(preserved.read_text(encoding='utf-8'))))
    for kind in ['vital','lab']:
        table(f'{kind} 정제 전후 관측·환자·결측률',pd.read_csv(root/f'{kind}_missingness_comparison.csv'))
        table(f'{kind} 항목·사유별 제외',pd.read_csv(root/f'{kind}_exclusion_summary.csv'))
        table(f'{kind} 유효 값 분포·보존된 극단값',pd.read_csv(root/f'{kind}_value_distribution.csv'))
        table(f'{kind} 측정 횟수·관측 구간 분포',pd.read_csv(root/f'{kind}_timing_distribution.csv'))
        timing = pd.read_csv(root/f'{kind}_timing_diagnostics.csv')
        table(f'{kind} 정보 가용 시각 진단',timing.groupby('feature')[[
            'storetime_late_count','storetime_missing_count']].sum().reset_index())
    sections.extend(['## 남아 있는 품질 문제와 해석',
                     '\n'.join('- '+s for s in validation['quality_notes']),
                     '- INR 복원 후에는 무차원 INR의 빈 단위를 ratio로 허용합니다. 다른 항목의 불명확 단위는 수치 집계에서 제외합니다. 원시 단위는 unit_inventory CSV에 보존합니다.' if repair_path.exists() else '- 단위가 비어 있는 INR을 포함하여 불명확 단위는 수치 집계에서 제외했습니다. 단위별 수는 unit_inventory CSV를 확인하세요.',
                     '- missingness의 before는 cohort ID와 관찰 구간이 일치한 관측을 기준으로 합니다. 원본 전체 행 수는 pipeline_summary의 source_rows입니다.',
                     '- 제거 사유는 단위 오류 → 비유한 수치 → 불가능값 순의 상호 배타적 주 사유입니다. invalid_numeric_observations는 단위 오류 동반 여부와 무관한 별도 집계입니다.',
                     '- 분포 표는 유효 관측에 가중한 평균·표본 표준편차입니다. 히스토그램은 별도 CSV에 저장합니다.',
                     '- 동일 원본 Lab 행이 복수 stay 구간에 들어가면 각 stay에 독립 배정됩니다. overlapping_lab_observations에 원본 행 수를 기록합니다.'])
    (root/'pipeline_report.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    logging.info('최종 보고서 저장: %s',root/'pipeline_report.md')


if __name__ == '__main__':
    args = arguments(__doc__).parse_args()
    setup(args)
    report(args)
