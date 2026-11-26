"""추출부터 특징 생성·병합·자동 검증까지 CPU에서 실행한다."""
from pipeline_common import arguments, setup, extract, aggregate, merge, validate
from write_report import report

if __name__ == '__main__':
    # 어느 단계에서든 오류가 나면 다음 단계를 실행하지 않는다.
    args = arguments(__doc__).parse_args()
    setup(args)
    if args.max_chunks is not None and args.output_dir.resolve() == (args.cohort.parent).resolve():
        raise ValueError('표본 실행은 --output-dir에 별도 디렉터리를 지정하세요.')
    for kind in ['vital', 'lab']:
        extract(args, kind)
        aggregate(args, kind)
    merge(args)
    validate(args)
    report(args)
