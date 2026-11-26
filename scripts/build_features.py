"""유효 관측으로만 7종 Vital·15종 Lab 통계 특징을 생성한다."""
from pipeline_common import arguments, setup, aggregate

if __name__ == '__main__':
    # 유효 관측이 없는 stay도 count=0으로 보존한다.
    args = arguments(__doc__).parse_args()
    setup(args)
    for kind in ['vital', 'lab']:
        aggregate(args, kind)
