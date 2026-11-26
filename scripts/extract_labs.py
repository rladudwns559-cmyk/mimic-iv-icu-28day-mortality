"""입원·환자 ID와 각 stay 관찰 구간으로 Lab 관측을 추출한다."""
from pipeline_common import arguments, setup, extract

if __name__ == '__main__':
    # 같은 입원의 여러 ICU stay에도 독립적인 시간 구간을 적용한다.
    args = arguments(__doc__).parse_args()
    setup(args)
    extract(args, 'lab')
