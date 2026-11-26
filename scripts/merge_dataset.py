"""코호트 전체를 보존하는 left join으로 모델링 테이블을 생성한다."""
from pipeline_common import arguments, setup, merge

if __name__ == '__main__':
    # 기존 라벨을 재계산하지 않고 원본 코호트 열을 유지한다.
    args = arguments(__doc__).parse_args()
    setup(args)
    merge(args)
