"""저장한 관측·특징·모델링 테이블의 시간·단위·ID를 재검증한다."""
from pipeline_common import arguments, setup, validate

if __name__ == '__main__':
    # 검증 실패는 명확한 오류로 종료하며 데이터를 자동 수정하지 않는다.
    args = arguments(__doc__).parse_args()
    setup(args)
    validate(args)
