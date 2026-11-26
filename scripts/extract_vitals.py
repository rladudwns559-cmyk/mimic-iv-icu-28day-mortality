"""로컬 사전 기준 Vital 관측을 청크 단위로 추출한다."""
from pipeline_common import arguments, setup, extract

if __name__ == '__main__':
    # 코호트와 로컬 사전 검증 후 관측 파티션을 저장한다.
    args = arguments(__doc__).parse_args()
    setup(args)
    extract(args, 'vital')
