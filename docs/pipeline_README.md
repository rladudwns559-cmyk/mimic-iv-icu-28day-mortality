# MIMIC 원본 데이터

`data/raw/`에 원본 데이터를 보관합니다. 원본은 수정하지 않습니다.

- `icustays.csv`, `chartevents.csv`, `labevents.csv`
- `patients.csv.gz`, `admissions.csv.gz`
- `d_items.csv`, `d_labitems.csv`, `d_icd_diagnoses.csv.gz`
- `patients.csv/`, `admissions.csv/`: 압축 해제된 CSV를 담은 폴더

원본 파일의 내용은 변경하지 않았으며, 압축본과 압축 해제본을 모두 보존했습니다.

## ICU 28일 사망예측 코호트 생성

Python 3.10 이상과 pandas가 필요합니다.

```powershell
python scripts/01_make_cohort.py
```

입력은 `icustays.csv`, `patients.csv.gz`, `admissions.csv.gz`이며 출력은 `data/processed/`에 저장합니다.

| 출력 파일 | 대상 |
|---|---|
| `cohort_base.csv` | 24시간 관찰 조건을 충족하고 라벨이 0/1로 확정된 성인 |
| `cohort_unknown.csv` | 24시간 조건은 충족하지만 28일 라벨이 unknown인 성인 |
| `cohort_audit.csv` | 첫 ICU 입실이 성인인 후보 전체와 제외 사유·라벨 근거 |

24시간 조건을 충족한 전체 코호트는 base와 unknown의 합입니다. unknown을 0으로 바꾸면 안 됩니다.

### 라벨 판정 원칙

- 전체 입원의 `deathtime`을 우선하여 재입원 중 사망도 확인합니다.
- `dod`만 있으면 `[당일 00:00, 다음 날 00:00)` 범위로 비교합니다. `death_time`은 비워 두고 날짜 범위를 별도 저장합니다.
- 사망 날짜가 24시간 경계에 걸리면 제외하고, 28일 경계에 걸리면 unknown으로 저장합니다.
- 0은 사망시각/사망일이 28일 이후이거나, 28일 시점 이후 실제 입원 또는 생존 퇴원(`hospital_expire_flag=0`) 기록이 있을 때만 부여합니다.
- ICU `outtime`은 생존 퇴실 여부가 따로 없어 단독 생존 근거로 사용하지 않습니다. 사망 기록이 없다는 이유나 추적기간에 관한 가정만으로 0을 부여하지 않습니다.
- `survival_evidence`, `survival_confirmed_at`, `label_reason`, `death_source`에 판정 근거를 기록합니다. 모순된 기록은 audit에 제외 사유를 남깁니다.

생존 확인용 미래 기록과 사망 관련 열은 라벨 작성·검증용이며 모델 feature로 사용하면 안 됩니다. 라벨 확정 환자만 사용하면 선택 편향이 생길 수 있으므로 unknown 비율과 처리 방식을 연구 결과에 보고하세요.

첫 stay는 전체 ICU 기록에서 먼저 정합니다. 첫 입실이 소아이거나 24시간 조건을 충족하지 못해도 이후 stay로 대체하지 않습니다. `intime` 누락으로 첫 입실 순서를 확정할 수 없는 환자는 제외합니다. 입실시각 동률이면 `stay_id`가 작은 한 건을 선택합니다.

경계 조건 검증:

```powershell
python scripts/test_cohort.py
```

공식 정의: [patients](https://mimic.mit.edu/docs/iv/modules/hosp/patients.html), [admissions](https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html).

## 첫 24시간 Vital/Lab 특징 생성

Windows 로컬 CPU에서 실행하며 원본과 기존 cohort 파일은 수정하지 않습니다.
Python 3.10 이상과 `requirements.txt`의 패키지가 필요합니다.

```powershell
python -m pip install -r requirements.txt
python scripts/test_feature_pipeline.py
python scripts/run_pipeline.py --max-chunks 1 --chunksize 50000 --output-dir data/processed/sample_checked
python scripts/run_pipeline.py
```

표본은 원본 각 파일의 앞 5만 행만 읽으므로 전체 환자의 대표 표본은 아닙니다.
복수 stay와 중첩 구간, 경계 시각, C/F 변환, 결측 관측, 중단 후 재실행은
별도의 합성 CSV 테스트에서 확인합니다. 전체 실행은 원본을 끝까지 읽습니다.

개별 단계 실행도 지원합니다. 모든 단계에서 `--raw-dir`, `--cohort`,
`--output-dir`, `--chunksize` 옵션을 동일하게 지정하세요.

```powershell
python scripts/extract_vitals.py
python scripts/extract_labs.py
python scripts/build_features.py
python scripts/merge_dataset.py
python scripts/validate_dataset.py
```

### 매핑과 정제 기준

`scripts/feature_config.json`에서 itemid, 예상 label, 허용 단위, 진단용
극단값 경계를 관리합니다. 매 실행 시 로컬 데이터 사전의 모든 label을 대조하고,
Vital은 사전의 unitname도 확인합니다. Lab 사전에는 단위 열이 없으므로
실제 관측의 valueuom을 허용 목록과 대조합니다.

수치 제외 경계는 `scripts/pipeline_common.py` 상단의 `NONNEGATIVE_MIN`,
`SPO2_MIN_EXCLUSIVE`, `SPO2_MAX_INCLUSIVE`, `TEMPERATURE_MIN_C` 상수로
관리합니다. 변경한 기준은 다른 출력 디렉터리에서 실행하세요.

- Temperature F=223761, C=223762이며 유효 단위의 F 관측만 섭씨로 변환합니다.
- Glucose=50931, Potassium=50971, AST=50878, ALT=50861입니다.
- 단위를 확인한 다음 NaN/inf, 명백히 불가능한 음수를 제외합니다.
  AnionGap은 음수가 가능하므로 보존하고 체온은 섭씨 절대영도 미만만 제외합니다.
- SpO2는 0 이하 또는 100 초과를 제외합니다. 그 외 극단값은 보존합니다.
- mEq/L과 mmol/L은 설정에 명시한 단가 이온/anion gap 항목에서 같은 수치로
  취급합니다. AST/ALT의 IU/L과 U/L, 혈구수의 K/uL과 10^3/uL도 명시적 동의 단위입니다.
- INR(51237)은 무차원 비율입니다. `_`, `ratio`, 빈 단위를 허용하고
  출력 단위는 `ratio`로 통일합니다. 빈 원시 단위는 그대로 남깁니다.
  이 예외는 INR에만 적용하며 비유한 수치와 음수는 계속 제외합니다.
- 극단값 경계는 임상 정상 범위나 추가 제외 기준이 아닙니다.
  `*_value_distribution.csv`와 `*_value_histogram.csv`에서 검토합니다.

`charttime`은 `[intime, prediction_time)`에 있어야 합니다. `storetime`이
예측 시각과 같거나 늦은 관측도 이 charttime 기준 산출물에는 포함되며,
`*_timing_diagnostics.csv`에 개수를 남깁니다. 실제 시점의 정보 가용성을
엄격히 적용하는 연구에서는 이 관측의 처리 방침을 별도로 결정해야 합니다.

### 저장 및 재시작

`vital_24h.parquet`와 `lab_24h.parquet`는 단일 파일이 아닌 **분할 Parquet
데이터셋 디렉터리**입니다. 파티션 하나씩 읽으면 전체 관측을 메모리에
올리지 않아도 됩니다. `pd.read_parquet(path)`로 전체를 읽을 수도 있으나
메모리 사용에 주의하세요. 특징 계산은 청크 평균과 M2를 결합해 표본
표준편차(ddof=1)를 계산합니다. 1개 이하 관측의 std는 NaN입니다.

`*_audit/manifest.json`이 완료된 청크를 기록합니다. 중단 후 **동일한
명령어**로 실행하면 완료된 청크는 재저장하지 않습니다. CSV를 체크포인트까지
순차 스캔하는 시간은 다시 필요합니다. 완료 직전 중단된 파티션은 같은 파일명으로
원자적으로 교체하므로 관측이 추가 중복되지 않습니다.

원본 경로·크기·수정시각, cohort·설정·공통 코드·사전 해시 또는 chunksize가
달라지면 기존 결과의 혼합을 막기 위해 오류를 냅니다. 기준을 바꾼 실행은
기존 결과를 보존하면서 다른 `--output-dir`에 생성하세요. 단일 출력 디렉터리에
여러 파이프라인을 동시에 실행하지 마세요.

동일 내용의 별도 원본 행은 임상 반복 측정일 수 있으므로 자동 삭제하지 않습니다.
전체 청크를 대상으로 동일 stay·시각·itemid·값·단위·storetime의 중복 후보를
디스크 기반 인덱스로 계산하여 `pipeline_summary.csv`에 기록합니다.

### 주요 산출물

| 파일 | 내용 |
|---|---|
| `item_mapping.csv`, `cleaning_rules.csv` | 실제 사전 label·단위·최종 변수·제외 기준 |
| `vital_24h.parquet/`, `lab_24h.parquet/` | 정제 관측, 원시 값·단위, 원본 행 번호 |
| `vital_features.csv`, `lab_features.csv` | 전체 stay의 mean/min/max/std/count/missing |
| `modeling_dataset.csv` | cohort 원본 열과 132개 통계 특징의 left join |
| `model_feature_columns.json` | age/gender 및 Vital/Lab 통계 특징 후보 목록 |
| `*_audit/rejected-*.parquet` | 제외 관측과 원시 값·단위·제외 사유 |
| `*_audit/linkage-*.csv` | ID 불일치 또는 파싱 불가능한 charttime 사례 |
| `*_exclusion_summary.csv` | 항목·사유별 제외 관측 수와 고유 stay 수 |
| `*_missingness_comparison.csv` | 정제 전후 관측·stay·환자 수와 결측률 |
| `*_unit_inventory.csv` | 항목별 실제 원시 단위 분포 |
| `*_value_distribution.csv`, `*_value_histogram.csv` | 유효 관측 분포·극단값 개수 |
| `*_timing_diagnostics.csv`, `*_timing_distribution.csv` | stay별 측정 횟수·최초/최종 시각·공백·storetime 진단 |
| `pipeline_summary.csv`, `validation_report.json`, `pipeline.log` | 전체 통계·검증 결과·진행 로그 |
| `pipeline_report.md` | 전체 결과를 표로 모은 한글 보고서 |

보고서만 다시 작성할 때는 `python scripts/write_report.py`를 실행합니다.

### 기존 제외 관측으로 INR만 복원

```powershell
python scripts/test_repair_inr.py
python scripts/repair_inr.py
```

`lab_audit/rejected-*.parquet`에 보존한 원시 수치·단위·ID·시각으로 INR만
복원합니다. `chartevents.csv`와 `labevents.csv`를 다시 읽지 않으며 Vital
추출·특징 생성과 전체 Lab 특징 생성도 호출하지 않습니다. 기존 Lab 특징의
INR 6개 열만 갱신하고 데이터셋을 재병합합니다. 다른 Lab 관측·특징·모델링
열은 정확히 같은지 대조하고, Vital 관측·특징은 SHA256으로 보호합니다.

수정 전 파일은 `inr_repair_backup/`에 보존합니다. staging에서 검증한 파일을
원자적으로 교체하며 중단 시 같은 명령으로 최초 백업에서 다시 구성합니다.
`inr_repair_report.json`은 복원 수·결측률·라벨 분포 및 변경 파일 지문을
기록합니다. 기존 전체 검증 증거와 보존 증명을 이용하는 **증분 검증**이며,
전체 Vital/Lab 전수 검증을 재실행하지 않습니다. 입력·설정 지문을 확인한 후
INR 정책 변경 이력만 두 manifest에 남겨 향후 단계별 실행과의 일관성을 유지합니다.

시간 진단 변수는 모델링 테이블에 자동 포함하지 않습니다. cohort의 사망시각,
사망일 범위, 생존 근거 등 미래 정보 열은 원본 보존을 위해 모델링 CSV에 남지만
**학습 입력으로 사용하면 안 됩니다**. 명시적 후보 목록을 기준으로 선택하세요.
결측치 대체·표준화·모델 학습은 수행하지 않습니다.
