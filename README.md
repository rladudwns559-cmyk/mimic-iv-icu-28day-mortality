# MIMIC-IV 성인 중환자실 환자의 28일 사망 위험 예측

데이터사이언스를 위한 프로그래밍 · 1트랙 2팀 최종 프로젝트 코드입니다.
성인 환자의 첫 ICU 입실 후 24시간 정보로 입실 후 28일 이내 사망을 예측합니다.

> 이 저장소에는 **코드와 실행 방법만** 있습니다. MIMIC-IV 원자료, 환자 단위 파생 자료(코호트·전처리·분할 파일),
> 학습된 모델 파일은 데이터 이용 협약(DUA)에 따라 올리지 않았습니다.

## 자료

- **MIMIC-IV 3.1** (PhysioNet, doi:10.13026/kpb9-mt58). PhysioNet 자격 인증과 DUA 서명 후 내려받을 수 있습니다.
- 사용 파일(`data/raw/`에 둠): `icustays.csv`, `chartevents.csv`, `labevents.csv`, `patients.csv.gz`, `admissions.csv.gz`,
  `d_items.csv`, `d_labitems.csv`
- `schemas/`에는 중간 파일의 **열 이름만** 담은 템플릿과 모델 입력 목록이 있습니다(환자 자료 없음).

## 환경

Python 3.12

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 실행 순서

| 순서 | 명령 | 하는 일 | 시간(참고) |
|---|---|---|---|
| 1 | `python scripts/01_make_cohort.py` | 성인 첫 ICU 입실, 24시간 관찰, 28일 결과 확정 조건으로 코호트 생성 | 미측정 |
| 2 | `python scripts/run_pipeline.py` | 활력징후 7종·혈액검사 12종 추출, 첫 24시간 변수 생성 | 약 43분 |
| 3 | `python scripts/clean_gross_errors.py` | 명백한 기록 오류 활력징후 관측 제거(176건) | 미측정 |
| 4 | `python scripts/split_dataset.py` | 층화 70/15/15 분할 (random_state 42) | 미측정 |
| 5 | `python scripts/reproduce_final_icu_xgboost.py` | 최종 XGBoost 재학습과 테스트 평가 | 미측정 |

시간은 팀원 노트북(Windows 11) 한 대에서 잰 값이며, 재지 않은 단계는 미측정으로 두었습니다.
5번까지 실행하면 테스트 4,517명에서 ROC-AUC 0.819, AP 0.580, Brier 0.131이 나옵니다.

### 개발 단계 비교 실험 (선택)

```powershell
python scripts/train_three_models_cv.py              # 세 모델 고정 설정 비교 (5-fold)
python scripts/compare_three_models_imputation_cv.py # 중앙값 vs MICE 대체
python scripts/tune_three_models_nested_cv.py        # 5×3 중첩 교차검증
python scripts/compare_tuned_xgboost_missingness.py  # 결측 처리·젖산 5조건
python scripts/analyze_lactate_subgroup_calibration.py
python scripts/compare_xgboost_indicators_icu.py     # 결측 지시변수 제거·첫 ICU 유형 추가
```

### 보고서 추가 집계와 대시보드 (`final-dashboard/`)

```powershell
cd final-dashboard
..\.venv\Scripts\python train.py                    # 최종 모델 저장 + 입력 변환 검증 (약 15초)
..\.venv\Scripts\python report_extras.py            # 부트스트랩 95% CI, 분류표, 테스트 곡선
..\.venv\Scripts\python report_feedback_extras.py   # 대상자 특성표·제외군 비교·교차검증 ±SD
..\.venv\Scripts\python app.py                      # 교육용 대시보드 (Gradio)
```

`report_*.py`는 최종 모델 성능이 팀 기록과 소수 여섯째 자리까지 같을 때만 결과를 만듭니다.
결과는 모두 `final-dashboard/outputs/`에 저장되며 환자 단위 자료는 저장하지 않습니다.
대시보드는 **교육용 연구 모델이며 실제 진료에 사용할 수 없습니다.**

## 고정한 난수

| 대상 | 값 |
|---|---|
| 학습·검증·테스트 분할 | 42 |
| 로지스틱 회귀, 랜덤 포레스트, XGBoost `random_state` | 42 |
| MICE 대체 자료 5개 | 42–46 |
| 부트스트랩 신뢰구간 (2,000회) | 42 |

## 최종 모델

XGBoost · 결측값 대체 없음(NaN) + 결측 지시변수 + 젖산 + 첫 ICU 유형 (입력 150개)
트리 1,000 · 깊이 4 · 학습률 0.03 · min_child_weight 10 · reg_lambda 10

## 생성형 AI 활용

OpenAI Codex와 Claude Code(Anthropic)를 코드 작성과 보고서 문단·수치 대조에 사용했습니다.
모든 수치는 이 저장소의 코드를 다시 실행해 확인했습니다.
