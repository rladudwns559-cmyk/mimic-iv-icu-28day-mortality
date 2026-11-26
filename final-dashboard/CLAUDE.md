# final-dashboard 규칙

- 모델은 팀 최종 설정(XGBoost · 대치 없음 + 결측 지시변수 + 젖산 + 첫 ICU 유형)만 쓴다. 설정을 바꾸지 않는다.
- 데이터는 상위 폴더 `data/processed/error_cleaned/splits/`와 `data/raw/icustays.csv`에서 읽는다. MIMIC-IV 제한 데이터이므로 복사·업로드하지 않는다.
- 파이썬은 상위 폴더의 `.venv\Scripts\python`을 쓴다. 새 패키지는 설치하지 않는다.
- 결과 파일은 모두 `outputs/`에 저장한다. 환자 개인 단위 자료는 저장하지 않고 집계값만 남긴다.
- 난수는 `random_state=42`로 고정한다.
- `app.py`는 `outputs/model.ubj`를 불러 쓰기만 하고 다시 학습하지 않는다(`.fit` 금지).
- 입력 변환은 `dashboard_core.py` 하나로 하고, `train.py`가 테스트 환자 전원으로 그 변환을 검증한다.
- 그림 안 글자는 영어로 쓴다(보고서 그림 1·2인 make_schematic.py·make_flowchart.py는 팀 요청으로 한국어). 설명과 답은 한국어로 한다.
- 화면 맨 위와 맨 아래에 "교육용 연구 모델, 실제 진료에 사용할 수 없음" 문구를 둔다.
