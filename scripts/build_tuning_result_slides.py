"""Create editable result slides from completed imputation and nested tuning runs."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.codex_deps'))
import pandas as pd
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor

def main():
    source=ROOT/'results/three_models_nested_tuning'
    assert json.loads((source/'COMPLETE.json').read_text())['checks_passed']
    before=pd.read_csv(source/'before_after.csv')
    best=json.loads((source/'final_parameters.json').read_text())
    imp=pd.read_csv(ROOT/'results/three_models_imputation_5fold/cv_summary.csv')
    names={'logistic_regression':'Logistic regression','random_forest':'Random forest','xgboost':'XGBoost'}
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    def box(slide,x,y,w,h,text,size=21):
        shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
        shape.text_frame.word_wrap=True
        for i,line in enumerate(text.split('\n')):
            p=shape.text_frame.paragraphs[0] if i==0 else shape.text_frame.add_paragraph()
            p.text=line;p.font.name='맑은 고딕';p.font.size=Pt(size)
            p.font.color.rgb=RGBColor.from_string('24354A');p.space_after=Pt(12)
    def slide(title,bullets,notes=''):
        s=prs.slides.add_slide(prs.slide_layouts[6])
        s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('F8FAFC')
        box(s,.55,.35,12.2,.9,title,28)
        box(s,.7,1.6,11.9,4.9,'\n'.join('• '+b for b in bullets),22)
        box(s,.65,7.0,11,.25,'Train 내부 탐색·평가 | ALT·AST·총빌리루빈 제외 | 이번 실험 Test 미사용',10)
        s.notes_slide.notes_text_frame.text=notes
        return s
    slide('결측 처리 비교 후, 모델별 설정을 탐색했다',[
        'Train 25,595명 · 고정된 바깥 층화 5-fold',
        'Logistic·RF: 중앙값+지표 / XGBoost: MICE+지표',
        '결측 처리와 입력 변수는 고정하고 하이퍼파라미터만 탐색',
        'Test는 이전 단계에서 평가했으며 이번 선택·평가에는 사용하지 않음'],
        '기존 발표자료에 추가하거나 모델 학습·결과 부분을 교체하기 위한 편집 가능한 슬라이드입니다. 전처리 선택은 이전 개발 데이터 탐색에 근거했으므로 전체 연구의 완전한 독립 평가라고 표현하지 않습니다.')
    bullets=[]
    for m in names:
        a=imp.query('model==@m and method=="median_indicator"').iloc[0]
        b=imp.query('model==@m and method=="mice_indicator"').iloc[0]
        bullets.append(f'{names[m]} ROC-AUC: 중앙값 {a.roc_auc_mean:.4f} / MICE {b.roc_auc_mean:.4f}')
    bullets.append('차이는 작음. MICE는 5개 모델 평균이므로 앙상블 효과도 포함')
    slide('먼저 결측 처리 2가지 × 모델 3가지를 비교했다',bullets,
          '같은 폴드·같은 입력 135개를 사용했습니다. 결측 처리 비교 단계의 XGB는 트리1000개 고정이며 조기 종료 없음. 값은 폴드 평균입니다.')
    slide('탐색 범위는 미리 고정했다',[
        'Logistic L2: C=.01, .1, 1, 10, 100 (5개)',
        'RF: 깊이 10·15·20·제한없음 / leaf 최소 2·5·10 / 특징 sqrt·0.5의 4개 조합',
        'XGB: 깊이2~4, 학습률 .02~.05, 트리600~1500, 자식가중치5·10, L2 1·5·10의 6개 조합',
        '클래스 가중치 미적용 · 임계값 0.5 유지'],
        'RF와 XGB는 나열된 모든 값의 완전 격자가 아니라 4개 및 6개의 명시적 조합입니다. 정확한 후보는 experiment.json의 candidates를 참고합니다. 원래 기준 설정도 후보에 포함했습니다.')
    slide('바깥 평가와 내부 선택을 분리했다',[
        '바깥 5-fold: 선택된 설정의 성능 평가',
        '각 바깥 훈련 안의 3-fold: 평균 ROC-AUC로 후보 선택',
        '대체·표준화도 매번 해당 훈련 부분에만 적합',
        'XGB 내부 탐색은 MICE 1회, 바깥 평가는 MICE 5회 평균'],
        '내부 탐색 점수를 최종 성능으로 제시하지 않습니다. 계산 효율을 위한 단일 대체 내부 탐색은 5회 앙상블 자체를 직접 최적화한 절차가 아니라는 한계를 명시합니다. 결측 방법 선택 자체는 nested loop 밖에서 고정했습니다.')
    bullets=[]
    for r in before.itertuples():
        bullets.append(f'{names[r.model]} ROC-AUC: {r.baseline_auc:.4f} → {r.tuned_auc:.4f} ({r.auc_difference:+.4f})')
    bullets.append('조정 후=각 바깥 폴드에서 선택 절차를 적용한 검증 성능')
    slide('하이퍼파라미터 조정 전후: 바깥 검증 비교',bullets,
          '조정 전과 후에 같은 바깥 환자·폴드를 사용했습니다. 탐색을 했다고 성능이 반드시 개선되는 것은 아닙니다. AP·Brier·log loss 및 폴드 표준편차는 training_report.md와 nested_cv_summary.csv에 있습니다.')
    slide('전체 Train 재학습에 선택한 설정',[
        f'Logistic: C={best["logistic_regression"]["params"]["C"]} (L2 유지)',
        'RF: '+json.dumps(best['random_forest']['params'],ensure_ascii=False),
        'XGB: '+json.dumps(best['xgboost']['params'],ensure_ascii=False),
        '최종 설정은 내부 점수 평균으로 선택; 바깥 검증 점수로 고르지 않음'],
        '최종 모델은 전체 Train에서 재학습했습니다. 바깥 검증은 폴드마다 다른 설정을 선택할 수 있으므로 이 단일 최종 설정의 독립 성능 보장은 아닙니다. Test를 새로 평가하지 않았습니다.')
    slide('남은 검증과 발표에서 구분할 점',[
        '모델·전처리 선택까지 포함한 전체 절차의 외부 검증은 아직 없음',
        'KNN 대체, L1 비교, ICU 유형 추가는 별도 후속 실험',
        '임계값·보정은 Train 내부 예측으로 검토',
        '성능 순위와 임상적 활용 가능성을 구분해서 해석'],
        '기존 Test는 이미 확인했으므로 미사용 Test라고 발표하지 않습니다. 미래 시점·다른 병원 등 새 검증 자료에서 확인이 필요합니다.')
    path=source/'결측처리_하이퍼파라미터_결과슬라이드.pptx'
    prs.save(path)
    print(path)

if __name__=='__main__':
    main()
