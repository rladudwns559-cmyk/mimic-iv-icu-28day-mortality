"""Create a Korean, searchable PDF summarizing the November 17 experiments."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.codex_deps'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.patches import Rectangle
from PIL import Image
from pypdf import PdfReader

OUT=ROOT/'results/daily_summary_2026-11-17'
TUNING=ROOT/'results/three_models_nested_tuning'
IMPUTE=ROOT/'results/three_models_imputation_5fold'
FONT_PATH=Path('C:/Windows/Fonts/malgun.ttf')
FONT=FontProperties(fname=str(FONT_PATH))
MEASURE=TextToPath()
plt.rcParams.update({'font.family':FONT.get_name(),'pdf.fonttype':42,'axes.unicode_minus':False})
WIDTH,HEIGHT=8.2677,11.6929
LEFT,RIGHT=.075,.925
NAVY='#152b46'; TEAL='#187f97'; INK='#24354a'; GRAY='#596a7f'
NAMES={'logistic_regression':'Logistic','random_forest':'Random forest','xgboost':'XGBoost'}


def wrap(text,width_pt,size):
    prop=FontProperties(fname=str(FONT_PATH),size=size)
    def width(s):
        return MEASURE.get_text_width_height_descent(s,prop,False)[0]
    result=[]
    for paragraph in str(text).split('\n'):
        current=''
        for word in paragraph.split(' '):
            proposed=(current+' '+word).strip()
            if current and width(proposed)>width_pt:
                result.append(current);current=''
            if width(word)>width_pt:
                for char in word:
                    if width(current+char)>width_pt and current:
                        result.append(current);current=''
                    current+=char
            else:
                current=(current+' '+word).strip()
        result.append(current)
    return result


class Report:
    def __init__(self,pdf):
        self.pdf=pdf; self.fig=None; self.page=0; self.y=0; self.markdown=[]; self.artists=[]
    def finish(self):
        if self.fig is None:return
        if self.y<.067:
            raise RuntimeError(f'Page {self.page}: content overflow at {self.y:.3f}')
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for artist in self.artists:
            bbox=artist.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            if bbox.x1>.945 or bbox.x0<.058 or bbox.y0<.058:
                raise RuntimeError(f'Page {self.page}: text outside margins: {artist.get_text()}')
        self.pdf.savefig(self.fig,facecolor='white')
        if self.page in [1,5,6]:
            self.fig.savefig(OUT/f'preview_page_{self.page}.png',dpi=125,facecolor='white')
        plt.close(self.fig);self.fig=None
    def new(self,title,subtitle=''):
        self.finish();self.page+=1
        self.fig=plt.figure(figsize=(WIDTH,HEIGHT),dpi=100,facecolor='white');self.artists=[]
        self.fig.text(LEFT,.968,'ICU 28일 사망 예측 | 분석 기록',fontproperties=FONT,fontsize=9,color=TEAL)
        self.fig.text(RIGHT,.968,'2026.11.17 · KST',fontproperties=FONT,fontsize=9,color=GRAY,ha='right')
        self.fig.add_artist(plt.Line2D([LEFT,RIGHT],[.951,.951],transform=self.fig.transFigure,color=TEAL,lw=1.5))
        self.y=.925
        self.p(title,20,color=NAVY,gap=.015)
        if subtitle:self.p(subtitle,10.5,color=GRAY,gap=.018)
        self.fig.text(LEFT,.035,'오늘의 설정 · 결측 처리 비교 · 하이퍼파라미터 탐색',fontproperties=FONT,fontsize=8,color=GRAY)
        self.fig.text(RIGHT,.035,str(self.page),fontproperties=FONT,fontsize=9,color=GRAY,ha='right')
        self.markdown.extend(['',f'## {self.page}. {title}',''])
    def p(self,text,size=11,color=INK,gap=.012):
        lines=wrap(text,(RIGHT-LEFT)*WIDTH*72,size)
        leading=size*1.6/(HEIGHT*72)
        for line in lines:
            a=self.fig.text(LEFT,self.y,line,fontproperties=FONT,fontsize=size,va='top',color=color)
            self.artists.append(a);self.y-=leading
        self.y-=gap
        self.markdown.extend([text,''])
    def h(self,text):
        self.y-=.008;self.p(text,13,color=TEAL,gap=.008)
    def table(self,headers,rows,widths=None,size=10):
        widths=np.array(widths if widths else [1/len(headers)]*len(headers))
        widths=widths/widths.sum()*(RIGHT-LEFT)
        xs=np.r_[LEFT,LEFT+np.cumsum(widths)]
        self.markdown.extend(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)])
        self.markdown.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
        self.markdown.append('')
        for i,row in enumerate([headers]+rows):
            wrapped=[wrap(value,(w*WIDTH*72)-12,size) for value,w in zip(row,widths)]
            leading=size*1.45/(HEIGHT*72)
            height=max(map(len,wrapped))*leading+.016
            for j,lines in enumerate(wrapped):
                self.fig.add_artist(Rectangle((xs[j],self.y-height),widths[j],height,
                    transform=self.fig.transFigure,facecolor='#e8f1f6' if i==0 else ('#f7f9fb' if i%2==0 else 'white'),
                    edgecolor='#dce4ec',lw=.5))
                for k,line in enumerate(lines):
                    a=self.fig.text(xs[j]+.009,self.y-.008-k*leading,line,fontproperties=FONT,
                        fontsize=size,va='top',color=NAVY if i==0 else INK)
                    self.artists.append(a)
            self.y-=height
        self.y-=.018
    def image(self,path):
        with Image.open(path) as im:ratio=im.width/im.height
        height=((RIGHT-LEFT)*WIDTH/ratio)/HEIGHT
        ax=self.fig.add_axes([LEFT,self.y-height,RIGHT-LEFT,height])
        ax.imshow(plt.imread(path));ax.axis('off');self.y-=height+.012


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((TUNING/'experiment.json').read_text(encoding='utf-8'))
    completed=json.loads((TUNING/'COMPLETE.json').read_text())
    assert completed['checks_passed']
    imp=pd.read_csv(IMPUTE/'cv_summary.csv')
    nested=pd.read_csv(TUNING/'nested_cv_summary.csv').set_index('model')
    before=pd.read_csv(TUNING/'before_after.csv')
    final=json.loads((TUNING/'final_parameters.json').read_text())
    freq=pd.read_csv(TUNING/'selection_frequency.csv')
    scores=pd.read_csv(TUNING/'search_scores.csv')
    assert len(scores)==225 and len(config['input_features'])==116
    pdf_path=OUT/'2026-11-17_결측처리_모델학습_하이퍼파라미터_분석정리.pdf'
    with PdfPages(pdf_path,metadata={'Title':'2026-11-17 결측 처리·모델 학습·하이퍼파라미터 탐색 정리',
                                   'Author':'분석 기록','Subject':'Train-only nested CV results'}) as pdf:
        r=Report(pdf)
        r.new('오늘의 분석 설정과 결과 요약','결측 처리 2가지 × 모델 3가지 비교 후, 모델별 하이퍼파라미터 탐색')
        r.h('핵심 결론')
        r.p('중앙값+결측 지표와 MICE+결측 지표를 동일한 5-fold에서 비교했습니다. 이후 Logistic·Random forest는 중앙값, XGBoost는 MICE로 고정해 소규모 후보 탐색을 수행했습니다. 조정 후 평균 ROC-AUC는 각각 0.7797, 0.8061, 0.8224로, 개선 폭은 작고 XGBoost가 가장 높았습니다.')
        r.table(['항목','오늘 적용한 조건'],[
            ['예측 목표','첫 ICU 입실+24시간 정보를 이용해 기존 mortality_28d 예측'],
            ['분석 대상','성인 첫 ICU 입실, 24시간 관찰 및 사망 라벨 확정 환자 30,112명'],
            ['개발 / Test','Train 25,595명(기존 train+validation 병합) / 기존 Test 4,517명'],
            ['변수 제외','ALT·AST·총빌리루빈의 mean/min/max/std/count/missing 총 18개'],
            ['입력 수','기존 134개 → 116개; std 결측 지표 19개 추가 후 135개'],
            ['환자 추가 제외','결측 개수만을 이유로 환자를 추가 제외하지 않음'],
            ['공통 기준','환자 단위 층화 분할, seed=42, 클래스 가중치 없음, 임계값=0.5']
        ],[.2,.8],10.2)
        r.h('오늘 진행한 순서')
        r.p('① 데이터·제외 변수 고정 → ② 결측 지표 정의 → ③ 두 대체 방법으로 세 모델 5-fold 비교 → ④ 모델별 대체 방법 고정 → ⑤ 바깥 5-fold / 내부 3-fold 탐색 → ⑥ 전체 Train 재학습 및 저장 검증')
        r.p('Test 주의: 기존 Test는 이전 연구 단계에서 이미 평가했습니다. 오늘의 결측 처리 비교와 이번 튜닝에서는 읽거나 재평가하지 않았습니다. 따라서 최종 튜닝 모델의 새로운 Test 성능은 아직 없습니다.',10.5,color=GRAY)

        r.new('결측 정의와 두 가지 처리 방법','대체 이전의 결측 정보를 보존하고, 모든 전처리는 해당 훈련 부분에서만 적합')
        r.h('결측 지시변수의 정의')
        r.p('검사 결측은 첫 ICU 입실 후 [0,24시간)에 유효한 정제 검사값이 없는 경우입니다. mean=NaN, count=0, missing=1의 일치를 확인했습니다. 기존 19개 missing 지표를 유지하고, std에는 별도 지표 19개를 추가했습니다. std는 측정값이 1개일 때도 결측이므로 검사 자체의 미측정과 구분합니다.')
        r.table(['방법','구현 및 적용'],[
            ['중앙값+지표','각 훈련 폴드의 중앙값으로 수치 NaN 대체. 검증 폴드에는 같은 중앙값 적용. 지표는 원래 NaN에서 미리 정의.'],
            ['MICE+지표','BayesianRidge 기반 IterativeImputer, posterior sampling. 5개 대체 자료로 각각 학습한 예측확률의 평균 사용.'],
            ['MICE 상세','10회 순환, 이웃 특징 20개, seed=42~46. 훈련 스케일로 변환 후 대체·원래 단위 복원. 비음수 하한 적용(AnionGap 값 제외).'],
            ['누수 방지','검증 환자는 대체기 학습에 사용하지 않고, 사망 라벨은 대체 입력에 넣지 않음. 관측된 원래 값은 유지.']
        ],[.22,.78],10)
        r.h('동일 5-fold 비교 결과: 평균 ROC-AUC / AP / Brier')
        rows=[]
        for m in NAMES:
            for method,label in [('median_indicator','중앙값'),('mice_indicator','MICE')]:
                a=imp.loc[imp.model.eq(m)&imp.method.eq(method)].iloc[0]
                rows.append([NAMES[m],label,f'{a.roc_auc_mean:.4f}',f'{a.average_precision_mean:.4f}',f'{a.brier_score_mean:.4f}'])
        r.table(['모델','처리+지표','ROC-AUC ↑','AP ↑','Brier ↓'],rows,[.25,.18,.19,.19,.19],10)
        r.p('차이가 작았지만 Logistic·RF는 중앙값, XGBoost는 MICE의 점추정치가 조금 좋아 후속 탐색에서 이 조합을 고정했습니다. MICE는 5개 모델 평균이므로 대체 효과와 앙상블 효과가 함께 포함됩니다. MICE가 보편적으로 더 우수하다는 결론은 아닙니다.',10.3,color=GRAY)

        r.new('모델별 학습 조건과 선택 이유','결측 처리 비교 단계는 고정 기준 모델, 그 다음 단계에서 하이퍼파라미터를 탐색')
        r.table(['모델','기준 학습 조건','전처리'],[
            ['Logistic','L2, C=1, lbfgs, max_iter=5000','중앙값+지표 또는 MICE+지표. 이진 특징을 제외한 수치 입력 표준화.'],
            ['Random forest','트리500, 깊이15, min_samples_leaf=5, max_features=sqrt','두 대체 방법을 각각 적용. 모델 입력 표준화 없음.'],
            ['XGBoost','트리1000, 깊이3, 학습률.03, min_child_weight=5, reg_lambda=1','두 대체 방법을 각각 적용. 모델 입력 표준화 및 조기 종료 없음.']
        ],[.19,.4,.41],10)
        r.h('왜 이 세 모델인가?')
        r.p('Logistic은 로그오즈에 대한 선형·가산 관계의 기준 모델입니다. RF는 부트스트랩과 특징 무작위화를 이용해 비선형·상호작용을 학습합니다. XGBoost는 이전 오차를 보완하는 boosting 모델로 유연하지만 과적합과 복잡도를 관리해야 합니다.')
        r.h('왜 L2를 사용했는가?')
        r.p('평균·최솟값·최댓값 및 SBP·DBP·MAP처럼 연관된 입력이 있습니다. L2는 계수를 축소해 과도한 계수 크기와 불안정성을 줄이려는 선택입니다. 변수를 일반적으로 정확한 0으로 만들지는 않습니다. L1은 일부 계수를 0으로 만들 수 있지만, 이번 실행에서는 L1 대 L2를 비교하지 않았습니다.')
        r.p('C는 규제 강도의 역방향 조절값입니다. C가 작을수록 규제가 강합니다. C=1은 기준값이었으며, 탐색 후 C=0.1을 선택했습니다. 이를 L2가 L1보다 우수하다는 증거로 해석하지 않습니다.')
        r.h('공통 고정 조건')
        r.p('random_state=42, n_jobs=4, 클래스 가중치·오버샘플링 없음, 분류 임계값0.5. XGBoost subsample=0.8, colsample_bytree=0.8, reg_alpha=0, scale_pos_weight=1, tree_method=hist, objective=binary:logistic를 유지했습니다.')
        r.p('MICE 계산을 위한 내부 스케일링과 Logistic 모델의 입력 표준화는 별개 단계입니다. RF·XGBoost는 대체 후 원래 단위로 복원한 값을 사용했습니다.',10.3,color=GRAY)
        r.p('XGBoost의 min_child_weight는 자식 노드 최소 Hessian 가중치 조건이며 RF의 min_samples_leaf(표본 수)와 같은 단위가 아닙니다.',10.3,color=GRAY)

        r.new('하이퍼파라미터 탐색 과정','평가용 바깥 5-fold와 선택용 내부 3-fold를 분리한 nested CV')
        r.table(['단계','수행 내용'],[
            ['1. 바깥 분할','이전과 동일한 StratifiedKFold(5, shuffle=True, seed=42). 바깥 검증 환자는 후보 선택에 사용하지 않음.'],
            ['2. 내부 분할','각 바깥 훈련에서 StratifiedKFold(3, shuffle=True, seed=100+바깥 폴드번호).'],
            ['3. 후보별 학습','내부 훈련에서만 대체·표준화 적합 → 후보 모델 학습 → 내부 검증 점수 계산.'],
            ['4. 설정 선택','내부 3개 ROC-AUC 평균이 가장 높은 후보 선택. 동률은 AP로 구분.'],
            ['5. 바깥 평가','선택 후보를 바깥 훈련 전체에 재학습하여 미사용 바깥 검증에 예측. MICE는 5회 평균.'],
            ['6. 반복·집계','5개 바깥 폴드에서 반복. 조정 전후는 같은 환자·같은 바깥 폴드의 성능으로 비교.'],
            ['7. 최종 재학습','15개 내부 검증 점수의 후보별 평균으로 최종 설정 선택 → 전체 Train으로 재학습. Test 평가 없음.']
        ],[.2,.8],10.4)
        r.h('계산량과 MICE의 적용 차이')
        r.p('후보는 Logistic 5개 + RF 4개 + XGBoost 6개 = 15개입니다. 15개 × 내부3-fold × 바깥5-fold = 225회 후보 학습을 수행했습니다. 여기에 선택 모델의 바깥 재학습, MICE 5회 평가, 최종 Train 재학습이 추가됩니다.')
        r.p('계산 효율을 위해 XGBoost 내부 탐색은 MICE seed42의 1회 대체를 사용했습니다. 바깥 평가와 최종 모델은 seed42~46의 5회 평균입니다. 따라서 후보 순위가 5회 앙상블 전체를 직접 최적화한 결과는 아닙니다.',10.5,color=GRAY)
        r.p('결측 처리 방법 자체는 이전 개발 데이터 탐색으로 고정했습니다. 이번 nested CV는 고정된 전처리 아래 하이퍼파라미터 선택을 평가하며, 변수·전처리 선택까지 완전히 독립 검증한 결과는 아닙니다.',10.5,color=GRAY)

        r.new('실제 탐색한 후보 전체','문서 후보 번호는 1부터 표시하며, 원본 파일 candidate 번호는 0부터 시작')
        r.h('Logistic: L2 규제 강도 5개')
        r.p('C = 0.01, 0.1, 1, 10, 100. solver=lbfgs와 max_iter=5000은 고정했습니다. C=1이 조정 전 기준 모델입니다.',10.4)
        r.h('Random forest: 4개 조합')
        rows=[]
        for i,p in enumerate(config['candidates']['random_forest'],1):
            rows.append([str(i),str(p['n_estimators']),'제한 없음' if p['max_depth'] is None else str(p['max_depth']),
                         str(p['min_samples_leaf']),str(p['max_features'])])
        r.table(['후보','트리 수','최대 깊이','최소 leaf 표본','분기 후보 특징'],rows,[.1,.16,.22,.24,.28],10)
        r.h('XGBoost: 6개 조합')
        rows=[]
        for i,p in enumerate(config['candidates']['xgboost'],1):
            rows.append([str(i),str(p['n_estimators']),str(p['max_depth']),f"{p['learning_rate']:.2f}",
                         str(p['min_child_weight']),str(p['reg_lambda'])])
        r.table(['후보','트리 수','깊이','학습률','자식 가중치','L2 λ'],rows,[.09,.19,.12,.18,.24,.18],10)
        r.h('탐색 범위의 의미')
        r.p('RF·XGBoost는 표의 명시적 조합만 비교했습니다. 각 열 값의 모든 조합을 탐색한 완전 격자가 아닙니다. 기준 설정은 두 표의 후보1에 포함됩니다.')
        r.p('“최적의 하이퍼파라미터”는 가능한 모든 설정의 전역 최적이라는 의미가 아닙니다. 정확한 표현은 “사전에 정한 후보 범위에서 교차검증으로 선택한 설정”입니다.',11,color=TEAL)

        r.new('조정 전후 성능과 해석','조정 후 수치는 바깥 폴드별 선택 절차의 평가 성능이며 Test 점수가 아님')
        rows=[]
        for a in before.itertuples():
            rows.append([NAMES[a.model],f'{a.baseline_auc:.4f}',f'{a.tuned_auc:.4f}',f'{a.auc_difference:+.4f}',f'{a.baseline_ap:.4f} → {a.tuned_ap:.4f}'])
        r.table(['모델','전 AUC','후 AUC','차이','AP 전 → 후'],rows,[.25,.15,.15,.14,.31],10.3)
        r.h('조정 후 바깥 5-fold 평균 ± 표준편차')
        rows=[]
        for m in NAMES:
            a=nested.loc[m]
            rows.append([NAMES[m],f'{a.roc_auc_mean:.4f} ± {a.roc_auc_std:.4f}',
                         f'{a.average_precision_mean:.4f} ± {a.average_precision_std:.4f}',
                         f'{a.brier_score_mean:.4f} ± {a.brier_score_std:.4f}'])
        r.table(['모델','ROC-AUC ↑','AP ↑','Brier ↓'],rows,[.23,.26,.26,.25],9.8)
        r.p('세 모델 모두 ROC-AUC가 소폭 상승했습니다. Logistic의 AP는 거의 같았고, RF와 XGBoost의 AP·Brier는 개선됐습니다. XGBoost가 가장 높은 점수를 보였지만, 개선 차이의 통계적 유의성이나 임상적 효용을 입증한 것은 아닙니다.',10.3)
        r.image(TUNING/'before_after.png')
        r.p('SD는 5개 폴드 지표의 표본 표준편차이며 95% 신뢰구간이 아닙니다. 폴드 간 훈련 집합이 겹치므로 독립 반복 실험으로 해석하지 않습니다. AP는 average precision이며 사다리꼴 PR-AUC와 별개입니다.',9.8,color=GRAY)

        r.new('최종 선택 설정과 저장 모델','바깥 성능으로 고르지 않고 내부 후보 점수 평균으로 최종 설정 선택')
        r.table(['모델','전체 Train 재학습에 사용한 설정'],[
            ['Logistic','L2, C=0.1, lbfgs, max_iter=5000. 중앙값+지표, 수치 입력 표준화.'],
            ['Random forest','n_estimators=500, max_depth=None, min_samples_leaf=5, max_features=0.5. 중앙값+지표.'],
            ['XGBoost','n_estimators=1000, max_depth=4, learning_rate=0.03, min_child_weight=10, reg_lambda=10. MICE+지표의 5개 모델 평균.']
        ],[.23,.77],10.8)
        r.h('바깥 폴드별 선택의 일관성')
        r.table(['모델','선택 후보','선택 횟수'],[
            ['Logistic','C=0.1','5 / 5'],
            ['Random forest','위 최종 설정','5 / 5'],
            ['XGBoost','문서 후보6: 깊이4, 자식가중치10, L2=10','3 / 5'],
            ['XGBoost','문서 후보3: 깊이4, 자식가중치5, L2=1','2 / 5']
        ],[.23,.57,.2],10)
        r.p('최종 설정은 후보별 15개 내부 검증 점수 평균을 기준으로 선택했습니다. 이 15개 검증은 중복 환자를 포함하므로 15개 독립 검증이 아닙니다. 조정 후 nested CV 성능은 폴드마다 다른 설정을 선택할 수 있는 절차의 성능이며, 최종 단일 설정의 독립 성능을 보장하지 않습니다.')
        r.h('저장 및 재현 확인')
        r.p('Logistic·RF는 모델/대체기/표준화기를 묶은 bundle 각 1개, XGBoost는 MICE 대체기·스케일·모델을 묶은 bundle 5개를 저장했습니다. 예측 시 XGBoost 5개 확률을 평균해야 합니다.')
        r.p('입력 SHA256 보존, 환자·폴드 배정 일치, 관측값 보존, 대체 후 유한값, LR 수렴, OOF 누락 부재를 확인했습니다. 저장 bundle을 다시 로드한 30행 예측도 재현됐습니다(수치 허용오차 1e-12).')
        r.p('MICE는 posterior 난수 상태를 사용합니다. 같은 배치를 재현하려면 저장 상태에서 bundle을 다시 로드해야 하며, 배치 구성이 달라지면 개별 확률이 달라질 수 있습니다.',10.2,color=GRAY)

        r.new('해석 범위, 후속 과제와 근거 파일','완료된 분석과 아직 수행하지 않은 검증을 구분')
        r.h('해석할 때 주의할 점')
        r.p('• 이번 탐색은 작은 명시적 후보 집합의 비교입니다. 더 넓은 범위나 다른 대체기·모델로 결과가 바뀔 수 있습니다.\n• 결측 처리 선택은 이전 개발 데이터 결과를 이용했습니다. 이를 포함한 전체 연구 절차의 외부 검증은 아닙니다.\n• MICE는 10회 순환을 고정했으며 체인 수렴을 입증하지 않았습니다. mean/min/max의 집계 일관성을 강제하지 않았습니다.\n• 라벨 미확정 환자 제외와 24시간 관찰 조건에 따른 선택 편향, charttime 기준상 늦은 storetime 허용 문제가 남습니다.\n• Test는 이전 단계에서 이미 확인했습니다. 이번 최종 모델의 미사용 독립 Test 결과라고 보고할 수 없습니다.',10.5)
        r.h('다음에 검토할 사항')
        r.p('KNN 대체, L1 대 L2, ICU 유형 추가, 다중 결측 환자 제외 정책의 민감도 분석은 이번 튜닝에 포함하지 않았습니다. 임계값·확률 보정은 Train 내부 예측으로 검토하고, 실제 적용 가능성은 새로운 시점·병원 등 미사용 자료에서 확인해야 합니다.',10.5)
        r.h('근거 파일과 재현 코드')
        r.table(['파일 / 위치','내용'],[
            ['three_models_imputation_5fold/cv_summary.csv','두 결측 처리 × 세 모델의 기준 성능'],
            ['three_models_nested_tuning/experiment.json','입력 해시·전체 후보·분할·버전'],
            ['search_scores.csv / candidate_rankings.csv','225회 내부 점수 / 최종 후보 순위'],
            ['selected_by_fold.json / selection_frequency.csv','폴드별 선택 설정 및 빈도'],
            ['nested_cv_summary.csv / before_after.csv','조정 후 평균±SD / 조정 전후 비교'],
            ['final_parameters.json / 모델별 model_bundle_*.joblib','최종 선택 설정 / 저장 모델·전처리'],
            ['scripts/tune_three_models_nested_cv.py','탐색·바깥 평가·최종 Train 재학습'],
            ['scripts/predict_tuned_models.py','저장 모델로 새 입력에 예측']
        ],[.58,.42],8.8)
        r.p('결과 파일 기본 위치: results/three_models_nested_tuning/. scikit-learn 1.9.0, XGBoost 3.4.1 사용. 저장된 CSV·JSON에서 작성했으며 PDF 생성 중 모델 재학습·Test 평가는 하지 않았습니다.',9.5,color=GRAY)
        r.finish()
        (OUT/'분석정리_원문.md').write_text('# 2026-11-17 분석 설정·결측 처리·탐색 결과\n'+'\n'.join(r.markdown),encoding='utf-8')
    reader=PdfReader(str(pdf_path))
    assert len(reader.pages)==8
    extracted=[page.extract_text() for page in reader.pages]
    assert all(len(t)>250 for t in extracted)
    assert '0.8224' in extracted[5] and '225' in extracted[3]
    assert '결측' in extracted[1] and 'C=0.1' in extracted[6]
    manifest=dict(pdf=str(pdf_path),pages=8,searchable_korean_text=True,
        embedded_font=str(FONT_PATH),margin_checks=True,models_retrained=False,test_evaluated=False,
        sources=[str(IMPUTE/'cv_summary.csv'),str(TUNING/'experiment.json'),str(TUNING/'before_after.csv'),
                 str(TUNING/'nested_cv_summary.csv'),str(TUNING/'final_parameters.json')])
    (OUT/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PDF:',pdf_path)
    print('Pages:',len(reader.pages),'| Korean text extraction: PASS | margin checks: PASS')


if __name__=='__main__':main()
