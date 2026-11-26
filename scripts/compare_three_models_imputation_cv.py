"""Shared 5-fold median/MICE comparison; train-only, fixed model settings."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'.codex_deps/xgboost').is_dir():
    sys.path.insert(0, str(ROOT/'.codex_deps'))
import copy
import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from train_three_models_cv import metrics, LR_SETTINGS, RF_SETTINGS, XGB_SETTINGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT/'data/processed/error_cleaned/splits'
OUT = ROOT/'results/three_models_imputation_5fold'
MODELS = ['logistic_regression', 'random_forest', 'xgboost']
METHODS = ['median_indicator', 'mice_indicator']
XGB_FIXED = {**XGB_SETTINGS, 'n_estimators':1000}


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def fit_model(name, train, y, valid, columns):
    scaler = None
    if name == 'logistic_regression':
        # Preserve binary indicator units; standardize the remaining original numeric columns.
        numeric_idx = [i for i,c in enumerate(columns) if c != 'gender' and not c.endswith('_missing')]
        scaler = StandardScaler().fit(train[:,numeric_idx])
        train, valid = train.copy(), valid.copy()
        train[:,numeric_idx] = scaler.transform(train[:,numeric_idx])
        valid[:,numeric_idx] = scaler.transform(valid[:,numeric_idx])
        model = LogisticRegression(**LR_SETTINGS)
    elif name == 'random_forest':
        model = RandomForestClassifier(**RF_SETTINGS)
    else:
        model = xgb.XGBClassifier(**XGB_FIXED)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
        warnings.simplefilter('always')
        model.fit(train,y)
    bad = [str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if bad:
        raise RuntimeError('Convergence failure: '+'; '.join(bad))
    for w in caught:
        print(f'WARNING {w.category.__name__}: {w.message}',flush=True)
    p = model.predict_proba(valid)[:,1]
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    # In-memory serialization verifies a save/load does not change the estimator predictions.
    import io
    buffer = io.BytesIO()
    joblib.dump(model,buffer)
    buffer.seek(0)
    np.testing.assert_allclose(joblib.load(buffer).predict_proba(valid)[:,1],p)
    detail = dict(iterations=int(model.n_iter_[0])) if name == 'logistic_regression' else {}
    return p,detail


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving existing results: {OUT}')
    all_columns = json.loads((SOURCE/'model_feature_columns.json').read_text())
    excluded = [c for c in all_columns if c.startswith(('lab_ALT_','lab_AST_','lab_BilirubinTotal_'))]
    columns = [c for c in all_columns if c not in excluded]
    assert len(excluded)==18 and len(columns)==116
    frames,labels,identities,hashes = [],[],[],{}
    baseline = json.loads((ROOT/'results/three_models_5fold_no_liver_labs/experiment.json').read_text(encoding='utf-8'))
    for split in ['train','validation']:
        for file in ['X_features.csv','y_target.csv','row_ids.csv']:
            key=f'{split}/{file}'
            hashes[key]=hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()
            assert hashes[key]==baseline['input_sha256'][key]
        frame=pd.read_csv(SOURCE/split/'X_features.csv')
        assert frame.columns.tolist()==all_columns
        frame=frame[columns].copy()
        assert frame.gender.isin(['F','M']).all()
        frame['gender']=frame.gender.map({'F':0,'M':1})
        y=pd.read_csv(SOURCE/split/'y_target.csv').mortality_28d
        ids=pd.read_csv(SOURCE/split/'row_ids.csv')
        assert len(frame)==len(y)==len(ids) and y.isin([0,1]).all()
        assert not np.isinf(frame.to_numpy()).any()
        for c in [c for c in columns if c.endswith('_mean')]:
            prefix=c[:-5]
            mask=frame[c].isna()
            assert mask.equals(frame[prefix+'_count'].eq(0))
            assert frame[prefix+'_missing'].eq(mask.astype(int)).all()
            assert all(mask.equals(frame[prefix+'_'+s].isna()) for s in ['min','max'])
        frames.append(frame); labels.append(y); identities.append(ids)
    data=pd.concat(frames,ignore_index=True)
    y=pd.concat(labels,ignore_index=True)
    ids=pd.concat(identities,ignore_index=True)
    assert len(data)==25595 and ids.subject_id.is_unique and ids.stay_id.is_unique
    assert ids[['subject_id','stay_id']].notna().all().all()
    std_columns=[c for c in columns if c.endswith('_std')]
    assert len(std_columns)==19
    names=columns+[c+'__was_missing' for c in std_columns]
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(data,y))
    fold_ids=np.full(len(y),-1)
    for fold,(_,va) in enumerate(folds,1):
        assert (fold_ids[va]==-1).all()
        fold_ids[va]=fold
    prior=pd.read_csv(ROOT/'results/three_models_5fold_no_liver_labs/train_folds.csv')
    np.testing.assert_array_equal(prior.subject_id,ids.subject_id)
    np.testing.assert_array_equal(prior.fold,fold_ids)
    OUT.mkdir(parents=True)
    ids.assign(mortality_28d=y,fold=fold_ids).to_csv(OUT/'train_folds.csv',index=False)
    settings=dict(logistic_regression=LR_SETTINGS,random_forest=RF_SETTINGS,xgboost=XGB_FIXED)
    metadata=dict(input_sha256=hashes,excluded_features=excluded,input_features=columns,
        transformed_features=names,train_n=len(y),folds=5,test_read=False,test_evaluated=False,
        versions=dict(sklearn=sklearn.__version__,xgboost=xgb.__version__),settings=settings,
        indicator_policy='original 19 missing flags plus 19 per-std indicators, defined before imputation',
        mice=dict(draws=5,seeds=list(range(42,47)),max_iter=10,n_nearest_features=20,
            sample_posterior=True,estimator='BayesianRidge',skip_complete=True,
            scale='train-fold StandardScaler, inverse after imputation',
            bounds='nonnegative continuous imputations except AnionGap mean/min/max',
            pooling='mean probabilities across five fitted models'),
        hyperparameter_search=False,early_stopping=False,threshold=.5,
        limitations=['Exploratory comparison of previously studied development patients.',
            'MICE uses a five-model ensemble; median uses one model per fold.',
            'Aggregate min/mean/max consistency is not enforced.',
            'No MICE convergence claim: posterior chains run for a fixed 10 iterations.',
            'No independent test evaluation in this experiment.'])
    write_json(OUT/'experiment.json',metadata)
    oof={(method,model):np.full(len(y),np.nan) for method in METHODS for model in MODELS}
    # Per-draw OOF permits uncertainty / stability inspection without refitting.
    mice_draw_oof={model:np.full((5,len(y)),np.nan) for model in MODELS}
    rows,draw_rows,timings=[],[],[]
    started=time.perf_counter()
    def augment(filled,original):
        values=np.column_stack([filled,original[std_columns].isna().to_numpy(dtype=np.float64)])
        assert values.shape[1]==135 and np.isfinite(values).all()
        observed=original.notna().to_numpy()
        np.testing.assert_allclose(values[:,:116][observed],original.to_numpy()[observed],rtol=1e-10,atol=1e-10)
        return values
    for fold,(tr,va) in enumerate(folds,1):
        raw_train,raw_valid=data.iloc[tr],data.iloc[va]
        assert not set(ids.iloc[tr].subject_id)&set(ids.iloc[va].subject_id)
        assert not raw_train.isna().all().any()
        folder=OUT/f'fold_{fold}'
        folder.mkdir()
        scaler=StandardScaler().fit(raw_train)
        scaled_train,scaled_valid=scaler.transform(raw_train),scaler.transform(raw_valid)
        minimum=np.full(len(columns),-np.inf)
        for j,c in enumerate(columns):
            if c.endswith(('_mean','_min','_max','_std')) and ('AnionGap' not in c or c.endswith('_std')):
                minimum[j]=-scaler.mean_[j]/scaler.scale_[j]
        joblib.dump(scaler,folder/'mice_scaler.joblib')
        for method in METHODS:
            per_model={model:[] for model in MODELS}
            for draw in range(5 if method=='mice_indicator' else 1):
                t=time.perf_counter()
                print(f'IMPUTE fold={fold}/5 method={method} draw={draw+1}',flush=True)
                if method=='median_indicator':
                    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
                    a=imputer.fit_transform(raw_train)
                    b=imputer.transform(raw_valid)
                    joblib.dump(imputer,folder/'median_imputer.joblib')
                else:
                    imputer=IterativeImputer(max_iter=10,n_nearest_features=20,initial_strategy='median',
                        sample_posterior=True,random_state=42+draw,skip_complete=True,
                        min_value=minimum,keep_empty_features=True)
                    with threadpool_limits(limits=1):
                        a=scaler.inverse_transform(imputer.fit_transform(scaled_train))
                        # Save the exact pre-validation random state, then verify replay.
                        joblib.dump(imputer,folder/f'mice_imputer_{draw+1}.joblib',compress=3)
                        replay_imputer=copy.deepcopy(imputer)
                        b=scaler.inverse_transform(imputer.transform(scaled_valid))
                        replay=scaler.inverse_transform(replay_imputer.transform(scaled_valid))
                        np.testing.assert_allclose(replay,b,rtol=1e-10,atol=1e-10)
                xt,xv=augment(a,raw_train),augment(b,raw_valid)
                timings.append(dict(fold=fold,method=method,draw=draw+1,step='imputation',seconds=time.perf_counter()-t))
                for model in MODELS:
                    t=time.perf_counter()
                    p,detail=fit_model(model,xt,y.iloc[tr],xv,columns)
                    per_model[model].append(p)
                    if method=='mice_indicator':
                        mice_draw_oof[model][draw,va]=p
                    result=metrics(y.iloc[va],p)
                    draw_rows.append(dict(fold=fold,method=method,model=model,draw=draw+1,**detail,**result))
                    timings.append(dict(fold=fold,method=method,draw=draw+1,step=model,seconds=time.perf_counter()-t))
                    print(f'FIT fold={fold} {method} draw={draw+1} {model}: AUC={result["roc_auc"]:.4f} AP={result["average_precision"]:.4f}',flush=True)
                pd.DataFrame(draw_rows).to_csv(OUT/'draw_metrics.csv',index=False)
                pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
            for model in MODELS:
                p=np.mean(per_model[model],axis=0)
                oof[(method,model)][va]=p
                rows.append(dict(method=method,model=model,fold=fold,train_n=len(tr),validation_n=len(va),**metrics(y.iloc[va],p)))
                # Per-fold predictions persist progress and provide an audit trail.
                ids.iloc[va].assign(mortality_28d=y.iloc[va],fold=fold,probability=p).to_csv(
                    folder/f'{method}_{model}_predictions.csv',index=False)
            pd.DataFrame(rows).to_csv(OUT/'cv_fold_metrics.csv',index=False)
            print(f'COMPLETE fold={fold} method={method}',flush=True)
    metric_names=['roc_auc','average_precision','pr_auc_trapezoidal','brier_score','log_loss','sensitivity','specificity','precision','accuracy']
    cv=pd.DataFrame(rows)
    summary=cv.groupby(['method','model'])[metric_names].agg(['mean','std'])
    summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary.reset_index().to_csv(OUT/'cv_summary.csv',index=False)
    pooled=[]
    for (method,model),p in oof.items():
        assert np.isfinite(p).all()
        pooled.append(dict(method=method,model=model,**metrics(y,p)))
        ids.assign(mortality_28d=y,fold=fold_ids,probability=p,prediction_at_0_5=(p>=.5).astype(int)).to_csv(
            OUT/f'oof_{method}_{model}.csv',index=False)
    pd.DataFrame(pooled).to_csv(OUT/'oof_metrics.csv',index=False)
    for model,values in mice_draw_oof.items():
        assert np.isfinite(values).all()
        frame=ids.assign(mortality_28d=y,fold=fold_ids)
        for draw in range(5):
            frame[f'probability_draw_{draw+1}']=values[draw]
        frame.to_csv(OUT/f'mice_draw_predictions_{model}.csv',index=False)
    # Fold-paired differences are descriptive, not independent repeated experiments.
    differences=[]
    for model in MODELS:
        for metric in metric_names:
            a=cv.query('method == "median_indicator" and model == @model').sort_values('fold')[metric].to_numpy()
            b=cv.query('method == "mice_indicator" and model == @model').sort_values('fold')[metric].to_numpy()
            delta=b-a
            differences.append(dict(model=model,metric=metric,mice_minus_median_mean=float(delta.mean()),
                                    fold_difference_sd=float(delta.std(ddof=1))))
    pd.DataFrame(differences).to_csv(OUT/'paired_fold_differences.csv',index=False)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    display={'logistic_regression':'Logistic','random_forest':'Random forest','xgboost':'XGBoost'}
    for ax,metric,title in zip(axes,['roc_auc','average_precision','brier_score'],['ROC-AUC ↑','Average precision ↑','Brier score ↓']):
        for j,method in enumerate(METHODS):
            values=[summary.loc[(method,m),metric+'_mean'] for m in MODELS]
            stds=[summary.loc[(method,m),metric+'_std'] for m in MODELS]
            positions=np.arange(3)+(j-.5)*.34
            ax.bar(positions,values,width=.32,yerr=stds,capsize=3,
                   label=['중앙값 + 지표','MICE(5회) + 지표'][j],color=['#187f97','#b87545'][j])
            for x,v in zip(positions,values):
                ax.annotate(f'{v:.4f}',(x,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(3),[display[m] for m in MODELS])
        ax.set_title(title)
        ax.set_ylim(0,max(summary[metric+'_mean'])*1.2)
        ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.suptitle('동일 5-fold: 결측 처리 2가지 × 모델 3가지')
    fig.text(.02,.015,'막대=폴드 평균, 오차선=폴드 간 SD. ALT·AST·총빌리루빈 제외. Test 미사용. MICE는 5개 모델 평균 예측.',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.85))
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    lines=['# 중앙값+지시변수 vs MICE+지시변수: 3모델 비교','',
        'Train 25,595명에서 기존과 같은 층화 5-fold를 사용했습니다. Test는 읽거나 평가하지 않았습니다. ALT·AST·총빌리루빈 관련 18개 특징을 제외했습니다.','',
        '| 모델 | 결측 처리 | ROC-AUC 평균±SD | AP 평균±SD | Brier 평균±SD | 민감도 @0.5 | 특이도 @0.5 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        for method in METHODS:
            r=summary.loc[(method,model)]
            lines.append(f'| {display[model]} | {method} | {r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f} | {r.average_precision_mean:.4f} ± {r.average_precision_std:.4f} | {r.brier_score_mean:.4f} ± {r.brier_score_std:.4f} | {r.sensitivity_mean:.2%} | {r.specificity_mean:.2%} |')
    lines += ['', '![비교 그림](comparison.png)', '', '## 모델별 차이: MICE − 중앙값', '',
              '| 모델 | ROC-AUC 차이 | AP 차이 | Brier 차이 |', '|---|---:|---:|---:|']
    for model in MODELS:
        a=summary.loc[('median_indicator',model)]
        b=summary.loc[('mice_indicator',model)]
        lines.append(f'| {display[model]} | {b.roc_auc_mean-a.roc_auc_mean:+.4f} | {b.average_precision_mean-a.average_precision_mean:+.4f} | {b.brier_score_mean-a.brier_score_mean:+.4f} |')
    lines += ['', '## 학습 조건 및 해석', '',
        '- 이번 실행은 결측 처리 비교를 위한 고정 기준 모델입니다. 하이퍼파라미터 탐색 및 최종 Test 평가는 수행하지 않았습니다. 모든 모델을 두 방법에서 새로 학습했습니다.',
        '- 기존 116개 입력 + std 결측 지표 19개 = 두 방법·세 모델 모두 135개. 기존 19개 missing 지표는 mean/min/max 결측을 나타내며 std 결측과 구별합니다. 지표는 대체 전 원래 NaN에서 정의했습니다.',
        '- 중앙값은 각 outer 훈련 폴드에서만 계산합니다. MICE는 같은 훈련 폴드에서만 스케일 및 조건부 모델을 적합하며 검증 환자를 대체기 학습이나 donor로 사용하지 않습니다. 사망 라벨은 대체기에 입력하지 않습니다.',
        '- MICE: BayesianRidge 기반 IterativeImputer, posterior sampling, 10회 순환, 이웃 특징 20개, seed 42~46의 5개 대체 데이터셋. 폴드 내 같은 대체 데이터를 세 모델에 공통 적용하고 모델별 5개 예측확률을 평균했습니다. 고정 10회 실행이며 수렴을 입증한 분석은 아닙니다.',
        '- MICE 연산 안정성을 위해 훈련 폴드 StandardScaler로 변환 후 대체하고 원래 단위로 복원했습니다. 대체 연속값의 비음수 하한을 적용(AnionGap mean/min/max 제외). 집계 min/mean/max 간 일관성은 강제하지 않았습니다.',
        '- Logistic: L2, C=1, lbfgs, max_iter=5000. 대체 후 연속·계수형 입력만 훈련 폴드에서 표준화하고 이진 성별/지표는 원 단위 유지. RF: 트리500, 깊이15, min_samples_leaf=5, max_features=sqrt. 클래스 가중치 없음.',
        '- XGB: 두 방법 모두 트리1000개 고정, 깊이3, learning_rate=.03, min_child_weight=5, subsample/colsample=.8, L2=1, 가중치1. 조기 종료 및 파라미터 선택을 하지 않아 outer 검증 라벨이 학습에 쓰이지 않았습니다. 이전 NaN 실험의 트리 수 선택 절차와 다르므로 직접 동등 비교하면 안 됩니다.',
        '- 중앙값은 폴드당 한 모델, MICE는 폴드당 다섯 모델이므로 차이에는 대체 방식과 앙상블 효과가 함께 포함됩니다. draw_metrics.csv와 mice_draw_predictions_*.csv로 개별 대체 결과도 확인할 수 있습니다. Rubin 규칙으로 계수를 합치는 추론 분석이 아닙니다.',
        '- SD는 5개 폴드 지표의 표본 표준편차이며 신뢰구간이 아닙니다. CV 훈련 집합이 겹치므로 5개의 독립 실험으로 해석하지 않습니다. AP는 average precision이며 사다리꼴 PR-AUC와 별도로 저장했습니다.',
        '- 검증: 이전과 동일한 fold 배정 및 입력 SHA256, 환자 중복/폴드 중첩 부재, 관측값 보존, 대체 후 유한값, MICE 검증 변환 RNG 재현, LR 수렴, 모델 메모리 직렬화 재현, OOF 환자별 1회 예측, 원본 불변성을 확인했습니다.',
        '- 폴드별 대체기와 MICE scaler를 저장했습니다. 모델별 폴드 추정기는 저장하지 않았으며 코드·설정·입력·시드로 재학습 가능합니다. 성능 비교용 OOF 예측이며 전체 Train 최종 모델을 새로 적합하지 않았습니다.',
        '- 기존 unknown 라벨 제외/24시간 관찰 조건의 선택 편향, late storetime 허용 정책 및 내부 무작위 검증 한계가 유지됩니다. 이미 탐색한 개발 데이터의 결과이며 외부 검증이 아닙니다.','',
        '재실행: `python scripts/compare_three_models_imputation_cv.py` (기존 결과가 있으면 덮어쓰지 않습니다).']
    (OUT/'training_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for key,digest in hashes.items():
        assert hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()==digest
    metadata.update(checks_passed=True,complete=True,elapsed_seconds=time.perf_counter()-started)
    write_json(OUT/'experiment.json',metadata)
    print(summary[['roc_auc_mean','average_precision_mean','brier_score_mean']].to_string(),flush=True)
    print('COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    main()
