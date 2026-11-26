"""Small reproducible nested search; no test access; fixed imputation per model."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'.codex_deps/xgboost').is_dir():
    sys.path.insert(0,str(ROOT/'.codex_deps'))
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
from sklearn.impute import SimpleImputer,IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from train_three_models_cv import metrics,LR_SETTINGS,RF_SETTINGS,XGB_SETTINGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=ROOT/'data/processed/error_cleaned/splits'
PRIOR=ROOT/'results/three_models_imputation_5fold'
OUT=ROOT/'results/three_models_nested_tuning'
MODELS=['logistic_regression','random_forest','xgboost']
CANDIDATES={
 'logistic_regression':[dict(C=c) for c in [.01,.1,1.,10.,100.]],
 'random_forest':[
    dict(n_estimators=500,max_depth=15,min_samples_leaf=5,max_features='sqrt'),
    dict(n_estimators=500,max_depth=10,min_samples_leaf=10,max_features='sqrt'),
    dict(n_estimators=500,max_depth=20,min_samples_leaf=2,max_features='sqrt'),
    dict(n_estimators=500,max_depth=None,min_samples_leaf=5,max_features=.5)],
 'xgboost':[
    dict(n_estimators=1000,max_depth=3,learning_rate=.03,min_child_weight=5,reg_lambda=1.),
    dict(n_estimators=1000,max_depth=2,learning_rate=.03,min_child_weight=5,reg_lambda=1.),
    dict(n_estimators=1000,max_depth=4,learning_rate=.03,min_child_weight=5,reg_lambda=1.),
    dict(n_estimators=600,max_depth=3,learning_rate=.05,min_child_weight=5,reg_lambda=1.),
    dict(n_estimators=1500,max_depth=3,learning_rate=.02,min_child_weight=10,reg_lambda=5.),
    dict(n_estimators=1000,max_depth=4,learning_rate=.03,min_child_weight=10,reg_lambda=10.)]
}

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare(a,b,columns,method,draw=0):
    std=[c for c in columns if c.endswith('_std')]
    scaler=None
    if method=='native':
        imputer=None
        ta,tb=a.to_numpy(),b.to_numpy()
    elif method=='median':
        imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        ta=imputer.fit_transform(a)
        tb=imputer.transform(b)
    else:
        scaler=StandardScaler().fit(a)
        sa,sb=scaler.transform(a),scaler.transform(b)
        lower=np.full(len(columns),-np.inf)
        for j,c in enumerate(columns):
            if c.endswith(('_mean','_min','_max','_std')) and ('AnionGap' not in c or c.endswith('_std')):
                lower[j]=-scaler.mean_[j]/scaler.scale_[j]
        imputer=IterativeImputer(max_iter=10,n_nearest_features=20,initial_strategy='median',
            sample_posterior=True,random_state=42+draw,skip_complete=True,min_value=lower,keep_empty_features=True)
        with threadpool_limits(limits=1):
            ta=scaler.inverse_transform(imputer.fit_transform(sa))
            # Persist this state for the next future inference batch.
            import copy
            inference_imputer=copy.deepcopy(imputer)
            tb=scaler.inverse_transform(imputer.transform(sb))
        imputer=inference_imputer
    for orig,filled in [(a,ta),(b,tb)]:
        observed=orig.notna().to_numpy()
        np.testing.assert_allclose(filled[observed],orig.to_numpy()[observed],rtol=1e-10,atol=1e-10)
    ta=np.column_stack([ta,a[std].isna().to_numpy(dtype=float)])
    tb=np.column_stack([tb,b[std].isna().to_numpy(dtype=float)])
    assert ta.shape[1]==tb.shape[1]==len(columns)+len(std)
    if method=='native':
        np.testing.assert_array_equal(np.isnan(ta[:,:len(columns)]),a.isna().to_numpy())
        np.testing.assert_array_equal(np.isnan(tb[:,:len(columns)]),b.isna().to_numpy())
    else:
        assert np.isfinite(ta).all() and np.isfinite(tb).all()
    return ta,tb,dict(imputer=imputer,mice_scaler=scaler,columns=columns,std_columns=std)


def fit(name,params,a,y,b,columns):
    scaler=None
    numeric=[i for i,c in enumerate(columns) if c!='gender' and not c.endswith('_missing')]
    if name=='logistic_regression':
        a,b=a.copy(),b.copy()
        scaler=StandardScaler().fit(a[:,numeric])
        a[:,numeric]=scaler.transform(a[:,numeric]); b[:,numeric]=scaler.transform(b[:,numeric])
        model=LogisticRegression(**{**LR_SETTINGS,**params})
    elif name=='random_forest':
        model=RandomForestClassifier(**{**RF_SETTINGS,**params})
    else:
        model=xgb.XGBClassifier(**{**XGB_SETTINGS,**params})
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=4):
        warnings.simplefilter('always')
        model.fit(a,y)
    failures=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    if failures:
        raise RuntimeError('LR convergence failure: '+'; '.join(failures))
    p=model.predict_proba(b)[:,1]
    assert np.isfinite(p).all()
    return p,dict(model=model,logistic_scaler=scaler,logistic_numeric_indices=numeric)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise FileExistsError('Completed results already exist; preserving them')
    previous=json.loads((PRIOR/'experiment.json').read_text(encoding='utf-8'))
    columns=previous['input_features']
    frames,targets,identities,hashes=[],[],[],{}
    for split in ['train','validation']:
        for filename in ['X_features.csv','y_target.csv','row_ids.csv']:
            key=f'{split}/{filename}'
            hashes[key]=hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()
            assert hashes[key]==previous['input_sha256'][key]
        x=pd.read_csv(SOURCE/split/'X_features.csv')[columns].copy()
        assert x.gender.isin(['F','M']).all()
        x['gender']=x.gender.map({'F':0,'M':1})
        frames.append(x)
        targets.append(pd.read_csv(SOURCE/split/'y_target.csv').mortality_28d)
        identities.append(pd.read_csv(SOURCE/split/'row_ids.csv'))
    data=pd.concat(frames,ignore_index=True); y=pd.concat(targets,ignore_index=True)
    ids=pd.concat(identities,ignore_index=True)
    assert len(data)==25595 and ids.subject_id.is_unique and ids.stay_id.is_unique
    assert y.isin([0,1]).all() and not np.isinf(data.to_numpy()).any()
    config=dict(input_sha256=hashes,input_features=columns,candidates=CANDIDATES,
        outer_folds=5,inner_folds=3,seed=42,selection_metric='roc_auc; AP tie-break',
        preprocessing=dict(logistic_regression='median+indicators+numeric scaling',random_forest='median+indicators',
            xgboost='MICE+indicators, one draw for inner search, five-draw ensemble for outer evaluation/final model'),
        test_read=False,test_evaluated=False,early_stopping=False,
        imputation_selection='fixed from prior exploratory CV; not reselected in nested loop',
        versions=dict(sklearn=sklearn.__version__,xgboost=xgb.__version__))
    config_path=OUT/'experiment.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8'))==config,'Resume configuration changed'
    else:
        write_json(config_path,config)
    start=time.perf_counter()
    results=pd.read_csv(OUT/'search_scores.csv').to_dict('records') if (OUT/'search_scores.csv').exists() else []
    done={(int(r['outer_fold']),int(r['inner_fold']),r['model'],int(r['candidate'])) for r in results}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(data,y))
    prior_folds=pd.read_csv(PRIOR/'train_folds.csv')
    np.testing.assert_array_equal(prior_folds.subject_id,ids.subject_id)
    fold_id=np.zeros(len(y),dtype=int)
    rows,selections=[],[]
    oof={m:np.full(len(y),np.nan) for m in MODELS}
    for outer,(tr,va) in enumerate(folds,1):
        fold_id[va]=outer
        assert (prior_folds.fold.iloc[va]==outer).all()
        folder=OUT/f'outer_{outer}'; folder.mkdir(exist_ok=True)
        print(f'OUTER {outer}/5 INNER SEARCH START',flush=True)
        for inner,(subtr,subva) in enumerate(StratifiedKFold(3,shuffle=True,random_state=100+outer).split(tr,y.iloc[tr]),1):
            itr,iva=tr[subtr],tr[subva]
            for method,models in [('median',['logistic_regression','random_forest']),('mice',['xgboost'])]:
                missing=[(m,cid,p) for m in models for cid,p in enumerate(CANDIDATES[m]) if (outer,inner,m,cid) not in done]
                if not missing:
                    continue
                print(f'INNER outer={outer} inner={inner}/3 preprocessing={method}',flush=True)
                a,b,_=prepare(data.iloc[itr],data.iloc[iva],columns,method)
                for model,cid,params in missing:
                    t=time.perf_counter()
                    p,_=fit(model,params,a,y.iloc[itr],b,columns)
                    result=metrics(y.iloc[iva],p)
                    results.append(dict(outer_fold=outer,inner_fold=inner,model=model,candidate=cid,
                        train_n=len(itr),validation_n=len(iva),seconds=time.perf_counter()-t,**result))
                    done.add((outer,inner,model,cid))
                    pd.DataFrame(results).to_csv(OUT/'search_scores.csv',index=False)
                    print(f'SEARCH outer={outer} inner={inner} {model} candidate={cid} AUC={result["roc_auc"]:.4f}',flush=True)
        search=pd.DataFrame(results)
        selected={}
        for model in MODELS:
            ranking=search.query('outer_fold==@outer and model==@model').groupby('candidate')[['roc_auc','average_precision']].mean()
            ranking=ranking.sort_values(['roc_auc','average_precision'],ascending=False)
            cid=int(ranking.index[0]); selected[model]=cid
            selections.append(dict(outer_fold=outer,model=model,candidate=cid,params=CANDIDATES[model][cid],
                inner_mean_auc=float(ranking.iloc[0].roc_auc)))
        write_json(folder/'selection.json',{m:dict(candidate=c,params=CANDIDATES[m][c]) for m,c in selected.items()})
        print(f'OUTER {outer} SELECTED '+json.dumps(selected),flush=True)
        for method,models in [('median',['logistic_regression','random_forest']),('mice',['xgboost'])]:
            draws=5 if method=='mice' else 1
            pending=[m for m in models if not (folder/f'{m}_predictions.csv').exists()]
            if pending:
                preds={m:[] for m in pending}
                for draw in range(draws):
                    print(f'OUTER EVALUATE {outer} {method} draw={draw+1}/{draws}',flush=True)
                    a,b,_=prepare(data.iloc[tr],data.iloc[va],columns,method,draw)
                    for model in pending:
                        p,_=fit(model,CANDIDATES[model][selected[model]],a,y.iloc[tr],b,columns)
                        preds[model].append(p)
                for model in pending:
                    p=np.mean(preds[model],axis=0)
                    ids.iloc[va].assign(mortality_28d=y.iloc[va],probability=p).to_csv(folder/f'{model}_predictions.csv',index=False)
            for model in models:
                saved=pd.read_csv(folder/f'{model}_predictions.csv')
                np.testing.assert_array_equal(saved.subject_id,ids.iloc[va].subject_id)
                p=saved.probability.to_numpy(); oof[model][va]=p
                result=metrics(y.iloc[va],p)
                rows.append(dict(model=model,outer_fold=outer,selected_candidate=selected[model],**result))
                print(f'OUTER RESULT {outer} {model} AUC={result["roc_auc"]:.4f} AP={result["average_precision"]:.4f}',flush=True)
        pd.DataFrame(rows).to_csv(OUT/'nested_fold_metrics.csv',index=False)
        write_json(OUT/'selected_by_fold.json',selections)
    summary=pd.DataFrame(rows).groupby('model')[['roc_auc','average_precision','brier_score','log_loss','sensitivity','specificity']].agg(['mean','std'])
    summary.columns=[f'{a}_{b}' for a,b in summary.columns]
    summary.reset_index().to_csv(OUT/'nested_cv_summary.csv',index=False)
    for model,p in oof.items():
        assert np.isfinite(p).all()
        ids.assign(mortality_28d=y,fold=fold_id,probability=p).to_csv(OUT/f'oof_{model}.csv',index=False)
    # Select deployment configuration using aggregate inner search evidence only, not outer scores.
    final={}
    search=pd.DataFrame(results)
    rankings=search.groupby(['model','candidate'])[['roc_auc','average_precision','brier_score']].agg(['mean','std'])
    rankings.columns=[f'{a}_{b}' for a,b in rankings.columns]
    rankings.reset_index().to_csv(OUT/'candidate_rankings.csv',index=False)
    for model in MODELS:
        r=rankings.loc[model].sort_values(['roc_auc_mean','average_precision_mean'],ascending=False)
        cid=int(r.index[0]); final[model]=dict(candidate=cid,params=CANDIDATES[model][cid])
    write_json(OUT/'final_parameters.json',final)
    pd.DataFrame(selections).groupby(['model','candidate']).size().rename('selected_folds').reset_index().to_csv(
        OUT/'selection_frequency.csv',index=False)
    print('FINAL PARAMETERS '+json.dumps(final),flush=True)
    for method,models in [('median',['logistic_regression','random_forest']),('mice',['xgboost'])]:
        for draw in range(5 if method=='mice' else 1):
            print(f'FULL TRAIN REFIT {method} draw={draw+1}',flush=True)
            # Dummy training-only probe is used solely for save/load validation, never selection.
            probe=data.iloc[:30]
            a,b,prep=prepare(data,probe,columns,method,draw)
            for model in models:
                folder=OUT/model;folder.mkdir(exist_ok=True)
                p,fitted=fit(model,final[model]['params'],a,y,b,columns)
                bundle={**prep,**fitted,'preprocessing':method,'parameters':final[model]['params'],
                        'feature_names':columns+[c+'__was_missing' for c in columns if c.endswith('_std')]}
                path=folder/f'model_bundle_{draw+1}.joblib'
                joblib.dump(bundle,path,compress=3)
                loaded=joblib.load(path)
                transformed=b.copy()
                if loaded['logistic_scaler'] is not None:
                    ix=loaded['logistic_numeric_indices']
                    transformed[:,ix]=loaded['logistic_scaler'].transform(transformed[:,ix])
                np.testing.assert_allclose(loaded['model'].predict_proba(transformed)[:,1],p)
    old=pd.read_csv(PRIOR/'cv_summary.csv').set_index(['method','model'])
    comparison=[]
    for model in MODELS:
        method='mice_indicator' if model=='xgboost' else 'median_indicator'
        b=old.loc[(method,model)]; t=summary.loc[model]
        comparison.append(dict(model=model,imputation=method,baseline_auc=b.roc_auc_mean,tuned_auc=t.roc_auc_mean,
            auc_difference=t.roc_auc_mean-b.roc_auc_mean,baseline_ap=b.average_precision_mean,tuned_ap=t.average_precision_mean,
            baseline_brier=b.brier_score_mean,tuned_brier=t.brier_score_mean))
    pd.DataFrame(comparison).to_csv(OUT/'before_after.csv',index=False)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric,title in zip(axes,['auc','ap'],['ROC-AUC','Average precision']):
        for j,stage in enumerate(['baseline','tuned']):
            values=[r[f'{stage}_{metric}'] for r in comparison]
            ax.bar(np.arange(3)+(j-.5)*.34,values,width=.32,label=stage)
            for i,v in enumerate(values):
                ax.text(i+(j-.5)*.34,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(3),['Logistic','Random forest','XGBoost'])
        ax.set_title(title);ax.set_ylim(0,max(r[f'tuned_{metric}'] for r in comparison)*1.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(OUT/'before_after.png',dpi=180);plt.close(fig)
    lines=['# 하이퍼파라미터 탐색: 5×3 nested CV','',
        'Train 25,595명, ALT·AST·총빌리루빈 제외. Logistic/RF 중앙값+지표, XGB MICE+지표. Test는 읽거나 평가하지 않았습니다.','',
        '| 모델 | 조정 전 AUC | 조정 후 nested AUC | 차이 | 조정 전 AP | 조정 후 AP |','|---|---:|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f"| {r['model']} | {r['baseline_auc']:.4f} | {r['tuned_auc']:.4f} | {r['auc_difference']:+.4f} | {r['baseline_ap']:.4f} | {r['tuned_ap']:.4f} |")
    lines+=['','![조정 전후](before_after.png)','','## 최종 전체 Train 모델 설정','']
    for model,params in final.items():
        lines += [f'- {model}: `{json.dumps(params["params"],ensure_ascii=False)}`']
    lines+=['','## 탐색 설계 및 한계','',
        '- 바깥 5-fold는 이전 비교와 동일하며 평가용입니다. 각 outer 훈련 안에서 층화 3-fold(seed=100+outer) 평균 ROC-AUC로 설정을 선택하고 AP를 동률 구분에 사용했습니다.',
        '- Logistic L2 C 5개(.01,.1,1,10,100), RF 4개, XGB 6개의 명시적 후보를 탐색했습니다. 후보에는 기존 기준 설정을 포함했습니다. 전수 가능한 모든 설정을 탐색한 것이 아니며 최적성은 이 후보 범위에 한정됩니다.',
        '- 중앙값·지표 및 Logistic 표준화는 각 inner/outer 훈련 안에서만 적합했습니다. MICE도 inner 검증 또는 outer 검증 환자를 학습/참조에 넣지 않았습니다. 사망 라벨은 대체 입력에 사용하지 않았습니다.',
        '- MICE 내부 탐색은 seed42 한 번의 posterior 대체를 사용합니다. outer 성능은 5개 대체 데이터에서 선택된 같은 설정을 각각 학습한 평균확률입니다. 전체 Train 최종 모델도 5개입니다. 따라서 내부 후보 순위는 5회 앙상블 전체를 직접 최적화한 결과가 아닙니다.',
        '- 결측 처리 방법 자체는 기존 개발 데이터 탐색 결과로 고정했으며 nested loop에서 재선택하지 않았습니다. 따라서 전체 연구의 변수·전처리 선택까지 독립 검증한 결과는 아닙니다.',
        '- 최종 파라미터는 15개 내부 검증 점수 평균에서 선택했습니다. 이 내부 분할들은 중복 표본을 포함하므로 15개 독립 검증이 아닙니다. outer 점수로 최종 설정을 고르지 않았습니다.',
        '- 조정 후 지표는 outer 폴드마다 다른 설정을 선택할 수 있는 탐색 절차의 성능입니다. 전체 Train에 적합한 하나의 최종 설정이 같은 성능을 낸다고 보장하지 않습니다. 최종 모델의 Test 성능은 새로 계산하지 않았습니다.',
        '- MICE 구현 및 범위: BayesianRidge IterativeImputer, posterior sampling, max_iter10, n_nearest_features20, 훈련 스케일 후 복원, 비음수 하한(AnionGap 값 제외), 원래 관측값 보존. aggregate min/mean/max 상호 일관성은 강제하지 않으며 체인 수렴을 입증하지 않았습니다.',
        '- 환자/폴드 배정 일치, 원래 입력 SHA256, 관측값 보존, 대체 후 유한값, LR 수렴, OOF 누락 부재, 저장 최종 모델 예측 재현 및 원본 불변성을 확인했습니다. 분류 지표는 임계값 .5이며 임계값 튜닝은 하지 않았습니다.',
        '- 전체 Train 최종 모델과 전처리는 모델별 폴더의 model_bundle_*.joblib에 저장했습니다. XGB는 5개 bundle의 예측확률을 평균해야 합니다. MICE 추론은 posterior 난수 상태를 사용하므로 같은 배치 재현을 위해 저장 상태에서 다시 로드해야 합니다.',
        '- search_scores.csv: 내부 후보별 성능. selected_by_fold.json: outer별 선택. nested_cv_summary.csv: 평가 평균±SD. candidate_rankings.csv: 최종 선택용 내부 평균. before_after.csv: 조정 전후. experiment.json: 설정·후보·입력 해시.',
        '- 기존 코호트 선택 편향·late storetime·내부 무작위 평가의 한계는 유지됩니다. Test는 이전 연구 단계에서 이미 사용됐으며 이번 탐색에 사용하지 않았습니다.','',
        '실행/중단 후 재시작: `python scripts/tune_three_models_nested_cv.py`. 완료 결과는 덮어쓰지 않습니다.']
    (OUT/'training_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for key,digest in hashes.items():
        assert hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()==digest
    write_json(OUT/'COMPLETE.json',dict(complete=True,checks_passed=True,elapsed_seconds=time.perf_counter()-start,test_evaluated=False))
    print(pd.DataFrame(comparison).to_string(index=False),flush=True)
    print('COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    main()
