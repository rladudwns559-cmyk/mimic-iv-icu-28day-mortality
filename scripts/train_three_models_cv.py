"""Fixed baseline comparison: shared five-fold CV, three liver labs excluded."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if (ROOT/'.codex_deps/xgboost').is_dir():
    sys.path.insert(0, str(ROOT/'.codex_deps'))

import hashlib
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, confusion_matrix, precision_recall_curve, roc_curve, auc)
from sklearn.calibration import calibration_curve
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT/'data/processed/error_cleaned/splits'
OUT = ROOT/'results/three_models_5fold_no_liver_labs'
METHODS = ['logistic_regression', 'random_forest', 'xgboost']
LR_SETTINGS = dict(C=1.0, l1_ratio=0.0, solver='lbfgs', max_iter=5000, random_state=42, class_weight=None)
RF_SETTINGS = dict(n_estimators=500, max_depth=15, min_samples_leaf=5,
                   max_features='sqrt', n_jobs=4, random_state=42, class_weight=None)
XGB_SETTINGS = dict(objective='binary:logistic', tree_method='hist', n_estimators=2000,
                    learning_rate=.03, max_depth=3, min_child_weight=5, subsample=.8,
                    colsample_bytree=.8, reg_lambda=1., reg_alpha=0., scale_pos_weight=1.,
                    eval_metric='logloss', random_state=42, n_jobs=4)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                pr_auc_trapezoidal=float(auc(recall, precision)), brier_score=float(brier_score_loss(y, p)),
                log_loss=float(log_loss(y, p)), sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
                precision=float(tp/(tp+fp)) if tp+fp else 0., accuracy=float((y == (p >= .5)).mean()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def make_pipeline(method, columns):
    binary = [c for c in columns if c.endswith('_missing') or c == 'gender']
    numeric = [c for c in columns if c not in binary]
    std = [c for c in columns if c.endswith('_std')]
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if method == 'logistic_regression':
        steps.append(('scale', StandardScaler()))
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline(steps), numeric), ('binary', 'passthrough', binary),
        # Existing missing flags cover mean/min/max absence. Std also misses for one observation.
        ('std_missing', MissingIndicator(features='all'), std),
    ], verbose_feature_names_out=True)
    model = LogisticRegression(**LR_SETTINGS) if method == 'logistic_regression' else RandomForestClassifier(**RF_SETTINGS)
    return Pipeline([('preprocessing', preprocessing), ('model', model)])


def checked_fit(model, x, y):
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
        warnings.simplefilter('always')
        model.fit(x, y)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError('Failed to converge: '+'; '.join(convergence))
    for w in caught:
        print(f'Warning: {w.message}', flush=True)
    return model


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Preserving existing results: {OUT}')
    source_columns = json.loads((SOURCE/'model_feature_columns.json').read_text())
    excluded = [c for c in source_columns if c.startswith(('lab_ALT_', 'lab_AST_', 'lab_BilirubinTotal_'))]
    columns = [c for c in source_columns if c not in excluded]
    assert len(excluded) == 18 and len(columns) == 116
    frames, targets, identities, hashes = {}, {}, {}, {}
    for split in ['train', 'validation', 'test']:
        for filename in ['X_features.csv', 'y_target.csv', 'row_ids.csv']:
            key = f'{split}/{filename}'
            hashes[key] = hashlib.sha256((SOURCE/key).read_bytes()).hexdigest()
        x = pd.read_csv(SOURCE/split/'X_features.csv')
        y = pd.read_csv(SOURCE/split/'y_target.csv').mortality_28d
        ids = pd.read_csv(SOURCE/split/'row_ids.csv')
        assert x.columns.tolist() == source_columns
        assert len(x) == len(y) == len(ids) and y.isin([0, 1]).all()
        assert ids[['subject_id', 'stay_id']].notna().all().all()
        assert ids.subject_id.is_unique and ids.stay_id.is_unique
        x = x[columns].copy()
        assert x.gender.isin(['F', 'M']).all()
        x['gender'] = x.gender.map({'F':0, 'M':1})
        assert not np.isinf(x.to_numpy()).any()
        for c in [c for c in columns if c.endswith('_mean')]:
            prefix = c[:-5]
            assert x[c].isna().equals(x[prefix+'_count'].eq(0))
            assert x[prefix+'_missing'].eq(x[c].isna().astype(int)).all()
            for suffix in ['min', 'max']:
                assert x[c].isna().equals(x[prefix+'_'+suffix].isna())
        frames[split], targets[split], identities[split] = x, y, ids
    all_ids = pd.concat(identities.values(), ignore_index=True)
    assert all_ids.subject_id.is_unique and all_ids.stay_id.is_unique
    train = pd.concat([frames['train'], frames['validation']], ignore_index=True)
    y = pd.concat([targets['train'], targets['validation']], ignore_index=True)
    ids = pd.concat([identities['train'], identities['validation']], ignore_index=True)
    test, yt, test_ids = frames['test'], targets['test'], identities['test']
    assert not train.isna().all().any()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, y))
    fold_id = np.full(len(train), -1)
    for fold, (_, valid_idx) in enumerate(folds, 1):
        assert (fold_id[valid_idx] == -1).all()
        fold_id[valid_idx] = fold
    assert (fold_id > 0).all()
    OUT.mkdir(parents=True)
    ids.assign(mortality_28d=y, fold=fold_id).to_csv(OUT/'train_folds.csv', index=False)
    test_ids.assign(mortality_28d=yt).to_csv(OUT/'test_patients.csv', index=False)
    metadata = dict(seed=42, excluded_features=excluded, input_features=columns, input_sha256=hashes,
                    versions=dict(sklearn=sklearn.__version__, xgboost=xgb.__version__),
                    split=dict(train=len(train), test=len(test), train_deaths=int(y.sum()), test_deaths=int(yt.sum()),
                               method='old train+validation merged; original test retained'),
                    protocol='fixed baseline settings; no hyperparameter search; shared stratified 5-fold',
                    threshold=.5, class_weights=False, settings=dict(logistic_regression=LR_SETTINGS,
                    random_forest=RF_SETTINGS, xgboost=XGB_SETTINGS),
                    preprocessing='train-fold-only median; existing missing flags plus per-std missing flag; LR numeric scaling; XGB raw NaN',
                    xgb_tree_selection='inner stratified 15% of each outer-training fold; early stop 50; refit entire outer-training fold; final median tree count',
                    models={})
    write_json(OUT/'experiment.json', metadata)
    cv_rows, oof_rows, oof_predictions = [], [], {}
    final_tree_counts = []
    # All model settings and XGB tree counts are fixed before any test performance is computed.
    for method in METHODS:
        started = time.perf_counter()
        folder = OUT/method
        folder.mkdir()
        p_oof = np.full(len(train), np.nan)
        details = []
        for fold, (tr, va) in enumerate(folds, 1):
            print(f'START {method} fold={fold}/5', flush=True)
            assert not set(ids.iloc[tr].subject_id) & set(ids.iloc[va].subject_id)
            if method == 'xgboost':
                inner_tr, inner_va = train_test_split(tr, test_size=.15, stratify=y.iloc[tr], random_state=42+fold)
                selector = xgb.XGBClassifier(**XGB_SETTINGS, early_stopping_rounds=50)
                selector.fit(train.iloc[inner_tr], y.iloc[inner_tr],
                             eval_set=[(train.iloc[inner_va], y.iloc[inner_va])], verbose=False)
                trees = int(selector.best_iteration)+1
                final_tree_counts.append(trees)
                write_json(folder/f'inner_learning_curve_fold_{fold}.json', selector.evals_result())
                model = xgb.XGBClassifier(**{**XGB_SETTINGS, 'n_estimators':trees})
                model.fit(train.iloc[tr], y.iloc[tr], verbose=False)
                detail = dict(fold=fold, trees=trees, inner_train_n=len(inner_tr), inner_stop_n=len(inner_va))
            else:
                model = checked_fit(make_pipeline(method, columns), train.iloc[tr], y.iloc[tr])
                detail = dict(fold=fold, transformed_features=len(model['preprocessing'].get_feature_names_out()))
                if method == 'logistic_regression':
                    detail['iterations'] = int(model['model'].n_iter_[0])
            p_oof[va] = model.predict_proba(train.iloc[va])[:, 1]
            result = metrics(y.iloc[va], p_oof[va])
            cv_rows.append(dict(model=method, fold=fold, train_n=len(tr), validation_n=len(va), **result))
            details.append(detail)
            pd.DataFrame(cv_rows).to_csv(OUT/'cv_fold_metrics.csv', index=False)
            print(f'FINISH {method} fold={fold} AUC={result["roc_auc"]:.4f} AP={result["average_precision"]:.4f}', flush=True)
        assert np.isfinite(p_oof).all()
        oof_predictions[method] = p_oof
        oof_rows.append(dict(model=method, **metrics(y, p_oof)))
        ids.assign(mortality_28d=y, fold=fold_id, probability=p_oof,
                   prediction_at_0_5=(p_oof >= .5).astype(int)).to_csv(folder/'oof_predictions.csv', index=False)
        metadata['models'][method] = dict(cv_details=details, cv_seconds=time.perf_counter()-started,
                                          pooled_oof_metrics=metrics(y, p_oof))
        write_json(OUT/'experiment.json', metadata)
    metric_names = ['roc_auc', 'average_precision', 'pr_auc_trapezoidal', 'brier_score', 'log_loss',
                    'sensitivity', 'specificity', 'precision', 'accuracy']
    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby('model')[metric_names].agg(['mean', 'std'])
    summary.columns = [f'{a}_{b}' for a,b in summary.columns]
    summary.reset_index().to_csv(OUT/'cv_summary.csv', index=False)
    pd.DataFrame(oof_rows).to_csv(OUT/'oof_metrics.csv', index=False)
    final_trees = int(np.median(final_tree_counts))
    metadata['final_xgb_trees'] = final_trees
    metadata['settings_frozen_before_test_evaluation'] = True
    write_json(OUT/'experiment.json', metadata)
    test_rows, test_predictions, final_models = [], {}, {}
    for method in METHODS:
        print(f'FINAL REFIT {method}', flush=True)
        folder = OUT/method
        if method == 'xgboost':
            model = xgb.XGBClassifier(**{**XGB_SETTINGS, 'n_estimators':final_trees})
            model.fit(train, y, verbose=False)
            model.save_model(folder/'model.ubj')
            loaded = xgb.XGBClassifier()
            loaded.load_model(folder/'model.ubj')
            names = columns
            importance = model.feature_importances_
        else:
            model = checked_fit(make_pipeline(method, columns), train, y)
            joblib.dump(model, folder/'pipeline.joblib', compress=3)
            loaded = joblib.load(folder/'pipeline.joblib')
            names = model['preprocessing'].get_feature_names_out().tolist()
            importance = model['model'].coef_[0] if method == 'logistic_regression' else model['model'].feature_importances_
            transformed = model['preprocessing'].transform(train.iloc[:100])
            assert np.isfinite(transformed).all()
        p = model.predict_proba(test)[:,1]
        np.testing.assert_allclose(loaded.predict_proba(test)[:,1], p)
        test_predictions[method], final_models[method] = p, model
        result = metrics(yt, p)
        test_rows.append(dict(model=method, n=len(yt), **result))
        test_ids.assign(mortality_28d=yt, probability=p,
                        prediction_at_0_5=(p >= .5).astype(int)).to_csv(folder/'test_predictions.csv', index=False)
        pd.DataFrame({'feature':names, 'coefficient' if method == 'logistic_regression' else 'importance':importance}).to_csv(
            folder/'feature_weights.csv', index=False)
        write_json(folder/'feature_columns.json', names)
        metadata['models'][method]['test_metrics'] = result
        metadata['models'][method]['final_feature_count'] = len(names)
        write_json(OUT/'experiment.json', metadata)
        pd.DataFrame(test_rows).to_csv(OUT/'test_metrics.csv', index=False)
        print(f'TEST {method}: '+json.dumps(result), flush=True)
    # Conditional uncertainty of fixed predictions, using shared patient bootstrap draws.
    rng = np.random.default_rng(42)
    distributions = {m:[] for m in METHODS}
    yy = yt.to_numpy()
    for _ in range(1000):
        ix = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[ix])) < 2:
            continue
        for method in METHODS:
            pp = test_predictions[method][ix]
            distributions[method].append([roc_auc_score(yy[ix], pp), average_precision_score(yy[ix], pp), brier_score_loss(yy[ix], pp)])
    ci_rows, pair_rows = [], []
    for method in METHODS:
        for j, metric in enumerate(['roc_auc', 'average_precision', 'brier_score']):
            low, high = np.quantile(np.asarray(distributions[method])[:, j], [.025, .975])
            ci_rows.append(dict(model=method, metric=metric, estimate=metadata['models'][method]['test_metrics'][metric],
                                ci95_low=low, ci95_high=high))
    from itertools import combinations
    for a, b in combinations(METHODS, 2):
        for j, metric in enumerate(['roc_auc', 'average_precision', 'brier_score']):
            delta = np.asarray(distributions[a])[:,j]-np.asarray(distributions[b])[:,j]
            low, high = np.quantile(delta, [.025, .975])
            pair_rows.append(dict(model_a=a, model_b=b, metric=metric,
                difference=metadata['models'][a]['test_metrics'][metric]-metadata['models'][b]['test_metrics'][metric],
                ci95_low=low, ci95_high=high))
    pd.DataFrame(ci_rows).to_csv(OUT/'test_bootstrap_ci.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(OUT/'test_paired_differences.csv', index=False)
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    display = {'logistic_regression':'Logistic regression', 'random_forest':'Random forest', 'xgboost':'XGBoost'}
    for method in METHODS:
        p = test_predictions[method]
        fpr, tpr, _ = roc_curve(yt, p)
        precision, recall, _ = precision_recall_curve(yt, p)
        fraction, mean_p = calibration_curve(yt, p, n_bins=10, strategy='quantile')
        axes[0].plot(fpr, tpr, label=f'{display[method]} ({roc_auc_score(yt,p):.3f})')
        axes[1].plot(recall, precision, label=f'{display[method]} (AP {average_precision_score(yt,p):.3f})')
        axes[2].plot(mean_p, fraction, 'o-', label=display[method])
    axes[0].plot([0,1], [0,1], '--', color='gray')
    axes[1].axhline(yt.mean(), ls='--', color='gray')
    axes[2].plot([0,1], [0,1], '--', color='gray')
    for ax, title, xlabel, ylabel in zip(axes, ['Test ROC', 'Test Precision-Recall', 'Test calibration'],
        ['False positive rate', 'Recall', 'Mean predicted probability'], ['True positive rate', 'Precision', 'Observed mortality']):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0,1), ylim=(0,1))
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)
    fig.suptitle('ALT·AST·총빌리루빈 제외: 5-fold 학습 후 최종 Test 평가')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT/f'test_curves.{ext}', dpi=180)
    plt.close(fig)
    lines = ['# 3개 모델: Train/Test + 층화 5-fold 비교', '',
        f'Train {len(train):,}명(기존 train+validation), Test {len(test):,}명(기존 test). ALT·AST·총빌리루빈의 18개 특징을 제외한 116개 입력을 사용했습니다.', '',
        '## 교차검증 결과', '', '| 모델 | ROC-AUC 평균±SD | AP 평균±SD | Brier 평균±SD |', '|---|---:|---:|---:|']
    for m in METHODS:
        s = summary.loc[m]
        lines.append(f'| {display[m]} | {s.roc_auc_mean:.4f} ± {s.roc_auc_std:.4f} | {s.average_precision_mean:.4f} ± {s.average_precision_std:.4f} | {s.brier_score_mean:.4f} ± {s.brier_score_std:.4f} |')
    lines += ['', 'SD는 5개 폴드 점수의 표본 표준편차이며 신뢰구간이 아닙니다. OOF 전체 지표는 oof_metrics.csv에 별도로 저장했습니다.', '',
        '## 최종 Test 결과', '', '| 모델 | ROC-AUC | AP | Brier | Log loss | 민감도 | 특이도 | 정밀도 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in test_rows:
        lines.append(f"| {display[r['model']]} | {r['roc_auc']:.4f} | {r['average_precision']:.4f} | {r['brier_score']:.4f} | {r['log_loss']:.4f} | {r['sensitivity']:.2%} | {r['specificity']:.2%} | {r['precision']:.2%} |")
    lines += ['', '![Test 곡선](test_curves.png)', '', '## 학습 조건 및 검증', '',
        '- 별도 고정 validation 세트 없이 동일한 StratifiedKFold(5, shuffle=True, seed=42)를 사용했습니다. 폴드 배정은 train_folds.csv에 보존했습니다.',
        '- 이번 실행은 튜닝 전 기준 모델입니다. Logistic regression: L2, C=1, lbfgs, max_iter=5000. Random forest: 트리 500개, 깊이 15, min_samples_leaf=5, max_features=sqrt. 클래스 가중치 및 오버샘플링은 적용하지 않았습니다.',
        '- Logistic/RF: 각 훈련 폴드에서 중앙값을 학습하고 결측 지표를 유지합니다. 기존 변수별 missing 지표는 mean/min/max의 결측을 나타내며, std에는 19개 별도 결측 지표를 추가했습니다. 최종 변환 입력은 135개입니다. Logistic에만 이진 특징을 제외한 수치 특징의 StandardScaler를 적용했습니다.',
        '- XGBoost: 수치 NaN 그대로, 기존 missing/count 유지, 추가 대체/표준화 없음. 깊이 3, 학습률 .03, min_child_weight=5, subsample/colsample=.8, L2=1, 최대 트리 2000.',
        f'- XGBoost 조기 종료는 각 outer 훈련 폴드의 내부 15%에서만 수행(50회)합니다. 선택 트리 수 {final_tree_counts}로 해당 outer 훈련 전체를 재학습했습니다. 최종 전체 Train 모델은 중앙값 {final_trees}개 트리로 학습했습니다. Test를 조기 종료에 사용하지 않았습니다.',
        '- Test 평가 전 세 모델의 모든 설정과 최종 트리 수를 고정했습니다. 전체 Train으로 재학습한 모델만 Test에서 평가했고 Test 기반 튜닝/임계값 선택은 하지 않았습니다. 분류 지표는 임계값 0.5입니다.',
        '- AP는 average precision이며 PR-AUC(사다리꼴)와 구별됩니다. CSV에는 두 지표를 모두 기록했습니다.',
        '- test_bootstrap_ci.csv와 test_paired_differences.csv는 동일 Test 환자 1000회 재표집에 기반합니다. 모델 재학습/분할 변동성을 반영하지 않으며 다중비교 보정 전입니다.',
        '- 검증: 환자/stay 중복 및 Train/Test·폴드 중첩 부재, 원래 특징 목록/결측 정의, OOF 각 환자 1회 예측, LR 수렴, 저장 모델 예측 일치, 전처리 후 유한값, 입력 파일 SHA256 보존을 확인했습니다.',
        '- 예측 파이프라인 전체를 비교하므로 XGBoost와 다른 두 모델 사이에는 결측 처리 방법 차이도 포함됩니다.',
        '- 기존 unknown 라벨 제외 및 24시간 관찰 조건에 따른 선택 편향, 늦은 storetime을 허용하는 charttime 정책, 내부 무작위 분할의 한계가 남습니다. 이번 Test는 최종 비교에 사용했으므로 이후 반복 튜닝의 기준으로 사용하면 안 됩니다.', '',
        '재실행: `python scripts/train_three_models_cv.py` (기존 결과가 있으면 덮어쓰지 않습니다).',
        '각 모델 폴더에 최종 모델/전처리, OOF·Test 예측, 특징 목록 및 계수/중요도를 저장했습니다.']
    (OUT/'training_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for key, digest in hashes.items():
        assert hashlib.sha256((SOURCE/key).read_bytes()).hexdigest() == digest
    metadata['checks_passed'] = True
    metadata['test_evaluated'] = True
    write_json(OUT/'experiment.json', metadata)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)
    print('COMPLETE:', OUT, flush=True)


if __name__ == '__main__':
    main()
