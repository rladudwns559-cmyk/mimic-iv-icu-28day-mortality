"""Predict with saved final model bundles; MICE bundles are loaded afresh per batch."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.codex_deps'))
import argparse
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def predict(frame,model_name,model_root):
    paths=sorted((model_root/model_name).glob('model_bundle_*.joblib'))
    expected=5 if model_name=='xgboost' else 1
    if len(paths)!=expected:
        raise FileNotFoundError(f'Expected {expected} completed model bundles: {model_name}; found {len(paths)}')
    probabilities=[]
    for path in paths:
        bundle=joblib.load(path)
        raw=frame[bundle['columns']].copy()
        if raw.gender.isin(['F','M']).all():
            raw['gender']=raw.gender.map({'F':0,'M':1})
        if not raw.gender.isin([0,1]).all() or np.isinf(raw.to_numpy()).any():
            raise ValueError('Invalid gender or infinite input values')
        with threadpool_limits(limits=1):
            if bundle['preprocessing']=='mice':
                transformed=bundle['mice_scaler'].transform(raw)
                filled=bundle['mice_scaler'].inverse_transform(bundle['imputer'].transform(transformed))
            else:
                filled=bundle['imputer'].transform(raw)
        values=np.column_stack([filled,raw[bundle['std_columns']].isna().to_numpy(dtype=float)])
        if bundle['logistic_scaler'] is not None:
            ix=bundle['logistic_numeric_indices']
            values[:,ix]=bundle['logistic_scaler'].transform(values[:,ix])
        assert np.isfinite(values).all()
        probabilities.append(bundle['model'].predict_proba(values)[:,1])
    return np.mean(probabilities,axis=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--model-root',type=Path,default=ROOT/'results/three_models_nested_tuning')
    args=parser.parse_args()
    if not (args.model_root/'COMPLETE.json').exists():
        raise RuntimeError('Training has not completed; refusing partial model bundles')
    if args.input.resolve()==args.output.resolve() or args.output.exists():
        raise FileExistsError('Output must be new; preserving existing files')
    frame=pd.read_csv(args.input)
    result=frame[[c for c in ['subject_id','hadm_id','stay_id'] if c in frame]].copy()
    if not len(result.columns):
        result['row_index']=np.arange(len(frame))
    for model in ['logistic_regression','random_forest','xgboost']:
        p=predict(frame,model,args.model_root)
        result[model+'_probability']=p
        result[model+'_prediction_at_0_5']=(p>=.5).astype(int)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print('Predictions:',args.output)


if __name__=='__main__':
    main()
