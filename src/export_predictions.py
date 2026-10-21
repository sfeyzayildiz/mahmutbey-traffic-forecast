'''
Retrain the full XGBoost and LSTM models (test-stage setup) and save their
quantile predictions for the validation AND test periods.

The validation predictions are needed for conformal calibration of the
prediction intervals (notebook 04). Training is identical to src/train.py,
so test predictions reproduce the numbers in results.csv.

Usage
  python -m src.export_predictions --out results
'''

import argparse
from pathlib import Path

import pandas as pd

from src.data import ROOT
from src.features import HORIZONS, horizon_dataset
from src.models import QUANTILES, fit_predict_xgb, fit_predict_lstm
from src.train import SEEDS, drop_cols, stage_masks, lstm_parts

QCOLS=[f'q{int(a*100):02d}' for a in QUANTILES]


def export(out_dir, models=('xgboost', 'lstm'), seeds=SEEDS):
    out=Path(out_dir)/'calibration'
    out.mkdir(parents=True, exist_ok=True)
    df=pd.read_parquet(ROOT/'data'/'processed'/'dataset.parquet')

    for h in HORIZONS:
        X, y, meta=horizon_dataset(df, h)
        tr, va, te=stage_masks(meta, 'test')
        ev=va|te
        for model in models:
            for seed in seeds:
                path=out/f'{model}_h{h}_s{seed}.parquet'
                if path.exists():
                    continue
                if model=='xgboost':
                    cols=drop_cols(X.columns, 'full')
                    q, _=fit_predict_xgb(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va], X.loc[ev, cols], seed)
                else:
                    parts, (c, s)=lstm_parts(df, X, y, (tr, va, ev), 'full')
                    q, _=fit_predict_lstm(parts, seed)
                    q=q*s+c
                pred=pd.DataFrame(q, columns=QCOLS, index=X.index[ev])
                pred['y']=y[ev].values
                pred['split']=meta.loc[ev, 'split'].values
                pred['target_time']=meta.loc[ev, 'target_time'].values
                pred.to_parquet(path)
                print(f'saved {path.name}', flush=True)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    args=ap.parse_args()
    export(args.out)


if __name__=='__main__':
    main()
