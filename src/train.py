'''
Run all modelling experiments and write results to one folder.

Stages
  test        : train Jan-Aug, early stopping on Sep-Oct, evaluate once on Nov-Dec
  walkforward : train on months 1..m, evaluate on month m+1 (m = 4..9, i.e. May-Oct)

Every finished run is appended to results.csv, so the script can be stopped and restarted
(finished runs are skipped). Test-stage predictions are saved for plotting in notebook 05.

Usage
  python -m src.train --out results                         # everything
  python -m src.train --out results --stage test            # only the final test runs
  python -m src.train --out results --quick                 # tiny smoke test (1 seed, few epochs)
'''

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT
from src.features import HORIZONS, CALENDAR_COLS, horizon_dataset, past_features
from src.models import (QUANTILES, WINDOW, SEQ_COLS_BASE, SEQ_COLS_NB, SEQ_COLS_WEATHER, Scaler,
                        persistence, seasonal_naive, add_error_quantiles,
                        fit_predict_xgb, fit_predict_lstm, make_windows, evaluate)

ABLATIONS=['full', 'no_neighbours', 'no_calendar', 'no_weather', 'no_fuel']
WF_ABLATIONS=['full', 'no_neighbours']
SEEDS=[1, 2, 3]
WF_MONTHS=[5, 6, 7, 8, 9, 10]


def drop_cols(columns, ablation):
    cols=list(columns)
    if ablation=='no_neighbours':
        cols=[c for c in cols if not c.startswith('nb')]
    elif ablation=='no_calendar':
        cols=[c for c in cols if c not in CALENDAR_COLS or c=='fuel_panic']
    elif ablation=='no_weather':
        cols=[c for c in cols if not c.startswith(('temperature', 'rain', 'snowfall', 'wind_speed'))]
    elif ablation=='no_fuel':
        cols=[c for c in cols if c!='fuel_panic']
    return cols


def stage_masks(meta, stage, month=None):
    t=meta['target_time']
    if stage=='test':
        return (meta['split']=='train').values, (meta['split']=='val').values, (meta['split']=='test').values
    # walk-forward: train on months < month, inner early-stopping set = last 14 days of that, evaluate on month
    train_all=(t.dt.month<month).values
    cut=t[train_all].max()-pd.Timedelta(days=14)
    tr=train_all&(t<=cut).values
    va=train_all&(t>cut).values
    ev=(t.dt.month==month).values
    return tr, va, ev


def lstm_parts(df, X, y, masks, ablation):
    '''Build scaled windows, context, hour-of-week and anchor for train/val/eval rows.'''
    seq_cols=SEQ_COLS_BASE+SEQ_COLS_NB+SEQ_COLS_WEATHER
    seq_cols=drop_cols(seq_cols, ablation)
    ctx_cols=[c for c in drop_cols(X.columns, ablation) if c not in seq_cols and c!='hour_of_week']

    full_past=past_features(df)                              # one row per hour of the year
    pos=pd.Series(np.arange(len(df)), index=df.index)
    origins=pos.loc[X.index].values

    tr=masks[0]
    seq_scaler=Scaler().fit(full_past.loc[X.index[tr], seq_cols])
    seq_table=np.nan_to_num(seq_scaler.transform(full_past[seq_cols]))
    ctx_scaler=Scaler().fit(X.loc[tr, ctx_cols])
    s_center, s_scale=seq_scaler.center['speed_t0'], seq_scaler.scale['speed_t0']

    parts={}
    for name, m in zip(['tr', 'va', 'ev'], masks):
        parts[name]={
            'seq':make_windows(seq_table, origins[m]),
            'ctx':ctx_scaler.transform(X.loc[m, ctx_cols]),
            'how':X.loc[m, 'hour_of_week'].values.astype('int32')[:, None],
            'anchor':((X.loc[m, 'speed_t0'].values-s_center)/s_scale).astype('float32')[:, None],
            'y':((y[m].values-s_center)/s_scale).astype('float32')[:, None],
        }
    return parts, (s_center, s_scale)


def run_one(df, data, stage, month, model, h, ablation, seed, quick):
    X, y, meta=data[h]
    masks=stage_masks(meta, stage, month)
    tr, va, ev=masks
    info={}
    t0=time.time()

    if model in ('persistence', 'seasonal_naive'):
        f=persistence if model=='persistence' else seasonal_naive
        q=add_error_quantiles(f(X[tr]), y[tr].values, f(X[ev]))
    elif model=='xgboost':
        cols=drop_cols(X.columns, ablation)
        q, info=fit_predict_xgb(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va], X.loc[ev, cols], seed)
    elif model=='lstm':
        parts, (c, s)=lstm_parts(df, X, y, masks, ablation)
        kw={'max_epochs':3, 'patience':2} if quick else {}
        q, info=fit_predict_lstm(parts, seed, **kw)
        q=q*s+c
    else:
        raise ValueError(model)

    res=evaluate(y[ev].values, q)
    res.update({'train_seconds':round(time.time()-t0, 1), **info})
    pred=pd.DataFrame(q, columns=[f'q{int(a*100):02d}' for a in QUANTILES], index=X.index[ev])
    pred['y']=y[ev].values
    pred['target_time']=meta.loc[ev, 'target_time'].values
    return res, pred


def plan(stages, quick):
    seeds=SEEDS[:1] if quick else SEEDS
    runs=[]
    for stage in stages:
        months=[None] if stage=='test' else WF_MONTHS
        abls=ABLATIONS if stage=='test' else WF_ABLATIONS
        sds=seeds if stage=='test' else seeds[:1]
        for month, h in itertools.product(months, HORIZONS):
            runs.append((stage, month, 'persistence', h, 'full', 0))
            runs.append((stage, month, 'seasonal_naive', h, 'full', 0))
            for model, abl, seed in itertools.product(['xgboost', 'lstm'], abls, sds):
                runs.append((stage, month, model, h, abl, seed))
    return runs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    ap.add_argument('--stage', choices=['all', 'test', 'walkforward'], default='all')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-mlflow', action='store_true')
    args=ap.parse_args()

    out=Path(args.out); (out/'predictions').mkdir(parents=True, exist_ok=True)
    csv=out/('results_quick.csv' if args.quick else 'results.csv')
    done=set()
    if csv.exists():
        prev=pd.read_csv(csv)
        done={tuple(r) for r in prev[['run_id']].astype(str).values}

    mlflow=None
    if not args.no_mlflow:
        import os
        os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')
        import mlflow
        mlflow.set_tracking_uri(f'sqlite:///{(out/"mlflow.db").resolve()}')
        mlflow.set_experiment('mahmutbey-quick' if args.quick else 'mahmutbey')

    df=pd.read_parquet(ROOT/'data'/'processed'/'dataset.parquet')
    data={h:horizon_dataset(df, h) for h in HORIZONS}
    stages=['test', 'walkforward'] if args.stage=='all' else [args.stage]
    runs=plan(stages, args.quick)
    if args.quick:
        runs=[r for r in runs if r[0]=='test' and r[4] in ('full', 'no_neighbours')]+[r for r in runs if r[0]=='walkforward' and r[1]==10 and r[4]=='full']

    print(f'{len(runs)} runs planned, {len(done)} already done')
    for i, (stage, month, model, h, abl, seed) in enumerate(runs, 1):
        run_id=f'{stage}|{month}|{model}|h{h}|{abl}|s{seed}'
        if (run_id,) in done:
            continue
        res, pred=run_one(df, data, stage, month, model, h, abl, seed, args.quick)
        row={'run_id':run_id, 'stage':stage, 'eval_month':month, 'model':model, 'horizon':h,
             'ablation':abl, 'seed':seed, **res}
        pd.DataFrame([row]).to_csv(csv, mode='a', header=not csv.exists(), index=False)
        if stage=='test':
            pred.to_parquet(out/'predictions'/f'{model}_h{h}_{abl}_s{seed}.parquet')
        if mlflow:
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params({k:row[k] for k in ['stage', 'eval_month', 'model', 'horizon', 'ablation', 'seed']})
                mlflow.log_metrics({k:v for k, v in res.items() if isinstance(v, (int, float))})
        print(f"[{i}/{len(runs)}] {run_id:45s} MAE {res['MAE']:.2f}  PICP90 {res['PICP90']:.1f}  ({res['train_seconds']}s)", flush=True)

    print(f'Done. Results: {csv}')


if __name__=='__main__':
    main()
