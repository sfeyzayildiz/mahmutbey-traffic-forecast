'''
Feature engineering.

Convention: each row is a FORECAST ORIGIN t (the last hour whose data is known).
The target is the measured speed at t+h, for horizons h=1, 2, 3.

- Past features use data up to and including t.
- Known-ahead features (time, calendar, fuel events) are taken at the target time t+h,
  because they are known in advance.
- Values learned from data (typical speed) come from the training period only.
'''

import numpy as np
import pandas as pd

from src.processing import typical_profile, split_labels

HORIZONS=(1, 2, 3)
N_NEIGHBOURS=6
WEATHER_COLS=['temperature', 'rain', 'snowfall', 'wind_speed']
CALENDAR_COLS=['is_public_holiday', 'is_holiday_day1', 'is_half_day', 'is_admin_leave', 'is_ramadan',
               'is_iftar_hour', 'is_after_iftar', 'is_school_day', 'fuel_panic']


def past_features(df):
    '''Features at origin t that only use data up to t.'''
    s=df['speed']
    f=pd.DataFrame(index=df.index)
    f['speed_t0']=s
    for k in (1, 2):
        f[f'speed_lag{k}']=s.shift(k)
    f['speed_diff1']=s.diff(1)
    for w in (3, 6, 24):
        f[f'speed_mean{w}']=s.rolling(w).mean()
        f[f'speed_std{w}']=s.rolling(w).std()
    f['vehicles_t0']=df['vehicles']
    f['speed_filled_t0']=df['speed_filled']
    f['filled_share24']=df['speed_filled'].rolling(24).mean()

    for i in range(1, N_NEIGHBOURS+1):
        f[f'nb{i}_speed_t0']=df[f'nb{i}_speed']
        f[f'nb{i}_diff1']=df[f'nb{i}_speed'].diff(1)

    for col in WEATHER_COLS:
        f[f'{col}_t0']=df[col]
    f['rain_sum3']=df['rain'].rolling(3).sum()
    return f


def known_ahead_features(index):
    '''Time features for a set of target times (fully known in advance).'''
    h=index.hour
    d=index.dayofweek
    f=pd.DataFrame(index=index)
    f['hour_sin']=np.sin(2*np.pi*h/24)
    f['hour_cos']=np.cos(2*np.pi*h/24)
    f['dow_sin']=np.sin(2*np.pi*d/7)
    f['dow_cos']=np.cos(2*np.pi*d/7)
    f['is_weekend']=(d>=5).astype(int)
    f['hour_of_week']=d*24+h
    return f


def horizon_dataset(df, h):
    '''X, y and metadata for horizon h. Rows: origins whose target hour has a measured speed.'''
    past=past_features(df)
    target_time=df.index+pd.Timedelta(hours=h)

    ahead=known_ahead_features(target_time)
    ahead.index=df.index
    cal=df[CALENDAR_COLS].reindex(target_time)
    cal.index=df.index

    prof=typical_profile(df['speed_raw'])
    typical_target=prof.reindex(list(zip(target_time.dayofweek, target_time.hour))).values
    typical_now=prof.reindex(list(zip(df.index.dayofweek, df.index.hour))).values

    X=pd.concat([past, ahead, cal], axis=1)
    X['typical_target']=typical_target
    X['deviation_t0']=df['speed'].values-typical_now
    X['speed_lag24_target']=df['speed'].shift(24-h).values      # same hour yesterday, relative to the target
    X['speed_lag168_target']=df['speed'].shift(168-h).values    # same hour last week, relative to the target

    y=df['speed_raw'].shift(-h).rename('y')
    meta=pd.DataFrame({'target_time':target_time, 'split':split_labels(target_time).values}, index=df.index)

    keep=y.notna()&X.notna().all(axis=1)&meta['split'].notna()
    return X[keep], y[keep], meta[keep]
