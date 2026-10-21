'''
Processing: data split, suspect hours, neighbour selection, gap filling and joining all sources.

Every value that is learned from data (thresholds, correlations, typical profiles)
is computed on the TRAINING period only, so nothing leaks from validation or test.
'''

import numpy as np
import pandas as pd

from src.data import TARGET_CELL, YEAR_INDEX
from src.calendar_2024 import build_calendar

SPLITS={'train':('2024-01-01', '2024-08-31'), 'val':('2024-09-01', '2024-10-31'), 'test':('2024-11-01', '2024-12-31')}
SUSPECT_RATIO=0.10          # hour is suspect if vehicles < 10% of the typical count for that weekday and hour
N_NEIGHBOURS=6
MAX_DIST_KM=3.0
MIN_COVERAGE=0.90
WEATHER_COLS=['temperature', 'rain', 'snowfall', 'wind_speed']


def split_labels(index):
    labels=pd.Series(index=index, dtype=object)
    for name, (start, end) in SPLITS.items():
        labels[(index>=pd.Timestamp(start))&(index<=pd.Timestamp(end)+pd.Timedelta(hours=23))]=name
    return labels


def train_mask(index):
    return split_labels(index)=='train'


def wide_table(traffic, column):
    return traffic.pivot_table(index='DATE_TIME', columns='GEOHASH', values=column).reindex(YEAR_INDEX)


def suspect_flags(vehicles):
    '''1 where the vehicle count is far below normal for that weekday and hour (typical count from training data).'''
    keys=list(zip(vehicles.index.dayofweek, vehicles.index.hour))
    typical=pd.Series(typical_profile(vehicles).reindex(keys).values, index=vehicles.index)
    return (vehicles<SUSPECT_RATIO*typical).astype(int)


def select_neighbours(speed, cells, k=N_NEIGHBOURS):
    '''Pick the k cells most correlated with the target on TRAINING data, within distance and coverage limits.'''
    tr=speed[train_mask(speed.index)]
    info=cells.copy()
    info['coverage']=speed.notna().mean()
    info['corr_train']=tr.corrwith(tr[TARGET_CELL])
    lat0, lon0=info.loc[TARGET_CELL, ['LATITUDE', 'LONGITUDE']]
    info['dist_km']=np.hypot((info['LATITUDE']-lat0)*111.32, (info['LONGITUDE']-lon0)*111.32*np.cos(np.radians(lat0)))
    pool=info.drop(TARGET_CELL).query('dist_km<=@MAX_DIST_KM and coverage>=@MIN_COVERAGE')
    chosen=pool.sort_values('corr_train', ascending=False).head(k).index.tolist()
    return chosen, info


def typical_profile(series):
    '''Median speed for each (weekday, hour), from training data only.'''
    s=series[train_mask(series.index)].dropna()
    return s.groupby([s.index.dayofweek, s.index.hour]).median()


def fill_with_profile(series, profile):
    keys=list(zip(series.index.dayofweek, series.index.hour))
    typical=pd.Series(profile.reindex(keys).values, index=series.index)
    return series.fillna(typical), series.isna().astype(int)


def build_dataset(traffic, weather, fuel):
    '''One row per hour of 2024 with target, neighbours, weather, calendar and flags.'''
    speed=wide_table(traffic, 'AVERAGE_SPEED')
    vehicles=wide_table(traffic, 'NUMBER_OF_VEHICLES')
    cells=traffic.groupby('GEOHASH')[['LATITUDE', 'LONGITUDE']].first()

    neighbours, cell_info=select_neighbours(speed, cells)

    df=pd.DataFrame(index=YEAR_INDEX)
    df.index.name='DATE_TIME'
    df['split']=split_labels(YEAR_INDEX).values

    raw=speed[TARGET_CELL]
    df['observed']=raw.notna().astype(int)
    df['suspect']=suspect_flags(vehicles[TARGET_CELL])
    df['usable']=((df['observed']==1)&(df['suspect']==0)).astype(int)
    df['speed_raw']=raw.where(df['usable']==1)                      # target for evaluation (NaN = not usable)
    df['speed'], df['speed_filled']=fill_with_profile(df['speed_raw'], typical_profile(df['speed_raw']))
    veh=vehicles[TARGET_CELL].where(df['usable']==1)
    df['vehicles'], _=fill_with_profile(veh, typical_profile(veh))

    for i, cell in enumerate(neighbours, 1):
        s=speed[cell].where(suspect_flags(vehicles[cell])==0)
        df[f'nb{i}_speed'], df[f'nb{i}_filled']=fill_with_profile(s, typical_profile(s))

    df=df.join(weather[WEATHER_COLS])
    df=df.join(build_calendar(YEAR_INDEX))
    df['fuel_panic']=df.index.isin(fuel['DATE_TIME']).astype(int)

    meta={'suspect_ratio':SUSPECT_RATIO, 'neighbours':neighbours}
    return df, meta, cell_info
