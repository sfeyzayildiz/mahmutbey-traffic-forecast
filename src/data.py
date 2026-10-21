'''
Data loading helpers shared by all notebooks and scripts.
'''

from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RAW_DIR=ROOT/'data'/'raw'
FIG_DIR=ROOT/'figures'

TARGET_CELL='sxk3xe'          # Mahmutbey Junction (verified against the original study file)
YEAR_INDEX=pd.date_range('2024-01-01 00:00', '2024-12-31 23:00', freq='h')


def load_traffic():
    '''All geohash cells in the study box, one row per cell and hour.'''
    df=pd.read_csv(RAW_DIR/'ibb_2024_mahmutbey_area.csv.gz', parse_dates=['DATE_TIME'])
    return df.sort_values(['GEOHASH', 'DATE_TIME']).reset_index(drop=True)


def load_target(traffic=None, reindex=True):
    '''Mahmutbey series. With reindex=True, missing hours appear as NaN rows.'''
    if traffic is None:
        traffic=load_traffic()
    df=traffic[traffic['GEOHASH']==TARGET_CELL].set_index('DATE_TIME')
    df=df[['MINIMUM_SPEED', 'MAXIMUM_SPEED', 'AVERAGE_SPEED', 'NUMBER_OF_VEHICLES']]
    if reindex:
        df=df.reindex(YEAR_INDEX)
        df.index.name='DATE_TIME'
    return df


def load_weather():
    df=pd.read_csv(RAW_DIR/'istanbul_weather_2024.csv', parse_dates=['DATE_TIME'])
    return df.set_index('DATE_TIME')


def load_fuel_events():
    return pd.read_csv(RAW_DIR/'fuel_panic_hours_2024.csv', parse_dates=['DATE_TIME'])
