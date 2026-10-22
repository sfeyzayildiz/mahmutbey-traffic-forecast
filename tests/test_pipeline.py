'''
Tests that protect the main rules of the project:
data quality, a strictly chronological split, measured-only targets,
no future information in the features, and calibrated intervals.

Run:  pytest -q
'''

import numpy as np
import pandas as pd
import pytest

from src.data import load_traffic, load_weather, load_fuel_events, TARGET_CELL, YEAR_INDEX
from src.processing import build_dataset, SPLITS
from src.features import HORIZONS, horizon_dataset, past_features
from src.calibration import cqr_offset


@pytest.fixture(scope='session')
def traffic():
    return load_traffic()


@pytest.fixture(scope='session')
def dataset(traffic):
    df, meta, _=build_dataset(traffic, load_weather(), load_fuel_events())
    return df


# ---------------------------------------------------------------- raw data

def test_raw_traffic_is_consistent(traffic):
    assert TARGET_CELL in set(traffic['GEOHASH'])
    assert not traffic.duplicated(['GEOHASH', 'DATE_TIME']).any()
    assert (traffic['AVERAGE_SPEED']>=traffic['MINIMUM_SPEED']).all()
    assert (traffic['AVERAGE_SPEED']<=traffic['MAXIMUM_SPEED']).all()
    coverage=traffic[traffic['GEOHASH']==TARGET_CELL]['DATE_TIME'].nunique()/len(YEAR_INDEX)
    assert coverage>0.85


def test_weather_covers_every_hour():
    weather=load_weather()
    assert weather.index.equals(pd.DatetimeIndex(YEAR_INDEX, name='DATE_TIME'))
    assert weather[['temperature', 'rain', 'snowfall', 'wind_speed']].notna().all().all()


# ---------------------------------------------------------------- processing

def test_split_is_chronological(dataset):
    order=['train', 'val', 'test']
    ends=[dataset.index[dataset['split']==s].max() for s in order]
    starts=[dataset.index[dataset['split']==s].min() for s in order]
    assert ends[0]<starts[1] and ends[1]<starts[2]
    assert starts[2]==pd.Timestamp(SPLITS['test'][0])


def test_inputs_are_complete(dataset):
    assert dataset.drop(columns='speed_raw').isna().sum().sum()==0


# ---------------------------------------------------------------- features

@pytest.mark.parametrize('h', HORIZONS)
def test_targets_are_measured_hours_only(dataset, h):
    X, y, meta=horizon_dataset(dataset, h)
    filled_at_target=dataset['speed_filled'].reindex(meta['target_time']).values
    assert y.notna().all()
    assert (filled_at_target==0).all()


def test_no_future_information_in_features(dataset):
    X, y, meta=horizon_dataset(dataset, 1)
    cols=past_features(dataset).columns
    rng=np.random.default_rng(0)
    for t in rng.choice(X.index, 25, replace=False):
        rebuilt=past_features(dataset.loc[:t]).loc[t, cols].astype(float).values
        assert np.allclose(rebuilt, X.loc[t, cols].astype(float).values, equal_nan=True)


# ---------------------------------------------------------------- calibration

def test_cqr_reaches_target_coverage():
    rng=np.random.default_rng(1)
    y=rng.normal(0, 1, 2000)
    lower, upper=np.full(2000, -1.0), np.full(2000, 1.0)      # too narrow for 90%
    off=cqr_offset(y, lower, upper, 0.90)
    coverage=np.mean((y>=lower-off)&(y<=upper+off))
    assert off>0 and coverage>=0.90
