'''
Turkish calendar for 2024: public holidays, Ramadan, iftar hours and school terms.

Sources (checked October 2026):
  - Public holidays: Law No. 2429 on national and religious holidays; 2024 religious holiday dates.
  - Administrative leave (idari izin): Presidential circulars for Ramadan Bayramı (8-9 April) and
    Kurban Bayramı (20-21 June). Applies to public employees only.
  - School terms: Ministry of National Education (MEB) 2023-24 and 2024-25 calendars.
  - Iftar: computed as astronomical sunset at Mahmutbey with the astral package.
    Official iftar times differ by only a few minutes, which does not change the hour.
'''

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

TZ=ZoneInfo('Europe/Istanbul')
LOCATION=LocationInfo('Mahmutbey', 'Turkey', 'Europe/Istanbul', 41.064, 28.812)

# Full-day public holidays
PUBLIC_HOLIDAYS={
    '2024-01-01':'New Year',
    '2024-04-10':'Ramadan Bayram day 1', '2024-04-11':'Ramadan Bayram day 2', '2024-04-12':'Ramadan Bayram day 3',
    '2024-04-23':'National Sovereignty and Children\'s Day',
    '2024-05-01':'Labour Day',
    '2024-05-19':'Youth and Sports Day',
    '2024-06-16':'Kurban Bayram day 1', '2024-06-17':'Kurban Bayram day 2',
    '2024-06-18':'Kurban Bayram day 3', '2024-06-19':'Kurban Bayram day 4',
    '2024-07-15':'Democracy and National Unity Day',
    '2024-08-30':'Victory Day',
    '2024-10-29':'Republic Day',
}
HOLIDAY_FIRST_DAYS=['2024-04-10', '2024-06-16']
HALF_DAYS=['2024-04-09', '2024-06-15', '2024-10-28']          # arife / eve, holiday from 13:00
ADMIN_LEAVE=['2024-04-08', '2024-04-09', '2024-06-20', '2024-06-21']
RAMADAN=('2024-03-11', '2024-04-09')

# School is in session on weekdays inside these ranges
SCHOOL_TERMS=[('2024-01-01', '2024-01-19'), ('2024-02-05', '2024-06-14'), ('2024-09-09', '2024-12-31')]
SCHOOL_BREAKS=[('2024-04-08', '2024-04-12'), ('2024-11-11', '2024-11-15')]


def _in_ranges(dates, ranges):
    mask=pd.Series(False, index=dates)
    for start, end in ranges:
        mask|=(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))
    return mask.values


def sunset_hour(day):
    return sun(LOCATION.observer, date=day, tzinfo=TZ)['sunset'].hour


def build_calendar(index):
    '''Return calendar flags for an hourly DatetimeIndex (local time).'''
    day=index.normalize()
    hour=index.hour
    dow=index.dayofweek
    cal=pd.DataFrame(index=index)

    cal['is_public_holiday']=day.strftime('%Y-%m-%d').isin(list(PUBLIC_HOLIDAYS)).astype(int)
    cal['is_holiday_day1']=day.strftime('%Y-%m-%d').isin(HOLIDAY_FIRST_DAYS).astype(int)
    cal['is_half_day']=(day.strftime('%Y-%m-%d').isin(HALF_DAYS)&(hour>=13)).astype(int)
    cal['is_admin_leave']=day.strftime('%Y-%m-%d').isin(ADMIN_LEAVE).astype(int)

    cal['is_ramadan']=((day>=pd.Timestamp(RAMADAN[0]))&(day<=pd.Timestamp(RAMADAN[1]))).astype(int)
    sunset={d:sunset_hour(d.date()) for d in day.unique()}
    sunset_h=pd.Series(day.map(sunset), index=index)
    cal['is_iftar_hour']=((cal['is_ramadan']==1)&(hour==sunset_h.values)).astype(int)
    cal['is_after_iftar']=((cal['is_ramadan']==1)&(hour==sunset_h.values+1)).astype(int)

    school_day=_in_ranges(day, SCHOOL_TERMS)&~_in_ranges(day, SCHOOL_BREAKS)&(dow<5)&(cal['is_public_holiday']==0).values
    cal['is_school_day']=school_day.astype(int)
    return cal
