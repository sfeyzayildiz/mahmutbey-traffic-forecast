'''
00 - Download IBB hourly traffic density data (2024, 12 months)

Source : Istanbul Metropolitan Municipality (IBB) Open Data Portal
         "Hourly Traffic Density Data Set"
         https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set

What it does:
  1. Queries the portal API for the list of files in the dataset.
  2. Selects the monthly CSV files for 2024.
  3. Reads each file in chunks and keeps only the rows inside a bounding box
     around Mahmutbey Junction. Full monthly files are several hundred MB,
     so they are never written to disk.
  4. Saves the result to data/raw/ibb_2024_mahmutbey_area.csv

Usage (from the project root):
    pip install requests pandas
    python src/00_download_data.py --list    # only list the files found
    python src/00_download_data.py           # download and filter
'''

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

API_URL='https://data.ibb.gov.tr/api/3/action/package_show'
DATASET_ID='hourly-traffic-density-data-set'
YEAR='2024'

# Mahmutbey Junction is at roughly (41.055 N, 28.820 E).
# The box is about 12 x 13 km. It is kept wide on purpose;
# neighbouring cells are selected later, in the processing step.
BBOX={'lat_min':41.00, 'lat_max':41.11, 'lon_min':28.74, 'lon_max':28.90}

OUTPUT_PATH=Path(__file__).resolve().parents[1]/'data'/'raw'/'ibb_2024_mahmutbey_area.csv'


def list_resources():
    '''Return the 2024 CSV resources of the dataset as [{'name', 'url'}].'''
    response=requests.get(API_URL, params={'id':DATASET_ID}, timeout=60)
    response.raise_for_status()
    resources=response.json()['result']['resources']
    selected=[]
    for res in resources:
        label=(res.get('name') or '')+' '+(res.get('url') or '')
        file_format=(res.get('format') or '').lower()
        if YEAR in label and file_format in ('csv', ''):
            selected.append({'name':res.get('name'), 'url':res['url']})
    return selected


def download_and_filter(url):
    '''Download one monthly file and return only the rows inside BBOX.'''
    response=requests.get(url, timeout=600)
    response.raise_for_status()
    raw=io.BytesIO(response.content)

    kept=[]
    for chunk in pd.read_csv(raw, chunksize=500_000):
        chunk.columns=[c.strip().upper() for c in chunk.columns]
        in_box=(
            chunk['LATITUDE'].between(BBOX['lat_min'], BBOX['lat_max'])
            &chunk['LONGITUDE'].between(BBOX['lon_min'], BBOX['lon_max'])
        )
        kept.append(chunk[in_box])
    return pd.concat(kept, ignore_index=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='only print the files found')
    args=parser.parse_args()

    resources=list_resources()
    print(f'Found {len(resources)} files for {YEAR}:')
    for res in resources:
        print(f"  - {res['name']}")
    if args.list:
        return
    if len(resources)!=12:
        print('Warning: expected 12 monthly files, check the list above.', file=sys.stderr)

    frames=[]
    for i, res in enumerate(resources, 1):
        print(f"[{i}/{len(resources)}] downloading {res['name']} ...", flush=True)
        df=download_and_filter(res['url'])
        print(f"      {len(df):,} rows in box, {df['GEOHASH'].nunique()} geohash cells")
        frames.append(df)

    result=pd.concat(frames, ignore_index=True)
    result['DATE_TIME']=pd.to_datetime(result['DATE_TIME'])
    result=(
        result.sort_values(['GEOHASH', 'DATE_TIME'])
        .drop_duplicates(['GEOHASH', 'DATE_TIME'])
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f'\nSaved: {OUTPUT_PATH}')
    print(
        f"Total {len(result):,} rows | {result['GEOHASH'].nunique()} geohash cells | "
        f"{result['DATE_TIME'].min()} -> {result['DATE_TIME'].max()}"
    )


if __name__=='__main__':
    main()
