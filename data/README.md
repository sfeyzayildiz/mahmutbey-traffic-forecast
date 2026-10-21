# Data

## Files in this repository

| File | Description | Rows | Source |
|---|---|---|---|
| `raw/ibb_2024_mahmutbey_area.csv.gz` | Hourly traffic speed and vehicle counts for 251 geohash cells around Mahmutbey Junction, 1 Jan – 31 Dec 2024 | 2,003,411 | IBB Open Data Portal |

The file is a filtered subset of the monthly 2024 files of the IBB
**Hourly Traffic Density Data Set**. Only rows inside the bounding box
`41.00–41.11 N, 28.74–28.90 E` were kept. No values were changed.
It can be regenerated with `python src/00_download_data.py`.

Columns: `DATE_TIME`, `LATITUDE`, `LONGITUDE`, `GEOHASH`, `MINIMUM_SPEED`,
`MAXIMUM_SPEED`, `AVERAGE_SPEED` (km/h), `NUMBER_OF_VEHICLES`.

## License and attribution

**IBB traffic data**
Source: Istanbul Metropolitan Municipality (İBB) Open Data Portal,
[Hourly Traffic Density Data Set](https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set).
Licensed under the [İBB Open Data License v1.0](https://data.ibb.gov.tr/license),
which permits copying, redistribution and adaptation, including commercial use,
provided the source is attributed.

> Contains public sector information licensed under the İBB Open Data License v1.0
> (İstanbul Büyükşehir Belediyesi Açık Veri Lisansı).

The data is provided "as is" by İBB. This project is independent and is not
endorsed by or affiliated with İstanbul Büyükşehir Belediyesi.

**Code vs. data**
The MIT license in the repository root applies to the **code only**.
The data files keep the license of their original source, as listed above.
