"""Small, reproducible public-GEFS data-access proof; does not train a model.

Fetch only two inventory files and the first GRIB message in each source using
HTTP byte ranges (~1.2 MB total), then decode the grid point nearest Dillon.
Existing raw samples are reused unless --refresh is specified.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import eccodes

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ('https://noaa-gefs-retrospective.s3.amazonaws.com/'
          'GEFSv12/reforecast/2019/2019110100/c00/Days:1-10/')
MEMBERS_URL = ('https://noaa-gefs-retrospective.s3.amazonaws.com/?list-type=2&delimiter=/'
               '&prefix=GEFSv12/reforecast/2019/2019110100/&max-keys=20')
LAT, LON = 39.6244, -106.0336
KEYS = ['edition', 'centre', 'subCentre', 'generatingProcessIdentifier',
        'dataDate', 'dataTime', 'significanceOfReferenceTime', 'stepType',
        'startStep', 'endStep', 'stepUnits', 'validityDate', 'validityTime',
        'shortName', 'name', 'units', 'typeOfLevel', 'level',
        'perturbationNumber', 'typeOfEnsembleForecast',
        'numberOfForecastsInEnsemble', 'gridType', 'Ni', 'Nj',
        'iDirectionIncrementInDegrees', 'jDirectionIncrementInDegrees',
        'latitudeOfFirstGridPointInDegrees', 'longitudeOfFirstGridPointInDegrees',
        'latitudeOfLastGridPointInDegrees', 'longitudeOfLastGridPointInDegrees']


def request(url, *, headers=None, method=None):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.status, dict(response.headers), response.read()


def iso_time(date, hhmm):
    return datetime.strptime(f'{int(date):08d}{int(hhmm):04d}', '%Y%m%d%H%M').replace(tzinfo=timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'data/forecast_input_proof'
    output.mkdir(exist_ok=True)
    listing_path = output / 'gefs_reforecast_member_listing.xml'
    if args.refresh or not listing_path.exists():
        _, _, listing = request(MEMBERS_URL)
        listing_path.write_bytes(listing)
    listing = ET.fromstring(listing_path.read_bytes())
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    members = [node.text.rstrip('/').split('/')[-1]
               for node in listing.findall('s3:CommonPrefixes/s3:Prefix', ns)]
    records = []
    for variable in ['apcp_sfc', 'tmp_2m']:
        source_url = PREFIX + variable + '_2019110100_c00.grib2'
        inventory_path = output / (variable + '.idx')
        if args.refresh or not inventory_path.exists():
            _, _, inventory = request(source_url + '.idx')
            inventory_path.write_bytes(inventory)
        lines = inventory_path.read_text().splitlines()
        start = int(lines[0].split(':')[1])
        end = int(lines[1].split(':')[1]) - 1
        message_path = output / (variable + '_first_message.grib2')
        headers_path = output / (variable + '_source_headers.json')
        if args.refresh or not message_path.exists():
            status, headers, message = request(source_url, headers={'Range': f'bytes={start}-{end}'})
            if status != 206 or len(message) != end - start + 1:
                raise ValueError('Source did not honor the exact byte range; refusing a full-file download.')
            message_path.write_bytes(message)
            headers_path.write_text(json.dumps(headers, indent=2))
        elif not headers_path.exists():
            _, headers, _ = request(source_url, method='HEAD')
            headers_path.write_text(json.dumps(headers, indent=2))
        headers = {k.lower(): v for k, v in json.loads(headers_path.read_text()).items()}
        message = message_path.read_bytes()
        if len(message) != end - start + 1:
            raise ValueError('Cached message size does not match its saved inventory.')
        with message_path.open('rb') as handle:
            gid = eccodes.codes_grib_new_from_file(handle)
        try:
            meta = {}
            for key in KEYS:
                try:
                    meta[key] = eccodes.codes_get(gid, key)
                except eccodes.CodesInternalError:
                    meta[key] = None
            nearest = dict(eccodes.codes_grib_find_nearest(gid, LAT, LON, npoints=1)[0])
            nearest['longitude_degrees_east_signed'] = (nearest['lon'] + 180) % 360 - 180
            if meta['shortName'] == '2t':
                nearest['value_celsius'] = nearest['value'] - 273.15
            if meta['shortName'] == 'tp':
                nearest['liquid_precipitation_mm'] = nearest['value']
        finally:
            eccodes.codes_release(gid)
        records.append({
            'source_url': source_url, 'inventory_url': source_url + '.idx',
            'inventory_record': lines[0], 'requested_byte_range': [start, end],
            'downloaded_bytes': len(message), 'sha256': hashlib.sha256(message).hexdigest(),
            'file': str(message_path.relative_to(ROOT)),
            'source_last_modified': headers.get('last-modified'),
            'nominal_grib_reference_time_utc': iso_time(meta['dataDate'], meta['dataTime']),
            'valid_time_utc': iso_time(meta['validityDate'], meta['validityTime']),
            'actual_historical_issue_or_publication_time': None,
            'archive_member': 'c00', 'metadata': meta,
            'requested_location': {'lat': LAT, 'lon': LON, 'station': 'USC00052281 Dillon 1 E'},
            'nearest_grid_point': nearest,
            'surface_grid_elevation_m': None,
        })
    proof = {
        'kind': 'Decoded historical atmospheric reforecast values, not just an inventory',
        'reader': 'eccodes ' + eccodes.__version__,
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'actual_archive_members': members,
        'records': records,
        'interpretation': [
            'Standardized retrospective model runs: not forecasts actually issued in November 2019. Source objects were posted in July 2021.',
            'The nominal reference time is decoded as 00 UTC. The GEFSv12 paper describes Phase 2 initialization at 03 UTC; preserve this unresolved convention and check forecast spin-up semantics before using the first interval.',
            'GRIB ensemble-count headers differ between variables (20 versus 10 here); the archive listing actually contains five members. Enumerate available member files.',
            'The nearest grid point is about 14 km from the station. These are coarse-grid forecast values, not measured Dillon weather or fresh snowfall.',
            'Precipitation is liquid-equivalent accumulation. Its record window must be deaccumulated correctly; it is not new-snow depth.',
            'No grid-surface elevation was downloaded. Temperature height is 2 m above model ground, not station elevation.',
            'No atmospheric model or forecast-informed snowfall model was trained in this proof.',
        ],
    }
    (output / 'decoded_gefs_sample.json').write_text(json.dumps(proof, indent=2))
    with (output / 'decoded_gefs_sample.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['variable', 'reference_time_utc', 'valid_time_utc', 'member',
                              'start_step_hours', 'end_step_hours', 'latitude', 'longitude', 'value', 'units',
                              'temperature_celsius', 'distance_from_station_km'])
        writer.writeheader()
        for row in records:
            m, p = row['metadata'], row['nearest_grid_point']
            writer.writerow({'variable': m['name'], 'reference_time_utc': row['nominal_grib_reference_time_utc'],
                             'valid_time_utc': row['valid_time_utc'], 'member': 'c00',
                             'start_step_hours': m['startStep'], 'end_step_hours': m['endStep'],
                             'latitude': p['lat'], 'longitude': p['longitude_degrees_east_signed'],
                             'value': p['value'], 'units': m['units'],
                             'temperature_celsius': p.get('value_celsius', ''),
                             'distance_from_station_km': p['distance']})
    print(json.dumps({'decoded_records': len(records), 'raw_bytes': sum(r['downloaded_bytes'] for r in records),
                      'members': members, 'proof': str((output / 'decoded_gefs_sample.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
