"""Build an hourly Mammoth disdrometer archive. Install h5py and numpy to run."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np


RECORD = 'https://zenodo.org/records/17179963'
FILENAME = 'CUES_1min_data_atmos_radiation_soiltemp_precip_2015to2025.nc'
URL = f'https://zenodo.org/api/records/17179963/files/{FILENAME}/content'
MD5 = '69af18ce8c3575d0c9c9646c502a0273'
SIZE = 333406355
DEFAULT = Path(__file__).resolve().parents[1] / 'data' / 'mammoth' / 'research'
SWE = 'snowfall_swe_rate'
DBZ = 'equivalent_radar_reflectivity'
COLUMNS = ('start_utc', 'snow_valid_minutes', 'snow_swe_mm', 'peak_snow_swe_mm_h',
           'snow_rate_at_cap_minutes', 'reflectivity_valid_minutes', 'peak_reflectivity_dbz')


def attr(obj, key):
    value = obj.attrs[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    return value.decode() if isinstance(value, bytes) else str(value)


def checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build(source, output):
    """Convert the versioned source and atomically replace each output file."""
    if source.stat().st_size != SIZE:
        raise ValueError('Unexpected source file length')
    if checksum(source, 'md5') != MD5:
        raise ValueError('Source failed its published MD5 checksum')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as tmp, h5py.File(source, 'r') as nc:
        directory = Path(tmp)
        csv_path = directory / 'cues_disdrometer_hourly.csv.gz'
        metadata_path = directory / 'cues_disdrometer_source.json'
        times = nc['datetime']
        source_metadata = (attr(nc, 'time_zone_offset'), attr(times, 'units'),
                           attr(nc[SWE], 'units'), attr(nc[DBZ], 'units'))
        if source_metadata != ('UTC-8', 'minutes since 2015-01-01', 'mm h-1', 'dBZ'):
            raise ValueError(f'Unrecognized source clock or units: {source_metadata}')
        if any(not np.all(np.isnan(nc[key].attrs['_FillValue'])) for key in (SWE, DBZ)):
            raise ValueError('Unexpected missing-data marker')
        # Source dates are fixed Pacific standard time throughout the year.
        origin = datetime(2015, 1, 1, 8, tzinfo=timezone.utc)
        n = len(times)
        if n % 60 or n != len(nc[SWE]) or n != len(nc[DBZ]):
            raise ValueError('Expected complete aligned one-minute time axes')
        complete = wet = rates = echoes = capped_minutes = capped_hours = echo_hours = 0
        first = last = None
        partial_hours = []
        with csv_path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed, io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
            writer = csv.writer(text, lineterminator='\n')
            writer.writerow(COLUMNS)
            for start in range(0, n, 60 * 24 * 30):
                end = min(n, start + 60 * 24 * 30)
                if not np.array_equal(times[start:end], np.arange(start, end)):
                    raise ValueError('Source minutes are missing, duplicated or unordered')
                snow = nc[SWE][start:end].astype('float64').reshape(-1, 60)
                dbz = nc[DBZ][start:end].astype('float64').reshape(-1, 60)
                snow_ok = np.isfinite(snow)
                # The publisher already masks the no-echo code before release.
                dbz_ok = np.isfinite(dbz)
                if (snow[snow_ok] < 0).any():
                    raise ValueError('Unexpected negative snow SWE rate')
                for i, (srow, drow) in enumerate(zip(snow, dbz)):
                    svals, dvals = srow[snow_ok[i]], drow[dbz_ok[i]]
                    ns, nd = len(svals), len(dvals)
                    stamp = (origin + timedelta(minutes=start + 60*i)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    if ns:
                        if first is None:
                            first = stamp
                        last = stamp
                    subtotal = math.fsum(svals) / 60 if ns else None
                    total = subtotal if ns == 60 else None
                    if 0 < ns < 60:
                        partial_hours.append((stamp, f'{subtotal:.6f}'))
                    cap = int(np.count_nonzero(np.isclose(svals, 999.999, rtol=0, atol=.0001)))
                    writer.writerow((stamp, ns, '' if total is None else f'{total:.6f}',
                                     '' if not ns else f'{max(svals):.3f}',
                                     cap, nd, '' if not nd else f'{max(dvals):.3f}'))
                    rates += ns
                    echoes += nd
                    capped_minutes += cap
                    capped_hours += cap > 0
                    echo_hours += nd > 0
                    complete += total is not None
                    wet += total is not None and total > 0
        digest = checksum(csv_path, 'sha256')
        partial_path = directory / 'cues_partial_hour_subtotals.csv.gz'
        with partial_path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped, io.TextIOWrapper(zipped, encoding='utf-8', newline='') as dest:
            writer = csv.writer(dest, lineterminator='\n')
            writer.writerow(('start_utc', 'observed_minutes_swe_mm'))
            writer.writerows(partial_hours)
        provenance = {
            'site': attr(nc, 'station_name'), 'latitude': float(attr(nc, 'station_latitude')),
            'longitude': float(attr(nc, 'station_longitude')), 'elevation': attr(nc, 'station_elevation'),
            'creator': 'Manda Chasteen (2025)', 'source': RECORD, 'doi': '10.5281/zenodo.17179963',
            'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
            'input': {'url': URL, 'size_bytes': SIZE, 'md5': MD5},
            'output': {'name': csv_path.name, 'sha256': digest},
            'partial_hour_subtotals': {'name': partial_path.name, 'sha256': checksum(partial_path, 'sha256')},
            'local_period_start': attr(nc, 'period_start'), 'local_period_end': attr(nc, 'period_end'),
            'archive_hours': n//60, 'snow_swe_complete_hours': complete, 'snow_swe_wet_complete_hours': wet,
            'snow_swe_first_observed_hour_utc': first, 'snow_swe_last_observed_hour_utc': last,
            'snow_valid_minutes': rates, 'reflectivity_valid_minutes': echoes,
            'snow_rate_at_cap_minutes': capped_minutes, 'hours_with_snow_rate_at_cap': capped_hours,
            'hours_with_detected_reflectivity': echo_hours, 'source_variables': [SWE, DBZ],
            'method': [
                'UTC hours include minutes 00 through 59. Source time is fixed UTC-8.',
                'For complete hours, snow_swe_mm integrates all 60 one-minute snow-water rates, each multiplied by 1/60 hour. Partial totals are blank. Snow depth requires an independent density measurement.',
                'observed_minutes_swe_mm integrates available minutes only, without imputing gaps. Use snow_swe_mm for strict hourly totals.',
                'Minute counts accompany each peak. Only finite observations enter the peaks. Missing values are blank. Snow rates at the 999.999 mm/h instrument cap are counted.',
                'Peak equivalent reflectivity is calculated from the Parsivel2 particle size distribution. The source marks no detected precipitation as missing reflectivity.',
                'The disdrometer was installed in October 2020. Earlier archive hours have zero coverage. Phase classification, wind exposure and terrain can affect comparisons with a radar grid cell.',
            ]}
        metadata_path.write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
        partial_path.replace(output / partial_path.name)
        csv_path.replace(output / csv_path.name)
        metadata_path.replace(output / metadata_path.name)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Use an existing copy of the published NetCDF file')
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.input:
        data = build(args.input, args.output)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / FILENAME
            with urlopen(URL, timeout=60) as response, path.open('wb') as dest:
                shutil.copyfileobj(response, dest, length=1024*1024)
            data = build(path, args.output)
    print(f"{data['snow_swe_complete_hours']} complete snow-water hours from {data['site']}")


if __name__ == '__main__':
    main()
