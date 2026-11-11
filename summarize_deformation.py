"""Summarize valid pointwise GC-M3C2 estimates without interpolated values."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame, interval=10.0, chainage_offset=0.0, magnitude=False):
    if not np.isfinite(interval) or interval <= 0 or not np.isfinite(chainage_offset):
        raise ValueError('Interval must be positive and offset must be finite.')
    column = 'abs_gc_m3c2_distance' if magnitude else 'signed_gc_m3c2_distance'
    required = ['valid_flag', 'mileage_m', column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    values = frame[required].apply(pd.to_numeric, errors='raise')
    if not values.valid_flag.isin([0, 1]).all():
        raise ValueError('valid_flag must contain only 0 or 1.')
    valid = values.valid_flag.eq(1)
    if not np.isfinite(values.loc[valid, ['mileage_m', column]].to_numpy()).all():
        raise ValueError('Rows marked valid must have finite chainage and displacement.')
    if not valid.any():
        raise ValueError('No valid points to summarize.')
    retained = values.loc[valid].copy()
    if (retained.mileage_m < 0).any():
        raise ValueError('mileage_m must be nonnegative distance from the baseline origin.')
    if magnitude and (retained[column] < 0).any():
        raise ValueError('Absolute distances must be nonnegative.')
    retained['bin'] = np.floor(retained.mileage_m / interval).astype(int)
    # Treat a point exactly on the final interval boundary as part of that last interval.
    upper = float(retained.mileage_m.max())
    last_bin = max(0, int(np.ceil(upper / interval)) - 1)
    retained['bin'] = retained['bin'].clip(upper=last_bin)
    retained['displacement_mm'] = retained[column] * 1000.0
    grouped = retained.groupby('bin').displacement_mm.agg(['count', 'mean', 'median', 'min', 'max'])
    grouped = grouped.reindex(range(last_bin + 1))
    grouped['count'] = grouped['count'].fillna(0).astype(int)
    grouped.insert(0, 'chainage_start_m', chainage_offset + grouped.index * interval)
    grouped.insert(1, 'chainage_end_m', chainage_offset + (grouped.index + 1) * interval)
    grouped = grouped.rename(columns={'count':'valid_count', 'mean':'mean_mm', 'median':'median_mm', 'min':'min_mm', 'max':'max_mm'})
    displacement = retained.displacement_mm.to_numpy()
    stats = {'mode':'absolute' if magnitude else 'signed_inward_positive',
             'total_count':len(frame), 'valid_count':int(valid.sum()),
             'valid_ratio':float(valid.mean()), 'mean_mm':float(displacement.mean()),
             'median_mm':float(np.median(displacement)), 'std_mm':float(displacement.std()),
             'min_mm':float(displacement.min()), 'max_mm':float(displacement.max()),
             'interval_m':interval, 'chainage_offset_m':chainage_offset}
    return grouped.reset_index(drop=True), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='GC-M3C2 pointwise output CSV; distances are in metres.')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--interval', type=float, default=10.0)
    parser.add_argument('--chainage_offset', type=float, default=0.0)
    parser.add_argument('--absolute', action='store_true', help='Summarize magnitudes instead of signed convergence.')
    args = parser.parse_args()
    intervals, stats = summarize(pd.read_csv(args.input), args.interval, args.chainage_offset, args.absolute)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(out / 'longitudinal_statistics.csv', index=False)
    (out / 'full_field_statistics.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
