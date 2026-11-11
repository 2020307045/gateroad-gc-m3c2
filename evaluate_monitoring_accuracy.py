"""Compare matched field/reference and algorithm displacements supplied in mm."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(frame):
    required = ['chainage_m', 'location', 'method', 'measured_mm', 'estimated_mm']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    result = frame[required].copy()
    if result.empty or result.isna().any().any():
        raise ValueError('Input must contain complete matched measurements.')
    for column in ['location', 'method']:
        result[column] = result[column].astype(str).str.strip()
        if result[column].eq('').any():
            raise ValueError(f'{column} must not be empty.')
    for column in ['chainage_m', 'measured_mm', 'estimated_mm']:
        result[column] = pd.to_numeric(result[column], errors='raise')
    if not np.isfinite(result[['chainage_m', 'measured_mm', 'estimated_mm']].to_numpy()).all():
        raise ValueError('Measurement columns must contain finite numbers.')
    keys = ['chainage_m', 'location']
    if result.duplicated(keys+['method']).any():
        raise ValueError('Duplicate location/method rows; evaluate one monitoring interval at a time.')
    if result.groupby(keys).measured_mm.nunique().gt(1).any():
        raise ValueError('Methods must use the same measured reference at each location.')
    locations = [set(zip(group.chainage_m, group.location)) for _, group in result.groupby('method')]
    if any(items != locations[0] for items in locations[1:]):
        raise ValueError('Methods must be evaluated at the same set of monitoring locations.')
    result['signed_error_mm'] = result.estimated_mm - result.measured_mm
    result['absolute_error_mm'] = result.signed_error_mm.abs()
    result['relative_error_pct'] = 100 * result.absolute_error_mm / result.measured_mm.abs().replace(0, np.nan)
    rows = []
    for method, group in result.groupby('method', sort=False):
        rows.append({'method':method, 'n_locations':len(group),
                     'mae_mm':group.absolute_error_mm.mean(),
                     'rmse_mm':float(np.sqrt(np.mean(group.signed_error_mm**2))),
                     'bias_mm':group.signed_error_mm.mean(),
                     'max_absolute_error_mm':group.absolute_error_mm.max()})
    return result, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    detailed, summary = evaluate(pd.read_csv(args.input))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out / 'monitoring_errors.csv', index=False)
    summary.to_csv(out / 'accuracy_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
