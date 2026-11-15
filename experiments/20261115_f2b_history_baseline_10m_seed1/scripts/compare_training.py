"""Compare the uninterrupted baseline prefix without modifying either run."""
from pathlib import Path
import csv
import datetime
import json
import math
import os
import time


def read_csv(path):
    if not path.is_file():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def compare_rows(reference_rows, candidate_rows, step_key):
    reference = {int(float(r[step_key])): r for r in reference_rows if r.get(step_key)}
    candidate = {int(float(r[step_key])): r for r in candidate_rows if r.get(step_key)}
    steps = sorted(reference.keys() & candidate.keys())
    differences = []
    compared = 0
    different_rows = 0
    missing_keys = set()
    for step in steps:
        left, right = reference[step], candidate[step]
        missing_keys.update(left.keys() ^ right.keys())
        mismatch = False
        for key in sorted(left.keys() & right.keys()):
            if key.startswith('time/') or key == step_key:
                continue
            a, b = left[key], right[key]
            if a == '' and b == '':
                continue
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                pass
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                continue
            compared += 1
            if a != b:
                mismatch = True
                if len(differences) < 15:
                    differences.append({'step': step, 'metric': key, 'reference': a, 'reproduction': b})
        different_rows += int(mismatch)
    return {'common_rows': len(steps), 'last_common_step': steps[-1] if steps else None,
            'values_compared': compared, 'different_rows': different_rows,
            'all_compared_values_equal': bool(steps) and different_rows == 0 and not missing_keys,
            'missing_metric_columns': sorted(missing_keys), 'first_differences': differences}


def main():
    exp = Path(os.environ['REPRO_EXPERIMENT_DIR'])
    reference = Path(os.environ['REPRO_BASELINE_DIR'])
    while True:
        record = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'reference': str(reference), 'comparison_boundary': 'Original uninterrupted segment only; stop at original interruption',
                  'equality': 'Exact parsed numeric equality; wall-clock metrics excluded'}
        for key, relative, step_key in [
            ('ppo', 'logs/ppo/progress.csv', 'time/total_timesteps'),
            ('reward_components', 'logs/reward_components.csv', 'timesteps'),
        ]:
            record[key] = compare_rows(read_csv(reference/'training'/relative),
                                       read_csv(exp/'training'/relative), step_key)
        temporary = exp/'comparison.json.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(exp/'comparison.json')
        status = exp/'status.json'
        if status.is_file() and json.loads(status.read_text()).get('phase') in ('COMPLETED', 'FAILED'):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
