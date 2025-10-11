# plan_greedy.py
# Greedy scheduling (earliest-first). Fixed I/O paths.
# Requires: pip install pandas numpy

import pandas as pd

# ---------- Fixed paths ----------
CANDS_CSV = 'candidates.csv'
TASKS_CSV = 'test.csv'
OUT_PLAN  = 'plan_greedy.csv'

# Fixed slewing buffer (simplified). You can later switch to per-satellite dynamic buffer.
SLEW_BUFFER_SEC = 60

def build_slots(row):
    """Create coverage slots in the window using required_interval_hours.
       Slots have center time and +/- half-interval tolerance."""
    step = pd.to_timedelta(float(row['required_interval_hours']), unit='h')
    s = row['window_start_utc']  # UTC-aware
    e = row['window_end_utc']    # UTC-aware
    slots = []
    t = s
    while t <= e:
        slots.append({
            'slot_center': t,
            'tol_minus': step / 2,
            'tol_plus':  step / 2
        })
        t = t + step
    return slots

def slot_hit(slot, tmid):
    return (tmid >= slot['slot_center'] - slot['tol_minus']) and \
           (tmid <= slot['slot_center'] + slot['tol_plus'])

def main():
    # Load candidates (UTC-aware)
    cand = pd.read_csv(CANDS_CSV)
    cand['t_start'] = pd.to_datetime(cand['t_start'], utc=True, errors='coerce')
    cand['t_end']   = pd.to_datetime(cand['t_end'],   utc=True, errors='coerce')
    cand['t_mid']   = pd.to_datetime(cand['t_mid'],   utc=True, errors='coerce')
    cand = cand.dropna(subset=['t_start','t_end','t_mid']).reset_index(drop=True)

    # Load tasks (UTC-aware)
    tasks = pd.read_csv(TASKS_CSV)
    tasks['window_start_utc'] = pd.to_datetime(tasks['window_start_utc'], utc=True, errors='coerce')
    tasks['window_end_utc']   = pd.to_datetime(tasks['window_end_utc'],   utc=True, errors='coerce')
    tasks = tasks.dropna(subset=['window_start_utc','window_end_utc']).reset_index(drop=True)

    # Build slots per (user, point)
    slots = {}
    for _, r in tasks.iterrows():
        key = (r['user'], r['point_id'])
        slots[key] = build_slots(r)

    # Chronological greedy: earliest first, tie-break by smaller off-nadir
    cand = cand.sort_values(['t_mid', 'off_nadir_deg'], ascending=[True, True]).reset_index(drop=True)

    last_end = {}           # per-satellite last occupied end (UTC-aware)
    selected = []
    covered = {k: [False]*len(v) for k, v in slots.items()}

    for _, c in cand.iterrows():
        sat = c['satellite']

        # No-overlap with fixed buffer, in chronological direction
        if sat in last_end and c['t_start'] < last_end[sat] + pd.Timedelta(seconds=SLEW_BUFFER_SEC):
            continue

        key = (c['user'], c['point_id'])
        if key not in slots:
            continue

        # find the first uncovered slot this candidate can satisfy
        for i, sl in enumerate(slots[key]):
            if not covered[key][i] and slot_hit(sl, c['t_mid']):
                covered[key][i] = True
                selected.append(c)
                last_end[sat] = c['t_end']
                break

    sel = pd.DataFrame(selected)
    sel.to_csv(OUT_PLAN, index=False)

    # Coverage summary
    coverage = {k: (sum(v)/len(v) if len(v) else 0.0) for k, v in covered.items()}
    avg_cov = round(sum(coverage.values()) / len(coverage) if coverage else 0.0, 3)
    print('selected:', len(sel), 'coverage(avg)=', avg_cov)
    print('->', OUT_PLAN)

if __name__ == '__main__':
    import time, csv
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 3)
    print(f"[METRIC] plan_greedy_runtime_sec={elapsed}")
    with open('metrics.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['plan_greedy', elapsed])
