# generate_candidates.py
# Purpose: Build candidate observation windows using per-satellite capabilities.
# Fixed I/O:
#   - tasks CSV: test.csv
#   - TLE: satellites_10.tle
#   - caps JSON: sat_capabilities.json
#   - output candidates: candidates.csv
# Requires: pip install pandas numpy skyfield sgp4

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite

# ------------- Fixed paths -------------
TASKS_CSV = 'test.csv'
TLE_PATH = 'satellites_10.tle'
CAPS_PATH = 'sat_capabilities.json'
OUT_CANDIDATES = 'candidates.csv'

# ------------- Tunables (edit here as needed) -------------
DT_SEC = 30                 # propagation step (s)
MIN_DWELL_SEC = 10          # minimal candidate duration (s)
MERGE_GAP_SEC = 90          # merge gap threshold between True samples (s)
OFF_PER_DEG_DEFAULT = 30.0  # fallback km/deg to translate off-nadir angle to ground distance
SWATH_SCALE = 1.0           # multiplier for swath_km
DEBUG = True                # print diagnostics

# ---------------- Helpers ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def to_utc(ts):
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.tz_convert('UTC').to_pydatetime()
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    raise TypeError(f"Unsupported datetime type: {type(ts)}")

def as_timescale(ts_obj, start_dt, end_dt, dt_sec):
    N = max(1, int((end_dt - start_dt).total_seconds() // dt_sec) + 1)
    pts = [start_dt + timedelta(seconds=i*dt_sec) for i in range(N)]
    return ts_obj.utc(pts), N

def contiguous_true_segments(mask, dt_sec, merge_gap_sec):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    segs = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if (k - prev) * dt_sec > merge_gap_sec:
            segs.append((start, prev))
            start = k
        prev = k
    segs.append((start, prev))
    return segs

def load_tle_map(tle_path, whitelist=None):
    ts = load.timescale()
    sats = {}
    with open(tle_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 3):
        name, l1, l2 = lines[i], lines[i+1], lines[i+2]
        if whitelist and name not in whitelist:
            continue
        sats[name] = EarthSatellite(l1, l2, name, ts)
    return sats, ts

# ---------------- Main ----------------
def main():
    # Load tasks (UTC-aware)
    tasks = pd.read_csv(TASKS_CSV)
    tasks['window_start_utc'] = pd.to_datetime(tasks['window_start_utc'], utc=True, errors='coerce')
    tasks['window_end_utc']   = pd.to_datetime(tasks['window_end_utc'],   utc=True, errors='coerce')
    tasks = tasks.dropna(subset=['window_start_utc','window_end_utc']).reset_index(drop=True)

    # Requested satellites (semicolon separated)
    requested = set()
    for s in tasks['satellites'].fillna(''):
        requested.update(t.strip() for t in str(s).split(';') if t.strip())

    # Load capabilities
    with open(CAPS_PATH, 'r', encoding='utf-8') as f:
        CAPS = json.load(f)

    # Load TLE (filtered by requested sats if provided)
    sats, ts = load_tle_map(TLE_PATH, None if len(requested)==0 else requested)

    if DEBUG:
        print(f"Tasks: {len(tasks)} rows | Users: {tasks['user'].nunique()} | Points: {tasks['point_id'].nunique()}")
        print("Window:", tasks['window_start_utc'].min(), "->", tasks['window_end_utc'].max())
        print("Requested sats:", sorted(requested) if requested else "(all from TLE)")
        print("Loaded sats:", len(sats), list(sats.keys())[:8], "..." if len(sats)>8 else "")

    grouped = tasks.groupby(['user','point_id','lat','lon','window_start_utc','window_end_utc'])
    out_rows = []

    for (user, pid, plat, plon, wstart, wend), g in grouped:
        req_int_h = float(g['required_interval_hours'].iloc[0])
        t0 = to_utc(wstart); t1 = to_utc(wend)

        for sat_name, sat in sats.items():
            cap = CAPS.get(sat_name, {})
            swath_km = float(cap.get('swath_km', 100.0)) * SWATH_SCALE
            max_off  = float(cap.get('max_off_nadir_deg', 30.0))
            access_km_per_deg = float(cap.get('access_km_per_deg', OFF_PER_DEG_DEFAULT))
            # Access radius (km): pragmatic approximation
            access_km = max(swath_km/2.0, access_km_per_deg * max_off)

            times, N = as_timescale(ts, t0, t1, DT_SEC)
            if N == 0:
                continue

            geoc = sat.at(times).subpoint()
            slat = geoc.latitude.degrees
            slon = geoc.longitude.degrees

            dist = haversine_km(slat, slon, float(plat), float(plon))
            hit = dist <= access_km
            if not np.any(hit):
                if DEBUG:
                    md = float(np.min(dist)) if dist.size else float('inf')
                    print(f"[diag] {sat_name}->{pid}: min_dist={md:.1f}km access={access_km:.1f}km -> no hit")
                continue

            segs = contiguous_true_segments(hit, DT_SEC, MERGE_GAP_SEC)
            for si, ei in segs:
                t_start = t0 + timedelta(seconds=int(si*DT_SEC))
                t_end   = t0 + timedelta(seconds=int(ei*DT_SEC))
                dur     = (t_end - t_start).total_seconds()
                if dur < MIN_DWELL_SEC:
                    continue
                mid_idx = (si + ei) // 2
                mid_t   = t0 + timedelta(seconds=int(mid_idx*DT_SEC))
                # Approximate off-nadir from ground distance at mid
                off_deg_est = min(90.0, float(dist[mid_idx]) / access_km_per_deg)
                if off_deg_est > max_off:
                    continue

                out_rows.append({
                    'user': user,
                    'point_id': pid,
                    'lat': float(plat),
                    'lon': float(plon),
                    'satellite': sat_name,
                    't_start': t_start.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    't_end':   t_end.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    't_mid':   mid_t.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    'duration_s': int(dur),
                    'off_nadir_deg': round(off_deg_est, 2),
                    'req_interval_h': req_int_h,
                    'cap_swath_km': swath_km,
                    'cap_max_off_deg': max_off,
                    'cap_access_km_per_deg': access_km_per_deg,
                    'access_radius_km_used': round(access_km,1),
                })

            if DEBUG:
                md = float(np.min(dist)) if dist.size else float('inf')
                print(f"[diag] {sat_name}->{pid}: min_dist={md:.1f}km | segs={len(segs)}")

    cand_df = pd.DataFrame(out_rows)
    cand_df.to_csv(OUT_CANDIDATES, index=False)
    print(f"candidates: {len(cand_df)} -> {OUT_CANDIDATES}")

if __name__ == '__main__':
    import time, csv
    t0 = time.perf_counter()
    main()  # 保持你当前实现
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 3)
    print(f"[METRIC] generate_candidates_runtime_sec={elapsed}")
    # 追加记录到 metrics.csv
    with open('metrics.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['generate_candidates', elapsed])
