#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate candidate observation windows from:
  - test.csv                (task requests)
  - satellites_10.tle       (TLE 3-line blocks)
  - sat_capabilities.json   (per-satellite capabilities)

Output:
  - candidates.csv

Each candidate row represents one feasible acquisition window for (user, point_id, satellite),
with t_start/t_mid/t_end in UTC, and geometry fields for downstream planning:

Columns:
  user, point_id, lat, lon, satellite,
  t_start, t_end, t_mid, duration_s, off_nadir_deg,
  req_interval_h,
  cap_swath_km, cap_max_off_deg, cap_access_km_per_deg, access_radius_km_used,
  sub_lat, sub_lon, cover_km, track_dist_km_mid

Key notes:
- We approximate visibility by checking great-circle distance from ground track (sub-satellite point)
  to the task point; distance threshold = max(swath/2, access_km_per_deg * max_off_nadir).
- cover_km is set to swath/2 (used later for multi-point coverage accounting).
- off_nadir_deg at t_mid is estimated as distance_mid / access_km_per_deg (clamped by max_off).
- All times are timezone-aware UTC and exported as ISO8601 with trailing 'Z'.

Dependencies:
  pip install pandas numpy skyfield sgp4
"""

import json
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite

# ---------------- Fixed I/O paths ----------------
TASKS_CSV = 'test.csv'
TLE_PATH = 'satellites_10.tle'
CAPS_PATH = 'sat_capabilities.json'
OUT_CANDIDATES = 'candidates.csv'

# ---------------- Tunable parameters ----------------
DT_SEC = 30                 # propagation step (s); use 10–20s for narrower swaths / higher fidelity
MIN_DWELL_SEC = 10          # minimal candidate duration (s)
MERGE_GAP_SEC = 90          # merge gap between True samples to form one window (s)
SWATH_SCALE = 1.0           # global multiplier for swath_km (e.g., 1.2 makes footprint 20% wider)
DEFAULT_ACCESS_KM_PER_DEG = 30.0  # fallback if not provided in capabilities
DEBUG = True                # print diagnostics

# ---------------- Helpers ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers (supports numpy arrays elementwise)."""
    R = 6371.0
    dlat = np.radians(np.array(lat2) - np.array(lat1))
    dlon = np.radians(np.array(lon2) - np.array(lon1))
    a = np.sin(dlat / 2.0) ** 2 + \
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def to_utc(ts):
    """Return a timezone-aware datetime in UTC."""
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return ts.tz_convert('UTC').to_pydatetime()
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    raise TypeError(f"Unsupported datetime type: {type(ts)}")

def ts_range(ts_obj, start_dt, end_dt, dt_sec):
    """Build a Skyfield timescale array from start to end (inclusive) with step dt_sec."""
    total = max(0.0, (end_dt - start_dt).total_seconds())
    n = int(total // dt_sec) + 1
    if n <= 0:
        return ts_obj.utc([]), 0
    pts = [start_dt + timedelta(seconds=i * dt_sec) for i in range(n)]
    return ts_obj.utc(pts), n

def contiguous_true_segments(mask, dt_sec, merge_gap_sec):
    """Group True samples into contiguous (or near-contiguous) segments based on time gap."""
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
    """Load TLEs into dict name->EarthSatellite (optionally filter by whitelist)."""
    ts = load.timescale()
    sats = {}
    lines = [line.strip() for line in Path(tle_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for i in range(0, len(lines), 3):
        name, l1, l2 = lines[i], lines[i+1], lines[i+2]
        if whitelist and name not in whitelist:
            continue
        sats[name] = EarthSatellite(l1, l2, name, ts)
    return sats, ts

# ---------------- Main ----------------
def main():
    # -------- 1) Load tasks --------
    if not Path(TASKS_CSV).exists():
        raise FileNotFoundError(f"Missing {TASKS_CSV}")
    tasks = pd.read_csv(TASKS_CSV)

    # Flexible column aliases
    ren = {}
    alias = {
        'user': ['user','User','username'],
        'point_id': ['point_id','point','PointID','id'],
        'lat': ['lat','latitude','lat_center','Lat','Latitude'],
        'lon': ['lon','longitude','lon_center','Lon','Longitude'],
        'window_start_utc': ['window_start_utc','start','window_start','WindowStartUTC','window_start_UTC'],
        'window_end_utc':   ['window_end_utc','end','window_end','WindowEndUTC','window_end_UTC'],
        'required_interval_hours': ['required_interval_hours','frequency_hours','interval_hours'],
        'satellites': ['satellites','satellite','satellites_used','Satellites'],
    }
    for std, names in alias.items():
        for k in names:
            if k in tasks.columns:
                ren[k] = std
                break
    tasks = tasks.rename(columns=ren)

    # Types
    tasks['window_start_utc'] = pd.to_datetime(tasks['window_start_utc'], utc=True, errors='coerce')
    tasks['window_end_utc']   = pd.to_datetime(tasks['window_end_utc'],   utc=True, errors='coerce')
    tasks['required_interval_hours'] = pd.to_numeric(tasks.get('required_interval_hours', np.nan), errors='coerce')

    # Keep essentials
    tasks = tasks.dropna(subset=['user','point_id','lat','lon','window_start_utc','window_end_utc']).reset_index(drop=True)

    # Requested satellites allowlist (from tasks 'satellites' column; union of all)
    requested = set()
    if 'satellites' in tasks.columns:
        for s in tasks['satellites'].fillna(''):
            requested.update(t.strip() for t in str(s).split(';') if t.strip())

    # -------- 2) Load capabilities --------
    if not Path(CAPS_PATH).exists():
        raise FileNotFoundError(f"Missing {CAPS_PATH}")
    with open(CAPS_PATH, 'r', encoding='utf-8') as f:
        CAPS = json.load(f)

    # -------- 3) Load TLE --------
    if not Path(TLE_PATH).exists():
        raise FileNotFoundError(f"Missing {TLE_PATH}")
    sats, ts = load_tle_map(TLE_PATH, None if len(requested) == 0 else requested)

    if DEBUG:
        print(f"[diag] tasks={len(tasks)} users={tasks['user'].nunique()} points={tasks['point_id'].nunique()}")
        if len(tasks):
            print(f"[diag] window: {tasks['window_start_utc'].min()} -> {tasks['window_end_utc'].max()}")
        print(f"[diag] requested sats: {sorted(requested) if requested else '(all)'}")
        print(f"[diag] loaded sats: {len(sats)} -> {list(sats.keys())[:8]}{'...' if len(sats)>8 else ''}")

    # Group identical task keys
    grouped = tasks.groupby(['user','point_id','lat','lon','window_start_utc','window_end_utc'], dropna=False)

    out_rows = []

    # -------- 4) For each task group & satellite, build candidates --------
    for (user, pid, plat, plon, wstart, wend), gdf in grouped:
        req_int_h = float(gdf['required_interval_hours'].iloc[0]) if 'required_interval_hours' in gdf.columns and pd.notna(gdf['required_interval_hours'].iloc[0]) else np.nan
        t0 = to_utc(wstart)
        t1 = to_utc(wend)
        plat = float(plat)
        plon = float(plon)

        # If task specified satellites, use intersection; else use all loaded sats
        task_allow_raw = ""
        if 'satellites' in gdf.columns and pd.notna(gdf['satellites'].iloc[0]):
            task_allow_raw = str(gdf['satellites'].iloc[0]).strip()
        task_allow = set(x.strip() for x in task_allow_raw.split(';') if x.strip())
        sat_iter = sats.items() if not task_allow else ((n, s) for (n, s) in sats.items() if n in task_allow)

        for sat_name, sat in sat_iter:
            cap = CAPS.get(sat_name, {})
            swath_km = float(cap.get('swath_km', 80.0)) * SWATH_SCALE
            max_off  = float(cap.get('max_off_nadir_deg', 30.0))
            km_per_deg = float(cap.get('access_km_per_deg', DEFAULT_ACCESS_KM_PER_DEG))

            # Effective access radius (km): allow either swath/2 or off-nadir reach
            access_km = max(swath_km / 2.0, km_per_deg * max_off)
            cover_km  = swath_km / 2.0

            # Build time vector
            times, n = ts_range(ts, t0, t1, DT_SEC)
            if n == 0:
                continue

            # Sub-satellite ground track
            geoc = sat.at(times).subpoint()
            slat = geoc.latitude.degrees
            slon = geoc.longitude.degrees

            # Distance array to the task point
            dist_km = haversine_km(slat, slon, plat, plon)
            hit = dist_km <= access_km
            if not np.any(hit):
                if DEBUG:
                    md = float(np.min(dist_km)) if dist_km.size else float('inf')
                    print(f"[diag] {sat_name}->{pid}: min_dist={md:.1f}km access={access_km:.1f}km -> no hit")
                continue

            # Merge contiguous hit-samples into windows
            segs = contiguous_true_segments(hit, DT_SEC, MERGE_GAP_SEC)
            for si, ei in segs:
                t_start = t0 + timedelta(seconds=int(si * DT_SEC))
                t_end   = t0 + timedelta(seconds=int(ei * DT_SEC))
                dur     = (t_end - t_start).total_seconds()
                if dur < MIN_DWELL_SEC:
                    continue

                mid_idx = (si + ei) // 2
                t_mid   = t0 + timedelta(seconds=int(mid_idx * DT_SEC))

                # Estimate off-nadir at mid by distance / km_per_deg; clamp by max_off
                d_mid = float(dist_km[mid_idx]) if mid_idx < dist_km.size else float('nan')
                off_deg_est = min(max_off, d_mid / km_per_deg if km_per_deg > 0 else max_off)

                # Sub-satellite lat/lon at mid
                sub_lat_mid = float(slat[mid_idx]) if mid_idx < len(slat) else float('nan')
                sub_lon_mid = float(slon[mid_idx]) if mid_idx < len(slon) else float('nan')

                out_rows.append({
                    'user': user,
                    'point_id': pid,
                    'lat': plat,
                    'lon': plon,
                    'satellite': sat_name,

                    't_start': t_start.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    't_end':   t_end.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    't_mid':   t_mid.astimezone(timezone.utc).isoformat().replace('+00:00','Z'),
                    'duration_s': int(dur),
                    'off_nadir_deg': round(off_deg_est, 2),

                    'req_interval_h': req_int_h,

                    'cap_swath_km': round(swath_km, 2),
                    'cap_max_off_deg': round(max_off, 2),
                    'cap_access_km_per_deg': round(km_per_deg, 2),
                    'access_radius_km_used': round(access_km, 2),

                    'sub_lat': round(sub_lat_mid, 6),
                    'sub_lon': round(sub_lon_mid, 6),
                    'cover_km': round(cover_km, 2),

                    'track_dist_km_mid': round(d_mid, 2),
                })

            if DEBUG:
                md = float(np.min(dist_km)) if dist_km.size else float('inf')
                print(f"[diag] {sat_name}->{pid}: min_dist={md:.1f}km | segs={len(segs)}")

    # -------- 5) Write candidates.csv (ordered columns) --------
    cols = [
        'user','point_id','lat','lon','satellite',
        't_start','t_end','t_mid','duration_s','off_nadir_deg',
        'req_interval_h',
        'cap_swath_km','cap_max_off_deg','cap_access_km_per_deg','access_radius_km_used',
        'sub_lat','sub_lon','cover_km','track_dist_km_mid'
    ]
    cand_df = pd.DataFrame(out_rows)
    # Ensure all columns present (fill missing with NaN)
    for c in cols:
        if c not in cand_df.columns:
            cand_df[c] = np.nan
    cand_df = cand_df[cols].sort_values(['t_mid','satellite','point_id']).reset_index(drop=True)
    cand_df.to_csv(OUT_CANDIDATES, index=False)

    print(f"candidates: {len(cand_df)} -> {OUT_CANDIDATES}")

if __name__ == '__main__':
    import time, csv
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 3)
    print(f"[METRIC] generate_candidates_runtime_sec={elapsed}")
    # Append a simple metric row
    try:
        with open('metrics.csv', 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['generate_candidates', elapsed])
    except Exception:
        pass
