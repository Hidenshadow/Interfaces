#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy scheduler with multi-point coverage per single acquisition.

Inputs:
  - candidates.csv  (from generate_candidates.py; should include sub_lat/sub_lon/cover_km)
  - test.csv        (task requests)

Output:
  - plan_greedy.csv

Behavior highlights:
  - One acquisition can satisfy multiple tasks if they are within the effective
    footprint radius around (sub_lat, sub_lon) AND their nearest uncovered slot
    is hit by t_mid.
  - Optional satellite allowlist per task (semicolon-separated).
  - Per-satellite minimal slew/settle buffer time.
  - UTC-aware timestamps throughout.

Tunable knobs:
  MERGE_RADIUS_MODE   : 'cover' (use cover_km=swath/2), 'access' (use access_radius_km_used),
                        'hybrid' (cover + off-nadir-dependent expansion, capped by access).
  COVER_SCALE         : scale factor on cover_km (e.g., 1.2 makes footprint 20% wider).
  ACCESS_SCALE        : scale factor on access_radius_km_used.
  HYBRID_ALPHA        : km/deg multiplier for off-nadir expansion in 'hybrid' mode.
  MERGE_SAME_USER_ONLY: restrict multi-cover to same user.
  MERGE_IGNORE_ALLOWLIST: ignore per-task satellite allowlist when multi-covering.
  VERBOSE             : print merge logs to stdout.

Requirements: pandas, numpy
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt

# ---------- Fixed paths ----------
CANDS_CSV = 'candidates.csv'
TASKS_CSV = 'test.csv'
OUT_PLAN  = 'plan_greedy.csv'

# ---------- Constraints / knobs ----------
SLEW_BUFFER_SEC = 60                 # minimal gap between acquisitions per satellite

# ---- Multi-cover knobs ----
MERGE_RADIUS_MODE = 'cover'          # 'cover' | 'access' | 'hybrid'
COVER_SCALE       = 4.0              # scale on cover_km (swath/2)
ACCESS_SCALE      = 1.0              # scale on access_radius_km_used
HYBRID_ALPHA      = 0.6              # km/deg * off_nadir for hybrid expansion (capped by access)
MERGE_SAME_USER_ONLY   = False       # True => only multi-cover within the same user
MERGE_IGNORE_ALLOWLIST = False       # True => ignore task satellite allowlist during multi-cover
VERBOSE                = True        # print [merge] logs

# Fallback if missing in candidates
DEFAULT_ACCESS_KM_PER_DEG = 30.0

# ---------- Helpers ----------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (scalar)."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(np.sqrt(a))

def build_slots(row):
    """Create coverage slots in [window_start_utc, window_end_utc] using required_interval_hours.
       Each slot has center time and +/- half-interval tolerance."""
    step = pd.to_timedelta(float(row['required_interval_hours']), unit='h')
    s = row['window_start_utc']  # UTC-aware
    e = row['window_end_utc']    # UTC-aware
    slots = []
    t = s
    while t <= e:
        slots.append({
            'slot_center': t,
            'tol_minus': step / 2,
            'tol_plus':  step / 2,
            'covered': False
        })
        t = t + step
    return slots

def slot_hit(slot, tmid):
    """Return True if candidate midpoint hits the slot (with tolerance)."""
    return (tmid >= slot['slot_center'] - slot['tol_minus']) and \
           (tmid <= slot['slot_center'] + slot['tol_plus'])

def first_hit_index(slot_list, tmid):
    """Return the first uncovered slot index that the given tmid can satisfy, else None."""
    for i, sl in enumerate(slot_list):
        if (not sl['covered']) and slot_hit(sl, tmid):
            return i
    return None

def parse_tasks():
    """Load and normalize tasks CSV (UTC-aware times)."""
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

    tasks = tasks.dropna(subset=['user','point_id','lat','lon','window_start_utc','window_end_utc','required_interval_hours']).reset_index(drop=True)
    return tasks

def parse_candidates():
    """Load candidates CSV (UTC-aware times)."""
    cand = pd.read_csv(CANDS_CSV)
    for col in ('t_start','t_end','t_mid'):
        if col in cand.columns:
            cand[col] = pd.to_datetime(cand[col], utc=True, errors='coerce')
    cand = cand.dropna(subset=['t_start','t_end','t_mid']).reset_index(drop=True)

    # Fill optional columns if missing
    if 'sub_lat' not in cand.columns: cand['sub_lat'] = np.nan
    if 'sub_lon' not in cand.columns: cand['sub_lon'] = np.nan
    if 'cover_km' not in cand.columns: cand['cover_km'] = np.nan
    if 'cap_swath_km' not in cand.columns: cand['cap_swath_km'] = np.nan
    if 'off_nadir_deg' not in cand.columns: cand['off_nadir_deg'] = np.nan
    if 'access_radius_km_used' not in cand.columns: cand['access_radius_km_used'] = np.nan
    if 'cap_access_km_per_deg' not in cand.columns: cand['cap_access_km_per_deg'] = np.nan

    # Fallback cover_km = cap_swath_km/2 if missing/zero
    def _cover_km_row(r):
        v = r.get('cover_km', np.nan)
        if pd.isna(v) or v <= 0:
            sw = r.get('cap_swath_km', np.nan)
            if pd.notna(sw) and sw > 0:
                return float(sw) / 2.0
            return 0.0
        return float(v)
    cand['cover_km'] = cand.apply(_cover_km_row, axis=1)

    return cand

def effective_radius_km(row):
    """Compute effective multi-cover radius for this candidate according to MERGE_RADIUS_MODE."""
    cover = float(row.get('cover_km', 0.0)) * COVER_SCALE
    access = float(row.get('access_radius_km_used', 0.0)) * ACCESS_SCALE
    if MERGE_RADIUS_MODE == 'cover':
        return max(0.0, cover)
    elif MERGE_RADIUS_MODE == 'access':
        return max(0.0, access)
    elif MERGE_RADIUS_MODE == 'hybrid':
        # cover + off-nadir-dependent expansion (bounded by access)
        off = float(row.get('off_nadir_deg', 0.0))
        km_per_deg = float(row.get('cap_access_km_per_deg', np.nan))
        if not np.isfinite(km_per_deg) or km_per_deg <= 0:
            km_per_deg = DEFAULT_ACCESS_KM_PER_DEG
        expanded = cover + HYBRID_ALPHA * off * km_per_deg
        return max(0.0, min(access if access > 0 else expanded, expanded))
    else:
        return max(0.0, cover)

def main():
    # ---------- Load inputs ----------
    tasks = parse_tasks()
    cand  = parse_candidates()

    # ---------- Build slots per (user, point_id) ----------
    slots = {}   # (user, point_id) -> [slot dicts]
    for _, r in tasks.iterrows():
        key = (r['user'], r['point_id'])
        slots[key] = build_slots(r)

    # ---------- Satellite allow-list per task (optional) ----------
    allow = {}   # (user, point_id) -> set() or empty set meaning "all allowed"
    if 'satellites' in tasks.columns:
        for _, r in tasks.iterrows():
            s = str(r.get('satellites') or '').strip()
            aset = set(x.strip() for x in s.split(';') if x.strip()) if s else set()
            allow[(r['user'], r['point_id'])] = aset
    else:
        for _, r in tasks.iterrows():
            allow[(r['user'], r['point_id'])] = set()

    # ---------- Geolocation map ----------
    geo = {(r['user'], r['point_id']): (float(r['lat']), float(r['lon'])) for _, r in tasks.iterrows()}

    # ---------- Greedy order: earliest first, tie by smaller off-nadir ----------
    cand = cand.sort_values(['t_mid', 'off_nadir_deg'], ascending=[True, True]).reset_index(drop=True)

    # ---------- State ----------
    last_end = {}          # satellite -> last occupied end time
    chosen_rows = []       # expanded rows (one line per covered (user,point))

    # ---------- Loop over candidates ----------
    for _, c in cand.iterrows():
        sat = c['satellite']
        t_start, t_mid, t_end = c['t_start'], c['t_mid'], c['t_end']

        # Slew buffer per satellite
        if sat in last_end and t_start < last_end[sat] + pd.Timedelta(seconds=SLEW_BUFFER_SEC):
            continue

        # Candidate footprint center and radius
        sub_lat = float(c.get('sub_lat', np.nan))
        sub_lon = float(c.get('sub_lon', np.nan))
        eff_radius = effective_radius_km(c)
        has_footprint = (not np.isnan(sub_lat)) and (not np.isnan(sub_lon)) and (eff_radius > 0.0)

        # Build a set of (user, point_id) covered by this acquisition
        covered_this_pick = set()

        # 1) Always try the candidate's own key first
        key0 = (c['user'], c['point_id'])
        if key0 in slots:
            # Check satellite allow list for the main point
            if not allow[key0] or sat in allow[key0]:
                idx = first_hit_index(slots[key0], t_mid)
                if idx is not None:
                    slots[key0][idx]['covered'] = True
                    covered_this_pick.add(key0)

        # 2) Multi-cover: try to cover neighboring tasks inside eff_radius
        if has_footprint and eff_radius > 0.0:
            for key, slist in slots.items():
                if key in covered_this_pick:
                    continue
                if MERGE_SAME_USER_ONLY and key[0] != c['user']:
                    continue
                if all(sl['covered'] for sl in slist):
                    continue
                if (not MERGE_IGNORE_ALLOWLIST) and allow[key] and (sat not in allow[key]):
                    continue
                lat2, lon2 = geo[key]
                if haversine_km(sub_lat, sub_lon, lat2, lon2) > eff_radius:
                    continue
                idx2 = first_hit_index(slist, t_mid)
                if idx2 is not None:
                    slist[idx2]['covered'] = True
                    covered_this_pick.add(key)

        # If nothing is covered by this acquisition, skip it
        if not covered_this_pick:
            continue

        # This acquisition is taken; block the satellite for a buffer after t_end
        last_end[sat] = t_end

        # (Optional) log how many points were covered together
        if VERBOSE:
            extra = len(covered_this_pick) - 1
            if extra > 0:
                print(f"[merge] {sat} @ {t_mid} covered {len(covered_this_pick)} point(s), +{extra} extra")

        # Expand to one CSV row per covered (user, point)
        for (uu, pp) in covered_this_pick:
            row = {
                'user': uu,
                'point_id': pp,
                'satellite': sat,
                't_start': t_start,
                't_mid':   t_mid,
                't_end':   t_end,
            }
            # optional carry-overs if present (helpful for analysis/viz)
            for k in ('off_nadir_deg','sub_lat','sub_lon','cover_km','track_dist_km_mid',
                      'cap_swath_km','cap_access_km_per_deg','access_radius_km_used'):
                if k in c:
                    row[k] = c[k]
            # also write geometry of the point
            latp, lonp = geo[(uu, pp)]
            row['lat'] = latp
            row['lon'] = lonp
            chosen_rows.append(row)

    # ---------- Save plan ----------
    sel = pd.DataFrame(chosen_rows)
    sel.to_csv(OUT_PLAN, index=False)

    # ---------- Coverage summary (simple) ----------
    # compute average fraction of slots covered per (user, point_id)
    covs = []
    for key, slist in slots.items():
        if len(slist) == 0:
            covs.append(0.0)
        else:
            covs.append(sum(1 for sl in slist if sl['covered']) / len(slist))
    avg_cov = round(sum(covs) / len(covs), 4) if covs else 0.0

    print(f"selected: {len(sel)}  avg_slot_coverage={avg_cov}")
    print(f"-> {OUT_PLAN}")

if __name__ == '__main__':
    import time, csv
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 3)
    print(f"[METRIC] plan_greedy_runtime_sec={elapsed}")
    try:
        with open('metrics.csv', 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['plan_greedy', elapsed])
    except Exception:
        pass
