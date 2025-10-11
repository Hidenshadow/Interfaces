#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MILP planner: choose a subset of candidate acquisitions to best cover users'
time-slotted task requests, allowing one acquisition to cover multiple nearby
points (within an effective footprint radius) at the same time.

Inputs (fixed names):
  - test.csv          : task requests (user, point_id, lat, lon, window_start_utc, window_end_utc,
                        required_interval_hours, optional 'satellites' allowlist)
  - candidates.csv    : candidate acquisitions (from generate_candidates.py), should include:
                        user, point_id, lat, lon, satellite, t_start, t_end, t_mid,
                        off_nadir_deg, sub_lat, sub_lon, cover_km, access_radius_km_used,
                        cap_swath_km, cap_access_km_per_deg, cap_max_off_deg, ...

Output:
  - plan_milp.csv     : one row per covered (user, point_id) using a selected acquisition
                        (i.e., expand a chosen acquisition to multiple covered points)

Model sketch:
  - For each candidate j: binary y_j (select acquisition or not)
  - For each feasible (candidate j, task i, slot k_i near t_mid): binary z_{j,i}
    (we map each (j,i) to the single nearest-hit slot; hence only one z per (j,i))
  - Slot uniqueness: each task slot can be covered at most once
  - Link: z_{j,i} <= y_j
  - Per-satellite deconfliction: y_j + y_k <= 1 when time windows overlap (with slew buffer)
  - Objective: maximize sum(w_{j,i} * z_{j,i}) - ALPHA * sum(y_j)
               (optional off-nadir penalty in w_{j,i})

Tunable knobs below control footprint mode, tolerances, and solver time limit.
"""

import math
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import pulp  # MILP solver (CBC)

# ----------------------- Fixed I/O -----------------------
TASKS_CSV = "test.csv"
CANDS_CSV = "candidates.csv"
OUT_PLAN  = "plan_milp.csv"

# ----------------------- Knobs ---------------------------
# Slot building: slot_center every 'interval' hours, tolerance = interval/2 (inclusive)
# If you prefer rolling-window semantics, widen the tolerance slightly (< interval).
TOL_FACTOR = 0.5                       # slot tolerance factor (±interval * TOL_FACTOR)

# Footprint for multi-cover: select one of:
#   'cover'  -> use cover_km (= swath/2)
#   'access' -> use access_radius_km_used (usually larger, good for quick validation)
#   'hybrid' -> cover_km + HYBRID_ALPHA * off_nadir_deg * cap_access_km_per_deg, capped by access
MERGE_RADIUS_MODE = "cover"
COVER_SCALE  = 2.0                      # scale cover_km
ACCESS_SCALE = 1.0                      # scale access_radius_km_used
HYBRID_ALPHA = 0.6                      # km/deg multiplier in 'hybrid' mode
DEFAULT_KM_PER_DEG = 30.0               # fallback if cap_access_km_per_deg missing

# Limit multi-cover to same user?
MERGE_SAME_USER_ONLY = False

# Ignore per-task satellites allowlist during multi-cover?
MERGE_IGNORE_ALLOWLIST = False

# Per-satellite minimal gap to avoid back-to-back conflicts (slew/settle)
SLEW_BUFFER_SEC = 60

# Objective weights
ALPHA_SELECT = 1e-3                     # small penalty per selected acquisition (sum y_j)
QUALITY_GAMMA = 0.0                     # weight in [0..1], penalize off-nadir: w = 1 - gamma * (off / max_off)

# Solver
TIME_LIMIT_SEC = 300                    # set to None for no limit
VERBOSE = True

# ----------------------- Helpers -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2.0)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2.0)**2
    return 2 * R * asin(math.sqrt(a))

def parse_tasks():
    if not Path(TASKS_CSV).exists():
        raise FileNotFoundError(f"Missing {TASKS_CSV}")
    df = pd.read_csv(TASKS_CSV)

    # Robust column remap
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
            if k in df.columns:
                ren[k] = std
                break
    df = df.rename(columns=ren)

    # Types
    df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True, errors='coerce')
    df['window_end_utc']   = pd.to_datetime(df['window_end_utc'],   utc=True, errors='coerce')
    df['required_interval_hours'] = pd.to_numeric(df.get('required_interval_hours', np.nan), errors='coerce')

    df = df.dropna(subset=['user','point_id','lat','lon','window_start_utc','window_end_utc','required_interval_hours']).reset_index(drop=True)

    # Collapse to unique (user, point_id)
    g = (df.sort_values(['user','point_id'])
           .groupby(['user','point_id'], as_index=False)
           .first())
    return g

def parse_candidates():
    if not Path(CANDS_CSV).exists():
        raise FileNotFoundError(f"Missing {CANDS_CSV}")
    cand = pd.read_csv(CANDS_CSV)

    for col in ('t_start','t_end','t_mid'):
        if col in cand.columns:
            cand[col] = pd.to_datetime(cand[col], utc=True, errors='coerce')
    cand = cand.dropna(subset=['t_start','t_end','t_mid']).reset_index(drop=True)

    # Ensure optional columns exist
    for c in ('sub_lat','sub_lon','cover_km','access_radius_km_used',
              'cap_swath_km','cap_access_km_per_deg','cap_max_off_deg','off_nadir_deg'):
        if c not in cand.columns: cand[c] = np.nan

    # Fallback cover_km from swath/2
    def _cover_row(r):
        v = r.get('cover_km', np.nan)
        if pd.isna(v) or v <= 0:
            sw = r.get('cap_swath_km', np.nan)
            if pd.notna(sw) and sw > 0:
                return float(sw) / 2.0
            return 0.0
        return float(v)
    cand['cover_km'] = cand.apply(_cover_row, axis=1)
    return cand

@dataclass
class TaskInfo:
    lat: float
    lon: float
    slots: List[pd.Timestamp]
    tol: pd.Timedelta
    allow: set  # empty => all allowed

def build_slots(row) -> Tuple[List[pd.Timestamp], pd.Timedelta]:
    step = pd.to_timedelta(float(row['required_interval_hours']), unit='h')
    tol  = step * TOL_FACTOR
    s = row['window_start_utc']
    e = row['window_end_utc']
    slots = []
    t = s
    while t <= e:
        slots.append(t)
        t = t + step
    return slots, tol

def nearest_hit_slot(slots: List[pd.Timestamp], tol: pd.Timedelta, tmid: pd.Timestamp) -> int:
    """Return index k of the nearest slot that |tmid - slot| <= tol; else -1."""
    if not slots:
        return -1
    # slots are monotonic; use numpy for speed
    arr = pd.Series(slots)
    diffs = (arr - tmid).abs()
    k = int(diffs.idxmin())
    if abs(arr.iloc[k] - tmid) <= tol:
        return k
    return -1

def effective_radius_km(row) -> float:
    cover  = float(row.get('cover_km', 0.0)) * COVER_SCALE
    access = float(row.get('access_radius_km_used', 0.0)) * ACCESS_SCALE
    if MERGE_RADIUS_MODE == 'cover':
        return max(0.0, cover)
    elif MERGE_RADIUS_MODE == 'access':
        return max(0.0, access)
    elif MERGE_RADIUS_MODE == 'hybrid':
        off = float(row.get('off_nadir_deg', 0.0))
        km_per_deg = float(row.get('cap_access_km_per_deg', DEFAULT_KM_PER_DEG))
        expanded = cover + HYBRID_ALPHA * off * (km_per_deg if km_per_deg > 0 else DEFAULT_KM_PER_DEG)
        return max(0.0, min(access if access > 0 else expanded, expanded))
    else:
        return max(0.0, cover)

# ----------------------- Build model ----------------------
def main():
    # Load
    tasks_df = parse_tasks()
    cands_df = parse_candidates()

    # Build task dict
    tasks: Dict[Tuple[str,str], TaskInfo] = {}
    for _, r in tasks_df.iterrows():
        key = (str(r['user']), str(r['point_id']))
        slots, tol = build_slots(r)
        allow_raw = str(r.get('satellites') or '').strip() if 'satellites' in tasks_df.columns else ""
        allow = set(x.strip() for x in allow_raw.split(';') if x.strip()) if allow_raw else set()
        tasks[key] = TaskInfo(
            lat=float(r['lat']),
            lon=float(r['lon']),
            slots=slots,
            tol=tol,
            allow=allow
        )

    # Pre-index slots -> list of edge IDs constraint; and edges list
    # We map ONE slot per (candidate, task): the nearest-hit slot.
    edges = []   # each = (edge_id, j_idx, task_key, slot_index, weight)
    slot2edges: Dict[Tuple[str,str,int], List[int]] = {}

    # Prepare candidates list & conflicts per satellite
    cands_df = cands_df.sort_values(['t_mid','satellite']).reset_index(drop=True)
    J = len(cands_df)

    # Build conflict edges (same satellite, time overlap with buffer)
    # We'll collect pairs (j, k) with j < k
    conflicts = []
    for sat, group in cands_df.groupby('satellite', sort=False):
        g = group.sort_values('t_start').reset_index()
        idxs = g['index'].to_numpy()          # original j indices
        tstarts = g['t_start'].to_numpy()
        tends   = g['t_end'].to_numpy()
        n = len(g)
        for a in range(n):
            ja = int(idxs[a]); ta0 = tstarts[a]; ta1 = tends[a]
            # move b until start_b beyond end_a + buffer
            b = a + 1
            while b < n and (tstarts[b] <= (ta1 + pd.Timedelta(seconds=SLEW_BUFFER_SEC))):
                jb = int(idxs[b])
                # symmetric check (if they overlap within buffer)
                if not (tstarts[b] >= (ta1 + pd.Timedelta(seconds=SLEW_BUFFER_SEC)) or
                        ta0 >= (tends[b] + pd.Timedelta(seconds=SLEW_BUFFER_SEC))):
                    conflicts.append((min(ja, jb), max(ja, jb)))
                b += 1

    # Build eligibility edges (candidate -> multiple tasks)
    for j, c in cands_df.iterrows():
        sat = str(c['satellite'])
        sub_lat = float(c.get('sub_lat', np.nan))
        sub_lon = float(c.get('sub_lon', np.nan))
        radius  = effective_radius_km(c)

        has_fp  = (not np.isnan(sub_lat)) and (not np.isnan(sub_lon)) and radius > 0.0
        if not has_fp:
            # fallback: use its own point location (always present)
            sub_lat = float(c.get('lat'))
            sub_lon = float(c.get('lon'))
            radius  = max(radius, float(c.get('cover_km', 0.0)))

        tmid = c['t_mid']

        # try all tasks (for large instances you can spatially index this)
        for key, info in tasks.items():
            # optional same-user restriction
            if MERGE_SAME_USER_ONLY and key[0] != str(c['user']):
                continue
            # respect per-task allowlist, unless ignoring
            if (not MERGE_IGNORE_ALLOWLIST) and info.allow and (sat not in info.allow):
                continue
            # quick geo check
            d = haversine_km(sub_lat, sub_lon, info.lat, info.lon)
            if d > radius:
                continue
            # nearest slot within tolerance
            k = nearest_hit_slot(info.slots, info.tol, tmid)
            if k < 0:
                continue

            # weight: favor coverage; optionally penalize off-nadir (small)
            off = float(c.get('off_nadir_deg', 0.0))
            max_off = float(c.get('cap_max_off_deg', 0.0))
            off_norm = 0.0
            if QUALITY_GAMMA > 0 and max_off > 0:
                off_norm = min(1.0, max(0.0, off / max_off))
            w = 1.0 - QUALITY_GAMMA * off_norm

            eid = len(edges)
            edges.append((eid, j, key, k, w))
            slot_key = (key[0], key[1], k)
            slot2edges.setdefault(slot_key, []).append(eid)

    if VERBOSE:
        print(f"[build] candidates: {J}, edges: {len(edges)}, slots: {sum(len(info.slots) for info in tasks.values())}")
        print(f"[build] conflicts: {len(conflicts)}")

    # ----------------------- MILP -----------------------
    model = pulp.LpProblem("TaskingMILP", pulp.LpMaximize)

    # Variables
    y = pulp.LpVariable.dicts("y", (j for j in range(J)), 0, 1, cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", (e[0] for e in edges), 0, 1, cat=pulp.LpBinary)

    # Objective
    obj_cover = pulp.lpSum(z[eid] * edges[eid][4] for eid in range(len(edges)))
    obj_select = pulp.lpSum(y[j] for j in range(J))
    model += obj_cover - ALPHA_SELECT * obj_select

    # Link z -> y
    for eid, j, key, k, w in edges:
        model += z[eid] <= y[j]

    # Slot uniqueness: each slot (user, point_id, k) can be covered at most once
    for slot_key, eids in slot2edges.items():
        model += pulp.lpSum(z[e] for e in eids) <= 1

    # Satellite deconfliction (pairwise)
    for (ja, jb) in set(conflicts):
        model += y[ja] + y[jb] <= 1

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=1 if VERBOSE else 0, timeLimit=TIME_LIMIT_SEC)
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    if VERBOSE:
        print(f"[solve] status={status}, obj={pulp.value(model.objective):.3f}")

    # ----------------------- Build plan_milp.csv -----------------------
    # A chosen acquisition expands to one row per covered task (edge chosen)
    chosen_edges = [eid for (eid, j, key, k, w) in edges if z[eid].varValue and z[eid].varValue > 0.5]
    if VERBOSE:
        print(f"[solve] chosen acquisitions: {sum(1 for j in range(J) if y[j].varValue and y[j].varValue>0.5)}")
        print(f"[solve] covered slots (rows): {len(chosen_edges)}")

    rows = []
    # For quick lookup
    tasks_latlon = {key: (info.lat, info.lon) for key, info in tasks.items()}
    for eid in chosen_edges:
        _, j, key, k, w = edges[eid]
        c = cands_df.iloc[j]
        latp, lonp = tasks_latlon[key]
        row = {
            'user':        key[0],
            'point_id':    key[1],
            'satellite':   c['satellite'],
            't_start':     c['t_start'],
            't_mid':       c['t_mid'],
            't_end':       c['t_end'],
            'lat':         latp,
            'lon':         lonp,
        }
        # carry over useful geometry/capability fields if present
        for kcol in ('off_nadir_deg','sub_lat','sub_lon','cover_km','track_dist_km_mid',
                     'cap_swath_km','cap_access_km_per_deg','cap_max_off_deg','access_radius_km_used'):
            if kcol in c:
                row[kcol] = c[kcol]
        rows.append(row)

    plan = pd.DataFrame(rows)
    plan = plan.sort_values(['t_mid','satellite','user','point_id']).reset_index(drop=True)
    plan.to_csv(OUT_PLAN, index=False)
    print(f"-> {OUT_PLAN} (rows={len(plan)})")

    # Simple coverage metric (fraction of slots covered)
    total_slots = sum(len(info.slots) for info in tasks.values())
    covered_slots = len(set((edges[eid][2][0], edges[eid][2][1], edges[eid][3]) for eid in chosen_edges))
    if total_slots > 0:
        print(f"[metric] slot_coverage = {covered_slots}/{total_slots} = {covered_slots/total_slots:.3%}")

if __name__ == "__main__":
    main()
