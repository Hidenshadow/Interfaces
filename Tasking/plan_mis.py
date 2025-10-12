#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIS (Maximum Independent Set) planner with the same knobs/semantics as plan_milp.py / plan_tabu.py.

- 固定默认 I/O：
    tasks:       test.csv
    candidates:  candidates.csv
    out plan:    plan_mis.csv

- 完全对齐的建模旋钮（Knobs）：
    TOL_FACTOR, MERGE_RADIUS_MODE, COVER_SCALE, ACCESS_SCALE, HYBRID_ALPHA,
    MERGE_SAME_USER_ONLY, MERGE_IGNORE_ALLOWLIST,
    SLEW_BUFFER_SEC, ALPHA_SELECT, QUALITY_GAMMA,
    TIME_LIMIT_SEC, VERBOSE

- 约束语义与 MILP/Tabu 一致：
    (1) 同卫星时间重叠（含缓冲）不可同时选
    (2) 同一“时间槽”（slot）至多被一个候选覆盖（slot 唯一性）

- 目标：
    maximize Σ_i ( w_i * x_i )   ，其中  w_i = 质量加权的“覆盖时隙数” - ALPHA_SELECT
    * 质量加权：可按 off_nadir 与 cap_max_off_deg 做可选惩罚 QUALITY_GAMMA
    * MILP 版：直接以 w_i 作为变量系数
    * Greedy 版：同样使用 w_i 进行 MWIS 启发式 + 局部改良

- 输出：
    plan_mis.csv（按“被选 acquisition × 所覆盖任务点/槽”展开为多行），
    列与 plan_milp/plan_tabu 一致：user, point_id, satellite, t_start, t_mid, t_end, lat, lon, …（附带常用几何/性能列若存在）
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import math, time
from collections import defaultdict

# ---------- Optional MILP via PuLP ----------
try:
    import pulp
    HAS_PULP = True
except Exception:
    HAS_PULP = False

# ----------------------- Fixed defaults -----------------------
TASKS_CSV = "test.csv"
CANDS_CSV = "candidates.csv"
OUT_PLAN  = "plan_mis.csv"

# ----------------------- Knobs (align with MILP/Tabu) --------
# Slot building: slots every interval, tolerance = ±interval * TOL_FACTOR
TOL_FACTOR = 0.5

# Footprint for multi-cover radius
MERGE_RADIUS_MODE = "cover"     # 'cover' | 'access' | 'hybrid'
COVER_SCALE  = 4.0
ACCESS_SCALE = 1.0
HYBRID_ALPHA = 0.6
DEFAULT_KM_PER_DEG = 30.0

# Multi-cover scope & allowlist behavior
MERGE_SAME_USER_ONLY   = False
MERGE_IGNORE_ALLOWLIST = False

# Per-satellite minimal gap (slew/settle buffer)
SLEW_BUFFER_SEC = 60

# Objective weights
ALPHA_SELECT  = 1e-3   # penalty per selected acquisition
QUALITY_GAMMA = 0.0    # 0..1, penalize off-nadir: w = w * (1 - gamma * off/max_off)

# MILP runtime
TIME_LIMIT_SEC = 300
VERBOSE        = True

# Greedy MIS knobs
SEED = 42

# ----------------------- Utilities ----------------------------
def re_split(s: str) -> List[str]:
    import re
    return re.split(r'[;,\|]+', s)

def parse_sat_list(s: str) -> Optional[Set[str]]:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    txt = str(s).strip()
    if not txt: return None
    parts = [p.strip() for p in re_split(txt) if p.strip()]
    return set(parts) if parts else None

def to_utc(s) -> Optional[pd.Timestamp]:
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    t = str(s).strip()
    if not t: return None
    if len(t) == 10 and t[4]=='-' and t[7]=='-':
        t += "T00:00:00Z"
    if 'T' not in t and len(t) >= 16:
        t = t.replace(' ', 'T')
    if 'Z' not in t and ('+' not in t and '-' in t[10:]) is False:
        t += 'Z'
    return pd.to_datetime(t, utc=True, errors='coerce')

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1 = np.radians([lat1, lon1])
    p2 = np.radians([lat2, lon2])
    d  = p2 - p1
    a = np.sin(d[0]/2.0)**2 + np.cos(p1[0])*np.cos(p2[0])*np.sin(d[1]/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return float(R*c)

# ----------------------- Data -------------------------------
@dataclass
class TaskInfo:
    lat: float
    lon: float
    slots: List[pd.Timestamp]   # slot centers (we use window start ticks with tolerance)
    tol: pd.Timedelta
    allow: Set[str]             # empty => all allowed

@dataclass
class Cand:
    idx: int
    sat: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    t_mid: pd.Timestamp
    slot_ids: Set[int]          # global slot ids that this candidate can cover
    weight: float               # w_i (already quality-weighted & penalized by ALPHA_SELECT)

# ----------------------- Parsers ----------------------------
def parse_tasks(path=TASKS_CSV) -> Tuple[Dict[Tuple[str,str], TaskInfo], Dict[Tuple[str,str,int], int], List[Tuple[str,str,int]]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)

    alias = {
        'user': ['user','User','username'],
        'point_id': ['point_id','point','id','PointID','pointId'],
        'lat': ['lat','latitude','lat_center','Lat','Latitude'],
        'lon': ['lon','longitude','lon_center','Lon','Longitude'],
        'window_start_utc': ['window_start_utc','start','window_start','WindowStartUTC','window_start_UTC'],
        'window_end_utc':   ['window_end_utc','end','window_end','WindowEndUTC','window_end_UTC'],
        'required_interval_hours': ['required_interval_hours','frequency_hours','interval_hours'],
        'satellites': ['satellites','satellite','satellites_used','Satellites'],
    }
    ren = {}
    for std, cands in alias.items():
        for c in cands:
            if c in df.columns:
                ren[c] = std; break
    df = df.rename(columns=ren)

    for col in ('window_start_utc','window_end_utc'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    if 'required_interval_hours' in df.columns:
        df['required_interval_hours'] = pd.to_numeric(df['required_interval_hours'], errors='coerce')

    df = df.dropna(subset=['user','point_id','lat','lon','window_start_utc','window_end_utc','required_interval_hours']).reset_index(drop=True)

    # collapse duplicates (first per user,point)
    g = (df.sort_values(['user','point_id'])
           .groupby(['user','point_id'], as_index=False)
           .first())

    tasks: Dict[Tuple[str,str], TaskInfo] = {}
    slot_key_list: List[Tuple[str,str,int]] = []

    def build_slots(row) -> Tuple[List[pd.Timestamp], pd.Timedelta]:
        step = pd.to_timedelta(float(row['required_interval_hours']), unit='h')
        tol  = step * TOL_FACTOR
        s = row['window_start_utc']; e = row['window_end_utc']
        slots = []
        t = s
        while t <= e + pd.Timedelta(microseconds=1):
            slots.append(t)
            t = t + step
        return slots, tol

    for _, r in g.iterrows():
        key = (str(r['user']).strip().lower(), str(r['point_id']).strip())
        slots, tol = build_slots(r)
        allow_raw = (str(r.get('satellites') or '').strip()) if 'satellites' in g.columns else ""
        allow = set(x.strip() for x in allow_raw.split(';') if x.strip()) if allow_raw else set()
        tasks[key] = TaskInfo(
            lat=float(r['lat']), lon=float(r['lon']),
            slots=slots, tol=tol, allow=allow
        )
        for k in range(len(slots)):
            slot_key_list.append((key[0], key[1], k))

    slot_id_of: Dict[Tuple[str,str,int], int] = { sk:i for i, sk in enumerate(slot_key_list) }
    return tasks, slot_id_of, slot_key_list

def parse_candidates(path=CANDS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for col in ('t_start','t_end','t_mid','t_start_utc','t_end_utc','t_mid_utc'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    if 't_start' not in df.columns and 't_start_utc' in df.columns: df['t_start'] = df['t_start_utc']
    if 't_end'   not in df.columns and 't_end_utc'   in df.columns: df['t_end']   = df['t_end_utc']
    if 't_mid'   not in df.columns and 't_mid_utc'   in df.columns: df['t_mid']   = df['t_mid_utc']

    # Fill optional columns if missing
    for c in ('sub_lat','sub_lon','cover_km','access_radius_km_used',
              'cap_swath_km','cap_access_km_per_deg','cap_max_off_deg','off_nadir_deg',
              'user','point_id','lat','lon','track_dist_km_mid'):
        if c not in df.columns: df[c] = np.nan

    # fallback cover_km = cap_swath_km/2
    def _cover_row(r):
        v = r.get('cover_km', np.nan)
        if pd.isna(v) or v <= 0:
            sw = r.get('cap_swath_km', np.nan)
            if pd.notna(sw) and sw > 0: return float(sw)/2.0
            return 0.0
        return float(v)
    df['cover_km'] = df.apply(_cover_row, axis=1)

    if 'user' in df.columns:
        df['user'] = df['user'].astype(str).str.strip().str.lower()
    if 'point_id' in df.columns:
        df['point_id'] = df['point_id'].astype(str).str.strip()
    return df

def effective_radius_km(row) -> float:
    cover  = float(row.get('cover_km', 0.0)) * COVER_SCALE
    access = float(row.get('access_radius_km_used', 0.0)) * ACCESS_SCALE
    if MERGE_RADIUS_MODE == 'cover':
        return max(0.0, cover)
    elif MERGE_RADIUS_MODE == 'access':
        return max(0.0, access)
    else:
        off = float(row.get('off_nadir_deg', 0.0))
        km_per_deg = float(row.get('cap_access_km_per_deg', DEFAULT_KM_PER_DEG))
        expanded = cover + HYBRID_ALPHA * off * (km_per_deg if km_per_deg > 0 else DEFAULT_KM_PER_DEG)
        return max(0.0, min(access if access>0 else expanded, expanded))

def nearest_hit_slot(slots: List[pd.Timestamp], tol: pd.Timedelta, tmid: pd.Timestamp) -> int:
    if not slots: return -1
    diffs = [abs((tmid - s)) for s in slots]
    k = int(np.argmin(diffs))
    if abs(slots[k] - tmid) <= tol:
        return k
    return -1

# ----------------------- Build candidates / conflicts ------
@dataclass
class BuildResult:
    cands: List[Cand]
    conflicts: List[Set[int]]
    slot2cands: Dict[int, List[int]]
    dfc_clean: pd.DataFrame
    tasks: Dict[Tuple[str,str], TaskInfo]
    slot_key_list: List[Tuple[str,str,int]]

def build_all(tasks, slot_id_of, slot_key_list, dfc) -> BuildResult:
    # Keep candidates whose primary (user, point) exists
    dfc = dfc[(dfc['user'].astype(str).str.len()>0) & (dfc['point_id'].astype(str).str.len()>0)]
    dfc = dfc[dfc.apply(lambda r: (r['user'], r['point_id']) in tasks, axis=1)]

    # Primary allowlist (always obey)
    def allowed_primary(row):
        t = tasks.get((row['user'], row['point_id']))
        if t is None: return False
        if not t.allow: return True
        sat = str(row['satellite']).strip()
        return sat in t.allow
    dfc = dfc[dfc.apply(allowed_primary, axis=1)]

    # For scanning: tasks grouped by user
    tasks_by_user = defaultdict(list)
    for key, ti in tasks.items():
        tasks_by_user[key[0]].append((key, ti))

    # Which users to scan in multi-cover
    all_users = list(set(u for (u,_) in tasks.keys()))

    C: List[Cand] = []
    slot2cands: Dict[int, List[int]] = defaultdict(list)

    for j, r in dfc.reset_index(drop=True).iterrows():
        sat = str(r['satellite']).strip()
        ts, te, tm = r['t_start'], r['t_end'], r['t_mid']
        if pd.isna(ts) or pd.isna(te) or pd.isna(tm):
            continue

        sub_lat = r['sub_lat'] if pd.notna(r['sub_lat']) else r.get('lat')
        sub_lon = r['sub_lon'] if pd.notna(r['sub_lon']) else r.get('lon')
        if pd.isna(sub_lat) or pd.isna(sub_lon):
            continue

        radius = effective_radius_km(r)
        slot_ids: Set[int] = set()

        users_scan = [r['user']] if MERGE_SAME_USER_ONLY else all_users
        for u in users_scan:
            for (key, ti) in tasks_by_user[u]:
                # Allowlist (ignore or obey) when multi-cover
                if (not MERGE_IGNORE_ALLOWLIST) and ti.allow and (sat not in ti.allow):
                    continue
                # Spatial
                dkm = haversine_km(float(sub_lat), float(sub_lon), ti.lat, ti.lon)
                if dkm > radius:
                    continue
                # Temporal slot hit
                k = nearest_hit_slot(ti.slots, ti.tol, tm)
                if k < 0:
                    continue
                sid = slot_id_of[(key[0], key[1], k)]
                slot_ids.add(sid)

        if not slot_ids:
            continue

        # quality weighted weight; then subtract ALPHA_SELECT
        w_base = 1.0
        off = float(r.get('off_nadir_deg', 0.0))
        max_off = float(r.get('cap_max_off_deg', 0.0))
        if QUALITY_GAMMA > 0 and max_off > 0:
            quality = max(0.0, 1.0 - QUALITY_GAMMA * min(1.0, max(0.0, off/max_off)))
        else:
            quality = 1.0
        w = quality * len(slot_ids) - ALPHA_SELECT

        cid = len(C)
        C.append(Cand(
            idx=int(j),
            sat=sat, t_start=ts, t_end=te, t_mid=tm,
            slot_ids=slot_ids,
            weight=float(w)
        ))
        for sid in slot_ids:
            slot2cands[sid].append(cid)

    # Conflicts: same-sat overlap + per-slot uniqueness
    conflicts = [set() for _ in range(len(C))]

    # per-satellite time overlap with slew buffer
    by_sat = defaultdict(list)
    for i,c in enumerate(C): by_sat[c.sat].append(i)
    for sat, idxs in by_sat.items():
        idxs = sorted(idxs, key=lambda i: C[i].t_start)
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a+1, len(idxs)):
                j = idxs[b]
                if C[j].t_start > C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC):
                    break
                if not (C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[j].t_start or
                        C[j].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[i].t_start):
                    conflicts[i].add(j); conflicts[j].add(i)

    # per-slot uniqueness (clique for MILP; pairwise for greedy)
    for sid, arr in slot2cands.items():
        if len(arr) <= 1: continue
        for a in range(len(arr)):
            i = arr[a]
            for b in range(a+1, len(arr)):
                j = arr[b]
                conflicts[i].add(j); conflicts[j].add(i)

    return BuildResult(C, conflicts, slot2cands, dfc, tasks, slot_key_list)

# ----------------------- Solvers ------------------------------
def solve_mwis_greedy(C: List[Cand], conflicts: List[Set[int]], seed=SEED) -> Set[int]:
    """Greedy MWIS + small local improvement."""
    rng = np.random.default_rng(seed)
    n  = len(C)
    w  = np.array([c.weight for c in C], dtype=float)
    deg= np.array([len(conflicts[i]) for i in range(n)], dtype=float)
    key= w / (deg + 1.0)

    remaining = set(range(n))
    picked: Set[int] = set()
    while remaining:
        i = max(remaining, key=lambda k: (key[k], w[k]))
        picked.add(i)
        remaining -= ({i} | conflicts[i])

    # tiny 2-for-1 local improvement
    not_picked = set(range(n)) - picked
    for _ in range(min(200, len(not_picked))):
        if not picked: break
        i = rng.choice(list(picked))
        allowed = [j for j in not_picked if conflicts[j].isdisjoint(picked - {i})]
        if len(allowed) < 2:
            continue
        allowed.sort(key=lambda j: w[j], reverse=True)
        best_pair=None; best_gain=0.0
        lim = min(12, len(allowed))
        for a in range(lim):
            for b in range(a+1, lim):
                j,k = allowed[a], allowed[b]
                if j in conflicts[k]:  # j-k conflict
                    continue
                gain = w[j] + w[k] - w[i]
                if gain > best_gain:
                    best_gain=gain; best_pair=(j,k)
        if best_pair:
            j,k = best_pair
            picked.remove(i); picked.add(j); picked.add(k)
            not_picked.add(i); not_picked.remove(j); not_picked.remove(k)

    return picked

def solve_mwis_milp(C: List[Cand], slot2cands: Dict[int, List[int]]) -> Set[int]:
    if not HAS_PULP:
        raise RuntimeError("PuLP not installed; cannot MILP")

    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=pulp.LpBinary)
         for i in range(len(C))}
    prob = pulp.LpProblem("MWIS_slots", pulp.LpMaximize)

    weights = {i: float(C[i].weight) for i in range(len(C))}
    prob += pulp.lpSum(weights[i]*x[i] for i in x)

    # per-satellite overlap with buffer
    by_sat = defaultdict(list)
    for i,c in enumerate(C): by_sat[c.sat].append(i)
    for sat, idxs in by_sat.items():
        idxs = sorted(idxs, key=lambda i: C[i].t_start)
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a+1, len(idxs)):
                j = idxs[b]
                # if separated beyond buffer, can break early
                if C[j].t_start > C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC):
                    break
                # overlap?
                if not (C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[j].t_start or
                        C[j].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[i].t_start):
                    prob += x[i] + x[j] <= 1

    # per-slot clique: sum x_i <= 1
    for s, arr in slot2cands.items():
        if len(arr) > 1:
            prob += pulp.lpSum(x[i] for i in arr) <= 1

    solver = pulp.PULP_CBC_CMD(msg=VERBOSE, timeLimit=TIME_LIMIT_SEC if TIME_LIMIT_SEC else None)
    prob.solve(solver)
    sel=set(i for i in x if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5)
    return sel

# ----------------------- Export ------------------------------
def export_plan(sel: Set[int], C: List[Cand], dfc: pd.DataFrame,
                tasks: Dict[Tuple[str,str], TaskInfo],
                slot_key_list: List[Tuple[str,str,int]],
                out_path=OUT_PLAN):
    picked_idx = sorted(list(sel))
    rows = []
    task_latlon = {(u,p):(ti.lat, ti.lon) for (u,p), ti in tasks.items()}

    for i in picked_idx:
        c = C[i]
        crow = dfc.iloc[c.idx]
        for sid in sorted(c.slot_ids):
            (u, pid, k) = slot_key_list[sid]
            latp, lonp = task_latlon[(u,pid)]
            row = {
                'user':      u,
                'point_id':  pid,
                'satellite': crow['satellite'],
                't_start':   crow['t_start'],
                't_mid':     crow['t_mid'],
                't_end':     crow['t_end'],
                'lat':       latp,
                'lon':       lonp
            }
            for col in ('off_nadir_deg','sub_lat','sub_lon','cover_km','track_dist_km_mid',
                        'cap_swath_km','cap_access_km_per_deg','cap_max_off_deg','access_radius_km_used'):
                if col in crow.index:
                    row[col] = crow[col]
            rows.append(row)

    plan = pd.DataFrame(rows).sort_values(['t_mid','satellite','user','point_id']).reset_index(drop=True)
    for col in ('t_start','t_mid','t_end'):
        if col in plan.columns:
            plan[col] = pd.to_datetime(plan[col], utc=True, errors='coerce').dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    plan.to_csv(out_path, index=False)
    print(f"-> {out_path} (rows={len(plan)})")

    # coverage metric by slots
    total_slots = sum(len(ti.slots) for ti in tasks.values())
    covered_slots = len(set(sum([list(C[i].slot_ids) for i in picked_idx], [])))
    if total_slots > 0:
        print(f"[metric] slot_coverage = {covered_slots}/{total_slots} = {covered_slots/total_slots:.3%}")

# ----------------------- Main -------------------------------
def main():
    ap = argparse.ArgumentParser(description="MIS planner (MILP if PuLP available, else Greedy), aligned with plan_milp/plan_tabu knobs")
    ap.add_argument("--tasks", default=TASKS_CSV)
    ap.add_argument("--candidates", default=CANDS_CSV)
    ap.add_argument("--out", default=OUT_PLAN)
    ap.add_argument("--solver", default="auto", choices=["auto","milp","greedy"])
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    t0 = time.time()
    tasks, slot_id_of, slot_key_list = parse_tasks(args.tasks)
    dfc = parse_candidates(args.candidates)
    br = build_all(tasks, slot_id_of, slot_key_list, dfc)

    if not br.cands:
        print("MIS: no feasible candidates; write empty plan.")
        pd.DataFrame(columns=['user','point_id','satellite','t_start','t_mid','t_end','lat','lon']).to_csv(args.out, index=False)
        return

    if args.solver == "milp" or (args.solver == "auto" and HAS_PULP):
        sel = solve_mwis_milp(br.cands, br.slot2cands)
        solver_used = "MILP(PuLP)"
    else:
        sel = solve_mwis_greedy(br.cands, br.conflicts, seed=args.seed)
        solver_used = "Greedy-MWIS"

    t1 = time.time()
    print(f"[MIS:{solver_used}] picked={len(sel)}  time={(t1-t0):.2f}s")

    export_plan(sel, br.cands, br.dfc_clean, br.tasks, br.slot_key_list, args.out)

if __name__ == "__main__":
    main()
