#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tabu Search planner (MILP-parity knobs & semantics)

- 固定输入输出：
    TASKS_CSV = "test.csv"
    CANDS_CSV = "candidates.csv"
    OUT_PLAN  = "plan_tabu.csv"

- 采用与 plan_milp.py 相同的建模旋钮（见下）与“每时隙至多一次覆盖”的约束语义：
    * 时间槽容差 TOL_FACTOR
    * 多点覆盖半径 MERGE_RADIUS_MODE / COVER_SCALE / ACCESS_SCALE / HYBRID_ALPHA
    * MERGE_SAME_USER_ONLY / MERGE_IGNORE_ALLOWLIST
    * 卫星最小时间缓冲 SLEW_BUFFER_SEC
    * 目标中的选择条数轻惩罚 ALPHA_SELECT 与可选离轴惩罚 QUALITY_GAMMA

- Tabu 的目标：
    maximize  sum(候选所覆盖的时隙权重) - ALPHA_SELECT * (#picked)
  我们在冲突图中加入两类边：
    (1) 同卫星时间重叠（含缓冲）不能同选
    (2) 覆盖同一时隙不能同选（对应 MILP 的 slot 唯一约束）

- 输出与 MILP 风格一致：对每个被选 acquisition，按其覆盖到的 (user, point_id, slot) 展开为多行。
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

# ----------------------- Fixed I/O -----------------------
TASKS_CSV = "test.csv"
CANDS_CSV = "candidates.csv"
OUT_PLAN  = "plan_tabu.csv"

# ----------------------- Knobs (与 MILP 对齐) ------------
# Slot building: slot every interval, tolerance = ±interval * TOL_FACTOR
TOL_FACTOR = 0.5

# Footprint for multi-cover
MERGE_RADIUS_MODE = "cover"     # 'cover' | 'access' | 'hybrid'
COVER_SCALE  = 4.0
ACCESS_SCALE = 1.0
HYBRID_ALPHA = 0.6
DEFAULT_KM_PER_DEG = 30.0

MERGE_SAME_USER_ONLY   = False
MERGE_IGNORE_ALLOWLIST = False

# Per-satellite minimal gap (slew/settle buffer), seconds
SLEW_BUFFER_SEC = 60

# Objective weights
ALPHA_SELECT   = 1e-3    # penalty per selected acquisition
QUALITY_GAMMA  = 0.0     # penalize off-nadir: w = 1 - gamma * (off/max_off)

# Tabu params
ITERS        = 1500
TABU_TENURE  = 25
SEED         = 42
VERBOSE      = True

# ----------------------- Utils --------------------------
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

# ----------------------- Data structures ----------------
@dataclass
class TaskInfo:
    lat: float
    lon: float
    slots: List[pd.Timestamp]     # slot centers
    tol: pd.Timedelta             # ± tolerance
    allow: Set[str]               # empty => all allowed

@dataclass
class Cand:
    idx: int                      # original index in candidates df
    sat: str
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    t_mid: pd.Timestamp
    # multi-cover results
    slot_ids: Set[int]            # global slot ids that this candidate can cover
    weight: float                 # sum of per-slot weights (1 - gamma*off/max_off)

# ----------------------- Parsers ------------------------
def parse_tasks(path=TASKS_CSV) -> Tuple[Dict[Tuple[str,str], TaskInfo], Dict[Tuple[str,str,int], int], List[Tuple[str,str,int]]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)

    # Robust remap
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
    ren = {}
    for std, cands in alias.items():
        for c in cands:
            if c in df.columns:
                ren[c] = std; break
    df = df.rename(columns=ren)

    df['window_start_utc'] = pd.to_datetime(df['window_start_utc'], utc=True, errors='coerce')
    df['window_end_utc']   = pd.to_datetime(df['window_end_utc'],   utc=True, errors='coerce')
    df['required_interval_hours'] = pd.to_numeric(df['required_interval_hours'], errors='coerce')

    df = df.dropna(subset=['user','point_id','lat','lon','window_start_utc','window_end_utc','required_interval_hours']).reset_index(drop=True)

    # 合并为 (user, point_id) 唯一
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

    # 全局 slot id
    slot_id_of: Dict[Tuple[str,str,int], int] = { sk:i for i, sk in enumerate(slot_key_list) }
    return tasks, slot_id_of, slot_key_list

def parse_candidates(path=CANDS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for col in ('t_start','t_end','t_mid','t_start_utc','t_end_utc','t_mid_utc'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    if 't_start' not in df.columns and 't_start_utc' in df.columns:
        df['t_start'] = df['t_start_utc']
    if 't_end' not in df.columns and 't_end_utc' in df.columns:
        df['t_end'] = df['t_end_utc']
    if 't_mid' not in df.columns and 't_mid_utc' in df.columns:
        df['t_mid'] = df['t_mid_utc']

    # 补充可选字段
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

    # 规范 user/point_id
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
    else:  # hybrid
        off = float(row.get('off_nadir_deg', 0.0))
        km_per_deg = float(row.get('cap_access_km_per_deg', DEFAULT_KM_PER_DEG))
        expanded = cover + HYBRID_ALPHA * off * (km_per_deg if km_per_deg > 0 else DEFAULT_KM_PER_DEG)
        return max(0.0, min(access if access>0 else expanded, expanded))

def nearest_hit_slot(slots: List[pd.Timestamp], tol: pd.Timedelta, tmid: pd.Timestamp) -> int:
    if not slots: return -1
    # 二分/向量化都可，这里简洁实现
    diffs = [abs((tmid - s)) for s in slots]
    k = int(np.argmin(diffs))
    if abs(slots[k] - tmid) <= tol:
        return k
    return -1

# ----------------------- Build candidates & conflicts ----
def build_cands_and_conflicts(tasks: Dict[Tuple[str,str], TaskInfo],
                              slot_id_of: Dict[Tuple[str,str,int], int],
                              dfc: pd.DataFrame):
    # 仅保留属于已知任务主键的候选
    dfc = dfc[(dfc['user'].astype(str).str.len()>0) & (dfc['point_id'].astype(str).str.len()>0)]
    dfc = dfc[dfc.apply(lambda r: (r['user'], r['point_id']) in tasks, axis=1)]

    # 主任务必须遵守 allowlist（多点合并是否忽略受 MERGE_IGNORE_ALLOWLIST 控制）
    def allowed_primary(row):
        t = tasks.get((row['user'], row['point_id']))
        if t is None: return False
        if not t.allow: return True
        sat = str(row['satellite']).strip()
        return sat in t.allow
    dfc = dfc[dfc.apply(allowed_primary, axis=1)]

    # 预索引：按用户分任务，便于多点合并扫描
    tasks_by_user = defaultdict(list)
    for key, ti in tasks.items():
        tasks_by_user[key[0]].append((key, ti))

    C: List[Cand] = []
    # 对于“同一时隙唯一”的冲突，我们构建 slot -> [cand_ids]
    slot2cands: Dict[int, List[int]] = defaultdict(list)

    # 构建候选的可覆盖 slot 集与权重
    for j, r in dfc.reset_index(drop=True).iterrows():
        sat = str(r['satellite']).strip()
        tmid, ts, te = r['t_mid'], r['t_start'], r['t_end']
        if pd.isna(ts) or pd.isna(te) or pd.isna(tmid):
            continue

        # footprint
        sub_lat = r['sub_lat'] if pd.notna(r['sub_lat']) else r.get('lat')
        sub_lon = r['sub_lon'] if pd.notna(r['sub_lon']) else r.get('lon')
        if pd.isna(sub_lat) or pd.isna(sub_lon):
            continue
        radius = effective_radius_km(r)

        # 候选可覆盖的 slot（可能跨多个任务点）
        slot_ids = set()
        weight   = 0.0

        # 任务扫描范围：同用户或所有用户
        users_to_scan = [r['user']] if MERGE_SAME_USER_ONLY else list(set(u for (u,_) in tasks.keys()))
        for u in users_to_scan:
            for (key, ti) in tasks_by_user[u]:
                # allowlist：是否忽略
                if (not MERGE_IGNORE_ALLOWLIST) and ti.allow and (sat not in ti.allow):
                    continue
                # 空间半径
                dkm = haversine_km(float(sub_lat), float(sub_lon), ti.lat, ti.lon)
                if dkm > radius:
                    continue
                # 时间槽命中
                k = nearest_hit_slot(ti.slots, ti.tol, tmid)
                if k < 0:
                    continue
                sid = slot_id_of[(key[0], key[1], k)]
                slot_ids.add(sid)

        if not slot_ids:
            continue

        # 每 slot 的质量权重（简化为相同 w_s），可按 off_nadir 惩罚
        w_base = 1.0
        off = float(r.get('off_nadir_deg', 0.0))
        max_off = float(r.get('cap_max_off_deg', 0.0))
        if QUALITY_GAMMA > 0 and max_off > 0:
            quality = max(0.0, 1.0 - QUALITY_GAMMA * min(1.0, max(0.0, off/max_off)))
        else:
            quality = 1.0
        weight = w_base * quality * len(slot_ids)

        cid = len(C)
        C.append(Cand(
            idx = int(j),
            sat = sat,
            t_start = ts, t_end = te, t_mid = tmid,
            slot_ids = slot_ids,
            weight = float(weight)
        ))
        for sid in slot_ids:
            slot2cands[sid].append(cid)

    # 冲突图：同卫星时间重叠 + 同 slot 覆盖（唯一性）
    conflicts = [set() for _ in range(len(C))]

    # 1) per-sat overlap with buffer
    by_sat = defaultdict(list)
    for i,c in enumerate(C):
        by_sat[c.sat].append(i)
    for sat, idxs in by_sat.items():
        idxs = sorted(idxs, key=lambda i: C[i].t_start)
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a+1, len(idxs)):
                j = idxs[b]
                if C[j].t_start > C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC):
                    break
                # overlap?
                if not (C[i].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[j].t_start or
                        C[j].t_end + pd.Timedelta(seconds=SLEW_BUFFER_SEC) <= C[i].t_start):
                    conflicts[i].add(j); conflicts[j].add(i)

    # 2) slot uniqueness: 同一 slot 的候选两两冲突
    for sid, arr in slot2cands.items():
        if len(arr) <= 1: continue
        for a in range(len(arr)):
            i = arr[a]
            for b in range(a+1, len(arr)):
                j = arr[b]
                conflicts[i].add(j); conflicts[j].add(i)

    return C, conflicts, dfc, slot2cands

# ----------------------- Tabu search ---------------------
@dataclass
class Solution:
    picked: Set[int]
    score: float
    w_sum: float

def tabu_search(C: List[Cand], conflicts: List[Set[int]],
                iters=ITERS, tabu_tenure=TABU_TENURE,
                alpha=ALPHA_SELECT, seed=SEED) -> Solution:
    rng = np.random.default_rng(seed)

    # Greedy 初始化（按 weight/(deg+1) 选）
    deg = np.array([len(conflicts[i]) for i in range(len(C))], dtype=float)
    wt  = np.array([c.weight for c in C], dtype=float)
    score_key = wt / (deg + 1.0)
    remaining = set(range(len(C)))
    picked: Set[int] = set()
    blocked = set()
    # 贪心挑选非冲突
    while remaining:
        i = max(remaining, key=lambda k: (score_key[k], wt[k]))
        picked.add(i)
        # 删掉 i 及其邻居
        rem = {i} | conflicts[i]
        remaining -= rem
    w_sum = float(sum(C[i].weight for i in picked))
    cur_score = w_sum - alpha * len(picked)
    best = Solution(set(picked), cur_score, w_sum)

    tabu = {}  # move -> last_iter

    def conflicted(i: int) -> bool:
        return any(j in picked for j in conflicts[i])

    for it in range(1, iters+1):
        not_picked = [i for i in range(len(C)) if i not in picked]
        if len(not_picked) > 800:
            not_picked = list(rng.choice(not_picked, size=800, replace=False))
        picked_list = list(picked)
        if len(picked_list) > 400:
            picked_list = list(rng.choice(picked_list, size=400, replace=False))

        # 当前 w_sum
        cur_w = float(sum(C[i].weight for i in picked))
        cur_s = cur_w - alpha * len(picked)

        move_best = None  # (new_score, ('add',i) / ('rem',j) / ('swap',j_out,i_in))

        # ADD
        for i in not_picked:
            if conflicted(i):
                continue
            new_s = (cur_w + C[i].weight) - alpha * (len(picked)+1)
            if ('add', i) in tabu and it - tabu[('add',i)] < tabu_tenure and new_s <= best.score:
                continue
            if (move_best is None) or (new_s > move_best[0]):
                move_best = (new_s, ('add', i))

        # SWAP
        if not_picked and picked_list:
            sample_in  = not_picked if len(not_picked) <= 300 else list(rng.choice(not_picked, size=300, replace=False))
            sample_out = picked_list if len(picked_list) <= 200 else list(rng.choice(picked_list, size=200, replace=False))
            for i_in in sample_in:
                # i_in 不得与 picked\{j_out} 冲突
                for j_out in sample_out:
                    # 检查 i_in 与 picked 中除 j_out 外的冲突
                    conflict_with = conflicts[i_in] & (picked - {j_out})
                    if conflict_with:
                        continue
                    new_s = (cur_w - C[j_out].weight + C[i_in].weight) - alpha * len(picked)
                    if (('add', i_in) in tabu and it - tabu[('add',i_in)] < tabu_tenure and new_s <= best.score) or \
                       (('rem', j_out) in tabu and it - tabu[('rem',j_out)] < tabu_tenure and new_s <= best.score):
                        continue
                    if (move_best is None) or (new_s > move_best[0]):
                        move_best = (new_s, ('swap', j_out, i_in))

        # REM（轻）
        for j in picked_list[:50]:
            new_s = (cur_w - C[j].weight) - alpha * (len(picked)-1)
            if ('rem', j) in tabu and it - tabu[('rem',j)] < tabu_tenure and new_s <= best.score:
                continue
            if (move_best is None) or (new_s > move_best[0]):
                move_best = (new_s, ('rem', j))

        if move_best is None:
            # 无改进动作：随机踢一下
            rng.shuffle(not_picked)
            moved = False
            for i in not_picked:
                if not conflicted(i):
                    picked.add(i)
                    tabu[('add', i)] = it
                    moved = True
                    break
            if not moved:
                continue
            else:
                cur_w = float(sum(C[i].weight for i in picked))
                cur_s = cur_w - alpha * len(picked)
                if cur_s > best.score:
                    best = Solution(set(picked), cur_s, cur_w)
                continue

        new_s, move = move_best
        if move[0] == 'add':
            i = move[1]
            picked.add(i); tabu[('add',i)] = it
        elif move[0] == 'swap':
            j_out, i_in = move[1], move[2]
            if j_out in picked:
                picked.remove(j_out)
            picked.add(i_in)
            tabu[('rem', j_out)] = it
            tabu[('add', i_in)] = it
        elif move[0] == 'rem':
            j = move[1]
            if j in picked:
                picked.remove(j)
            tabu[('rem', j)] = it

        cur_w = float(sum(C[i].weight for i in picked))
        cur_s = cur_w - alpha * len(picked)
        if cur_s > best.score:
            best = Solution(set(picked), cur_s, cur_w)

        if VERBOSE and it % 200 == 0:
            print(f"[tabu] iter={it} picked={len(picked)} bestScore={best.score:.3f}")

    return best

# ----------------------- Export plan ---------------------
def export_plan(best: Solution, C: List[Cand], dfc: pd.DataFrame,
                tasks: Dict[Tuple[str,str], TaskInfo],
                slot_key_list: List[Tuple[str,str,int]],
                out_path=OUT_PLAN):
    picked_idx = sorted(list(best.picked))
    rows = []
    # 便于定位任务点经纬度
    task_latlon = {(u,p):(ti.lat, ti.lon) for (u,p), ti in tasks.items()}

    for i in picked_idx:
        c = C[i]
        crow = dfc.iloc[c.idx]
        # 对该候选覆盖到的所有 slot 展开
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
            # 携带常用性能/几何列（若存在）
            for col in ('off_nadir_deg','sub_lat','sub_lon','cover_km','track_dist_km_mid',
                        'cap_swath_km','cap_access_km_per_deg','cap_max_off_deg','access_radius_km_used'):
                if col in crow.index:
                    row[col] = crow[col]
            rows.append(row)

    plan = pd.DataFrame(rows).sort_values(['t_mid','satellite','user','point_id']).reset_index(drop=True)
    # 统一时间格式
    for col in ('t_start','t_mid','t_end'):
        if col in plan.columns:
            plan[col] = pd.to_datetime(plan[col], utc=True, errors='coerce').dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    plan.to_csv(out_path, index=False)
    print(f"-> {out_path} (rows={len(plan)})")

    # 覆盖率（以时隙为单位）
    total_slots = sum(len(ti.slots) for ti in tasks.values())
    covered_slots = len(set(sum([list(C[i].slot_ids) for i in picked_idx], [])))
    if total_slots > 0:
        print(f"[metric] slot_coverage = {covered_slots}/{total_slots} = {covered_slots/total_slots:.3%}")

# ----------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Tabu Search planner with MILP-parity knobs/semantics")
    parser.add_argument("--tasks", default=TASKS_CSV)
    parser.add_argument("--candidates", default=CANDS_CSV)
    parser.add_argument("--out", default=OUT_PLAN)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--tabu-tenure", type=int, default=TABU_TENURE)
    parser.add_argument("--alpha", type=float, default=ALPHA_SELECT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t0 = time.time()
    tasks, slot_id_of, slot_key_list = parse_tasks(args.tasks)
    dfc = parse_candidates(args.candidates)
    C, conflicts, dfc_clean, slot2cands = build_cands_and_conflicts(tasks, slot_id_of, dfc)

    if not C:
        print("tabu: no feasible candidates; write empty plan.")
        pd.DataFrame(columns=['user','point_id','satellite','t_start','t_mid','t_end','lat','lon']).to_csv(args.out, index=False)
        return

    best = tabu_search(C, conflicts, iters=args.iters, tabu_tenure=args.tabu_tenure, alpha=args.alpha, seed=args.seed)
    t1 = time.time()
    print(f"[tabu] picked={len(best.picked)} score={best.score:.3f} time={(t1-t0):.2f}s")

    export_plan(best, C, dfc_clean, tasks, slot_key_list, args.out)

if __name__ == "__main__":
    main()
