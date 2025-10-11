#!/usr/bin/env python3
"""
Build a compact "Coverage Summary" from:
  - test.csv          (task requests)
  - plan_greedy.csv   (greedy plan results)

Output (coverage_summary_minimal.csv), one row per (user, point_id) with:
  user, point_id, satellites_used, observation_count,
  avg_observation_gap_h, max_observation_gap_h,
  observation_times_utc, observation_times_sat_utc

Notes:
- Times are parsed as UTC-aware. Observation times are formatted "YYYY-MM-DD HH:MM:SSZ".
- If a task has no observations, it still appears with empty satellites/times and NaN gaps.
"""

import pandas as pd
import numpy as np
from pathlib import Path

TASKS_CSV = "test.csv"
PLAN_CSV  = "plan_greedy.csv"
OUT_CSV   = "coverage_summary_minimal.csv"

def main():
    # ---------- Load inputs ----------
    if not Path(TASKS_CSV).exists():
        raise FileNotFoundError(f"Missing file: {TASKS_CSV}")
    if not Path(PLAN_CSV).exists():
        raise FileNotFoundError(f"Missing file: {PLAN_CSV}")

    tasks = pd.read_csv(TASKS_CSV)
    plan  = pd.read_csv(PLAN_CSV)

    # ---------- Parse timestamps (UTC-aware) ----------
    for col in ("window_start_utc", "window_end_utc"):
        if col in tasks.columns:
            tasks[col] = pd.to_datetime(tasks[col], utc=True, errors="coerce")

    for col in ("t_start", "t_end", "t_mid"):
        if col in plan.columns:
            plan[col] = pd.to_datetime(plan[col], utc=True, errors="coerce")

    # ---------- Basic cleanup ----------
    tasks = tasks.dropna(subset=["user", "point_id"]).copy()
    if "satellite" not in plan.columns:
        plan["satellite"] = ""
    plan = plan.dropna(subset=["user", "point_id"]).copy()

    # Unique (user, point_id) from tasks to drive the output rows
    task_keys = tasks[["user", "point_id"]].drop_duplicates()

    # Group plan by (user, point_id) for quick lookup
    plan_sorted = plan.sort_values("t_mid")
    plan_grp = plan_sorted.groupby(["user", "point_id"], dropna=False)

    rows = []
    for _, trow in task_keys.iterrows():
        user = trow["user"]
        pid  = trow["point_id"]

        # Observations for this (user, point)
        if (user, pid) in plan_grp.groups:
            g = plan_grp.get_group((user, pid)).copy()
        else:
            g = plan.iloc[0:0].copy()

        # Keep only rows with valid mid time
        if "t_mid" in g.columns:
            g = g.dropna(subset=["t_mid"]).sort_values("t_mid").reset_index(drop=True)
        obs_count = len(g)

        # Satellites used (unique, sorted)
        satellites_used = ""
        if obs_count > 0:
            satellites_used = ";".join(sorted(g["satellite"].dropna().astype(str).str.strip().unique()))

        # Times only
        times_str = ""
        # Times with satellite
        times_sat_str = ""
        if obs_count > 0 and "t_mid" in g.columns:
            mids_utc = g["t_mid"].dt.tz_convert("UTC")
            times_str = " | ".join(ts.strftime("%Y-%m-%d %H:%M:%SZ") for ts in mids_utc)

            # pair with satellite, fill unknown if missing
            sats = g["satellite"].astype(str).fillna("").str.strip()
            sat_filled = sats.where(sats != "", other="UNKNOWN")
            times_sat_str = " | ".join(
                f"{ts.strftime('%Y-%m-%d %H:%M:%SZ')}@{sat}"
                for ts, sat in zip(mids_utc, sat_filled)
            )

        # Gaps (hours) between consecutive observations by t_mid
        avg_gap_h = np.nan
        max_gap_h = np.nan
        if obs_count >= 2 and "t_mid" in g.columns:
            mids = g["t_mid"].sort_values().reset_index(drop=True)
            gaps_h = (mids.diff().dropna().dt.total_seconds() / 3600.0).to_numpy()
            if gaps_h.size > 0:
                avg_gap_h = float(np.mean(gaps_h))
                max_gap_h = float(np.max(gaps_h))

        rows.append({
            "user": user,
            "point_id": pid,
            "satellites_used": satellites_used,
            "observation_count": obs_count,
            "avg_observation_gap_h": avg_gap_h,
            "max_observation_gap_h": max_gap_h,
            "observation_times_utc": times_str,
            "observation_times_sat_utc": times_sat_str,   # NEW: time@satellite
        })

    summary = pd.DataFrame(rows).sort_values(["user", "point_id"]).reset_index(drop=True)
    summary.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  (rows={len(summary)})")

if __name__ == "__main__":
    main()
