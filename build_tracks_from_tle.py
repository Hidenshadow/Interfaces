#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-day satellite ground tracks from a TLE file (ALL satellites).
- Robust to NumPy arrays: converts to Python lists, no ambiguous truth checks.
- Filters NaN/Inf, splits polyline at ±180° dateline, writes per-day JSON.

Output:
  tracks/
    index.json
    YYYY-MM-DD/<SAT>.json

Usage:
  python build_tracks_from_tle.py \
    --tle satellites_10.tle \
    --out-dir tracks \
    --start 2025-01-01 --end 2025-01-07 \
    --step-min 2
"""

import os
import json
import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
from skyfield.api import load, EarthSatellite, wgs84


def parse_args():
    p = argparse.ArgumentParser(description="Generate per-day ground tracks from a TLE (all satellites)")
    p.add_argument("--tle", required=True, help="Path to TLE file (3-line groups: name + L1 + L2)")
    p.add_argument("--out-dir", default="tracks", help="Output directory (default: tracks)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC), inclusive")
    p.add_argument("--step-min", type=int, default=2, help="Sampling step in minutes (default: 2)")
    p.add_argument("--thin", type=int, default=1, help="Keep every N-th sample (default: 1)")
    return p.parse_args()


def date_range(start_str, end_str):
    d0 = datetime.strptime(start_str, "%Y-%m-%d").date()
    d1 = datetime.strptime(end_str,   "%Y-%m-%d").date()
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def read_tles(path):
    """Read 3-line TLEs and return {name: EarthSatellite}. Accepts '0 NAME' header."""
    sats = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i + 2 < len(lines):
        name_line = lines[i]
        l1 = lines[i + 1]
        l2 = lines[i + 2]

        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            # try to realign if formatting is off
            j = i
            found = False
            while j < len(lines) - 2:
                if lines[j].startswith("1 ") and lines[j + 1].startswith("2 "):
                    if j - 1 >= 0:
                        name_line = lines[j - 1]
                        l1 = lines[j]
                        l2 = lines[j + 1]
                        i = j - 1
                        found = True
                        break
                j += 1
            if not found:
                i += 1
                continue

        if name_line.startswith("0 "):
            name = name_line[2:].strip()
        else:
            name = name_line.strip()

        try:
            sats[name] = EarthSatellite(l1, l2, name)
        except Exception as e:
            print(f"[WARN] Skip TLE group '{name}': {e}")

        i += 3

    return sats


def times_for_day(day, step_min):
    """UTC datetimes for [day, day+1) sampled every step_min minutes."""
    t0 = datetime(day.year, day.month, day.day, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=1)
    times = []
    t = t0
    while t < t1:
        times.append(t)
        t += timedelta(minutes=step_min)
    if len(times) == 0 or (t1 - times[-1]) > timedelta(minutes=step_min / 2):
        times.append(t1 - timedelta(seconds=1))
    return times


def thin_list(seq, stride):
    if stride <= 1 or len(seq) <= 2:
        return list(seq)
    out = []
    for i, x in enumerate(seq):
        if i == 0 or i == len(seq) - 1 or (i % stride == 0):
            out.append(x)
    return out


def clean_and_align(lat_list, lon_list, dt_list):
    """Convert to arrays, drop NaN/Inf, return aligned Python lists of equal length."""
    la = np.asarray(lat_list, dtype=float)
    lo = np.asarray(lon_list, dtype=float)
    # normalize lon into [-180, 180] for plotting
    lo = ((lo + 180.0) % 360.0) - 180.0
    if len(la) != len(lo) or len(la) != len(dt_list):
        n = min(len(la), len(lo), len(dt_list))
        la = la[:n]; lo = lo[:n]; dt_list = dt_list[:n]
    ok = np.isfinite(la) & np.isfinite(lo)
    la = la[ok]; lo = lo[ok]
    dt_list = [dt_list[i] for i, flag in enumerate(ok) if flag]
    return la.tolist(), lo.tolist(), dt_list


def wrap_segments(lat_list, lon_list, time_list):
    """Split polyline when crossing the dateline (lon jump > 180°)."""
    if len(lat_list) == 0 or len(lon_list) == 0:
        return []

    segs = []
    cur_lat = [float(lat_list[0])]
    cur_lon = [float(lon_list[0])]
    cur_time = [time_list[0]]

    for i in range(1, len(lat_list)):
        prev_lon = float(lon_list[i - 1])
        lon = float(lon_list[i])
        if abs(lon - prev_lon) > 180.0:
            segs.append((cur_lat, cur_lon, cur_time))
            cur_lat = [float(lat_list[i])]
            cur_lon = [lon]
            cur_time = [time_list[i]]
        else:
            cur_lat.append(float(lat_list[i]))
            cur_lon.append(lon)
            cur_time.append(time_list[i])

    segs.append((cur_lat, cur_lon, cur_time))

    out = []
    for la, lo, ti in segs:
        out.append([[la[j], lo[j], ti[j]] for j in range(len(la))])
    return out


def bbox(lat_list, lon_list):
    if len(lat_list) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    la = np.asarray(lat_list, dtype=float)
    lo = np.asarray(lon_list, dtype=float)
    return [float(la.min()), float(lo.min()), float(la.max()), float(lo.max())]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sats = read_tles(args.tle)
    if not sats:
        raise SystemExit("No valid satellites found in TLE. Check the file format.")

    ts = load.timescale()

    index = {}
    for day in date_range(args.start, args.end):
        day_str = day.strftime("%Y-%m-%d")
        day_dir = os.path.join(args.out_dir, day_str)
        os.makedirs(day_dir, exist_ok=True)
        index[day_str] = []

        dt_list = times_for_day(day, args.step_min)
        if args.thin > 1:
            dt_list = thin_list(dt_list, args.thin)
        t = ts.from_datetimes(dt_list)

        for name, sat in sats.items():
            try:
                geoc = sat.at(t)
                sp = wgs84.subpoint(geoc)

                # Convert to Python lists early (avoid ambiguous truth on NumPy arrays)
                lat_list = sp.latitude.degrees.tolist()
                lon_list = sp.longitude.degrees.tolist()
                time_list = [dt.strftime("%H:%M:%SZ") for dt in dt_list]

                # Clean and align (drop NaN/Inf, enforce equal length)
                lat_list, lon_list, time_list = clean_and_align(lat_list, lon_list, time_list)

                if len(lat_list) == 0:
                    print(f"[WARN] {day_str} {name} has no valid points (NaN/Inf filtered). Skipped.")
                    continue

                segments = wrap_segments(lat_list, lon_list, time_list)
                if len(segments) == 0:
                    print(f"[WARN] {day_str} {name} segments empty. Skipped.")
                    continue

                out = {
                    "satellite": name,
                    "date": day_str,
                    "step_min": int(args.step_min * (args.thin if args.thin > 1 else 1)),
                    "bbox": bbox(lat_list, lon_list),
                    "segments": segments
                }
                with open(os.path.join(day_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
                index[day_str].append(name)
                print(f"[OK] {day_str} {name} pts={sum(len(s) for s in segments)} segs={len(segments)}")

            except Exception as e:
                print(f"[WARN] {day_str} {name} failed: {e}")

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Tracks written to: {args.out_dir}")


if __name__ == "__main__":
    main()
