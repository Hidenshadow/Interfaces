#!/usr/bin/env python3
"""
Slice RAW hotspot CSV into many small spatial tiles per day (NO spatial aggregation).

Input CSV (min columns used):
  latitude, longitude, acq_date, acq_time (optional), brightness (optional)

Outputs under ./tiles:
  - meta.json:
      {"lat_min":..., "lat_max":..., "lon_min":..., "lon_max":..., "nx":N, "ny":M}
  - index.json:
      {"2025-01-01":[[row,col,count], ...], "2025-01-02":[...], ...}
  - YYYY-MM-DD/r{row}_c{col}.json:
      { "date":"YYYY-MM-DD", "row":row, "col":col,
        "points":[ [lat,lon,brightness,acq_time], ... ] }   # RAW rows (no aggregation)

Usage:
  python slice_raw_to_grid_tiles.py --input hotspotdata.csv --out-dir tiles \
    --min-date 2025-01-01 --max-date 2025-01-07 --grid 16x16

Notes:
- Grid covers the data's bounding box (or a user-specified bbox).
- Increase grid (e.g., 24x24) => more slices (smaller files, more requests).
- We only write tiles that actually contain points.
"""

import csv
import os
import json
import math
import argparse
from collections import defaultdict
from datetime import datetime

def parse_date(s: str) -> str:
    s = (s or "").strip()[:10]
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return s
    except Exception:
        return ""

def in_range(d: str, dmin: str, dmax: str) -> bool:
    return (not dmin or d >= dmin) and (not dmax or d <= dmax)

def clamp(v, a, b): return max(a, min(b, v))

def pass1_bbox(path, dmin, dmax, lat_keys, lon_keys, date_keys):
    lat_min = +1e9; lat_max = -1e9
    lon_min = +1e9; lon_max = -1e9
    rows = 0; kept = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        def k(cands):
            for x in cands:
                if x in r.fieldnames: return x
            return None
        lat_k, lon_k, date_k = k(lat_keys), k(lon_keys), k(date_keys)
        if not lat_k or not lon_k or not date_k:
            raise SystemExit("CSV header missing latitude/longitude/acq_date (or aliases).")

        for row in r:
            rows += 1
            try:
                lat = float(row.get(lat_k,""))
                lon = float(row.get(lon_k,""))
            except Exception:
                continue
            d = parse_date(row.get(date_k,""))
            if not d or not math.isfinite(lat) or not math.isfinite(lon):
                continue
            if not in_range(d, dmin, dmax):
                continue
            kept += 1
            if lat < lat_min: lat_min = lat
            if lat > lat_max: lat_max = lat
            if lon < lon_min: lon_min = lon
            if lon > lon_max: lon_max = lon
    if kept == 0:
        raise SystemExit("No rows found in the specified date window.")
    # add a tiny padding
    pad = 1e-6
    return lat_min - pad, lat_max + pad, lon_min - pad, lon_max + pad, kept

def row_col(lat, lon, lat_min, lon_min, dlat, dlon, ny, nx):
    r = int(math.floor((lat - lat_min) / dlat))
    c = int(math.floor((lon - lon_min) / dlon))
    r = clamp(r, 0, ny-1); c = clamp(c, 0, nx-1)
    return r, c

def main():
    ap = argparse.ArgumentParser(description="Slice RAW hotspot CSV to per-day spatial tiles (no aggregation).")
    ap.add_argument("--input", required=True, help="Path to hotspotdata.csv")
    ap.add_argument("--out-dir", default="tiles", help="Output dir (default: tiles)")
    ap.add_argument("--min-date", default="2025-01-01")
    ap.add_argument("--max-date", default="2025-01-07")
    ap.add_argument("--grid", default="16x16", help="Grid as NxNy, e.g. 16x16, 20x20")
    # optional fixed bbox; if omitted, we compute from data in pass 1
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    args = ap.parse_args()

    nx, ny = [int(x) for x in args.grid.lower().split("x")]
    os.makedirs(args.out_dir, exist_ok=True)

    lat_keys = ["latitude","lat","Latitude","LAT"]
    lon_keys = ["longitude","lon","Longitude","LON"]
    date_keys = ["acq_date","date","Date","ACQ_DATE"]

    # Pass 1: bbox
    if None in (args.lat_min, args.lat_max, args.lon_min, args.lon_max):
        lat_min, lat_max, lon_min, lon_max, kept = pass1_bbox(
            args.input, args.min_date, args.max_date, lat_keys, lon_keys, date_keys
        )
    else:
        lat_min, lat_max, lon_min, lon_max = args.lat_min, args.lat_max, args.lon_min, args.lon_max

    dlat = (lat_max - lat_min) / ny if ny > 0 else 1.0
    dlon = (lon_max - lon_min) / nx if nx > 0 else 1.0

    meta = {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "nx": nx, "ny": ny
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Pass 2: slice into day-tiles
    day_tiles = defaultdict(lambda: defaultdict(list))  # day -> (r,c) -> list of points
    day_counts = defaultdict(lambda: defaultdict(int))  # day -> (r,c) -> count

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        def k(cands):
            for x in cands:
                if x in r.fieldnames: return x
            return None
        lat_k, lon_k, date_k = k(lat_keys), k(lon_keys), k(date_keys)
        time_k = "acq_time" if "acq_time" in r.fieldnames else ("time" if "time" in r.fieldnames else None)
        bright_k = "brightness" if "brightness" in r.fieldnames else ("Brightness" if "Brightness" in r.fieldnames else None)

        for row in r:
            d = parse_date(row.get(date_k,""))
            if not d or not in_range(d, args.min_date, args.max_date):
                continue
            try:
                lat = float(row.get(lat_k,""))
                lon = float(row.get(lon_k,""))
            except Exception:
                continue
            if not math.isfinite(lat) or not math.isfinite(lon):
                continue

            r_idx, c_idx = row_col(lat, lon, lat_min, lon_min, dlat, dlon, ny, nx)
            # raw point: keep minimal fields to reduce file size
            acq_time = row.get(time_k,"") if time_k else ""
            brightness = row.get(bright_k,"") if bright_k else ""
            day_tiles[d][(r_idx,c_idx)].append([lat, lon, brightness, acq_time])
            day_counts[d][(r_idx,c_idx)] += 1

    # Write out tiles & index
    index = {}
    for day, tiles in day_tiles.items():
        day_dir = os.path.join(args.out_dir, day)
        os.makedirs(day_dir, exist_ok=True)
        idx_list = []
        for (r_idx, c_idx), pts in tiles.items():
            out_path = os.path.join(day_dir, f"r{r_idx}_c{c_idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"date": day, "row": r_idx, "col": c_idx, "points": pts},
                          f, ensure_ascii=False, separators=(",", ":"))
            idx_list.append([r_idx, c_idx, len(pts)])
        index[day] = sorted(idx_list)

    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Wrote tiles to {args.out_dir}")
    print(f"Grid: {nx}x{ny} covering lat[{lat_min},{lat_max}] lon[{lon_min},{lon_max}]")
    print("Example:", os.path.join(args.out_dir, args.min_date, "r0_c0.json"))

if __name__ == "__main__":
    main()
