# make_sat_caps_from_tle.py
# 用法: python make_sat_caps_from_tle.py --tle satellites_10.tle --out sat_capabilities.json
# 说明: 从 TLE 生成每星一个能力条目。新增 access_km_per_deg，便于 generate_candidates 精细控制可达半径。

import json
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tle', required=True, help='Path to TLE file (3-line blocks: name, L1, L2)')
    ap.add_argument('--out', default='sat_capabilities.json', help='Output JSON path')
    # 允许通过命令行调整几个全局默认
    ap.add_argument('--default-swath-km', type=float, default=80.0, help='Default ground swath (km)')
    ap.add_argument('--default-max-off-deg', type=float, default=30.0, help='Default max off-nadir (deg)')
    ap.add_argument('--default-access-km-per-deg', type=float, default=30.0,
                    help='Default ground distance per off-nadir degree (km/deg) for access radius approximation')
    args = ap.parse_args()

    with open(args.tle, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    names = []
    for i in range(0, len(lines), 3):
        names.append(lines[i])

    # 统一默认（后续可手工在 JSON 里微调各星）
    default = {
        "swath_km": args.default_swath_km,
        "max_off_nadir_deg": args.default_max_off_deg,
        "access_km_per_deg": args.default_access_km_per_deg,  # ★ 新增：供 generate_candidates 使用
        "min_elevation_deg": 0,
        "slew_rate_deg_s": 1.0,
        "settle_time_s": 8,
        "max_imaging_s_per_orbit": 900,
        "quality_penalty_per_deg": 0.01
    }

    caps = {name: dict(default) for name in names}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(caps, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out} with {len(names)} satellites.")
    print(f"Defaults -> swath_km={args.default_swath_km}, "
          f"max_off_nadir_deg={args.default_max_off_deg}, "
          f"access_km_per_deg={args.default_access_km_per_deg}")

if __name__ == '__main__':
    main()
