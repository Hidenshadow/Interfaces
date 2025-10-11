import os
import csv
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite, wgs84, utc

def generate_satellite_tracks(tle_path, output_dir, start_date, end_date, step_minutes=2):
    # Load timescale and define datetime range with UTC timezone
    ts = load.timescale()
    start_dt = datetime(*start_date).replace(tzinfo=utc)
    end_dt   = datetime(*end_date).replace(tzinfo=utc)

    # Generate time list: every 10 minutes
    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    time_list = [start_dt + timedelta(minutes=i) for i in range(0, total_minutes + 1, step_minutes)]
    times = ts.utc(time_list)
    print(f"[INFO] Total time steps generated: {len(times)} ({step_minutes} min interval)")

    # Read TLE entries
    with open(tle_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) % 3 == 0, "TLE file format error: not a multiple of 3 lines"

    entries = []
    for i in range(0, len(lines), 3):
        name = lines[i]
        line1 = lines[i+1]
        line2 = lines[i+2]
        entries.append((name, line1, line2))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each satellite
    for name, line1, line2 in entries:
        sat = EarthSatellite(line1, line2, name, ts)
        print(f"[INFO] Processing {name} (TLE epoch: {sat.epoch.utc_iso()})")

        subpoints = sat.at(times).subpoint()
        lats = subpoints.latitude.degrees
        lons = subpoints.longitude.degrees

        csv_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_track.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'lat', 'lon'])
            for t, lat, lon in zip(times, lats, lons):
                writer.writerow([t.utc_iso(), f"{lat:.6f}", f"{lon:.6f}"])
        print(f"[DONE] Track for {name} saved to {csv_path}")

if __name__ == '__main__':
    # Example usage
    tle_file = '/Users/jianglinqiao/Desktop/Fire_location/satellites_10.tle'
    output_folder = '/Users/jianglinqiao/Desktop/Fire_location/tracks'
    generate_satellite_tracks(tle_file, output_folder, (2025, 6, 25, 0, 0, 0), (2025, 7, 25, 23, 59, 0), step_minutes=2)
