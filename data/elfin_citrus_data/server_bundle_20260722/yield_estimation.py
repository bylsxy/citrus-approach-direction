#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Estimate fruit yield from fruit volume.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--size-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    density = float(cfg["yield"]["density_kg_per_m3"])
    factor = float(cfg["yield"]["calibration_factor"])
    rows = []
    with open(args.size_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            volume = float(row["volume_m3"])
            mass = density * volume * factor
            row["estimated_mass_kg"] = mass
            rows.append(row)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["fruit_id", "volume_m3", "estimated_mass_kg", "diameter_m", "method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "fruit_id": row["fruit_id"],
                    "volume_m3": row["volume_m3"],
                    "estimated_mass_kg": f"{row['estimated_mass_kg']:.6f}",
                    "diameter_m": row["diameter_m"],
                    "method": row["method"],
                }
            )

    total_mass = sum(r["estimated_mass_kg"] for r in rows)
    count = len(rows)
    with open(args.summary_md, "w", encoding="utf-8") as f:
        f.write("# Yield Summary\n\n")
        f.write(f"- fruit count: {count}\n")
        f.write(f"- density: {density:.3f} kg/m^3\n")
        f.write(f"- calibration factor: {factor:.3f}\n")
        f.write(f"- estimated total mass: {total_mass:.6f} kg\n")
        f.write("\nNote: density and calibration factor are placeholders until cultivar-specific weighing calibration is provided.\n")
    print(f"fruit_count={count} total_mass_kg={total_mass:.6f}")


if __name__ == "__main__":
    main()
