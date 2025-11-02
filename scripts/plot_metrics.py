
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

def plot_from_csv(csv_path: str, out_png: str):
    ts, ys = [], []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(int(row["t"]))
            ys.append(float(row["roll_fail_rate"]))
    plt.figure(figsize=(6,4), dpi=140)
    plt.plot(ts, ys, marker="o")
    plt.title("Rolling Failure Rate")
    plt.xlabel("Step (t)")
    plt.ylabel("Rolling Failure Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs_prof_demo")
    args = ap.parse_args()
    csv_path = os.path.join(args.runs_dir, "rolling.csv")
    png_path = os.path.join(args.runs_dir, "rolling_replot.png")
    plot_from_csv(csv_path, png_path)
    print("[plot_metrics] Saved:", png_path)

if __name__ == "__main__":
    main()
