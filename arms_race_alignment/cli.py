
import argparse, json, os, subprocess, sys
from .pipeline import DefenderPipeline, load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Seed toy data if missing
    data_train = os.path.join(cfg["data"]["root"], cfg["data"]["train_file"])
    if not os.path.exists(data_train):
        print("Seeding toy data...")
        subprocess.run([sys.executable, "scripts/seed_data.py"], check=True)

    pipe = DefenderPipeline(cfg)
    metrics = pipe.run()
    print("\\n=== FINAL METRICS (Pure-ML) ===")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
