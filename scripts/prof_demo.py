
import os, json, argparse, subprocess, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arms_race_alignment.pipeline import DefenderPipeline, load_config

def run_prof_demo(cfg_path: str, safety_threshold: float = 0.7, ood_threshold: float = None):
    cfg = load_config(cfg_path)

    data_train = os.path.join(cfg["data"]["root"], cfg["data"]["train_file"])
    if not os.path.exists(data_train):
        print("[prof_demo] Seeding toy data...")
        subprocess.run([sys.executable, "scripts/seed_data.py"], check=True)

    pipe = DefenderPipeline(cfg)

    # Re-threshold the safety model using its returned probability
    orig_infer = pipe.safety.infer
    def stricter_infer(prompt: str):
        res = orig_infer(prompt)
        decision = "refuse" if res.score >= safety_threshold else "comply"
        return type(res)(decision=decision, score=res.score)
    pipe.safety.infer = stricter_infer

    if ood_threshold is not None:
        try:
            pipe.detector.threshold = ood_threshold
        except Exception:
            pass

    out = pipe.run()

    out_dir = cfg["logging"]["dir"]
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    csv_path = os.path.join(out_dir, "rolling.csv")
    with open(csv_path, "w") as f:
        f.write("t,roll_fail_rate\n")
        for r in out["rolling"]:
            f.write(f'{r["t"]},{r["roll_fail_rate"]}\n')

    ts = [r["t"] for r in out["rolling"]]
    ys = [r["roll_fail_rate"] for r in out["rolling"]]
    plt.figure(figsize=(6,4), dpi=140)
    plt.plot(ts, ys, marker="o")
    plt.title("Rolling Failure Rate (Professor Demo)")
    plt.xlabel("Step (t)")
    plt.ylabel("Rolling Failure Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "rolling.png")
    plt.savefig(png_path)

    print("\n[prof_demo] Saved:")
    print("  -", json_path)
    print("  -", csv_path)
    print("  -", png_path)
    print("\n[prof_demo] Summary:")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/professor_demo.yaml")
    ap.add_argument("--safety_threshold", type=float, default=0.7)
    ap.add_argument("--ood_threshold", type=float, default=None)
    args = ap.parse_args()
    run_prof_demo(args.config, safety_threshold=args.safety_threshold, ood_threshold=args.ood_threshold)
