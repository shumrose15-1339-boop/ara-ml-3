import argparse
from arms_race_alignment.pipeline import DefenderPipeline, load_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--text", required=True, help="Prompt to test")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--mutation", default="manual")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe = DefenderPipeline(cfg)  # trains models / loads data

    ex = {"text": args.text, "language": args.lang, "mutation": args.mutation, "category": "unknown"}
    tag, score = pipe.detector.decide(ex)
    res = pipe.safety.infer(args.text)

    print("\n=== Single-Prompt Check ===")
    print("Text      :", args.text)
    print("Language  :", args.lang, "  Mutation:", args.mutation)
    print("Detector  :", tag, f"(score={score:.3f})")
    print("Safety    :", res.decision, f"(p_harm={res.score:.3f})")

if __name__ == "__main__":
    main()
