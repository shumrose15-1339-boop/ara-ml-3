
# Professor Demo Add-ons

## Quick Run (from your ARA_ML project root)
```bash
# 1) Copy these folders into your project root:
#    - configs/demo.yaml
#    - scripts/demo.py
#    - scripts/plot_metrics.py
#    - slides/Professor_Demo_Slides.md

# 2) Ensure venv is active and deps installed:
python -m pip install --upgrade pip setuptools wheel
pip install numpy pandas scikit-learn pyyaml tqdm regex matplotlib joblib scipy

# 3) Run the professor demo (saves JSON/CSV/PNG under runs_demo/)
python scripts/demo.py --config configs/demo.yaml

# (Optional) Re-plot from saved CSV
python scripts/plot_metrics.py --runs_dir runs_demo
```

## What You Get
- `runs_demo/metrics.json` — summary metrics
- `runs_demo/rolling.csv` — rolling failure rate (for tables/plots)
- `runs_demo/rolling.png` — figure for your slides
