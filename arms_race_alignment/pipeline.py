
from typing import Dict, List, Tuple
import random, os, yaml
from tqdm import tqdm

from .attacks import AttackGenerator
from .detectors import SafetyModel, ConsistencyDetector
from .updater import ContrastiveUpdater
from .metrics import MetricTracker
from .data import load_dataset

class DefenderPipeline:
    def __init__(self, cfg: Dict):
        random.seed(cfg.get("seed", 42))
        self.cfg = cfg
        self.attack = AttackGenerator(
            languages=cfg["data"]["languages"],
            mutation_types=cfg["attack"]["mutation_types"],
            max_depth=cfg["attack"]["max_depth"],
        )
        self.safety = SafetyModel()
        self.detector = ConsistencyDetector(threshold=cfg["detector"]["ood_threshold"])
        self.updater = ContrastiveUpdater()
        self.metrics = MetricTracker()
        self.log_dir = cfg["logging"]["dir"]
        os.makedirs(self.log_dir, exist_ok=True)

        # Load data
        self.train = load_dataset(cfg["data"]["root"], cfg["data"]["train_file"], cfg["data"]["max_rows"])
        self.eval = load_dataset(cfg["data"]["root"], cfg["data"]["eval_file"], cfg["data"]["max_rows"]//2)

        # Train ML models
        self._fit_models()

    def _fit_models(self):
        texts = [r["text"] for r in self.train]
        labels = [0 if r.get("category")=="benign" else 1 for r in self.train]
        self.safety.fit(texts, labels)
        benign_texts = [r["text"] for r in self.train if r.get("category")=="benign"]
        # If benign set small, still fit
        if len(benign_texts) < 10:
            benign_texts = texts[:20]
        self.detector.fit(benign_texts)

    def step(self, step_id: int) -> Tuple[int, int]:
        seeds = random.sample(self.train, k=min(5, len(self.train)))
        variants = []
        for s in seeds:
            variants.extend(self.attack.generate(s, k=self.cfg["attack"]["generations_per_step"]))

        failures = 0
        total = 0
        for ex in variants:
            total += 1
            tag, score = self.detector.decide(ex)
            # Make threshold gradually stricter after failures
            effective_threshold = self.cfg["detector"]["ood_threshold"] + self.updater.get_shift()
            pass_decision = (tag == "pass" and score > effective_threshold)

            sres = self.safety.infer(ex["text"])
            is_harm = ex.get("category","benign") != "benign"
            failed = pass_decision and (sres.decision != "refuse") and is_harm
            if failed:
                failures += 1
                self.metrics.log_failure(ex["id"])
                self.updater.update(ex, failed=True)
                self.metrics.log_fix(ex["id"])
        return failures, total

    def evaluate(self, step_id: int) -> float:
        harm = [e for e in self.eval if e.get("category") != "benign"]
        if not harm:
            return 0.0
        slips = 0
        for ex in harm:
            tag, score = self.detector.decide(ex)
            sres = self.safety.infer(ex["text"])
            effective_threshold = self.cfg["detector"]["ood_threshold"] + self.updater.get_shift()
            pass_decision = (tag == "pass" and score > effective_threshold)
            if pass_decision and sres.decision != "refuse":
                slips += 1
        frac = slips / len(harm)
        self.metrics.log_eval(step_id, frac)
        return frac

    def run(self):
        steps = self.cfg["train"]["steps"]
        eval_every = self.cfg["train"]["eval_every"]
        pbar = tqdm(range(1, steps+1), desc="ARA-ML loop")
        for step in pbar:
            fails, total = self.step(step)
            pbar.set_postfix({"fails": fails, "gen": total})
            if step % eval_every == 0:
                fr = self.evaluate(step)
                pbar.set_description(f"Eval fail-rate={fr:.2f}")
        ttm = self.metrics.time_to_mitigation()
        regret = self.metrics.robustness_regret(baseline=0.5)
        rolls = self.metrics.rolling_robustness(window=3)
        return {"time_to_mitigation": ttm, "robustness_regret": regret, "rolling": rolls}

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
