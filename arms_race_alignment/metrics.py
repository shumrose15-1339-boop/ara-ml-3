
from dataclasses import dataclass, field
from typing import List, Dict
import time
import numpy as np

@dataclass
class MetricTracker:
    start_time: float = field(default_factory=time.time)
    failures: List[Dict] = field(default_factory=list)
    fixes: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

    def log_eval(self, step: int, failure_rate: float):
        self.history.append({"t": step, "failure_rate": failure_rate, "ts": time.time()})

    def log_failure(self, example_id: str):
        self.failures.append({"id": example_id, "ts": time.time()})

    def log_fix(self, example_id: str):
        self.fixes.append({"id": example_id, "ts": time.time()})

    def time_to_mitigation(self) -> float:
        pairs = []
        for f in self.failures:
            for fx in self.fixes:
                if fx["id"] == f["id"] and fx["ts"] > f["ts"]:
                    pairs.append(fx["ts"] - f["ts"])
                    break
        return float(np.mean(pairs)) if pairs else float("nan")

    def rolling_robustness(self, window: int = 3) -> List[Dict]:
        out = []
        for i in range(len(self.history)):
            lo = max(0, i - window + 1)
            vals = [h["failure_rate"] for h in self.history[lo:i+1]]
            out.append({"t": self.history[i]["t"], "roll_fail_rate": float(np.mean(vals))})
        return out

    def robustness_regret(self, baseline: float = 0.5) -> float:
        regret = 0.0
        for h in self.history:
            regret += max(0.0, h["failure_rate"] - baseline)
        return regret
