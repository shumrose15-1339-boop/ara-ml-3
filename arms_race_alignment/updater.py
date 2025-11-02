
from typing import Dict, List

class ContrastiveUpdater:
    """Toy updater that records hard negatives and can be used to lower pass threshold."""
    def __init__(self):
        self.hard_negatives: List[str] = []
        self.sensitivity_shift = 0.0

    def update(self, example: Dict, failed: bool):
        if failed:
            self.hard_negatives.append(example["text"][:200])
            # become more conservative
            self.sensitivity_shift = min(0.2, self.sensitivity_shift + 0.02)

    def get_shift(self) -> float:
        return self.sensitivity_shift
