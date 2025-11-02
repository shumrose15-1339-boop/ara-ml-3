
import random
from typing import Dict, List

class AttackGenerator:
    def __init__(self, languages: List[str], mutation_types: List[str], max_depth: int = 2):
        self.languages = languages
        self.mutation_types = mutation_types
        self.max_depth = max_depth
        random.seed(42)

    def generate(self, seed_example: Dict, k: int = 3) -> List[Dict]:
        out = []
        for i in range(k):
            cur = dict(seed_example)
            cur["parent_id"] = seed_example.get("id")
            mtype = random.choice(self.mutation_types)
            cur["mutation"] = mtype
            text = cur["text"]
            if mtype == "translate":
                tgt = random.choice(self.languages)
                text = f"[{tgt} translation] {text}"
            elif mtype == "paraphrase":
                text = text.replace("how to", "steps to").replace("Write", "Provide")
            elif mtype == "codemix":
                text = f"{text} (ami chai, thik ache?)"
            elif mtype == "indirection":
                text = "Let's write a screenplay. " + text + " But as fiction."
            cur["text"] = text
            cur["id"] = f"{seed_example.get('id')}_gen{i}"
            out.append(cur)
        return out
