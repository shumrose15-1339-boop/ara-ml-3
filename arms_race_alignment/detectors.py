
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
import numpy as np

# ---------------- Safety model: TF-IDF + Logistic Regression ----------------
@dataclass
class SafetyResult:
    decision: str   # "refuse" or "comply"
    score: float    # probability it's harmful

class SafetyModel:
    def __init__(self):
        # word + char ngrams for multilingual/code-mix robustness
        self.word_vec = TfidfVectorizer(min_df=2, ngram_range=(1,2))
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
        self.clf = LogisticRegression(max_iter=2000, n_jobs=None)

        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]):
        Xw = self.word_vec.fit_transform(texts)
        Xc = self.char_vec.fit_transform(texts)
        from scipy.sparse import hstack
        X = hstack([Xw, Xc])
        self.clf.fit(X, labels)
        self._fitted = True

    def infer(self, prompt: str) -> SafetyResult:
        if not self._fitted:
            # Conservative fallback
            return SafetyResult("refuse", 0.5)
        Xw = self.word_vec.transform([prompt])
        Xc = self.char_vec.transform([prompt])
        from scipy.sparse import hstack
        X = hstack([Xw, Xc])
        p = float(self.clf.predict_proba(X)[0][1])  # prob harmful
        return SafetyResult("refuse" if p >= 0.5 else "comply", p)

# ------------- Consistency/OOD: IsolationForest on benign distribution ---------
class ConsistencyDetector:
    def __init__(self, threshold: float = 0.0):
        # decision_function > threshold => "pass"
        self.threshold = threshold
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2)
        self.iforest = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        self._fitted = False

    def fit(self, benign_texts: List[str]):
        X = self.vec.fit_transform(benign_texts)
        # IsolationForest expects dense; small demo so toarray() is fine
        self.iforest.fit(X.toarray())
        self._fitted = True

    def decide(self, example: Dict) -> Tuple[str, float]:
        if not self._fitted:
            return ("pass", 0.0)
        X = self.vec.transform([example.get("text","")]).toarray()
        score = float(self.iforest.decision_function(X)[0])  # higher is more normal
        return ("pass" if score > self.threshold else "flag", score)
