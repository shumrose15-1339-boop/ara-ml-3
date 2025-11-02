
# Slide 1 — Problem & Motivation
- LLM jailbreaks evolve rapidly across languages; static defenses decay.
- Need an **adaptive, multilingual, longitudinal** defense pipeline.

# Slide 2 — Approach (Arms-Race Alignment)
- Defender-in-the-loop: Attack generator → OOD/Consistency detector → Safety classifier → Update → Rolling eval.
- **ML Safety** (TF-IDF+LogReg) + **ML OOD** (IsolationForest).
- Mutation lineage (translate / paraphrase / code-mix / indirection).

# Slide 3 — System Diagram
- Boxes: Attack Generator → OOD Detector → Safety Classifier → Updater → Metrics Store.
- Logs: {language, code-mix, mutation, parent_id, timestamp, outcome}.

# Slide 4 — Results (Demo)
- Rolling failure curve; Time-to-Mitigation (TTM); Robustness-Regret.
- Show 2–3 multilingual/code-mix examples that were caught vs slipped.

# Slide 5 — Next Steps
- Semantic/embedding-based consistency reasoning.
- Real LMJD dataset (10+ languages, lineage over months).
- Baseline comparisons; ablations; visualization dashboard.
