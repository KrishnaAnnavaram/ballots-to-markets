"""Three-class sentiment (negative, neutral, positive) behind one interface, and its validation.

- ``LexiconSentiment``: offline word lists with negation. A text with a small net score is neutral.
- ``HFSentiment``: a 3-class Hugging Face model (default ``cardiffnlp/twitter-roberta-base-sentiment-latest``),
  extra ``nlp``. A binary model (for example SST-2) cannot output ``neutral``, so it is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

LABELS = ("negative", "neutral", "positive")
SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

POSITIVE = set("""great good best win wins winning strong hope hopeful excited exciting proud support love
progress optimistic growth boost thrilled inspiring brilliant success successful victory confident amazing
leading lead better improve improving gain gains rally""".split())
NEGATIVE = set("""bad worst lose losing loss weak fear afraid angry anger disaster corrupt crisis fail failed
failure terrible awful scary chaos lies lie lying threat danger dangerous worse decline collapse wrong
disappointed disappointing pathetic sad""".split())
NEGATORS = {"not", "no", "never", "n't", "cannot", "hardly"}
_TOKEN = re.compile(r"[a-z']+")


class SentimentModel(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[str]: ...


class LexiconSentiment:
    name = "lexicon"

    def __init__(self, neutral_band: float = 0.0):
        self.neutral_band = neutral_band

    def score(self, text: str) -> float:
        toks = _TOKEN.findall(text.lower().replace("n't", " n't"))
        total = 0.0
        for i, t in enumerate(toks):
            v = 1.0 if t in POSITIVE else -1.0 if t in NEGATIVE else 0.0
            if v and any(w in NEGATORS for w in toks[max(0, i - 3):i]):
                v = -v
            total += v
        return total / max(1.0, np.sqrt(len(toks)))

    def predict(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            s = self.score(t)
            out.append("positive" if s > self.neutral_band else "negative" if s < -self.neutral_band else "neutral")
        return out


class HFSentiment:  # pragma: no cover - needs the optional extra and a model download
    def __init__(self, model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size: int = 32):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError('transformers is not installed: pip install -e ".[nlp]"') from exc
        self.pipe = pipeline("text-classification", model=model, truncation=True)
        labels = {v.lower() for v in self.pipe.model.config.id2label.values()}
        if not set(LABELS) <= labels:
            raise ValueError(f"{model} is not a 3-class sentiment model (labels {sorted(labels)})")
        self.name, self.batch_size = model, batch_size

    def predict(self, texts: list[str]) -> list[str]:
        return [r["label"].lower() for r in self.pipe(texts, batch_size=self.batch_size)]


def build(name: str) -> SentimentModel:
    return LexiconSentiment() if name == "lexicon" else HFSentiment(name)


@dataclass
class Validation:
    n: int
    accuracy: float
    macro_f1: float
    recall: dict[str, float]
    confusion: list[list[int]]  # rows = true label, columns = predicted label, order LABELS

    def as_dict(self) -> dict:
        return {"n": self.n, "accuracy": self.accuracy, "macro_f1": self.macro_f1, "recall": self.recall,
                "confusion": self.confusion, "labels": list(LABELS)}


def validate(model: SentimentModel, texts: list[str], labels: list[str]) -> Validation:
    """Compare the model with labelled texts. Every class, also ``neutral``, gets a recall."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    keep = [i for i, l in enumerate(labels) if l in LABELS]
    if not keep:
        raise ValueError("no label in negative/neutral/positive")
    y = [labels[i] for i in keep]
    p = model.predict([texts[i] for i in keep])
    rec = recall_score(y, p, labels=list(LABELS), average=None, zero_division=0)
    return Validation(len(y), float(accuracy_score(y, p)),
                      float(f1_score(y, p, labels=list(LABELS), average="macro", zero_division=0)),
                      {l: float(r) for l, r in zip(LABELS, rec)},
                      confusion_matrix(y, p, labels=list(LABELS)).tolist())
