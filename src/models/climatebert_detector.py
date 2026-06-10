from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.common import project_path


@dataclass
class DetectorResult:
    is_environmental_claim: bool
    score: float
    raw_label: str


class EnvironmentalClaimDetector:
    def __init__(self, model_name: str, cache_dir: str | None = None, device: int = -1):
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = str(project_path(cache_dir))
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
        )
        self.id2label = model.config.id2label

    def predict(self, text: str) -> DetectorResult:
        scores = self.pipe(text, truncation=True, max_length=512)[0]
        positive = self._positive_score(scores)
        best = max(scores, key=lambda row: row["score"])
        return DetectorResult(
            is_environmental_claim=positive >= 0.5,
            score=float(positive),
            raw_label=str(best["label"]),
        )

    def _positive_score(self, scores: list[dict]) -> float:
        for row in scores:
            label = str(row["label"]).lower()
            if "environmental" in label or "claim" in label or label in {"label_1", "1", "true"}:
                return float(row["score"])
        if len(scores) == 2:
            return float(scores[1]["score"])
        return float(max(row["score"] for row in scores))
