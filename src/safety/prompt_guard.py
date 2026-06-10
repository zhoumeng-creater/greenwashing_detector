from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.common import project_path, resolve_model_ref


SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "ignore the rules",
    "system prompt",
    "developer message",
    "do not cite evidence",
    "always say high risk",
    "always say legal violation",
]


def simple_prompt_guard(text: str) -> dict:
    lower = text.lower()
    matches = [pattern for pattern in SUSPICIOUS_PATTERNS if pattern in lower]
    return {
        "guard_type": "rule",
        "is_suspicious": bool(matches),
        "score": 1.0 if matches else 0.0,
        "label": "suspicious" if matches else "benign",
        "matches": matches,
    }


def is_local_model_ref(model_ref: str) -> bool:
    if not model_ref:
        return False
    path = Path(model_ref).expanduser()
    return (path.is_absolute() and path.exists()) or project_path(path).exists()


def _normalize_label(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def _suspicious_label_indices(labels: dict[int, str]) -> list[int]:
    attack_hints = ("jailbreak", "injection", "attack", "malicious", "unsafe", "harmful", "label_1")
    safe_hints = ("benign", "safe", "legitimate", "clean", "not_", "no_", "label_0")
    suspicious = []
    for idx, label in labels.items():
        normalized = _normalize_label(label)
        if any(hint in normalized for hint in attack_hints) and not any(hint in normalized for hint in safe_hints):
            suspicious.append(idx)
    if not suspicious and len(labels) == 2:
        suspicious = [1]
    return suspicious


class ModelPromptGuard:
    def __init__(self, model_ref: str, threshold: float = 0.5, device: str | None = None):
        if not is_local_model_ref(model_ref):
            raise FileNotFoundError(
                "Prompt Guard must be a local directory for the offline workflow. "
                f"Received: {model_ref}"
            )

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        path = Path(model_ref).expanduser()
        resolved = path if path.is_absolute() else project_path(path)
        self.model_ref = str(resolved)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ref, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str) -> dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()

        id2label = getattr(self.model.config, "id2label", {}) or {}
        labels = {idx: str(id2label.get(idx, f"LABEL_{idx}")) for idx in range(len(probabilities))}
        suspicious_indices = _suspicious_label_indices(labels)
        suspicious_score = max((probabilities[idx] for idx in suspicious_indices), default=max(probabilities))
        top_idx = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
        return {
            "guard_type": "model",
            "model_ref": self.model_ref,
            "is_suspicious": suspicious_score >= self.threshold,
            "score": float(suspicious_score),
            "label": labels[top_idx],
            "top_score": float(probabilities[top_idx]),
            "probabilities": {labels[idx]: float(probabilities[idx]) for idx in range(len(probabilities))},
        }


def build_prompt_guard(
    config: dict[str, Any],
    mode: str | None = None,
    model_ref: str = "",
    threshold: float | None = None,
    device: str | None = None,
) -> Callable[[str], dict[str, Any]]:
    mode = mode or config.get("safety", {}).get("prompt_guard_mode", "auto")
    threshold = threshold if threshold is not None else float(config.get("safety", {}).get("prompt_guard_threshold", 0.5))
    model_ref = model_ref or resolve_model_ref(config, "prompt_guard", required=False)

    if mode not in {"auto", "model", "rule"}:
        raise ValueError("prompt guard mode must be one of: auto, model, rule")

    if mode == "rule":
        return simple_prompt_guard

    if mode == "auto" and not is_local_model_ref(model_ref):
        return simple_prompt_guard

    return ModelPromptGuard(model_ref=model_ref, threshold=threshold, device=device)
