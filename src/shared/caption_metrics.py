"""Shared caption-level metrics and human-readable quality bands."""
from __future__ import annotations

from collections import Counter

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score

QUALITY_BAD = "bad"
QUALITY_PARTIAL = "partial"
QUALITY_GOOD = "good"
QUALITY_ORDER = [QUALITY_BAD, QUALITY_PARTIAL, QUALITY_GOOD]
QUALITY_DISPLAY = {
    QUALITY_BAD: "bad (red)",
    QUALITY_PARTIAL: "partial (orange)",
    QUALITY_GOOD: "good (green)",
}

_SMOOTH = SmoothingFunction().method1


def _best_token_overlap(refs: list[list[str]], hyp: list[str]) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for the best-overlap reference."""
    hyp_counts = Counter(hyp)
    best = (0.0, 0.0, 0.0)
    for ref in refs:
        ref_counts = Counter(ref)
        overlap = sum((hyp_counts & ref_counts).values())
        precision = overlap / max(len(hyp), 1)
        recall = overlap / max(len(ref), 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best[2]:
            best = (precision, recall, f1)
    return best


def classify_caption_quality(bleu1: float, meteor: float, overlap_f1: float) -> str:
    """Map token-level metrics to a coarse human-friendly caption quality band."""
    if meteor >= 0.45 or overlap_f1 >= 0.75 or (bleu1 >= 0.70 and overlap_f1 >= 0.55):
        return QUALITY_GOOD
    if meteor >= 0.25 or overlap_f1 >= 0.40 or bleu1 >= 0.45:
        return QUALITY_PARTIAL
    return QUALITY_BAD


def score_caption_prediction(refs: list[list[str]], hyp: list[str]) -> dict:
    """Compute sentence-level metrics plus the heuristic quality band."""
    bleu1 = sentence_bleu(refs, hyp, weights=(1, 0, 0, 0), smoothing_function=_SMOOTH)
    bleu4 = sentence_bleu(refs, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=_SMOOTH)
    meteor = meteor_score(refs, hyp)
    overlap_precision, overlap_recall, overlap_f1 = _best_token_overlap(refs, hyp)
    quality_label = classify_caption_quality(bleu1, meteor, overlap_f1)
    return {
        "bleu1": bleu1,
        "bleu4": bleu4,
        "meteor": meteor,
        "overlap_precision": overlap_precision,
        "overlap_recall": overlap_recall,
        "overlap_f1": overlap_f1,
        "quality_label": quality_label,
        "quality_display": QUALITY_DISPLAY[quality_label],
    }


def empty_caption_metrics(prefix: str) -> tuple[dict, dict]:
    counts = {label: 0 for label in QUALITY_ORDER}
    percentages = {label: float("nan") for label in QUALITY_ORDER}
    metrics = {
        f"{prefix}/bleu1": float("nan"),
        f"{prefix}/bleu4": float("nan"),
        f"{prefix}/meteor": float("nan"),
        f"{prefix}/token_overlap_f1": float("nan"),
        f"{prefix}/caption_accuracy_pct": float("nan"),
        f"{prefix}/caption_partial_pct": float("nan"),
        f"{prefix}/caption_bad_pct": float("nan"),
        f"{prefix}/caption_acceptable_pct": float("nan"),
        f"{prefix}/caption_quality_score_pct": float("nan"),
    }
    summary = {"total": 0, "counts": counts, "percentages": percentages}
    return metrics, summary


def aggregate_caption_scores(samples: list[dict], prefix: str) -> tuple[dict, dict]:
    if not samples:
        return empty_caption_metrics(prefix)

    all_refs = [sample["refs"] for sample in samples]
    all_hyps = [sample["hyp"] for sample in samples]
    counts = {label: 0 for label in QUALITY_ORDER}
    for sample in samples:
        counts[sample["quality_label"]] += 1

    total = len(samples)
    percentages = {label: 100.0 * counts[label] / total for label in QUALITY_ORDER}
    metrics = {
        f"{prefix}/bleu1": corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0)),
        f"{prefix}/bleu4": corpus_bleu(all_refs, all_hyps, weights=(0.25, 0.25, 0.25, 0.25)),
        f"{prefix}/meteor": float(np.mean([sample["meteor"] for sample in samples])),
        f"{prefix}/token_overlap_f1": float(np.mean([sample["overlap_f1"] for sample in samples])),
        f"{prefix}/caption_accuracy_pct": percentages[QUALITY_GOOD],
        f"{prefix}/caption_partial_pct": percentages[QUALITY_PARTIAL],
        f"{prefix}/caption_bad_pct": percentages[QUALITY_BAD],
        f"{prefix}/caption_acceptable_pct": percentages[QUALITY_GOOD] + percentages[QUALITY_PARTIAL],
        f"{prefix}/caption_quality_score_pct": percentages[QUALITY_GOOD] + 0.5 * percentages[QUALITY_PARTIAL],
    }
    summary = {"total": total, "counts": counts, "percentages": percentages}
    return metrics, summary


def quality_breakdown_rows(summary: dict) -> list[list[float | int | str]]:
    return [
        [
            QUALITY_DISPLAY[label],
            summary["counts"][label],
            round(summary["percentages"][label], 2),
        ]
        for label in QUALITY_ORDER
    ]
