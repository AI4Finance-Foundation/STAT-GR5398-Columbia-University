from __future__ import annotations
import re
import json
from statistics import mean
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

from signal_module import (
    SignalOutput,
    DIRECTION_TO_NUM,
    normalize_bucket,
    bucket_to_signal,
    BUCKET_TO_SIGNAL,
)



def parse_gt_direction(answer_text: str) -> str:
    """Parse long-form teacher answer -> {Bullish, Neutral, Bearish}."""
    text = (answer_text or "").lower()

    m = re.search(
        r"prediction[^\n]*?(up by|down by|up more than|down more than|up\b|down\b|flat|unchanged|mixed|neutral)",
        text,
    )
    if m:
        tok = m.group(1)
        if "up" in tok:
            return "Bullish"
        if "down" in tok:
            return "Bearish"
        return "Neutral"

    if "up by" in text or "increase" in text or "positive outlook" in text:
        return "Bullish"
    if "down by" in text or "decrease" in text or "negative outlook" in text:
        return "Bearish"
    return "Neutral"


def parse_gt_magnitude(answer_text: str) -> float:
    """Signed % magnitude parsed from teacher answer (0 if not found)."""
    text = (answer_text or "").lower()
    direction = parse_gt_direction(answer_text)
    m = re.search(r"(\d+\.?\d*)\s*%", text)
    mag = float(m.group(1)) if m else 0.0
    sign = 1.0 if direction == "Bullish" else (-1.0 if direction == "Bearish" else 0.0)
    return sign * mag


def parse_gt_bucket_simplified(target_text: str) -> Optional[str]:
    """
    Parse the retrained-dataset target_text (simplified format).
    It's a short JSON/text like  {"prediction":"Up by 2-3%", ...}  or
    "Up by 1-2%" / "Down by 3-4%" / "Neutral".
    Returns the canonical 5-class bucket (Up 1-3%, Up 3-5%, Down 1-3%,
    Down 3-5%, Neutral, Up >5%, Down >5%) or None if unparsable.
    """
    if target_text is None:
        return None
    s = target_text.strip()
    # try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "prediction" in obj:
            s = str(obj["prediction"])
    except Exception:
        # try to pull from "prediction": "..."
        m = re.search(r'"prediction"\s*:\s*"([^"]+)"', s)
        if m:
            s = m.group(1)
    return normalize_bucket(s)


def bucket_to_magnitude(bucket: str) -> float:
    """5-class bucket -> signed midpoint % used for MSE vs sentiment."""
    m = {
        "Up 1-3%":   2.0,
        "Up 3-5%":   4.0,
        "Up >5%":    6.0,
        "Neutral":   0.0,
        "Down 1-3%": -2.0,
        "Down 3-5%": -4.0,
        "Down >5%":  -6.0,
    }
    return m.get(bucket, 0.0)


# Core metrics
def binary_and_ternary_accuracy(preds: List[str], gts: List[str]) -> Dict[str, float]:
    n = max(1, len(preds))
    ternary = sum(p == g for p, g in zip(preds, gts)) / n

    pairs = [(p, g) for p, g in zip(preds, gts) if g != "Neutral" and p != "Neutral"]
    binary = (sum(p == g for p, g in pairs) / len(pairs)) if pairs else 0.0

    # inclusive binary
    pairs_incl = [(p, g) for p, g in zip(preds, gts) if g != "Neutral"]
    binary_incl = (
        sum(p == g for p, g in pairs_incl) / len(pairs_incl) if pairs_incl else 0.0
    )

    return {
        "ternary_accuracy": round(ternary, 4),
        "binary_accuracy": round(binary, 4),
        "binary_accuracy_inclusive": round(binary_incl, 4),
        "n_ternary": n,
        "n_binary_strict": len(pairs),
        "n_binary_inclusive": len(pairs_incl),
    }


def bucket_accuracy(preds: List[str], gts: List[str]) -> Dict[str, float]:
    """5-class bucket accuracy (e.g. Up 1-3% vs Up 3-5%)."""
    pairs = [(p, g) for p, g in zip(preds, gts) if g is not None and p is not None]
    if not pairs:
        return {"bucket_accuracy": 0.0, "n": 0}
    acc = sum(p == g for p, g in pairs) / len(pairs)
    # near-miss accuracy: pred is in the same direction (Up vs Down) as GT
    def _dir(b):
        if b == "Neutral":
            return "N"
        if b.startswith("Up"):
            return "U"
        if b.startswith("Down"):
            return "D"
        return "?"
    near = sum(_dir(p) == _dir(g) for p, g in pairs) / len(pairs)
    return {
        "bucket_accuracy": round(acc, 4),
        "same_direction_accuracy": round(near, 4),
        "n": len(pairs),
    }


def mse_sentiment_vs_return(pred_sent_scores: List[float],
                            gt_magnitudes: List[float],
                            scale: float = 2.5) -> float:
    """sentiment_score in [-2,+2]; gt_magnitudes in %."""
    if not pred_sent_scores:
        return 0.0
    sq = [((p * scale) - g) ** 2 for p, g in zip(pred_sent_scores, gt_magnitudes)]
    return round(mean(sq), 4)


def rouge_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """ROUGE-1/2/L between rationale and teacher answer."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        if not p or not r:
            continue
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    if not r1:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0}
    return {
        "rouge1": round(mean(r1), 4),
        "rouge2": round(mean(r2), 4),
        "rougeL": round(mean(rl), 4),
        "n": len(r1),
    }


def information_coefficient(scores: List[float], returns: List[float]) -> Dict[str, float]:
    """Spearman IC between composite signal and realized returns."""
    try:
        from scipy.stats import spearmanr, pearsonr
    except ImportError:
        return {}
    if len(scores) < 3:
        return {"IC_spearman": 0.0, "IC_pearson": 0.0, "n": len(scores)}
    rho_s, p_s = spearmanr(scores, returns)
    rho_p, p_p = pearsonr(scores, returns)
    return {
        "IC_spearman": round(float(rho_s), 4), "IC_spearman_p": round(float(p_s), 4),
        "IC_pearson":  round(float(rho_p), 4), "IC_pearson_p":  round(float(p_p), 4),
        "n": len(scores),
    }


def reliability_diagram(confs: List[float], correct: List[int],
                        n_bins: int = 5) -> List[Dict]:
    """Per-bin (avg_confidence, accuracy, count)."""
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, y))
    out = []
    for i, b in enumerate(bins):
        if not b:
            continue
        cs, ys = zip(*b)
        out.append({
            "bin": f"[{i/n_bins:.2f}, {(i+1)/n_bins:.2f})",
            "avg_confidence": round(mean(cs), 4),
            "accuracy": round(mean(ys), 4),
            "count": len(b),
        })
    return out


def expected_calibration_error(confs: List[float], correct: List[int],
                               n_bins: int = 10) -> float:
    """ECE — lower is better-calibrated."""
    if not confs:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(confs, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, y))
    n = len(confs)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        cs, ys = zip(*b)
        ece += (len(b) / n) * abs(mean(cs) - mean(ys))
    return round(ece, 4)


# Top-level driver
def evaluate(
    signals: List[SignalOutput],
    gt_answers: List[str],
    realized_returns: Optional[List[float]] = None,
    mode: str = "simplified",
) -> Dict:
    """
    Parameters
    ----------
    signals            : list of SignalOutput  (from FinGPTSignalModule)
    gt_answers         : list of ground-truth strings:
                          - "simplified" mode: target_text from retrained dataset
                            (e.g. '{"prediction":"Up by 2-3%","analysis":"..."}')
                          - "legacy" mode: long-form teacher answer
    realized_returns   : optional list of real % returns for IC computation
    mode               : "simplified" or "legacy"
    """
    assert mode in ("simplified", "legacy")
    assert len(signals) == len(gt_answers), "length mismatch"

    preds_dir = [s.direction for s in signals]
    preds_buckets = [s.bucket for s in signals]
    preds_sent = [s.sentiment_score for s in signals]
    preds_text = [s.rationale or s.raw_text[:300] for s in signals]
    confs = [s.confidence for s in signals]
    inf_times = [s.inference_time for s in signals]

    if mode == "simplified":
        gt_buckets = [parse_gt_bucket_simplified(a) for a in gt_answers]
        gts_dir = [bucket_to_signal(b)[0] if b else "Neutral" for b in gt_buckets]
        gts_mag = [bucket_to_magnitude(b) if b else 0.0 for b in gt_buckets]
        # parse teacher analysis text for ROUGE
        gt_analysis_texts: List[str] = []
        for a in gt_answers:
            try:
                obj = json.loads(a)
                gt_analysis_texts.append(str(obj.get("analysis", a)))
            except Exception:
                m = re.search(r'"analysis"\s*:\s*"([^"]*)', a or "")
                gt_analysis_texts.append(m.group(1) if m else (a or ""))
    else:
        gts_dir = [parse_gt_direction(a) for a in gt_answers]
        gts_mag = [parse_gt_magnitude(a) for a in gt_answers]
        gt_buckets = [None] * len(gts_dir)
        gt_analysis_texts = list(gt_answers)

    acc = binary_and_ternary_accuracy(preds_dir, gts_dir)
    mse = mse_sentiment_vs_return(preds_sent, gts_mag)
    rouge = rouge_scores(preds_text, gt_analysis_texts)
    bucket_metrics = (
        bucket_accuracy(preds_buckets, gt_buckets) if mode == "simplified" else {}
    )

    correct = [1 if p == g else 0 for p, g in zip(preds_dir, gts_dir)]
    ece = expected_calibration_error(confs, correct)
    rd  = reliability_diagram(confs, correct)

    result = {
        "n_samples": len(signals),
        "mode": mode,
        "direction_accuracy": acc,
        "bucket_accuracy": bucket_metrics,
        "mse_vs_return_proxy": mse,
        "rouge": rouge,
        "avg_inference_time_s": round(mean(inf_times), 3) if inf_times else 0.0,
        "median_inference_time_s": round(sorted(inf_times)[len(inf_times)//2], 3) if inf_times else 0.0,
        "expected_calibration_error": ece,
        "reliability_bins": rd,
        "prediction_distribution": dict(Counter(preds_dir)),
        "ground_truth_distribution": dict(Counter(gts_dir)),
        "prediction_bucket_distribution": dict(Counter(preds_buckets)),
        "ground_truth_bucket_distribution": dict(Counter(gt_buckets)),
    }

    if realized_returns is not None:
        from signal_module import composite_score
        composites = [composite_score(s) for s in signals]
        result["information_coefficient"] = information_coefficient(
            composites, realized_returns
        )
        paired = list(zip(composites, realized_returns))
        paired.sort(key=lambda x: x[0])
        k = max(1, len(paired) // 10)
        bot = mean(r for _, r in paired[:k])
        top = mean(r for _, r in paired[-k:])
        result["long_short_return_pct"] = round(top - bot, 4)
        result["top_decile_mean_return_pct"] = round(top, 4)
        result["bottom_decile_mean_return_pct"] = round(bot, 4)

    return result


def pretty_print(metrics: Dict) -> None:
    print("=" * 70)
    print(" FinGPT Signal Module — Evaluation Report ")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))
