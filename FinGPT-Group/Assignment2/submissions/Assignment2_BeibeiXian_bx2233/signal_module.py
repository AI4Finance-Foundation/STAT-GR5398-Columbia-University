"""
{
    "sentiment_score": float  in [-2.0, +2.0],
    "direction"      : str    in {"Bullish", "Neutral", "Bearish"},
    "confidence"     : float  in [0.0, 1.0],
    "event_type"     : str    e.g. "earnings", "M&A", "macro", "regulation",
                                  "product", "guidance", "other",
    "urgency"        : float  in [0.0, 1.0]   (short-term impact urgency),
    "rationale"      : str    the 'analysis' free-text,
    "bucket"         : str    raw 5-class label (simplified mode only),
}
"""

from __future__ import annotations
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Constants
DIRECTION_CANON = {"Bullish", "Neutral", "Bearish"}

DIRECTION_MAP = {
    "bullish": "Bullish", "positive": "Bullish", "up": "Bullish",
    "buy": "Bullish", "long": "Bullish", "rise": "Bullish", "rising": "Bullish",
    "bearish": "Bearish", "negative": "Bearish", "down": "Bearish",
    "sell": "Bearish", "short": "Bearish", "fall": "Bearish", "falling": "Bearish",
    "neutral": "Neutral", "flat": "Neutral", "hold": "Neutral",
    "mixed": "Neutral", "unchanged": "Neutral",
}
DIRECTION_TO_NUM = {"Bullish": 1, "Neutral": 0, "Bearish": -1}

EVENT_TYPES = {
    "earnings", "m&a", "macro", "regulation",
    "product", "guidance", "legal", "management", "other"
}

# 5-class bucket -> (direction, sentiment_score in [-2, +2])
BUCKET_TO_SIGNAL: Dict[str, Tuple[str, float]] = {
    "Up 3-5%":   ("Bullish", 1.5),
    "Up 1-3%":   ("Bullish", 0.75),
    "Neutral":   ("Neutral", 0.0),
    "Down 1-3%": ("Bearish", -0.75),
    "Down 3-5%": ("Bearish", -1.5),
    "Up >5%":    ("Bullish", 2.0),
    "Down >5%":  ("Bearish", -2.0),
}

VALID_BUCKETS = list(BUCKET_TO_SIGNAL.keys())

SIMPLIFIED_EVAL_PROMPT = (
    'Return ONLY a JSON object.\n'
    'Use exactly this schema: {"prediction":"...", "analysis":"..."}\n'
    'prediction must be exactly one of: "Up 1-3%", "Up 3-5%", "Down 1-3%", '
    '"Down 3-5%", "Neutral".\n'
    'Do not include positive developments, concerns, explanations before the JSON, '
    'markdown, or commentary.\n\n'
    '__CLEAN_INPUT__\n\n'
    'Output: {"prediction":"'
)

SYSTEM_PROMPT = """You are FinGPT-Signal, a specialized financial signal-generation module.
You read a piece of financial news or company context and return ONLY a JSON
object. Do NOT output any text before or after the JSON. Do NOT wrap it in
markdown code fences. Use the exact schema below.

{
  "sentiment_score": <float in [-2, 2], -2=very bearish, 0=neutral, +2=very bullish>,
  "direction": <one of "Bullish", "Neutral", "Bearish">,
  "confidence": <float in [0, 1], how confident you are in this signal>,
  "event_type": <"earnings"|"M&A"|"macro"|"regulation"|"product"|"guidance"|"legal"|"management"|"other">,
  "urgency": <float in [0, 1], short-term impact urgency within the next 5 trading days>,
  "rationale": <<=40 words single sentence explaining the signal>
}
"""


# Structured signal container
@dataclass
class SignalOutput:
    sentiment_score: float
    direction: str
    confidence: float
    event_type: str
    urgency: float
    rationale: str
    bucket: str = ""             
    raw_text: str = ""
    inference_time: float = 0.0
    samples: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def direction_num(self) -> int:
        return DIRECTION_TO_NUM.get(self.direction, 0)


# Helpers — prompt cleaning + bucket parsing
def strip_llama_wrapper(text: str) -> str:
    """Strip Llama-2 [INST]<<SYS>>...<</SYS>>[/INST] wrapping if present."""
    if text is None:
        return ""
    t = text
    t = re.sub(r"\[INST\]\s*", "", t)
    t = re.sub(r"\s*\[/INST\]", "", t)
    t = re.sub(r"<<SYS>>[\s\S]*?<</SYS>>", "", t)
    return t.strip()


def normalize_bucket(text: str) -> Optional[str]:
    """Normalize any label (ground-truth or pred) into one of VALID_BUCKETS."""
    if text is None:
        return None
    t = str(text).strip().lower()
    t = t.replace("—", "-").replace("–", "-")
    if "neutral" in t:
        return "Neutral"
    direction_word = "up" if "up" in t else ("down" if "down" in t else None)
    if direction_word is None:
        return None
    if "more than 5" in t or ">5" in t or "more 5" in t:
        return "Up >5%" if direction_word == "up" else "Down >5%"
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if len(nums) >= 2:
        hi = float(nums[1])
        bucket_suffix = "1-3%" if hi <= 3 else "3-5%"
    elif len(nums) == 1:
        v = float(nums[0])
        bucket_suffix = "1-3%" if v <= 3 else "3-5%"
    else:
        return None
    return f"{'Up' if direction_word == 'up' else 'Down'} {bucket_suffix}"


def bucket_to_signal(bucket: str) -> Tuple[str, float]:
    """5-class bucket -> (direction, sentiment_score)."""
    return BUCKET_TO_SIGNAL.get(bucket, ("Neutral", 0.0))


# Core module
class FinGPTSignalModule:
    """Reusable LLM-to-signal module built on a LoRA-fine-tuned FinGPT."""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = True,          # <-- default matches retrain notebook
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        mode: str = "simplified",           # "simplified" or "structured"
    ):
        assert mode in ("simplified", "structured")
        self.mode = mode

        print(f"[FinGPT-Signal] Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if load_in_4bit and load_in_8bit:
            raise ValueError("choose one of load_in_4bit / load_in_8bit")
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[FinGPT-Signal] Loading base model (4bit={load_in_4bit}, 8bit={load_in_8bit})")
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

        if adapter_path:
            from peft import PeftModel
            print(f"[FinGPT-Signal] Attaching LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[FinGPT-Signal] Ready.  mode={self.mode}  total_params={n_params:,}")

    # ----------------- prompt building -----------------
    def _build_prompt(self, news_text: str, company_info: Optional[str] = None) -> str:
        if self.mode == "simplified":
            # strip any [INST]/<<SYS>> wrapping from the raw Dow30 input_text
            clean = strip_llama_wrapper(news_text)
            if company_info:
                clean = f"[Company Context]\n{company_info}\n\n{clean}"
            return SIMPLIFIED_EVAL_PROMPT.replace("__CLEAN_INPUT__", clean)

        # ---- structured (old 6-field) path ----
        user_msg = ""
        if company_info:
            user_msg += f"[Company Context]\n{company_info}\n\n"
        user_msg += f"[News]\n{news_text}\n\n"
        user_msg += "Return ONLY the JSON signal object described in the system prompt."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{user_msg}\n"
                f"<|assistant|>\n"
            )

    # ----------------- simplified-mode parser -----------------
    @staticmethod
    def _parse_simplified(raw_text: str) -> Dict:
        """
        Parse the completion that follows  Output: {"prediction":"
        The model emits something like:   Up 1-3%", "analysis":"..."}
        Returns a unified signal dict.
        """
        txt = raw_text or ""
        # Remove any accidental <think>...</think> leakage (deepseek habit)
        txt = re.sub(r"</?think>", "", txt, flags=re.IGNORECASE)

        # 1) try to reconstruct the full JSON: the prompt ended with
        #    {"prediction":"   so the model output continues that string.
        reconstructed = '{"prediction":"' + txt
        # Trim at first closing brace after the analysis field if present.
        end = reconstructed.find("}")
        json_candidate = reconstructed[: end + 1] if end != -1 else reconstructed
        json_candidate = re.sub(r",\s*}", "}", json_candidate)

        pred_label: Optional[str] = None
        analysis: str = ""
        try:
            obj = json.loads(json_candidate)
            pred_label = obj.get("prediction")
            analysis = obj.get("analysis", "") or ""
        except Exception:
            # 2) regex fallback
            m_pred = re.search(
                r'(Up\s+\d(?:\.\d)?-\d(?:\.\d)?%|Down\s+\d(?:\.\d)?-\d(?:\.\d)?%|Neutral|Up\s+>\s*5%|Down\s+>\s*5%)',
                txt,
                flags=re.IGNORECASE,
            )
            if m_pred:
                pred_label = m_pred.group(1)
            m_an = re.search(r'"analysis"\s*:\s*"([^"]*)', txt)
            analysis = m_an.group(1) if m_an else txt[:400]

        bucket = normalize_bucket(pred_label) or "Neutral"
        direction, sent = bucket_to_signal(bucket)

        return {
            "bucket": bucket,
            "direction": direction,
            "sentiment_score": sent,
            "confidence": 0.5,           
            "event_type": "other",
            "urgency": 0.5,
            "rationale": analysis.strip()[:800],
        }

    # ----------------- structured-mode parser -----------------
    @staticmethod
    def _parse_structured(raw_text: str) -> Dict:
        """Best-effort 6-field JSON parsing with regex fallback."""
        t = re.sub(r"<think>[\s\S]*?</think>", "", raw_text or "", flags=re.IGNORECASE)
        m = re.search(r"\{[\s\S]*?\}", t)
        if m:
            candidate = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            try:
                return json.loads(candidate)
            except Exception:
                pass

        data: Dict = {}
        m = re.search(r'"?direction"?\s*[:=]\s*"?([A-Za-z]+)"?', t)
        if m:
            data["direction"] = DIRECTION_MAP.get(m.group(1).lower(), "Neutral")
        else:
            tokens = re.findall(
                r"\b(bullish|bearish|neutral|positive|negative|up|down|rise|fall)\b",
                t.lower(),
            )
            data["direction"] = DIRECTION_MAP.get(tokens[0], "Neutral") if tokens else "Neutral"

        m = re.search(r'"?sentiment_score"?\s*[:=]\s*(-?\d+\.?\d*)', t)
        data["sentiment_score"] = float(m.group(1)) if m else {
            "Bullish": 1.0, "Neutral": 0.0, "Bearish": -1.0
        }[data["direction"]]
        m = re.search(r'"?confidence"?\s*[:=]\s*(\d+\.?\d*)', t)
        data["confidence"] = float(m.group(1)) if m else 0.5
        m = re.search(r'"?urgency"?\s*[:=]\s*(\d+\.?\d*)', t)
        data["urgency"] = float(m.group(1)) if m else 0.5
        m = re.search(r'"?event_type"?\s*[:=]\s*"?([A-Za-z&]+)"?', t)
        data["event_type"] = m.group(1).lower() if m else "other"
        m = re.search(r'"?rationale"?\s*[:=]\s*"([^"\n]+)"', t)
        data["rationale"] = m.group(1) if m else t.strip()[:200]
        return data

    @staticmethod
    def _validate(data: Dict) -> Dict:
        """Clip + canonicalize field values to guarantee schema compliance."""
        out = dict(data)
        d = str(out.get("direction", "Neutral"))
        out["direction"] = d if d in DIRECTION_CANON else DIRECTION_MAP.get(d.lower(), "Neutral")

        def _f(x, default):
            try:
                return float(x)
            except Exception:
                return default

        out["sentiment_score"] = max(-2.0, min(2.0, _f(out.get("sentiment_score"), 0.0)))
        out["confidence"]      = max(0.0,  min(1.0, _f(out.get("confidence"), 0.5)))
        out["urgency"]         = max(0.0,  min(1.0, _f(out.get("urgency"), 0.5)))
        out["event_type"]      = str(out.get("event_type", "other"))[:32].lower()
        out["rationale"]       = str(out.get("rationale", ""))[:800]
        out["bucket"]          = str(out.get("bucket", ""))[:32]
        return out

    # ----------------- raw generation -----------------
    @torch.inference_mode()
    def generate_raw(
        self,
        news_text: str,
        company_info: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> Tuple[str, float]:
        prompt = self._build_prompt(news_text, company_info)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3500
        ).to(self.model.device)

        t0 = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        dt = time.time() - t0
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        gen = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return gen, dt

    # ----------------- single-shot signal -----------------
    def generate_signal(
        self,
        news_text: str,
        company_info: Optional[str] = None,
        max_new_tokens: int = 160,
        temperature: float = 0.2,
        do_sample: bool = False,
    ) -> SignalOutput:
        gen, dt = self.generate_raw(
            news_text, company_info,
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=do_sample,
        )
        if self.mode == "simplified":
            data = self._validate(self._parse_simplified(gen))
        else:
            data = self._validate(self._parse_structured(gen))
        data["raw_text"] = gen
        data["inference_time"] = dt
        return SignalOutput(**data)

    # ----------------- self-consistency calibration -----------------
    def generate_with_calibration(
        self,
        news_text: str,
        company_info: Optional[str] = None,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 160,
    ) -> SignalOutput:
        """
        Sample N times at non-zero temperature and aggregate.

        direction / bucket = majority vote
        confidence         = majority share  (ECE-style calibration)
        sentiment_score    = mean among agreeing samples
        rationale          = first agreeing sample's analysis
        """
        samples: List[Dict] = []
        total_time = 0.0
        for _ in range(n_samples):
            gen, dt = self.generate_raw(
                news_text, company_info,
                temperature=temperature, do_sample=True,
                max_new_tokens=max_new_tokens,
            )
            if self.mode == "simplified":
                data = self._validate(self._parse_simplified(gen))
            else:
                data = self._validate(self._parse_structured(gen))
            data["raw_text"] = gen
            samples.append(data)
            total_time += dt

        # majority vote on bucket when in simplified mode, else direction
        vote_key = "bucket" if self.mode == "simplified" else "direction"
        labels = [s[vote_key] for s in samples]
        top_label, top_count = Counter(labels).most_common(1)[0]
        conf = top_count / n_samples
        agreeing = [s for s in samples if s[vote_key] == top_label]

        if self.mode == "simplified":
            direction, _ = bucket_to_signal(top_label)
            bucket = top_label
        else:
            direction = top_label
            bucket = ""

        mean_sent = sum(s["sentiment_score"] for s in agreeing) / len(agreeing)
        mean_urg  = sum(s["urgency"] for s in samples) / n_samples
        event = Counter(s["event_type"] for s in samples).most_common(1)[0][0]
        rationale = agreeing[0]["rationale"] if agreeing else ""

        return SignalOutput(
            sentiment_score=mean_sent,
            direction=direction,
            confidence=conf,
            event_type=event,
            urgency=mean_urg,
            rationale=rationale,
            bucket=bucket,
            raw_text=json.dumps([{k: v for k, v in s.items() if k != "raw_text"} for s in samples]),
            inference_time=total_time,
            samples=samples,
        )

    # ----------------- batch convenience -----------------
    def batch_generate(
        self,
        news_list: List[str],
        company_list: Optional[List[str]] = None,
        calibrate: bool = False,
        n_samples: int = 5,
        verbose: bool = True,
    ) -> List[SignalOutput]:
        if company_list is None:
            company_list = [None] * len(news_list)
        out = []
        for i, (news, comp) in enumerate(zip(news_list, company_list)):
            if verbose and i % 10 == 0:
                print(f"  [{i}/{len(news_list)}]")
            if calibrate:
                sig = self.generate_with_calibration(news, comp, n_samples=n_samples)
            else:
                sig = self.generate_signal(news, comp)
            out.append(sig)
        return out


# Composite signal example — simple linear combination
def composite_score(sig: SignalOutput,
                    w_sent: float = 0.5,
                    w_dir: float  = 0.3,
                    w_urg: float  = 0.2) -> float:
    """Single scalar quant-signal (higher = more bullish), confidence-weighted."""
    s = (w_sent * (sig.sentiment_score / 2.0)          
         + w_dir * sig.direction_num                    
         + w_urg * sig.urgency * sig.direction_num)     
    return float(s * sig.confidence)                    
