"""
Post-call analysis (hybrid):

  - compute_metrics():  cheap, deterministic signals from transcript timestamps.
  - ConversationAnalyzer: a small local instruct LLM (llama.cpp / GGUF) that reads
                        the diarized transcript and returns a rich structured
                        analysis, in the language of your choice.

build_messages() / parse_llm_json() are pure and unit-tested without the model.
"""
import os
import re
import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger("analyze")

# Default language for the analysis output (free-text fields). German by default.
DEFAULT_ANALYSIS_LANG = os.environ.get("ANALYSIS_LANG", "de")

# ISO code -> language name, to instruct the LLM clearly.
_LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pt": "Portuguese", "pl": "Polish", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek", "tr": "Turkish", "ru": "Russian", "uk": "Ukrainian",
}


def language_name(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    return _LANG_NAMES.get(code.lower(), code)


# ----------------------------- deterministic metrics -----------------------------
def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def compute_metrics(transcript: Dict) -> Dict:
    segs = transcript.get("segments", []) or []
    duration = transcript.get("duration") or max((s["end"] for s in segs), default=0.0)

    by_spk: Dict[str, Dict] = {}
    total_speech = 0.0
    for s in segs:
        dur = max(0.0, s["end"] - s["start"])
        total_speech += dur
        d = by_spk.setdefault(s["speaker"], {"talk_time": 0.0, "words": 0, "turns": 0})
        d["talk_time"] += dur
        d["words"] += _word_count(s.get("text", ""))
        d["turns"] += 1

    speakers = {}
    for spk, d in by_spk.items():
        tt = d["talk_time"]
        speakers[spk] = {
            "talk_time": round(tt, 2),
            "talk_ratio": round(tt / total_speech, 3) if total_speech else 0.0,
            "words": d["words"],
            "turns": d["turns"],
            "words_per_min": round(d["words"] / (tt / 60.0), 1) if tt > 0 else 0.0,
        }

    ordered = sorted(segs, key=lambda s: s["start"])
    gaps = [round(b["start"] - a["end"], 2) for a, b in zip(ordered, ordered[1:]) if b["start"] > a["end"]]

    return {
        "duration": round(float(duration), 2),
        "total_speech": round(total_speech, 2),
        "total_silence": round(max(0.0, float(duration) - total_speech), 2),
        "longest_silence": max(gaps, default=0.0),
        "num_turns": len(segs),
        "num_speakers": len(speakers),
        "speakers": speakers,
    }


# ----------------------------- LLM prompt + parsing (pure) -----------------------------
_FIELDS = """JSON keys and allowed values:
- call_reason: text
- call_category: one of [billing, technical, sales, complaint, information, other]
- urgency: one of [low, medium, high]
- overall_sentiment: one of [positive, negative, neutral]
- customer_sentiment: one of [positive, negative, neutral]
- sentiment_start: one of [positive, negative, neutral]
- sentiment_end: one of [positive, negative, neutral]
- speaker_roles: object mapping each speaker label to one of [agent, customer, other]
- is_complaint: boolean
- complaint_reason: text (empty string if not a complaint)
- tone: text (one or two words)
- agent_professionalism: one of [poor, adequate, good, excellent, unclear]
- resolution: one of [resolved, unresolved, follow-up required, unclear]
- customer_satisfaction: one of [high, medium, low, unclear]
- escalation_needed: boolean
- complaint_severity: one of [none, low, medium, high]
- call_outcome: one of [resolved, unresolved, follow-up required, transferred, escalated, unclear]
- topics: array of short text
- key_points: array of text
- unanswered_questions: array of text
- commitments: array of text
- entities: object with products (array), order_or_article_numbers (array), people (array)
- summary: text, 3-5 sentences
- details: text, 2-4 sentences giving a fuller explanation/assessment of the call
- action_items: array of text
- follow_up_required: boolean"""

# One-shot examples anchor BOTH the JSON format and the output language.
_EXAMPLE = {
    "de": """{"call_reason":"Kunde fragt nach der Lederklasse seines Sofas","call_category":"information","urgency":"low","overall_sentiment":"neutral","customer_sentiment":"neutral","sentiment_start":"neutral","sentiment_end":"neutral","speaker_roles":{"Speaker 1":"agent","Speaker 2":"customer"},"is_complaint":false,"complaint_reason":"","tone":"sachlich","agent_professionalism":"good","resolution":"follow-up required","customer_satisfaction":"medium","escalation_needed":false,"complaint_severity":"none","call_outcome":"follow-up required","topics":["Sofa","Lederklasse"],"key_points":["Kunde nennt die Artikelnummer"],"unanswered_questions":["Welche Lederklasse genau?"],"commitments":["Agent stellt eine Anfrage"],"entities":{"products":["Sofa"],"order_or_article_numbers":["444-877"],"people":["Sabine Raming"]},"summary":"Der Kunde erkundigt sich nach der Lederklasse seines Sofas. Der Agent erklärt die Unterschiede und stellt eine Anfrage.","details":"Es handelt sich um eine reine Informationsanfrage ohne Beschwerde. Der Agent bleibt freundlich, kann die genaue Lederklasse aber nicht sofort nennen und sagt eine Rückmeldung zu.","action_items":["Lederklasse ermitteln und zurückmelden"],"follow_up_required":true}""",
    "en": """{"call_reason":"Customer asks about their sofa's leather class","call_category":"information","urgency":"low","overall_sentiment":"neutral","customer_sentiment":"neutral","sentiment_start":"neutral","sentiment_end":"neutral","speaker_roles":{"Speaker 1":"agent","Speaker 2":"customer"},"is_complaint":false,"complaint_reason":"","tone":"matter-of-fact","agent_professionalism":"good","resolution":"follow-up required","customer_satisfaction":"medium","escalation_needed":false,"complaint_severity":"none","call_outcome":"follow-up required","topics":["sofa","leather class"],"key_points":["Customer gives the article number"],"unanswered_questions":["Which exact leather class?"],"commitments":["Agent will raise a request"],"entities":{"products":["sofa"],"order_or_article_numbers":["444-877"],"people":["Sabine Raming"]},"summary":"The customer asks about the leather class of their sofa. The agent explains and raises a request.","details":"This is a pure information request with no complaint. The agent stays friendly but cannot state the exact leather class and promises to follow up.","action_items":["Find the leather class and report back"],"follow_up_required":true}""",
}

# strong, native-language directives (the most reliable way to force output language)
_DIRECTIVE = {
    "de": "Antworte ausschließlich auf Deutsch. Alle Freitext-Werte müssen auf Deutsch sein. Verwende kein Englisch.",
    "fr": "Réponds uniquement en français. Toutes les valeurs de texte libre doivent être en français.",
    "es": "Responde únicamente en español. Todos los valores de texto libre deben estar en español.",
    "it": "Rispondi esclusivamente in italiano. Tutti i valori di testo libero devono essere in italiano.",
    "nl": "Antwoord uitsluitend in het Nederlands. Alle vrije-tekstwaarden moeten in het Nederlands zijn.",
}

_ENUMS = {
    "call_category": {"billing", "technical", "sales", "complaint", "information", "other"},
    "urgency": {"low", "medium", "high"},
    "overall_sentiment": {"positive", "negative", "neutral"},
    "customer_sentiment": {"positive", "negative", "neutral"},
    "sentiment_start": {"positive", "negative", "neutral"},
    "sentiment_end": {"positive", "negative", "neutral"},
    "agent_professionalism": {"poor", "adequate", "good", "excellent", "unclear"},
    "resolution": {"resolved", "unresolved", "follow-up required", "unclear"},
    "customer_satisfaction": {"high", "medium", "low", "unclear"},
    "complaint_severity": {"none", "low", "medium", "high"},
    "call_outcome": {"resolved", "unresolved", "follow-up required", "transferred", "escalated", "unclear"},
}
_BOOLS = {"is_complaint", "escalation_needed", "follow_up_required"}


def sanitize_insights(d: Dict) -> Dict:
    """Drop junk enum values (e.g. a model echoing the option list) and coerce
    booleans, so the UI never shows things like 'billing | technical | ...'."""
    if not isinstance(d, dict):
        return {}
    for key, allowed in _ENUMS.items():
        v = d.get(key)
        if isinstance(v, str):
            vv = v.strip().lower()
            d[key] = vv if vv in allowed else None
        elif v is not None:
            d[key] = None
    for key in _BOOLS:
        v = d.get(key)
        if isinstance(v, str):
            d[key] = v.strip().lower() in ("true", "yes", "1", "ja")
        elif not isinstance(v, bool) and v is not None:
            d[key] = bool(v)
    return d


def build_messages(transcript: Dict, output_language: str = "auto", max_chars: int = 14000) -> List[Dict]:
    lines = [f'{s["speaker"]}: {s.get("text","")}' for s in transcript.get("segments", []) or []]
    convo = "\n".join(lines).strip() or "(empty transcript)"
    if len(convo) > max_chars:
        convo = convo[-max_chars:]

    lang = (output_language or "auto").lower()
    if lang == "auto":
        lang = (transcript.get("language") or DEFAULT_ANALYSIS_LANG or "de").lower()
    lang_full = language_name(lang) or "German"
    directive = _DIRECTIVE.get(lang, f"Respond only in {lang_full}. All free-text values must be in {lang_full}.")
    example = _EXAMPLE.get(lang, _EXAMPLE["en"])

    system = (
        "You are a contact-center call analyst. Respond with EXACTLY ONE JSON object "
        "and nothing else — no markdown, no commentary. Choose one value for each enum "
        "field (never output the list of options). " + directive
    )
    user = (
        f"{_FIELDS}\n\n{directive}\n\n"
        f"Example of a correctly formatted answer (copy this structure and language, "
        f"but base the content on the transcript below):\n{example}\n\n"
        f"Transcript:\n{convo}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)

# canonical per-line emotion labels (UI localizes them for display)
EMOTIONS = {"neutral", "frustrated", "angry", "happy", "satisfied", "confused"}

# Smart keyword alerts — canonical -> case-insensitive substrings (multilingual)
DEFAULT_KEYWORDS = {
    "refund": ["refund", "erstattung", "rückerstattung", "geld zurück", "rückzahlung",
               "remboursement", "reembolso", "rimborso"],
    "cancel": ["cancel", "kündig", "storniere", "storno", "widerruf", "abbestell",
               "annuler", "résilier", "cancelar", "annullare", "disdire"],
    "complaint": ["complaint", "beschwerde", "reklamation", "beschwer",
                  "réclamation", "reclamación", "reclamo"],
    "urgent": ["urgent", "dringend", "sofort", "eilig", "asap", "urgente"],
    "supervisor": ["supervisor", "vorgesetzt", "teamleiter", "manager", "chef",
                   "responsable", "superviseur", "superior"],
    "lawyer": ["lawyer", "anwalt", "rechtsanwalt", "juristisch", "klage", "gericht",
               "avocat", "abogado", "avvocato", "legal"],
}


def detect_alerts(segments, keywords=None):
    """Flag segments containing important keywords. Pure/deterministic."""
    kw = keywords or DEFAULT_KEYWORDS
    alerts = []
    for s in segments or []:
        text = s.get("text") or ""
        low = text.lower()
        for canon, variants in kw.items():
            for v in variants:
                if v in low:
                    alerts.append({
                        "keyword": canon,
                        "match": v,
                        "speaker": s.get("speaker"),
                        "start": s.get("start"),
                        "snippet": text.strip()[:160],
                    })
                    break  # at most one hit per keyword per segment
    return alerts


def parse_json_array(text):
    if not text:
        return []
    t = re.sub(r"```(?:json)?", "", text.strip()).replace("```", "").strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, list) else []
    except Exception:
        m = _ARR_RE.search(t)
        if m:
            try:
                v = json.loads(m.group(0))
                return v if isinstance(v, list) else []
            except Exception:
                return []
    return []


def coerce_emotions(raw, n):
    """Force a model's line-emotion output into exactly n valid labels."""
    out = []
    for i in range(n):
        v = raw[i] if i < len(raw) else None
        v = v.strip().lower() if isinstance(v, str) else ""
        out.append(v if v in EMOTIONS else "neutral")
    return out


def parse_llm_json(text: str) -> Dict:
    if not text:
        return {}
    t = re.sub(r"```(?:json)?", "", text.strip()).replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        m = _JSON_RE.search(t)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


# ----------------------------- LLM engine -----------------------------
class ConversationAnalyzer:
    def __init__(self, model_path: str, n_threads: int = 4, n_ctx: int = 8192):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"LLM model not found: {model_path}")
        from llama_cpp import Llama

        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, verbose=False)

    def analyze(self, transcript: Dict, output_language: str = "auto") -> Dict:
        messages = build_messages(transcript, output_language)
        try:
            resp = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception as e:
            log.exception("LLM analysis failed")
            return {"error": str(e)}
        return sanitize_insights(parse_llm_json(content))

    def analyze_lines(self, transcript, output_language="auto", max_lines=120):
        """Return a canonical emotion label for each transcript segment, in order.
        Labels stay English (the UI localizes them). Skips very long calls."""
        segs = transcript.get("segments", []) or []
        if not segs or len(segs) > max_lines:
            return []
        numbered = "\n".join(
            f'{i+1}. {s.get("speaker","")}: {s.get("text","")}' for i, s in enumerate(segs)
        )
        system = ("You label the emotion of the speaker on each transcript line. "
                  "Respond with ONLY a JSON array of strings, nothing else.")
        user = (
            "Allowed emotions (use these exact English words): "
            "[neutral, frustrated, angry, happy, satisfied, confused].\n"
            f"Return a JSON array of EXACTLY {len(segs)} strings, one per numbered line, in order.\n\n"
            f"{numbered}"
        )
        try:
            resp = self.llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=min(2000, len(segs) * 8 + 100),
            )
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            log.exception("line-emotion analysis failed")
            return []
        return coerce_emotions(parse_json_array(content), len(segs))


def analyze_call(transcript: Dict, analyzer: Optional[ConversationAnalyzer],
                 output_language: str = "auto") -> Dict:
    segments = transcript.get("segments", []) or []
    result = {
        "confidence": transcript.get("confidence"),
        "language": transcript.get("language"),
        "metrics": compute_metrics(transcript),
        "voice": transcript.get("voice"),
        "alerts": detect_alerts(segments),
    }
    if analyzer is not None:
        result["insights"] = analyzer.analyze(transcript, output_language)
        if os.environ.get("LINE_EMOTIONS", "1") == "1":
            result["line_emotions"] = analyzer.analyze_lines(transcript, output_language)
        else:
            result["line_emotions"] = []
    else:
        result["insights"] = None
        result["line_emotions"] = []
    return result
