"""Tests for the pure analysis helpers (no LLM needed)."""
from app.voice.analyze import compute_metrics, build_messages, parse_llm_json


TRANSCRIPT = {
    "duration": 20.0,
    "segments": [
        {"start": 0.0, "end": 4.0, "speaker": "Speaker 1", "text": "hello how can i help you today"},   # 6 words / 4s
        {"start": 5.0, "end": 9.0, "speaker": "Speaker 2", "text": "my internet keeps dropping again"},   # 5 words / 4s, 1s gap
        {"start": 9.0, "end": 12.0, "speaker": "Speaker 1", "text": "i am sorry to hear that"},
    ],
}


def test_metrics_talk_time_and_silence():
    m = compute_metrics(TRANSCRIPT)
    assert m["num_turns"] == 3
    assert m["total_speech"] == 11.0          # 4 + 4 + 3
    assert m["total_silence"] == 9.0          # 20 - 11
    assert m["longest_silence"] == 1.0        # the 4->5 gap
    assert set(m["speakers"].keys()) == {"Speaker 1", "Speaker 2"}
    s1 = m["speakers"]["Speaker 1"]
    assert s1["turns"] == 2 and s1["talk_time"] == 7.0
    assert s1["words"] == 13
    assert s1["words_per_min"] == round(13 / (7 / 60.0), 1)


def test_metrics_empty():
    m = compute_metrics({"segments": []})
    assert m["num_turns"] == 0 and m["speakers"] == {}


def test_build_messages_includes_speakers():
    msgs = build_messages(TRANSCRIPT)
    assert msgs[0]["role"] == "system"
    assert "Speaker 1: hello how can i help you today" in msgs[1]["content"]


def test_build_messages_german_directive():
    msgs = build_messages(TRANSCRIPT, "de")
    blob = msgs[0]["content"] + msgs[1]["content"]
    assert "Deutsch" in blob                      # native-language directive present
    assert "Speaker 1: hello how can i help you today" in msgs[1]["content"]


def test_build_messages_auto_uses_detected_language():
    t = dict(TRANSCRIPT); t["language"] = "fr"
    blob = "".join(m["content"] for m in build_messages(t, "auto"))
    assert "français" in blob


def test_sanitize_drops_option_list_echo():
    from app.voice.analyze import sanitize_insights
    bad = {"call_category": "billing | technical | sales | complaint | information | other",
           "urgency": "medium", "overall_sentiment": "neutral",
           "is_complaint": "false", "escalation_needed": "ja"}
    out = sanitize_insights(bad)
    assert out["call_category"] is None        # junk echo rejected
    assert out["urgency"] == "medium"          # valid kept
    assert out["overall_sentiment"] == "neutral"
    assert out["is_complaint"] is False        # string coerced to bool
    assert out["escalation_needed"] is True


def test_parse_plain_json():
    assert parse_llm_json('{"overall_sentiment": "negative"}')["overall_sentiment"] == "negative"


def test_parse_fenced_json():
    txt = '```json\n{"is_complaint": true, "tone": "frustrated"}\n```'
    out = parse_llm_json(txt)
    assert out["is_complaint"] is True and out["tone"] == "frustrated"


def test_parse_json_with_surrounding_text():
    txt = 'Here is the analysis: {"overall_sentiment": "neutral"} hope this helps'
    assert parse_llm_json(txt)["overall_sentiment"] == "neutral"


def test_parse_garbage_returns_empty():
    assert parse_llm_json("not json at all") == {}




def test_detect_alerts_multilingual():
    from app.voice.analyze import detect_alerts
    segs=[{"speaker":"Speaker 2","start":5.0,"text":"Ich möchte eine Erstattung und sonst kündige ich"},
          {"speaker":"Speaker 1","start":9.0,"text":"Alles gut, kein Problem"}]
    al=detect_alerts(segs)
    kinds={a["keyword"] for a in al}
    assert "refund" in kinds and "cancel" in kinds
    assert all("start" in a and "speaker" in a for a in al)


def test_parse_json_array_and_coerce():
    from app.voice.analyze import parse_json_array, coerce_emotions
    assert parse_json_array('```json\n["neutral","angry"]\n```')==["neutral","angry"]
    assert parse_json_array("junk")==[]
    assert coerce_emotions(["angry","bogus"],4)==["angry","neutral","neutral","neutral"]


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
