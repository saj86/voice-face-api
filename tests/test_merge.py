"""Quick sanity tests for the merge logic — no model dependencies needed."""
from app.voice.merge import assign_speaker, merge_words


def test_assign_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": 0},
        {"start": 5.0, "end": 10.0, "speaker": 1},
    ]
    assert assign_speaker({"start": 1.0, "end": 2.0}, turns) == 0
    assert assign_speaker({"start": 6.0, "end": 7.0}, turns) == 1
    # straddles the boundary, more time in turn 1
    assert assign_speaker({"start": 4.5, "end": 6.5}, turns) == 1


def test_assign_no_overlap_snaps_nearest():
    turns = [{"start": 0.0, "end": 2.0, "speaker": 0}, {"start": 8.0, "end": 10.0, "speaker": 1}]
    assert assign_speaker({"start": 8.4, "end": 8.6}, turns) == 1


def test_assign_no_turns():
    assert assign_speaker({"start": 0.0, "end": 1.0}, []) == 0


def test_merge_groups_consecutive():
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": 0},
        {"start": 3.0, "end": 6.0, "speaker": 1},
    ]
    words = [
        {"start": 0.0, "end": 1.0, "word": " Hello"},
        {"start": 1.0, "end": 2.0, "word": " there"},
        {"start": 3.1, "end": 4.0, "word": " Hi"},
        {"start": 4.0, "end": 5.0, "word": " back"},
    ]
    out = merge_words(words, turns)
    assert len(out) == 2
    assert out[0] == {"start": 0.0, "end": 2.0, "speaker": "Speaker 1", "text": "Hello there"}
    assert out[1] == {"start": 3.1, "end": 5.0, "speaker": "Speaker 2", "text": "Hi back"}


def test_merge_empty():
    assert merge_words([], []) == []


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
