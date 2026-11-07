from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

from app import sentences
from app.ai import AIClient
from app.database import Database, utc_now
from app.sentences import SplitDecision


GUARDED = 0.5  # a coverage under recordings.COVERAGE_MIN


def line(text: str, gap_ms: int, number: int, coverage: float = 1.0) -> dict:
    return {"text": text, "runs": [number], "start_ms": number * 1000, "end_ms": number * 1000 + 800,
            "gap_after_ms": gap_ms, "break_s": None, "coverage": coverage}


def segment(index: int, narration: str = "", faithful: str = "", lines: list[dict] | None = None, kind: str = "narration") -> dict:
    if lines:
        narration = "\n\n".join(item["text"] for item in lines)
    return {"id": f"seg-{index}", "segment_index": index, "kind": kind, "narration_en": narration,
            "faithful_en": faithful, "lines": lines or []}


def back(narration: str = "", faithful: str = "") -> SplitDecision:
    return SplitDecision(narration_direction="back" if narration else "", fragment_narration=narration,
                         faithful_direction="back" if faithful else "", fragment_faithful=faithful)


def forward(narration: str = "", faithful: str = "") -> SplitDecision:
    return SplitDecision(narration_direction="forward" if narration else "", fragment_narration=narration,
                         faithful_direction="forward" if faithful else "", fragment_faithful=faithful)


# --- pairs -------------------------------------------------------------------------------------------


def test_pairs_are_neighbouring_narration_segments_with_english():
    rows = [
        segment(1, "One.", "One."),
        segment(2, "Two and", "Two and"),
        segment(3, "three.", "three."),
        segment(4, kind="original"),
        segment(5, "Five.", "Five."),
        segment(6, "", ""),
        segment(7, "Seven.", "Seven."),
    ]

    found = sentences.pairs(list(reversed(rows)))

    assert [(a["segment_index"], b["segment_index"]) for a, b in found] == [(1, 2), (2, 3)]


def test_a_decision_without_a_direction_is_no_split():
    assert not SplitDecision().split
    assert SplitDecision(faithful_direction="back", fragment_faithful="and").split


# --- plain text moves ---------------------------------------------------------------------------------


def test_a_back_move_carries_the_start_of_the_next_segment_onto_this_one():
    a = segment(1, "He drove home and", "He drove home and")
    b = segment(2, "parked the car. Then he slept.", "parked the car.  Then he\nslept.")

    a_fields, b_fields, problems = sentences.apply_move(a, b, back("parked the car.", "parked   the car."))

    assert problems == []
    assert a_fields == {"narration_en": "He drove home and parked the car.", "faithful_en": "He drove home and parked the car.", "lines": []}
    assert b_fields == {"narration_en": "Then he slept.", "faithful_en": "Then he\nslept.", "lines": []}


def test_a_forward_move_carries_the_end_of_this_segment_onto_the_next():
    a = segment(1, "It was late.\n\nThe police", "It was late. The police")
    b = segment(2, "arrived at dawn.", "arrived at dawn.")

    a_fields, b_fields, problems = sentences.apply_move(a, b, forward("The police", "The police"))

    assert problems == []
    assert a_fields["narration_en"] == "It was late." and a_fields["faithful_en"] == "It was late."
    assert b_fields["narration_en"] == "The police arrived at dawn." and b_fields["faithful_en"] == "The police arrived at dawn."


# --- lines --------------------------------------------------------------------------------------------


def test_a_back_move_joins_the_last_line_and_removes_the_line_it_emptied():
    a = segment(1, faithful="She opened the door. Inside, she found", lines=[
        line("She opened the door.", 600, 1), line("Inside, she found", 200, 2)])
    b = segment(2, faithful="a letter on the table. It was signed by Gacy.", lines=[
        line("a letter on the table.", 900, 1), line("It was signed by Gacy.", 500, 2)])

    a_fields, b_fields, problems = sentences.apply_move(a, b, back("a letter on the table.", "a letter on the table."))

    assert problems == []
    assert [item["text"] for item in a_fields["lines"]] == ["She opened the door.", "Inside, she found a letter on the table."]
    # The edited edge line now ends a sentence: its 200 ms silence earns the sentence minimum.
    assert a_fields["lines"][1]["break_s"] == 0.4
    assert a_fields["lines"][1]["runs"] == [2] and a_fields["lines"][1]["gap_after_ms"] == 200
    assert a_fields["lines"][0] == a["lines"][0]  # untouched
    assert a_fields["narration_en"] == "She opened the door.\n\nInside, she found a letter on the table."
    # The emptied line's spoken runs stay in the segment: they join the line that is now its first.
    assert [item["text"] for item in b_fields["lines"]] == ["It was signed by Gacy."]
    assert b_fields["lines"][0]["runs"] == [1, 2] and b_fields["lines"][0]["start_ms"] == 1000
    assert b_fields["lines"][0]["end_ms"] == 2800 and b_fields["lines"][0]["gap_after_ms"] == 500
    assert b_fields["narration_en"] == "It was signed by Gacy."


def test_a_forward_move_across_two_lines_trims_one_and_removes_the_other():
    a = segment(1, faithful="He ran. Then he stopped and turned", lines=[
        line("He ran.", 700, 1), line("Then he stopped", 500, 2), line("and turned", 150, 3)])
    b = segment(2, faithful="around slowly.", lines=[line("around slowly.", 800, 1)])

    a_fields, b_fields, problems = sentences.apply_move(a, b, forward("stopped and turned", "stopped and turned"))

    assert problems == []
    assert [item["text"] for item in a_fields["lines"]] == ["He ran.", "Then he"]
    # The emptied last line's runs join the trimmed line, which now ends the segment: it takes that
    # line's end and its short silence, so no break is written in the middle of the sentence.
    assert a_fields["lines"][1]["runs"] == [2, 3] and a_fields["lines"][1]["end_ms"] == 3800
    assert a_fields["lines"][1]["gap_after_ms"] == 150 and a_fields["lines"][1]["break_s"] is None
    assert [item["text"] for item in b_fields["lines"]] == ["stopped and turned around slowly."]
    assert b_fields["lines"][0]["break_s"] == 0.8
    assert a_fields["faithful_en"] == "He ran. Then he" and b_fields["faithful_en"] == "stopped and turned around slowly."


def test_a_thinly_heard_edge_line_keeps_the_guarded_break_cap():
    a = segment(1, faithful="She waited and", lines=[line("She waited and", 2500, 1, coverage=GUARDED)])
    b = segment(2, faithful="waited. Nothing.", lines=[line("waited.", 300, 1), line("Nothing.", 400, 2)])

    a_fields, _b_fields, problems = sentences.apply_move(a, b, back("waited.", "waited."))

    assert problems == []
    assert a_fields["lines"][0]["break_s"] == 1.0


# --- each text decided on its own ---------------------------------------------------------------------


def test_only_the_faithful_text_moves_when_the_narration_is_already_whole():
    a = segment(1, faithful="The card did not belong to Gacy, and", lines=[
        line("The card wasn't Gacy's, along with another identification card.", 600, 1)])
    b = segment(2, faithful="another identification card. Also, a ring.", lines=[line("There was also a ring.", 500, 1)])

    a_fields, b_fields, problems = sentences.apply_move(a, b, back(faithful="another identification card."))

    assert problems == []
    assert a_fields == {"faithful_en": "The card did not belong to Gacy, and another identification card."}
    assert b_fields == {"faithful_en": "Also, a ring."}


def test_a_rejected_narration_move_leaves_the_narration_and_still_moves_the_faithful():
    a = segment(1, "He drove home and", "He drove home and")
    b = segment(2, "parked the car. Then he slept.", "parked the car. Then he slept.")

    a_fields, b_fields, problems = sentences.apply_move(a, b, back("parked his car.", "parked the car."))

    assert a_fields == {"faithful_en": "He drove home and parked the car."}
    assert b_fields == {"faithful_en": "Then he slept."}
    assert len(problems) == 1 and "narration" in problems[0] and "start of segment 2" in problems[0]


def test_moves_that_are_not_exact_whole_words_or_empty_a_segment_are_rejected():
    a = segment(1, "He drove home and", "He drove home and")
    b = segment(2, "parked the car.", "parked the car.")

    for decision in (
        back("parked the car.", "parked the car."),    # the whole of segment 2
        back("parked the ca", "parked the ca"),        # half a word
        forward("drove home", "drove home"),           # not the end of segment 1
        back("   ", "   "),                             # empty
    ):
        a_fields, b_fields, problems = sentences.apply_move(a, b, decision)
        assert (a_fields, b_fields) == ({}, {})
        assert len(problems) == 2


def test_a_cut_after_a_finished_sentence_is_not_a_split_whatever_the_model_says():
    # Real case (Gacy 78/79): the narration already ended a sentence, and the model moved the next one back.
    a = segment(1, "Beneath the house.", "Beneath the house, and")
    b = segment(2, "Four more in the river.\n\nPolice tore it apart.", "four more in the river. Police tore it apart.")

    a_fields, b_fields, problems = sentences.apply_move(a, b, back("Four more in the river.", "four more in the river."))

    assert "narration_en" not in a_fields and "narration_en" not in b_fields
    assert a_fields["faithful_en"] == "Beneath the house, and four more in the river."
    assert problems == ["narration: segment 1 already ends a sentence"]


def test_a_cut_off_ending_marked_with_dots_or_a_quote_inside_a_sentence_still_counts():
    for ending in ("Detective Summer...", "Detective Summer…", "and he said"):
        a = segment(1, ending, ending)
        b = segment(2, "followed him. Then he left.", "followed him. Then he left.")
        _a_fields, _b_fields, problems = sentences.apply_move(a, b, back("followed him.", "followed him."))
        assert problems == [], ending
    for ending in ("He said, \u201cI came.\u201d", "Was it him?", "(It was.)"):
        a = segment(1, ending, ending)
        b = segment(2, "Then he left. Later.", "Then he left. Later.")
        _a_fields, _b_fields, problems = sentences.apply_move(a, b, back("Then he left.", "Then he left."))
        assert len(problems) == 2, ending


def test_the_translations_own_break_marks_count_as_a_split_even_after_a_full_stop():
    # Real cases (Gacy 35/36 and 78/79): "[incomplete]." ends one side, or "…" opens the next.
    cases = [
        ("After that, Gacy [incomplete].", "took him home. Later he left.", back("took him home.", "took him home.")),
        ("He had hidden four more bodies.", "…in the river. Police came.", back("…in the river.", "…in the river.")),
    ]
    for a_text, b_text, decision in cases:
        a = segment(1, a_text, a_text)
        b = segment(2, b_text, b_text)
        _a_fields, _b_fields, problems = sentences.apply_move(a, b, decision)
        assert problems == [], a_text


def test_no_split_changes_nothing():
    a = segment(1, "One.", "One.")
    b = segment(2, "Two.", "Two.")

    assert sentences.apply_move(a, b, SplitDecision()) == ({}, {}, [])


# --- excerpts -----------------------------------------------------------------------------------------


def test_the_model_sees_two_edge_lines_and_the_faithful_tail_and_head():
    a = segment(1, faithful="x " * 400 + "end of A", lines=[line("One.", 500, 1), line("Two.", 500, 2), line("Three", 100, 3)])
    b = segment(2, faithful="start of B " + "y " * 400, lines=[line("four.", 500, 1), line("Five.", 500, 2), line("Six.", 500, 3)])

    a_narration, a_faithful, b_narration, b_faithful = sentences.excerpts(a, b)

    assert a_narration == "Two.\n\nThree"
    assert b_narration == "four.\n\nFive."
    assert a_faithful.endswith("end of A") and len(a_faithful) <= sentences.FAITHFUL_EXCERPT_CHARS
    assert b_faithful.startswith("start of B") and len(b_faithful) <= sentences.FAITHFUL_EXCERPT_CHARS
    assert not a_faithful.startswith(" ") and not b_faithful.endswith(" ")


# --- fix_job ------------------------------------------------------------------------------------------


def seed(tmp_path, rows: list[dict], voiced: set[int] = frozenset()):
    db = Database(tmp_path / "app.db")
    db.initialize()
    project_id, job_id, now = str(uuid.uuid4()), str(uuid.uuid4()), utc_now()
    db.execute("INSERT INTO projects(id,title,topic,created_at,updated_at) VALUES(?,?,?,?,?)", (project_id, "T", "History", now, now))
    db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,created_at) VALUES(?,?,?,?,?,?,?)",
        (job_id, project_id, "awaiting_review", "review", 95, "{}", now),
    )
    for row in rows:
        index = row["segment_index"]
        db.execute(
            "INSERT INTO segments(id,job_id,project_id,segment_index,start_ms,end_ms,source_audio_path,kind,transcript_si,"
            "faithful_en,narration_en,lines_json,tts_audio_path,qa_json,qa_status,status,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"{job_id}-{index}", job_id, project_id, index, index * 1000, index * 1000 + 1000, "x.wav", row["kind"], "සිංහල",
             row["faithful_en"], row["narration_en"], json.dumps(row["lines"]),
             "voice.wav" if index in voiced else None, json.dumps({"passed": True, "issues": []}), "passed", "passed", now),
        )
    return db, job_id


def by_index(db, job_id) -> dict[int, dict]:
    return {row["segment_index"]: row for row in db.all("SELECT * FROM segments WHERE job_id=?", (job_id,))}


def test_fix_job_moves_records_notes_and_flags_voiced_segments(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "He drove home and", "He drove home and"),
        segment(2, "parked the car. Then he slept.", "parked the car. Then he slept."),
        segment(3, "Morning came.", "Morning came."),
    ], voiced={1, 2})
    warnings, progress = [], []

    def decide(a, b):
        return back("parked the car.", "parked the car.") if a["segment_index"] == 1 else SplitDecision()

    changed = sentences.fix_job(db, job_id, decide, warnings.append, on_progress=lambda done, total: progress.append((done, total)))

    rows = by_index(db, job_id)
    assert changed == [1, 2]
    assert warnings == []
    assert sorted(progress) == [(1, 2), (2, 2)]
    assert rows[1]["narration_en"] == "He drove home and parked the car."
    assert rows[2]["narration_en"] == "Then he slept." and rows[2]["faithful_en"] == "Then he slept."
    assert rows[1]["sentence_fix"]["notes"] == ["End of sentence moved in from segment 2 (narration and faithful)"]
    assert rows[2]["sentence_fix"]["notes"] == ["End of sentence moved out to segment 1 (narration and faithful)"]
    assert rows[1]["sentence_fix"]["at"]
    for index in (1, 2):
        assert rows[index]["qa_status"] == "needs_review"
        assert rows[index]["qa"]["passed"] is False
        assert sentences.REVOICE_NOTE in rows[index]["qa"]["issues"]
    # Text only: the transcript, times and audio never change.
    assert rows[2]["transcript_si"] == "සිංහල" and rows[2]["start_ms"] == 2000 and rows[2]["tts_audio_path"] == "voice.wav"
    assert rows[3]["sentence_fix"] == {} and rows[3]["qa_status"] == "passed"


def test_a_faithful_only_move_does_not_send_a_voiced_segment_to_review(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "The card wasn't his.", "The card was not his, and"),
        segment(2, "There was a ring.", "a ring. There was more."),
    ], voiced={1, 2})

    changed = sentences.fix_job(db, job_id, lambda a, b: forward(faithful="and"), lambda message: None)

    rows = by_index(db, job_id)
    assert changed == [1, 2]
    assert rows[1]["faithful_en"] == "The card was not his," and rows[2]["faithful_en"] == "and a ring. There was more."
    assert rows[1]["narration_en"] == "The card wasn't his."
    assert rows[1]["sentence_fix"]["notes"] == ["Start of sentence moved out to segment 2 (faithful)"]
    assert rows[2]["sentence_fix"]["notes"] == ["Start of sentence moved in from segment 1 (faithful)"]
    assert rows[1]["qa_status"] == "passed" and rows[2]["qa_status"] == "passed"


def test_fix_job_decides_on_the_original_text_and_moves_both_ends_of_one_segment(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "It was cold and", "It was cold and"),
        segment(2, "dark. He waited. Then the", "dark. He waited. Then the"),
        segment(3, "door opened.", "door opened."),
    ])
    seen = []

    def decide(a, b):
        seen.append((a["narration_en"], b["narration_en"]))
        if a["segment_index"] == 1:
            return back("dark.", "dark.")
        return forward("Then the", "Then the")

    changed = sentences.fix_job(db, job_id, decide, lambda message: None)

    rows = by_index(db, job_id)
    assert changed == [1, 2, 3]
    assert sorted(seen) == [("It was cold and", "dark. He waited. Then the"), ("dark. He waited. Then the", "door opened.")]
    assert [rows[i]["narration_en"] for i in (1, 2, 3)] == ["It was cold and dark.", "He waited.", "Then the door opened."]
    assert rows[2]["sentence_fix"]["notes"] == [
        "End of sentence moved out to segment 1 (narration and faithful)",
        "Start of sentence moved out to segment 3 (narration and faithful)",
    ]


def test_a_second_move_that_would_empty_a_segment_is_rejected_with_a_warning(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "It was", "It was"),
        segment(2, "cold. And", "cold. And"),
        segment(3, "dark.", "dark."),
    ])
    warnings = []

    def decide(a, b):
        return back("cold.", "cold.") if a["segment_index"] == 1 else forward("And", "And")

    sentences.fix_job(db, job_id, decide, warnings.append)

    rows = by_index(db, job_id)
    assert rows[1]["narration_en"] == "It was cold." and rows[2]["narration_en"] == "And"
    assert rows[3]["narration_en"] == "dark."
    assert len(warnings) == 2 and all("segments 2 and 3" in warning for warning in warnings)


def test_a_failed_decision_is_a_warning_and_the_other_pairs_still_move(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "One.", "One."),
        segment(2, "Two and", "Two and"),
        segment(3, "three.", "three."),
    ])
    warnings = []

    def decide(a, b):
        if a["segment_index"] == 1:
            raise RuntimeError("model unavailable")
        return forward("and", "and")

    changed = sentences.fix_job(db, job_id, decide, warnings.append, passes=1)

    assert changed == [2, 3]
    assert warnings == ["Sentence fix skipped segments 1 and 2: model unavailable"]
    assert by_index(db, job_id)[3]["narration_en"] == "and three."


def test_recordings_break_the_pairs(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "He said", "He said"),
        segment(2, kind="original"),
        segment(3, "nothing.", "nothing."),
    ])
    calls = []

    changed = sentences.fix_job(db, job_id, lambda a, b: calls.append(1) or SplitDecision(), lambda message: None)

    assert changed == [] and calls == []


# --- the model call -----------------------------------------------------------------------------------


def test_split_sentence_asks_for_exact_fragments_per_text_and_no_duplicates():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=SplitDecision(faithful_direction="back", fragment_faithful="a ring."))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    decision = ai.split_sentence("The card wasn't his.", "not his, and", "There was a ring.", "a ring. More.", "gpt-text")

    assert decision.fragment_faithful == "a ring."
    assert captured["model"] == "gpt-text"
    assert captured["text_format"] is SplitDecision
    system = captured["input"][0]["content"]
    user = captured["input"][1]["content"]
    assert "exactly" in system and "smaller" in system
    assert "already" in system and "both sides" in system
    assert "no split" in system.lower()
    for text in ("The card wasn't his.", "not his, and", "There was a ring.", "a ring. More."):
        assert text in user


# --- merge ---------------------------------------------------------------------------------------------


def merge(a_tail: str, b_head: str, sentence: str, place: str = "a", faithful: bool = False) -> SplitDecision:
    name = "faithful" if faithful else "narration"
    return SplitDecision(**{f"{name}_direction": "merge", f"fragment_{name}": sentence,
                            f"a_tail_{name}": a_tail, f"b_head_{name}": b_head, f"place_{name}": place})


def test_a_broken_sentence_and_its_echo_become_one_sentence_where_the_broken_half_was():
    # Real case (Bundy 63/64): A breaks off, B finishes the sentence in its own words.
    a = segment(1, faithful="x", lines=[
        line("They thought it could be Ted.", 600, 1), line("At that time, Ted had never…", 100, 2)])
    b = segment(2, faithful="y", lines=[
        line("He'd never been caught by police for anything.", 700, 1), line("So the task force went on.", 500, 2)])

    a_fields, b_fields, problems = sentences.apply_move(a, b, merge(
        "At that time, Ted had never…", "He'd never been caught by police for anything.",
        "At that time, Ted had never been caught by police for anything."))

    assert problems == []
    assert [item["text"] for item in a_fields["lines"]] == [
        "They thought it could be Ted.", "At that time, Ted had never been caught by police for anything."]
    # The merged sentence takes the broken line's place: its runs, its silence and a break it now earns.
    assert a_fields["lines"][1]["runs"] == [2] and a_fields["lines"][1]["break_s"] == 0.4
    assert a_fields["lines"][0] == a["lines"][0]
    # The echo goes; its runs join B's new first line.
    assert [item["text"] for item in b_fields["lines"]] == ["So the task force went on."]
    assert b_fields["lines"][0]["runs"] == [1, 2]
    assert b_fields["narration_en"] == "So the task force went on."


def test_a_merge_can_land_in_the_later_segment_and_keeps_the_rest_of_a_partly_used_line():
    a = segment(1, "He ran home. Then he took her to his…", "f")
    b = segment(2, "vehicle and drove off. Police came.", "f")

    a_fields, b_fields, problems = sentences.apply_move(a, b, merge(
        "Then he took her to his…", "vehicle and drove off.", "Then he took her to his vehicle and drove off.", place="b"))

    assert problems == []
    assert a_fields["narration_en"] == "He ran home."
    assert b_fields["narration_en"] == "Then he took her to his vehicle and drove off. Police came."


def test_a_merge_works_on_the_faithful_english_on_its_own():
    a = segment(1, "Whole.", "He was found guilty of trying to abduct Carol...")
    b = segment(2, "Also whole.", "Guilty. He went to prison.")

    a_fields, b_fields, problems = sentences.apply_move(a, b, merge(
        "He was found guilty of trying to abduct Carol...", "Guilty.",
        "He was found guilty of trying to abduct Carol.", faithful=True))

    assert problems == []
    assert a_fields == {"faithful_en": "He was found guilty of trying to abduct Carol."}
    assert b_fields == {"faithful_en": "He went to prison."}


def test_a_merge_that_brings_in_a_new_name_number_or_word_is_refused():
    a = segment(1, "They said Ted had never…", "f")
    b = segment(2, "He'd never been caught. Then more.", "f")

    for sentence in ("They said Ted Bundy had never been caught.",       # a name not in either half
                     "They said Ted had never been caught in 1974.",     # a number
                     "They said Ted had never been arrested."):          # a new word
        a_fields, b_fields, problems = sentences.apply_move(a, b, merge("They said Ted had never…", "He'd never been caught.", sentence))
        assert (a_fields, b_fields) == ({}, {}), sentence
        assert len(problems) == 1 and "not in the words it replaces" in problems[0], sentence


def test_a_merge_must_name_exact_edges_and_leave_both_segments_something():
    a = segment(1, "One. Two halves\u2026", "f")
    b = segment(2, "of it. Three.", "f")

    for decision in (
        merge("Two half\u2026", "of it.", "Two halves of it."),                         # not A's exact end
        merge("Two halves\u2026", "of that.", "Two halves of that."),                   # not B's exact start
        merge("One. Two halves\u2026", "of it.", "One. Two halves of it.", place="b"),  # empties A
        merge("Two halves\u2026", "of it. Three.", "Two halves of it. Three."),         # empties B
        merge("", "of it.", "of it."),                                                # no broken half named
    ):
        a_fields, b_fields, problems = sentences.apply_move(a, b, decision)
        assert (a_fields, b_fields) == ({}, {}), decision
        assert len(problems) == 1, decision


def test_a_merge_says_so_on_both_segments(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "First. Ted had never\u2026", "f one."),
        segment(2, "He'd never been caught. Then more.", "f two."),
    ])

    changed = sentences.fix_job(db, job_id, lambda a, b: merge("Ted had never\u2026", "He'd never been caught.", "Ted had never been caught."), lambda message: None)

    assert changed == [1, 2]
    rows = by_index(db, job_id)
    assert rows[1]["narration_en"] == "First. Ted had never been caught."
    assert rows[2]["narration_en"] == "Then more."
    assert rows[1]["sentence_fix"]["notes"] == ["Sentence joined across the cut with segment 2 (narration)"]
    assert rows[2]["sentence_fix"]["notes"] == ["Sentence joined across the cut into segment 1 (narration)"]


def test_a_joined_sentence_may_change_a_words_ending_but_not_bring_a_new_word():
    a = segment(1, "First. He said his name…", "f")
    b = segment(2, "He introduced himself as Ted. Then more.", "f")

    _a_fields, _b_fields, problems = sentences.apply_move(a, b, merge(
        "He said his name…", "He introduced himself as Ted.", "He said his name was Ted, to introduce himself."))
    assert problems == []

    a = segment(1, "First. Police were able to…", "f")
    b = segment(2, "A body was found. Then more.", "f")
    _a_fields, _b_fields, problems = sentences.apply_move(a, b, merge(
        "Police were able to…", "A body was found.", "Police were able to find a body."))
    assert len(problems) == 1 and "find" in problems[0]  # "find" is not "found": a different word, refused


def test_split_sentence_tells_the_model_when_a_break_mark_shows_the_sentence_is_broken():
    captured = []

    class Responses:
        def parse(self, **kwargs):
            captured.append(kwargs["input"][1]["content"])
            return SimpleNamespace(output_parsed=SplitDecision())

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.split_sentence("Ted had never…", "f", "He'd never been caught.", "f", "m")
    ai.split_sentence("Whole.", "f", "…that's what they say.", "f", "m")
    ai.split_sentence("Whole.", "f", "Also whole.", "f", "m")

    assert "narration of A ends with a break mark" in captured[0]
    assert "narration of B starts with a break mark" in captured[1]
    assert "break mark" not in captured[2]


def test_cuts_that_still_look_broken_get_another_pass_and_whole_ones_are_not_asked_again(tmp_path):
    db, job_id = seed(tmp_path, [
        segment(1, "One. Ted had never…", "f one."),
        segment(2, "He'd never been caught. Two.", "F two."),
        segment(3, "Three.", "F three."),
    ])
    asked: list[int] = []

    def decide(a, b):
        asked.append(a["segment_index"])
        # The model misses the broken cut the first time, then sees it.
        if a["segment_index"] == 1 and asked.count(1) > 1:
            return merge("Ted had never…", "He'd never been caught.", "Ted had never been caught.")
        return SplitDecision()

    changed = sentences.fix_job(db, job_id, decide, lambda message: None)

    assert changed == [1, 2]
    assert by_index(db, job_id)[1]["narration_en"] == "One. Ted had never been caught."
    # Pass one asks both cuts; pass two only the cut ending in "…"; pass three finds nothing broken.
    assert asked.count(1) == 2 and asked.count(2) == 1


def test_a_segment_that_loses_its_first_words_starts_with_a_capital_again():
    # Real case (Bundy 23/24): the merge took the repeated "Anyway," off segment 24, leaving "while Ted…".
    a = segment(1, faithful="F.", lines=[line("I think you should watch this…", 500, 1)])
    b = segment(2, faithful="F.", lines=[line("Anyway, while Ted was living in Seattle, he met Liz.", 500, 1)])

    _a_fields, b_fields, problems = sentences.apply_move(a, b, merge(
        "I think you should watch this…", "Anyway,", "I think you should watch this."))

    assert problems == []
    assert b_fields["narration_en"] == "While Ted was living in Seattle, he met Liz."
    assert b_fields["lines"][0]["text"] == "While Ted was living in Seattle, he met Liz."


def test_a_possessive_is_the_same_word_as_the_name():
    # Real case (Bundy 82/83): "Liz's" in the joined sentence, "Liz" in the halves.
    a = segment(1, "First. Liz thought again about Ted…", "F.")
    b = segment(2, "That suspicion was confirmed again. More.", "F.")

    _a_fields, _b_fields, problems = sentences.apply_move(a, b, merge(
        "Liz thought again about Ted…", "That suspicion was confirmed again.",
        "Liz's suspicion about Ted was confirmed again."))

    assert problems == []
