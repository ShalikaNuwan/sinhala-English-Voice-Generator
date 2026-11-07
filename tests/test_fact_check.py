from __future__ import annotations

import json
import threading
import uuid
from types import SimpleNamespace

import pytest

from app import fact_check
from app.database import Database, utc_now
from app.fact_check import Fact, FactReport, Source


GACY = "https://www.upi.com/Archives/1978/12/22/Bodies-of-five-youths-found/5871529432936/"
PATCH = "https://patch.com/illinois/desplaines/gacy-case-detailed-by-cook-county-sheriff"


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    # Another task adds facts_json to ADDED_COLUMNS; until it lands, add it here.
    with database.connect() as connection:
        existing = {row["name"] for row in connection.execute("PRAGMA table_info(segments)")}
        if "facts_json" not in existing:
            connection.execute("ALTER TABLE segments ADD COLUMN facts_json TEXT NOT NULL DEFAULT '{}'")
    return database


def make_job(db: Database, segments: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """A project, a job and its segments as (kind, narration_en) in order; returns the job and segment ids."""
    project_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        (project_id, "Gacy", "John Wayne Gacy", "uploaded", now, now),
    )
    db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,created_at) VALUES(?,?,?,?,?,?)",
        (job_id, project_id, "completed", "completed", 100, now),
    )
    ids = []
    for index, (kind, text) in enumerate(segments):
        segment_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO segments(id,job_id,project_id,segment_index,start_ms,end_ms,source_audio_path,kind,"
            "narration_en,faithful_en,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (segment_id, job_id, project_id, index, index * 1000, index * 1000 + 1000, f"{index}.wav", kind,
             text, text, now),
        )
        ids.append(segment_id)
    return job_id, ids


def facts_of(db: Database, segment_id: str) -> dict:
    return db.one("SELECT facts_json FROM segments WHERE id=?", (segment_id,))["facts"]


def report(*facts: Fact) -> FactReport:
    return FactReport(facts=list(facts))


def fact(claim="Gacy was arrested in 1978.", verdict="correct", evidence="He was arrested in December 1978.",
         urls=(GACY,)) -> Fact:
    return Fact(claim=claim, verdict=verdict, evidence=evidence,
                sources=[Source(url=url, title="UPI") for url in urls])


# clean_url


def test_clean_url_drops_tracking_parameters_and_keeps_the_rest():
    assert fact_check.clean_url("https://site.com/page?utm_source=openai") == "https://site.com/page"
    assert (fact_check.clean_url("https://site.com/p?id=4&utm_medium=x&UTM_Campaign=y&q=a")
            == "https://site.com/p?id=4&q=a")
    assert fact_check.clean_url("  https://site.com/a#part  ") == "https://site.com/a#part"


# screen


def test_screen_keeps_only_links_the_search_returned_comparing_cleaned_urls():
    fake = "https://made-up.example/gacy"
    items = fact_check.screen(
        report(fact(urls=(f"{GACY}?utm_source=openai", fake))),
        {GACY + "?utm_source=openai"},
    )
    assert items == [{
        "claim": "Gacy was arrested in 1978.",
        "verdict": "correct",
        "evidence": "He was arrested in December 1978.",
        "sources": [{"url": GACY, "title": "UPI"}],
    }]


def test_screen_ignores_a_trailing_slash_and_drops_duplicates_and_non_web_links():
    items = fact_check.screen(
        report(fact(urls=(GACY.rstrip("/"), GACY, "javascript:alert(1)"))),
        {GACY, "javascript:alert(1)"},
    )
    assert [source["url"] for source in items[0]["sources"]] == [GACY.rstrip("/")]


def test_screen_turns_a_wrong_fact_without_a_surviving_link_into_unverified():
    items = fact_check.screen(
        report(fact(verdict="wrong", evidence="He was arrested at home in Norwood Park.",
                    urls=("https://made-up.example/x",))),
        {GACY},
    )
    assert items[0]["verdict"] == "unverified"
    assert items[0]["evidence"] == "No source link found. He was arrested at home in Norwood Park."
    assert items[0]["sources"] == []


def test_screen_keeps_a_wrong_fact_that_has_a_returned_link():
    items = fact_check.screen(report(fact(verdict="wrong", urls=(PATCH,))), {PATCH})
    assert items[0]["verdict"] == "wrong"
    assert items[0]["sources"] == [{"url": PATCH, "title": "UPI"}]


def test_screen_strips_markdown_links_from_the_evidence():
    evidence = (
        "He was arrested at his home in [Norwood Park](https://a.com/x?utm_source=openai), not in Des Plaines. "
        "([patch.com](https://patch.com/page?utm_source=openai))"
    )
    items = fact_check.screen(report(fact(evidence=evidence)), {GACY})
    assert items[0]["evidence"] == "He was arrested at his home in Norwood Park, not in Des Plaines."


def test_screen_gives_an_untitled_source_its_site_name():
    untitled = Fact(claim="c", verdict="correct", evidence="e", sources=[Source(url=PATCH, title=" ")])
    assert fact_check.screen(report(untitled), {PATCH})[0]["sources"] == [{"url": PATCH, "title": "patch.com"}]


# ask_model


class FakeResponses:
    def __init__(self, parsed, output):
        self.parsed, self.output, self.calls = parsed, output, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed, output=self.output)


def fake_client(parsed, output):
    return SimpleNamespace(responses=FakeResponses(parsed, output))


def test_ask_model_searches_the_web_and_collects_every_returned_link():
    parsed = report(fact())
    output = [
        SimpleNamespace(type="reasoning"),
        SimpleNamespace(type="web_search_call", action=SimpleNamespace(
            type="search", sources=[SimpleNamespace(type="url", url=f"{GACY}?utm_source=openai")])),
        # An opened page has no sources list, only its url.
        SimpleNamespace(type="web_search_call", action=SimpleNamespace(type="open_page", sources=None,
                                                                       url="https://opened.example/page")),
        # Dicts as well as objects, and items with missing attributes.
        {"type": "web_search_call", "action": {"type": "search", "sources": [{"type": "url", "url": PATCH}]}},
        SimpleNamespace(type="web_search_call"),
        SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", annotations=[
            SimpleNamespace(type="url_citation", url="https://cited.example/a?utm_source=openai", title="A"),
            SimpleNamespace(type="file_citation", file_id="f"),
        ])]),
    ]
    client = fake_client(parsed, output)

    result, urls = fact_check.ask_model("Gacy was arrested in 1978.", "Earlier text.", "John Wayne Gacy",
                                        "gpt-test", client)

    assert result is parsed
    assert urls == {GACY, "https://opened.example/page", PATCH, "https://cited.example/a"}
    call = client.responses.calls[0]
    assert call["model"] == "gpt-test"
    assert call["tools"] == [{"type": "web_search"}]
    assert call["include"] == ["web_search_call.action.sources"]
    assert call["text_format"] is FactReport
    prompt = json.dumps(call["input"])
    for expected in ("Gacy was arrested in 1978.", "Earlier text.", "John Wayne Gacy", "unverified",
                     "opinion", "attributed"):
        assert expected in prompt


def test_ask_model_fails_without_a_structured_result():
    with pytest.raises(RuntimeError):
        fact_check.ask_model("text", "", "", "gpt-test", fake_client(None, []))


# check_segment


def test_check_segment_stores_the_screened_facts_and_the_text_it_checked(db):
    _job_id, (first, second) = make_job(db, [("narration", "First part."), ("narration", "Gacy was arrested.")])
    calls = []

    def ask(text, context):
        calls.append((text, context))
        assert facts_of(db, second)["state"] == "checking"
        return report(fact(verdict="wrong", urls=("https://made-up.example/",)), fact()), {GACY}

    stored = fact_check.check_segment(db, second, ask, now=lambda: "2026-10-29T10:00:00+00:00")

    assert calls == [("Gacy was arrested.", "First part.")]
    assert stored == facts_of(db, second)
    assert stored["state"] == "done"
    assert stored["checked_text"] == "Gacy was arrested."
    assert stored["at"] == "2026-10-29T10:00:00+00:00"
    assert stored["error"] == ""
    assert [item["verdict"] for item in stored["items"]] == ["unverified", "correct"]
    row = db.one("SELECT narration_en, faithful_en, revision FROM segments WHERE id=?", (second,))
    assert row == {"narration_en": "Gacy was arrested.", "faithful_en": "Gacy was arrested.", "revision": 1}
    assert facts_of(db, first) == {}


def test_check_segment_takes_context_from_the_previous_narration_with_english(db):
    _job_id, ids = make_job(db, [("narration", "Before the call."), ("original", ""), ("narration", "After it.")])
    seen = []
    fact_check.check_segment(db, ids[2], lambda text, context: (seen.append(context), (report(), set()))[1])
    assert seen == ["Before the call."]


def test_check_segment_stores_a_failure(db):
    _job_id, (segment_id,) = make_job(db, [("narration", "Gacy was arrested.")])

    def ask(_text, _context):
        raise RuntimeError("search timed out")

    stored = fact_check.check_segment(db, segment_id, ask)

    assert stored["state"] == "failed"
    assert stored["error"] == "search timed out"
    assert stored["items"] == []
    assert stored["checked_text"] == "Gacy was arrested."
    assert facts_of(db, segment_id) == stored


def test_check_segment_skips_recordings_and_empty_english(db):
    _job_id, ids = make_job(db, [("original", "A recorded call."), ("narration", "  ")])

    def ask(_text, _context):
        raise AssertionError("nothing to check")

    assert fact_check.check_segment(db, ids[0], ask) is None
    assert fact_check.check_segment(db, ids[1], ask) is None
    assert facts_of(db, ids[0]) == {} and facts_of(db, ids[1]) == {}


def test_check_segment_rejects_an_unknown_segment(db):
    with pytest.raises(LookupError):
        fact_check.check_segment(db, "missing", lambda text, context: (report(), set()))


# check_job


def test_check_job_checks_only_narration_with_english_in_parallel_with_progress(db):
    job_id, ids = make_job(db, [
        ("narration", "One."), ("original", "A call."), ("narration", "Two."), ("narration", ""),
        ("narration", "Three."),
    ])
    lock = threading.Lock()
    asked, threads, progress = [], set(), []
    barrier = threading.Barrier(3, timeout=5)

    def ask(text, context):
        barrier.wait()  # all three run at once, or this times out
        with lock:
            asked.append((text, context))
            threads.add(threading.get_ident())
        return report(fact(claim=text)), {GACY}

    results = fact_check.check_job(db, job_id, ask, workers=4,
                                   on_progress=lambda done, total: progress.append((done, total)))

    assert sorted(asked) == [("One.", ""), ("Three.", "Two."), ("Two.", "One.")]
    assert len(threads) == 3
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]
    assert [result["checked_text"] for result in results] == ["One.", "Two.", "Three."]
    assert [facts_of(db, segment_id).get("state") for segment_id in ids] == ["done", None, "done", None, "done"]


def test_check_job_keeps_going_when_one_segment_fails(db):
    job_id, ids = make_job(db, [("narration", "One."), ("narration", "Two.")])

    def ask(text, _context):
        if text == "One.":
            raise ValueError("bad answer")
        return report(fact()), {GACY}

    results = fact_check.check_job(db, job_id, ask)

    assert [result["state"] for result in results] == ["failed", "done"]
    assert facts_of(db, ids[0])["error"] == "bad answer"


def test_check_job_can_check_one_segment(db):
    job_id, ids = make_job(db, [("narration", "One."), ("narration", "Two.")])
    asked = []
    progress = []
    results = fact_check.check_job(db, job_id, lambda text, context: (asked.append((text, context)), (report(), set()))[1],
                                   segment_id=ids[1], on_progress=lambda done, total: progress.append((done, total)))
    assert asked == [("Two.", "One.")]
    assert progress == [(0, 1), (1, 1)]
    assert len(results) == 1 and facts_of(db, ids[0]) == {}


def test_check_job_rejects_a_segment_from_another_job(db):
    job_id, _ids = make_job(db, [("narration", "One.")])
    _other_job, other_ids = make_job(db, [("narration", "Elsewhere.")])
    with pytest.raises(LookupError):
        fact_check.check_job(db, job_id, lambda text, context: (report(), set()), segment_id=other_ids[0])


# resolved


def test_set_resolved_marks_a_fact_and_undo_clears_it(db):
    _job_id, (segment,) = make_job(db, [("narration", "Gacy was arrested.")])
    fact_check.check_segment(db, segment, lambda text, context: (report(fact(), fact(claim="He was 36.", verdict="wrong")), {GACY}))

    fact_check.set_resolved(db, segment, 1, "He was 36.", True)
    assert facts_of(db, segment)["items"][1]["resolved"] is True
    assert "resolved" not in facts_of(db, segment)["items"][0]

    fact_check.set_resolved(db, segment, 1, "He was 36.", False)
    assert "resolved" not in facts_of(db, segment)["items"][1]


def test_set_resolved_refuses_a_moved_fact_and_a_correct_one(db):
    _job_id, (segment,) = make_job(db, [("narration", "Gacy was arrested.")])
    fact_check.check_segment(db, segment, lambda text, context: (report(fact(), fact(claim="He was 36.", verdict="wrong")), {GACY}))

    with pytest.raises(LookupError):
        fact_check.set_resolved(db, segment, 1, "A claim from an older check.", True)
    with pytest.raises(LookupError):
        fact_check.set_resolved(db, segment, 5, "He was 36.", True)
    with pytest.raises(ValueError):
        fact_check.set_resolved(db, segment, 0, "Gacy was arrested in 1978.", True)


def test_check_again_keeps_a_resolved_claim_resolved(db):
    _job_id, (segment,) = make_job(db, [("narration", "Gacy was arrested.")])
    first = report(fact(claim="He was 36.", verdict="wrong"), fact(claim="It was cold.", verdict="unverified"))
    fact_check.check_segment(db, segment, lambda text, context: (first, {GACY}))
    fact_check.set_resolved(db, segment, 0, "He was 36.", True)

    again = report(fact(claim="It was cold.", verdict="unverified"), fact(claim="He was 36!", verdict="wrong"))
    stored = fact_check.check_segment(db, segment, lambda text, context: (again, {GACY}))

    assert [item.get("resolved", False) for item in stored["items"]] == [False, True]
