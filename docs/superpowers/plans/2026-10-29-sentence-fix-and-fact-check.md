# Sentence fix and fact check: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD for every step.

**Goal:** a sentence-fixer agent that moves split-sentence fragments between neighbouring segments' English
(text only), and a fact-checker agent that checks each fact on the web and reports wrong ones with links.

**Spec:** `docs/superpowers/specs/2026-10-29-sentence-fix-and-fact-check-design.md` (the contract).
**Branch:** `sentence-facts` (from `main` 280df0f). **Owner rule:** never commit or push.
**Tests:** `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q <files>`; full suite must stay green
(1161 passed before this work). Never call the network in tests.

## File ownership (agents run in parallel; touch only your files)

| Task | Owns |
|---|---|
| A: sentence fixer + pipeline + API | `app/sentences.py` (new), `app/ai.py`, `app/pipeline.py`, `app/database.py`, `app/main.py`, `tests/test_sentences.py` (new), `tests/test_pipeline.py`, `tests/test_api.py` |
| B: fact checker module | `app/fact_check.py` (new), `tests/test_fact_check.py` (new) |
| C: front end | `app/static/app.js`, `app/static/style.css`, `app/static/index.html` |
| D (lead, after A and B): wire B into A's pipeline stage and endpoints | — |

## Task A

1. `app/database.py`: `ADDED_COLUMNS` gains `("segments", "sentence_fix_json", "TEXT NOT NULL DEFAULT '{}'")`
   and `("segments", "facts_json", "TEXT NOT NULL DEFAULT '{}'")`.
2. `app/sentences.py`:
   - pydantic `SplitDecision{split: bool, direction: "back"|"forward"|"", fragment_narration: str = "",
     fragment_faithful: str = "", reason: str = ""}`.
   - `pairs(segments)`: consecutive narration pairs by `segment_index` with English, no recording between.
   - `apply_move(a, b, decision) -> (a_fields, b_fields) | str`: pure; returns the new `narration_en`,
     `lines`, `faithful_en` for both, or a rejection reason (fragment not an exact prefix/suffix with
     whitespace collapsed, empty fragment, fragment is the whole text). Line rules from the spec.
   - `fix_job(db, job_id, decide, warn, workers=6) -> list[int]`: decisions in parallel on the original
     text, moves applied in segment order, notes in `sentence_fix_json`, needs-review for voiced segments,
     returns changed segment indices; progress callback `on_progress(done, total)`.
3. `app/ai.py`: `AIClient.split_sentence(a_narration, a_faithful, b_narration, b_faithful, model) ->
   SplitDecision` (responses.parse; the prompt asks for the smaller half of a sentence that crosses the cut,
   quoted exactly, and `split: false` when the cut falls between sentences).
4. `app/pipeline.py`: split `_narrate_segment` into a text pass (transcribe, translate, adapt) and a finish
   pass (voice when on, QA); `process_job` runs text pass for all, then stage "fixing sentences" (when
   `config.get("fix_sentences", True)`), then finish pass for all, then a call to `self._check_facts(job_id)`
   hook (when `config.get("check_facts", True)`; for now a method that does nothing; Task D fills it), then
   the review gate. A task tracker `self.tasks: dict[job_id, dict]` with a lock, and
   `start_fix_sentences(job_id)` / `start_check_facts(job_id, segment_id=None)` launching daemon threads and
   raising a `TaskBusy` error when that task already runs for the job.
5. `app/main.py`: the four endpoints from the spec; `present_segment` adds `facts_stale`.
6. Tests: FakeAI in `tests/test_pipeline.py` gains `split_sentence` (default: no split); new tests for the
   pairing, every move/reject case, pipeline order, endpoints and 409.

## Task B

`app/fact_check.py`:
- pydantic `Source{url, title}`, `Fact{claim, verdict: "correct"|"wrong"|"unverified", evidence, sources}`,
  `FactReport{facts: list[Fact]}`.
- `ask_model(text, context, topic, model, client) -> (FactReport, set[str] returned_urls)`: one
  `client.responses.parse(model, tools=[{"type": "web_search"}], include=["web_search_call.action.sources"],
  input=..., text_format=FactReport)`; collect returned URLs from `web_search_call.action.sources` and the
  message's `url_citation` annotations. Prompt: checkable facts only; one fact per claim; verdict rules;
  evidence one plain sentence (no markdown links); sources must be pages the search returned.
- `clean_url(url)`: drop `utm_*` query parameters.
- `screen(report, returned_urls) -> list[dict]`: keep only returned links (compare cleaned); `wrong` with no
  link → `unverified` with evidence prefixed "No source link found. "; strip markdown links from evidence.
- `check_segment(db, segment_id, ask, now)`: sets `facts_json` state checking → done/failed, stores
  `checked_text` = the narration it checked, items, `at`, `error`.
- `check_job(db, job_id, ask, segment_id=None, workers=4, on_progress=None)`: narration segments with
  English (or the one segment), previous segment's English as context, in parallel.
- Tests with a fake client/ask: parsing, URL cleaning, link screening, downgrade, states, failure stored,
  progress, only narration segments.

## Task C

Front end per the spec's "Front end" section, against the spec's API and segment fields. Keep the existing
style of `app.js` (template strings, `escapeHtml`, `request()`); never re-render a card while its textarea
is focused or edited (existing guard in `loadSegments`). Poll `GET /api/jobs/{id}/tasks` every 2 s while a
task runs, then reload segments. Links: `target="_blank" rel="noopener noreferrer"`, only http(s) URLs.
Verify with a jsdom smoke test in the scratchpad (like `tests/ui_check.mjs`).

## Task D (lead)

Fill `Pipeline._check_facts` and `start_check_facts` with `fact_check.check_job` and
`ai.check_facts`-style injection (FakeAI gains a no-facts method), full suite, real run on a copy of an
owner project, show the diff.
