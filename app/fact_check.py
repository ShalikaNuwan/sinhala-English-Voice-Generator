"""The fact checker: every checkable fact in a segment's English, searched on the web.

It reports what the sources say and never edits any text. Links are only pages the search really
returned, so a link the model made up is dropped, and a `wrong` fact left without a link becomes
`unverified`.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

from .database import Database, utc_now


NO_LINK = "No source link found. "
# The context is only there to resolve pronouns; its tail is enough.
CONTEXT_CHARS = 2000

# A markdown link, with one level of brackets allowed inside the URL (Wikipedia pages have them).
_LINK = r"\[([^\]]*)\]\(((?:[^()\s]|\([^()\s]*\))*)\)"
_CITATION = re.compile(r"\(\s*" + _LINK + r"\s*\)")
_INLINE_LINK = re.compile(_LINK)

SYSTEM_PROMPT = (
    "You check the facts in one passage of a true-crime documentary narration, using web search.\n"
    "- List only checkable facts: names, dates, times, ages, numbers, places, events and causes. Do not list "
    "opinions, mood, suspense, figures of speech or the narration's style.\n"
    "- One fact per claim: split a sentence holding several facts into separate claims. Quote or closely "
    "paraphrase the passage in each claim.\n"
    "- A claim the passage attributes to someone (\"police believed X\", \"his mother said Y\") is checked as "
    "what was attributed: is it true that they believed or said it, not whether X itself was true.\n"
    "- Search the web for each fact. verdict is correct when reliable sources support it, wrong when reliable "
    "sources contradict it, and unverified when the sources disagree, are silent, or you found none.\n"
    "- evidence is one plain sentence saying what the sources say. No markdown, no links, no URLs in it.\n"
    "- sources lists the pages you found during this search that support the verdict, with their URL exactly "
    "as the search gave it and the page title. Never invent or guess a URL.\n"
    "- The previous passage is context only, to know who and what the passage means; do not check its facts.\n"
    "- A passage with no checkable facts returns an empty list."
)


class Source(BaseModel):
    url: str
    title: str


class Fact(BaseModel):
    claim: str
    verdict: Literal["correct", "wrong", "unverified"]
    evidence: str
    sources: list[Source]


class FactReport(BaseModel):
    facts: list[Fact]


# ask(text, context) -> the model's report and every URL its search returned.
Ask = Callable[[str, str], tuple[FactReport, set[str]]]


def _field(item, name: str):
    """Read a field from either an SDK object or a plain dict."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def clean_url(url: str) -> str:
    """The URL without tracking parameters (utm_*); everything else is kept as it was."""
    url = (url or "").strip()
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    kept = [(key, value) for key, value in pairs if not key.lower().startswith("utm_")]
    if len(kept) == len(pairs):
        return url
    return urlunsplit(parts._replace(query=urlencode(kept)))


def _url_key(url: str) -> str | None:
    """What two links are compared by: cleaned, no fragment, no trailing slash. None for a non-web link."""
    parts = urlsplit(clean_url(url))
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def plain_evidence(text: str) -> str:
    """The evidence without markdown links: a bracketed citation goes, an inline link keeps its words."""
    text = _CITATION.sub("", text or "")
    text = _INLINE_LINK.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def _site(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host.removeprefix("www.") or url


def screen(report: FactReport, returned_urls: set[str]) -> list[dict]:
    """The report as stored: only links the search returned, plain evidence, unlinked `wrong` downgraded."""
    returned = {key for key in (_url_key(url) for url in returned_urls) if key}
    items = []
    for fact in report.facts:
        sources, seen = [], set()
        for source in fact.sources:
            key = _url_key(source.url)
            if key is None or key not in returned or key in seen:
                continue
            seen.add(key)
            url = clean_url(source.url)
            sources.append({"url": url, "title": source.title.strip() or _site(url)})
        verdict, evidence = fact.verdict, plain_evidence(fact.evidence)
        if verdict == "wrong" and not sources:
            verdict, evidence = "unverified", NO_LINK + evidence
        items.append({"claim": fact.claim.strip(), "verdict": verdict, "evidence": evidence, "sources": sources})
    return items


def _returned_urls(output) -> set[str]:
    """Every page the search gave the model: search results, opened pages and the answer's citations."""
    urls: set[str] = set()
    for item in output or []:
        kind = _field(item, "type")
        if kind == "web_search_call":
            action = _field(item, "action")
            if action is None:
                continue
            for source in _field(action, "sources") or []:
                if isinstance(_field(source, "url"), str):
                    urls.add(clean_url(_field(source, "url")))
            # An open_page or find_in_page action names its page instead of listing sources.
            if isinstance(_field(action, "url"), str):
                urls.add(clean_url(_field(action, "url")))
        elif kind == "message":
            for content in _field(item, "content") or []:
                for annotation in _field(content, "annotations") or []:
                    if _field(annotation, "type") == "url_citation" and isinstance(_field(annotation, "url"), str):
                        urls.add(clean_url(_field(annotation, "url")))
    return urls


def ask_model(text: str, context: str, topic: str, model: str, client) -> tuple[FactReport, set[str]]:
    """One web-search call that checks a passage; returns the report and the URLs the search returned."""
    context_block = ""
    if context.strip():
        context_block = f"Previous passage, context only:\n<<<\n{context.strip()[-CONTEXT_CHARS:]}\n>>>\n"
    response = client.responses.parse(
        model=model,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic}\n{context_block}Passage to check:\n<<<\n{text}\n>>>"},
        ],
        text_format=FactReport,
    )
    if not response.output_parsed:
        raise RuntimeError("Fact check model did not return a structured result")
    return response.output_parsed, _returned_urls(_field(response, "output"))


def _checkable(segment: dict) -> bool:
    return segment["kind"] == "narration" and bool((segment["narration_en"] or "").strip())


def _previous_english(db: Database, segment: dict) -> str:
    previous = db.one(
        "SELECT narration_en FROM segments WHERE job_id=? AND segment_index<? AND kind='narration' "
        "AND TRIM(narration_en)!='' ORDER BY segment_index DESC LIMIT 1",
        (segment["job_id"], segment["segment_index"]),
    )
    return previous["narration_en"] if previous else ""


def _store(db: Database, segment_id: str, facts: dict) -> None:
    db.execute("UPDATE segments SET facts_json=? WHERE id=?", (json.dumps(facts), segment_id))


def facts_of(db: Database, segment_id: str) -> dict:
    row = db.one("SELECT facts_json FROM segments WHERE id=?", (segment_id,))
    return (row or {}).get("facts") or {}


def _claim_key(claim: str) -> str:
    return " ".join(re.findall(r"\w+", claim.lower()))


def _resolved_claims(facts: dict) -> set[str]:
    return {_claim_key(item.get("claim", "")) for item in facts.get("items") or [] if item.get("resolved")}


def set_resolved(db: Database, segment_id: str, index: int, claim: str, resolved: bool) -> dict:
    """Mark one wrong or unverified fact as resolved by the user (or not). The fact itself stays.

    `claim` must match the item at `index`, so a click on a block drawn before a new check never marks
    another fact. Raises LookupError for no such fact and ValueError for a fact that cannot be resolved.
    """
    facts = facts_of(db, segment_id)
    items = facts.get("items") or []
    if facts.get("state") != "done" or not 0 <= index < len(items) or items[index].get("claim") != claim:
        raise LookupError("That fact is no longer there; the facts were checked again")
    if items[index].get("verdict") == "correct":
        raise ValueError("A correct fact has nothing to resolve")
    if resolved:
        items[index]["resolved"] = True
    else:
        items[index].pop("resolved", None)
    _store(db, segment_id, facts)
    return facts


def check_segment(db: Database, segment_id: str, ask: Ask, now: Callable[[], str] = utc_now) -> dict | None:
    """Check one narration segment and store the result in facts_json. None for a segment with nothing to check.

    Only facts_json is written: the text, revision and updated_at stay as they are. A failed check is stored
    with its error, not raised.
    """
    segment = db.one(
        "SELECT id, job_id, segment_index, kind, narration_en FROM segments WHERE id=?", (segment_id,),
    )
    if segment is None:
        raise LookupError(f"Segment {segment_id} not found")
    if not _checkable(segment):
        return None
    text = segment["narration_en"]
    context = _previous_english(db, segment)
    resolved = _resolved_claims(facts_of(db, segment_id))
    _store(db, segment_id, {"state": "checking", "checked_text": text, "at": now(), "error": "", "items": []})
    try:
        report, returned_urls = ask(text, context)
        items = screen(report, returned_urls)
        for item in items:
            # The user already resolved this claim: a check again keeps it resolved.
            if item["verdict"] != "correct" and _claim_key(item["claim"]) in resolved:
                item["resolved"] = True
        facts = {"state": "done", "checked_text": text, "at": now(), "error": "", "items": items}
    except Exception as error:  # noqa: BLE001 - a failed check is shown on the segment; the others go on
        facts = {"state": "failed", "checked_text": text, "at": now(), "error": str(error) or type(error).__name__,
                 "items": []}
    _store(db, segment_id, facts)
    return facts


def check_job(
    db: Database,
    job_id: str,
    ask: Ask,
    segment_id: str | None = None,
    workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Check the job's narration segments with English (or the one segment), `workers` at a time.

    Returns the stored facts per checked segment in segment order; on_progress(done, total) runs on the
    calling thread, first with 0 done.
    """
    columns = "id, job_id, segment_index, kind, narration_en"
    if segment_id is not None:
        segment = db.one(f"SELECT {columns} FROM segments WHERE id=? AND job_id=?", (segment_id, job_id))
        if segment is None:
            raise LookupError(f"Segment {segment_id} not found in job {job_id}")
        segments = [segment]
    else:
        segments = db.all(f"SELECT {columns} FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
    targets = [segment for segment in segments if _checkable(segment)]
    total = len(targets)
    if on_progress:
        on_progress(0, total)
    if not targets:
        return []
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as pool:
        futures = {pool.submit(check_segment, db, segment["id"], ask): segment["id"] for segment in targets}
        for done, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if on_progress:
                on_progress(done, total)
    return [results[segment["id"]] for segment in targets]
