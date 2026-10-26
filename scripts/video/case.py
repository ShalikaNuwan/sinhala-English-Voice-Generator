"""The case profile: everything the picture research needs to know about one film's story, in case.json.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/case.py <video_dir> <script.txt> [--force]

The pipeline is general; what differs between films is where their pictures are: the case's Wikimedia
Commons category and Wikipedia articles, the region and era its period photographs and footage come
from, the people famous enough to search by name, and the names too broad to search at all. A model reads
the script and proposes these; code keeps only the Wikipedia articles and Commons categories that exist,
and writes `case.json` for the owner to read (and edit) before the harvest. harvest.py, caption.py,
footage.py and plan_edit.py read it with `load(video_dir)`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class Category(BaseModel):
    category: str = Field(description="A Wikimedia Commons category title, starting 'Category:'.")
    depth: int = Field(default=0, description="How many levels of subcategories to include (0-1).")


class Entity(BaseModel):
    name: str
    kind: Literal["person", "place", "institution", "business", "object"]


class CaseProfile(BaseModel):
    title: str = Field(description="What the film is about, as a phrase: 'the John Wayne Gacy murders'.")
    subject: str = Field(description="The English Wikipedia article at the centre of the case, exactly as titled.")
    keyword: str = Field(description="One word that ties a search to this case (usually the culprit's surname), "
                                     "added to searches for private people.")
    region: str = Field(description="Where the story happens, most important first, with suburbs or towns.")
    region_short: str = Field(description="The state, province or country, e.g. 'Illinois'.")
    main_places: list[str] = Field(default_factory=list,
                                   description="The cities or towns most of the story happens in, as Wikimedia "
                                               "Commons names them, e.g. ['Chicago', 'Des Plaines, Illinois'].")
    states: list[str] = Field(default_factory=list,
                              description="The states or provinces of the story, e.g. ['Illinois', 'Iowa'].")
    era: str = Field(description="When the story happens and its turning points, e.g. '1960s to 1994, centred "
                                 "on the December 1978 arrest and the 1980 trial'.")
    decade: str = Field(description="The decade most of the story is set in, e.g. '1970s'.")
    story_settings: str = Field(description="The kinds of events the film shows, in one line: 'a police "
                                            "investigation, a house search, a 1980 trial, a 1994 execution'.")
    settings: list[str] = Field(description="8-14 kinds of b-roll the film needs, plain words: 'night roads', "
                                            "'snow and winter driving', 'courtrooms'.")
    commons_categories: list[str] = Field(description="Commons categories about the case itself.")
    period_categories: list[Category] = Field(description="Commons categories of the region in the era "
                                                          "(e.g. 'Category:Chicago in the 1970s').")
    wikipedia_articles: list[str] = Field(description="English Wikipedia articles whose pictures may serve: "
                                                      "the case, its places, institutions and public figures.")
    starter_entities: list[Entity] = Field(description="The names a picture researcher would search first.")
    public_figures: list[str] = Field(default_factory=list,
                                      description="People famous enough to search by name alone.")
    skip_entities: list[str] = Field(default_factory=list,
                                     description="Names in the script too broad or unrelated to search (states, "
                                                 "big cities, brands, parties).")
    archive_queries: list[str] = Field(description="25-45 Internet Archive searches for period footage of the "
                                                   "SETTINGS, each ONE TO THREE plain words, as film catalogues index "
                                                   "topics: 'police', 'police car', 'traffic', 'winter', 'courtroom', "
                                                   "'highway patrol', 'campus', 'seattle'. Never the case or its "
                                                   "people, never a year: the archives hold old educational and "
                                                   "government films, not news of the case, and long searches find "
                                                   "nothing.")
    commons_footage_queries: list[str] = Field(description="10-20 Commons searches for period video of the "
                                                           "settings, likewise never of the case itself.")
    fallback_query: str = Field(description="One search for a period street of the main place, e.g. "
                                            "'chicago 1970s street'.")
    reveals: list[list[str]] = Field(default_factory=list,
                                     description="Optional: [name, regex] pairs for the sound mix's reveal "
                                                 "drones; empty lets sound.py take them from the storyboard.")
    footage_years: list[int] = Field(description="The first and last year period footage may come from: from "
                                                 "about ten years before the main events (cars and streets change "
                                                 "slowly) to the story's end, e.g. [1955, 1994].")


def brief(profile: CaseProfile) -> str:
    """One line for a prompt: what the film is about, where and when."""
    return f"{profile.title} ({profile.region}; {profile.era})"


def fallback_query(video_dir: Path, default: str = "city street") -> str:
    """The search for a period street of the story's main place, for a shot nothing else covers."""
    path = path_of(video_dir)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8")).get("fallback_query") or default


def path_of(video_dir: Path) -> Path:
    return Path(video_dir) / "case.json"


def load(video_dir: Path) -> CaseProfile:
    path = path_of(video_dir)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run scripts/video/case.py {video_dir} <script.txt> first")
    return CaseProfile(**json.loads(path.read_text(encoding="utf-8")))


SYSTEM = """You prepare the picture research for a documentary built from real archival photographs and
footage (Wikimedia Commons, Wikipedia, the Internet Archive). Read the narration script and fill in the
case profile. Use the titles Wikipedia and Commons actually use; never invent a category. Be concrete
about the region and era the pictures must come from."""


def propose(client, model: str, script: str) -> CaseProfile:
    r = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": script[:200_000]}],
                               text_format=CaseProfile)
    return r.output_parsed


def existing_categories(titles: list[str]) -> set[str]:
    """The Commons categories among `titles` that exist."""
    import harvest
    out: set[str] = set()
    for start in range(0, len(titles), 40):
        data = harvest.fetch_json(harvest.COMMONS_API, {"action": "query", "titles": "|".join(titles[start:start + 40]),
                                                        "redirects": 1, "format": "json"})
        for page in ((data.get("query") or {}).get("pages") or {}).values():
            if "missing" not in page and "invalid" not in page:
                out.add(page["title"])
    return out


def search_categories(query: str, limit: int = 10) -> list[str]:
    import harvest
    data = harvest.fetch_json(harvest.COMMONS_API, {"action": "query", "list": "search", "srnamespace": 14,
                                                    "srlimit": limit, "srsearch": query, "format": "json"})
    return [x["title"] for x in (data.get("query") or {}).get("search") or []]


PERIOD_MAX_CATEGORIES = 10


def period_categories(places: list[str], states: list[str], decade: str, years: list[int],
                      search=search_categories, exists=existing_categories) -> list[Category]:
    """Commons categories of the story's places in its era, as Commons names them: "Seattle in the 1970s"
    (with its subcategories), "1976 in Salt Lake City" (a year of the story), and for the 1970s the
    government's DOCUMERICA photographs of each state. Found by search; only real titles are kept."""
    decade_re = re.escape(decade)
    found: list[Category] = []
    for place in places:
        name = re.escape(place.split(",")[0])
        for title in search(f"{place} {decade}"):
            if re.match(rf"^Category:{name}[^:]* in the {decade_re}$", title):
                found.append(Category(category=title, depth=1))
            elif (m := re.match(rf"^Category:(\d{{4}}) in {name}[^:]*$", title)) and int(m.group(1)) in years:
                found.append(Category(category=title, depth=0))
    if decade == "1970s":                       # DOCUMERICA ran 1971-1977, public domain (US government)
        wanted = [f"Category:DOCUMERICA photographs of {s}" for s in states]
        found += [Category(category=c, depth=0) for c in wanted if c in exists(wanted)]
    out, seen = [], set()
    rank = lambda c: (0 if c.depth else 1 if "DOCUMERICA" in c.category else 2, c.category)  # noqa: E731
    for c in sorted(found, key=rank):          # a decade's category, then DOCUMERICA, then single years
        if c.category not in seen:
            seen.add(c.category)
            out.append(c)
    return out[:PERIOD_MAX_CATEGORIES]


def story_years(profile: CaseProfile) -> list[int]:
    years = [int(y) for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", profile.era)]
    if not years:
        return []
    return list(range(min(years), max(years) + 1))


def verified(profile: CaseProfile, search=search_categories, exists=existing_categories) -> CaseProfile:
    """Only the Wikipedia articles and Commons categories that exist (under their current titles), with the
    period categories found on Commons added to those the model named."""
    import harvest
    articles = harvest._existing_articles(list(dict.fromkeys([profile.subject] + profile.wikipedia_articles)), None)
    subject = next((a for a in articles if a.lower() == profile.subject.lower()), articles[0] if articles else profile.subject)
    cats = exists(list(dict.fromkeys(profile.commons_categories + [c.category for c in profile.period_categories])))
    period = [c for c in profile.period_categories if c.category in cats]
    period += [c for c in period_categories(profile.main_places, profile.states, profile.decade, story_years(profile),
                                            search, exists) if c.category not in {p.category for p in period}]
    return profile.model_copy(update={
        "subject": subject,
        "wikipedia_articles": articles,
        "commons_categories": [c for c in profile.commons_categories if c in cats],
        "period_categories": period[:PERIOD_MAX_CATEGORIES],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("script")
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true", help="replace an existing case.json")
    args = parser.parse_args()

    from openai import OpenAI

    from app.config import settings

    path = path_of(Path(args.video_dir))
    if path.exists() and not args.force:
        print(f"{path} exists (edit it, or pass --force to propose a new one)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(args.script).read_text(encoding="utf-8")
    proposed = propose(OpenAI(api_key=settings.openai_api_key), args.model or settings.text_model, script)
    profile = verified(proposed)
    path.write_text(json.dumps(profile.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8")
    dropped = (set(proposed.wikipedia_articles) - set(profile.wikipedia_articles)) | \
        ({c for c in proposed.commons_categories} - set(profile.commons_categories)) | \
        ({c.category for c in proposed.period_categories} - {c.category for c in profile.period_categories})
    print(f"wrote {path}: {brief(profile)}")
    print(f"  {len(profile.wikipedia_articles)} Wikipedia articles, {len(profile.commons_categories)} case and "
          f"{len(profile.period_categories)} period categories, {len(profile.starter_entities)} starter names")
    if dropped:
        print(f"  dropped (not found): {sorted(dropped)}")


if __name__ == "__main__":
    main()
