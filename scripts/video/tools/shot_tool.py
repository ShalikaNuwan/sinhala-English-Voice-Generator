"""Claude-agent tool for one film: inspect a shot, find or add a real picture, caption it, apply it and render.

No OpenAI calls: the agent using this tool does the looking and deciding. Every write (library, storyboard,
regen.json, renders) happens under one file lock, so several agents can work at once.

Run with the project's venv:
  export FILM=<name>; PY=.venv/bin/python
  $PY shot_tool.py info N
  $PY shot_tool.py find WORD [WORD ...]
  $PY shot_tool.py add-web --page URL [--image URL] --subject "what it shows"
  $PY shot_tool.py caption ID "caption" [--kind photo|document|newspaper|poster] [--era 1974] [--graphic]
  $PY shot_tool.py apply N (--pick ID | --keep) [--card "Name" --sub "line" [--hold] | --no-card] --why "..."
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, ROOT  # noqa: E402
os.chdir(ROOT)

from app import video_regen as regen  # noqa: E402
from app import video_review as review  # noqa: E402

LOCK = D / "review" / ".shot_tool.lock"


@contextlib.contextmanager
def locked():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def library_items() -> list[dict]:
    return json.loads((D / "library.json").read_text(encoding="utf-8"))["items"]


def write_library(items: list[dict]) -> None:
    tmp = D / "library.json.tmp"
    tmp.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, D / "library.json")


def exclusions() -> dict:
    p = D / "library" / "exclusions.json"
    return json.loads(p.read_text()) if p.exists() else {}


def usable_rights(item: dict) -> bool:
    import plan_edit  # noqa: F401 - loaded by review.engine()
    return item.get("rights") == "fair_use" or plan_edit.free_to_use(item)


def words_between(a: float, b: float) -> str:
    w = json.loads((D / "words.json").read_text(encoding="utf-8"))
    w = w["words"] if isinstance(w, dict) else w
    return " ".join(x.get("word", "") for x in w if a <= x["start"] < b and x["end"] > x["start"])


def describe(beat: dict, lib: dict) -> str:
    if beat.get("kind") == "generated":
        return f"AI DRAWING: {beat.get('detail') or '-'}"
    item = lib.get(beat.get("library_id") or "") or {}
    return f"{beat.get('library_id')} [{beat.get('rights')}] {(item.get('caption') or item.get('title') or '')[:140]}"


def cmd_info(a) -> None:
    review.engine()
    sb = review.load_storyboard(D)
    beats = {b["index"]: b for b in sb["beats"]}
    b = beats[a.n]
    lib = {i["id"]: i for i in library_items()}
    r = review.load_regen(D).get(str(a.n), {})
    mm = lambda t: f"{int(t // 60)}:{int(t % 60):02d}"  # noqa: E731
    print(f"shot {a.n}  {mm(b['start'])}-{mm(b['end'])}  kind={b.get('kind')}")
    print(f"said: {words_between(b['start'], b['end'])}")
    print(f"note: {r.get('review', '-')}")
    print(f"now:  {describe(b, lib)}")
    print(f"card: {b.get('overlay')}")
    for k in range(a.n - 3, a.n + 4):
        if k != a.n and k in beats:
            print(f"  {k:+d}: {describe(beats[k], lib)[:120]}  card={((beats[k].get('overlay') or {}).get('text'))}".replace(f"{k:+d}", f"#{k}"))


def cmd_find(a) -> None:
    review.engine()
    terms = [t.lower() for t in a.words]
    ex = exclusions()
    hits = []
    for i in library_items():
        if i["id"] in ex or i.get("graphic") or not i.get("caption") or not usable_rights(i):
            continue
        text = " ".join(str(i.get(k) or "") for k in ("id", "title", "caption", "found_by", "places", "subjects")).lower()
        if all(t in text for t in terms):
            hits.append(i)
    for i in hits[:30]:
        print(f"{i['id']} | {i.get('rights')} | {i.get('era') or '-'} | {i.get('kind') or '-'} | "
              f"{i.get('width')}x{i.get('height')} | {(i.get('caption') or '')[:130]} | {D / i['file']}")
    print(f"({len(hits)} matches)")


def cmd_add_web(a) -> None:
    seed = {"page_url": a.page, "subject": a.subject}
    if a.image:
        seed["image_url"] = a.image
    with locked():
        before = {i["id"] for i in library_items()}
        pages = D / "library" / "pages.json"
        pages.write_text(json.dumps([seed], ensure_ascii=False), encoding="utf-8")
        try:
            out = subprocess.run([sys.executable, "-u", str(ROOT / "scripts/video/harvest.py"), str(D),
                                  "--stages", "pages"], capture_output=True, text=True, timeout=600)
        finally:
            pages.unlink(missing_ok=True)
        items = library_items()
        new = [i for i in items if i["id"] not in before]
        cache = json.loads((D / "library" / "harvest_cache.json").read_text())
        web_pages = json.loads((D / "library" / "web_pages.json").read_text()) if (D / "library" / "web_pages.json").exists() else {}
        url = a.image or (web_pages.get(a.page) or {}).get("image")
        rec = cache.get(url or "", {})
    if new:
        for i in new:
            print(f"ADDED {i['id']} {i.get('width')}x{i.get('height')} file={D / i['file']} (caption it before use)")
    elif rec.get("id"):
        i = next((x for x in items if x["id"] == rec["id"]), None)
        state = "in library" if i else f"not added ({rec.get('dropped') or rec.get('failed') or 'unknown'})"
        print(f"ALREADY {rec['id']} {state}" + (f" file={D / i['file']} caption={'yes' if i.get('caption') else 'NO'}" if i else ""))
    else:
        print("NOTHING added: the page gave no picture or it was refused/too small.")
        print(out.stdout[-600:], out.stderr[-300:])


def cmd_caption(a) -> None:
    with locked():
        items = library_items()
        for i in items:
            if i["id"] == a.id:
                i["caption"] = a.text
                i["caption_model"] = "claude"
                i["graphic"] = bool(a.graphic)
                if a.kind:
                    i["kind"] = a.kind
                if a.era:
                    i["era"] = a.era
                i.setdefault("subjects", [])
                i.setdefault("places", [])
                i.setdefault("quality", 4)
                i.setdefault("colour", "colour")
                write_library(items)
                print(f"captioned {a.id}")
                return
    sys.exit(f"no library item {a.id}")


def cmd_apply(a) -> None:
    with locked():
        storyboard = review.engine()
        if a.pick:
            review.choose(D, a.n, a.pick, fair_use=regen.web_finds(D))
        sb = review.load_storyboard(D)
        beat = review.find_beat(sb, a.n)
        if a.no_card:
            beat["overlay"] = None
        elif a.card:
            beat["overlay"] = {"style": "name", "text": a.card.strip(), "subtext": (a.sub or "").strip()}
            if a.hold:
                beat["overlay"]["hold"] = True
        beat["owner"] = dict(beat.get("owner") or {"choice": beat.get("library_id") or beat.get("asset")}, at=time.time())
        storyboard.save(D, sb)
        clip = regen.render_preview(D, a.n)
        review.set_mark(D, a.n, None)
        regen.note(D, a.n, state="done", step="Done", clip=clip, check=None, error="",
                   explanation=f"(Claude) {a.why}")
    print(f"done {a.n}: {clip}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info"); s.add_argument("n", type=int); s.set_defaults(f=cmd_info)
    s = sub.add_parser("find"); s.add_argument("words", nargs="+"); s.set_defaults(f=cmd_find)
    s = sub.add_parser("add-web"); s.add_argument("--page", required=True); s.add_argument("--image", default="")
    s.add_argument("--subject", required=True); s.set_defaults(f=cmd_add_web)
    s = sub.add_parser("caption"); s.add_argument("id"); s.add_argument("text"); s.add_argument("--kind", default="")
    s.add_argument("--era", default=""); s.add_argument("--graphic", action="store_true"); s.set_defaults(f=cmd_caption)
    s = sub.add_parser("apply"); s.add_argument("n", type=int)
    g = s.add_mutually_exclusive_group(required=True); g.add_argument("--pick"); g.add_argument("--keep", action="store_true")
    s.add_argument("--card", default=""); s.add_argument("--sub", default=""); s.add_argument("--hold", action="store_true")
    s.add_argument("--no-card", action="store_true"); s.add_argument("--why", required=True); s.set_defaults(f=cmd_apply)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
