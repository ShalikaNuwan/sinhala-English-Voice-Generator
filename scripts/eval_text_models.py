"""Compare text models on real segments: faithful translation, then story adaptation, judged blind.

    OPENAI_API_KEY=... .venv/bin/python scripts/eval_text_models.py <project_id> \
        --models gpt-5.6-terra,gpt-5.6-sol,gpt-6-sol --segments 10 --judge gpt-6-astra

For each sampled segment every candidate runs the pipeline's own two stages (translate, then adapt
over the spoken runs) with the same context. A judge model that is not a candidate scores the
candidates blind, labels shuffled, on accuracy against the Sinhala, story-mode quality, and
closeness to the reviewer's edited English, twice with different shuffles. Writes results.json and
report.md under data/evals/<stamp>/. Spends API credit: segments x models x 2 calls, plus the judge.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import AIClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402


class CandidateScore(BaseModel):
    label: str
    accuracy: int = Field(description="1-10: every fact, name, number, date and causal link of the Sinhala preserved; no additions.")
    story: int = Field(description="1-10: reads as a storyteller's spoken voice-over, not a translation; natural, tense, for the ear.")
    closeness: int = Field(description="1-10: how close in content and tone to the reviewer's edited English.")
    problems: list[str] = Field(default_factory=list, description="Concrete omissions, additions or errors, short.")


class Judgement(BaseModel):
    scores: list[CandidateScore]
    ranking: list[str] = Field(description="Labels from best to worst overall.")


JUDGE_SYSTEM = (
    "You are grading English voice-over scripts produced from a Sinhala true-crime narration. "
    "You read Sinhala fluently. Grade each candidate independently and then rank them. Accuracy is "
    "judged against the Sinhala source only. Story quality means it sounds like a storyteller speaking "
    "to a listener: short spoken sentences, natural flow, tension kept, nothing bookish. Closeness is "
    "judged against the reviewer's edited English, which is what the owner wanted; it may itself have "
    "small flaws, so do not treat it as the accuracy reference. Be strict and specific."
)


def usage_recorder(ai: AIClient, sink: list[dict], lock: threading.Lock):
    """Wrap the client's parse call so token usage is captured without changing the pipeline code."""
    original = ai.client.responses.parse

    def parse(**kwargs):
        response = original(**kwargs)
        usage = getattr(response, "usage", None)
        with lock:
            sink.append({
                "model": kwargs.get("model"),
                "input": getattr(usage, "input_tokens", 0) or 0,
                "output": getattr(usage, "output_tokens", 0) or 0,
            })
        return response

    ai.client.responses.parse = parse


def with_retries(call, attempts: int = 3):
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - rate limits and transient failures
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up: {last}")


def sample_segments(db: Database, project_id: str, count: int) -> list[dict]:
    job = db.one("SELECT id FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
    if not job:
        sys.exit("No job for that project")
    rows = db.all("SELECT * FROM segments WHERE job_id=? AND kind='narration' ORDER BY segment_index", (job["id"],))
    edited = [r for r in rows if r.get("status") in ("edited", "approved") and r.get("phrases")] or rows
    if len(edited) <= count:
        chosen = edited
    else:
        step = len(edited) / count
        chosen = [edited[int(i * step)] for i in range(count)]
    by_index = {r["segment_index"]: r for r in rows}
    for seg in chosen:
        previous = by_index.get(seg["segment_index"] - 1) or {}
        seg["previous_faithful"] = (previous.get("faithful_en") or "")[-1000:]
        seg["previous_narration"] = (previous.get("narration_en") or "")[-600:]
    return chosen


def run_candidate(ai: AIClient, model: str, seg: dict, project: dict) -> dict:
    started = time.monotonic()
    faithful = with_retries(lambda: ai.translate(seg["transcript_si"], model, project["topic"], project["glossary"] or {}, seg["previous_faithful"]))
    runs = [p["text"] for p in seg["phrases"] if (p.get("text") or "").strip()] or None
    adaptation = with_retries(lambda: ai.adapt(faithful.english_faithful, model, project["narrator_profile"] or {}, seg["previous_narration"], runs=runs))
    lines = [line.text for line in adaptation.lines]
    narration = "\n\n".join(lines) if lines else adaptation.narration_text
    return {
        "model": model,
        "segment_index": seg["segment_index"],
        "faithful": faithful.english_faithful,
        "narration": narration,
        "lines_follow_runs": bool(runs) and [l.runs for l in adaptation.lines] == [[i] for i in range(1, len(runs) + 1)],
        "seconds": round(time.monotonic() - started, 1),
    }


def judge(ai: AIClient, judge_model: str, seg: dict, outputs: list[dict], seed: int) -> dict:
    order = list(outputs)
    random.Random(seed).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    shown = "\n\n".join(f"=== Candidate {label} ===\n{out['narration']}" for label, out in zip(labels, order))
    user = (
        f"Sinhala source:\n<<<\n{seg['transcript_si']}\n>>>\n\n"
        f"Reviewer's edited English (what the owner wanted):\n<<<\n{seg['narration_en']}\n>>>\n\n"
        f"Candidates:\n{shown}"
    )
    response = with_retries(lambda: ai.client.responses.parse(
        model=judge_model,
        input=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        text_format=Judgement,
    ))
    parsed = response.output_parsed
    mapping = {label: out["model"] for label, out in zip(labels, order)}
    return {
        "segment_index": seg["segment_index"],
        "seed": seed,
        "scores": [dict(s.model_dump(), model=mapping.get(s.label, "?")) for s in parsed.scores],
        "ranking": [mapping.get(label, "?") for label in parsed.ranking],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id")
    parser.add_argument("--models", default="gpt-5.6-terra,gpt-5.6-sol,gpt-6-sol")
    parser.add_argument("--segments", type=int, default=10)
    parser.add_argument("--judge", default="gpt-6-astra")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    db = Database(settings.database_path)
    project = db.one("SELECT * FROM projects WHERE id=?", (args.project_id,))
    if not project:
        sys.exit("No such project")
    segments = sample_segments(db, args.project_id, args.segments)
    print(f"{len(segments)} segments: {[s['segment_index'] for s in segments]}; models: {models}; judge: {args.judge}")

    ai = AIClient(settings.openai_api_key, None)
    usage: list[dict] = []
    usage_recorder(ai, usage, threading.Lock())

    out_dir = settings.data_dir / "evals" / f"text-models-{datetime.now(timezone.utc):%Y%m%d-%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(model, seg) for seg in segments for model in models]
    outputs: list[dict] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (model, seg), result in zip(tasks, pool.map(lambda t: _safe(run_candidate, ai, t[0], t[1], project), tasks)):
            if isinstance(result, dict):
                outputs.append(result)
                print(f"  done {model} segment {seg['segment_index']} in {result['seconds']}s", flush=True)
            else:
                failures.append(f"{model} segment {seg['segment_index']}: {result}")
                print(f"  FAILED {model} segment {seg['segment_index']}: {result}", flush=True)

    judgements: list[dict] = []
    for seg in segments:
        mine = [o for o in outputs if o["segment_index"] == seg["segment_index"]]
        if len(mine) < 2:
            continue
        for pass_number in range(args.passes):
            try:
                judgements.append(judge(ai, args.judge, seg, mine, seed=seg["segment_index"] * 100 + pass_number))
                print(f"  judged segment {seg['segment_index']} pass {pass_number + 1}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"judge segment {seg['segment_index']} pass {pass_number + 1}: {exc}")

    (out_dir / "results.json").write_text(json.dumps({
        "project": args.project_id, "models": models, "judge": args.judge,
        "segments": [s["segment_index"] for s in segments], "outputs": outputs,
        "judgements": judgements, "usage": usage, "failures": failures,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "report.md").write_text(report(models, args.judge, segments, outputs, judgements, usage, failures), encoding="utf-8")
    print(f"\nwritten to {out_dir}")
    print((out_dir / "report.md").read_text(encoding="utf-8"))


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return str(exc)[:300]


def report(models, judge_model, segments, outputs, judgements, usage, failures) -> str:
    lines = [f"# Text model evaluation ({judge_model} as judge, {len(segments)} segments, {len(judgements)} judging passes)", ""]
    lines += ["| Model | Accuracy | Story | Closeness | Overall | Ranked first | Lines follow runs | Median seconds | Tokens in/out per segment |", "|---|---|---|---|---|---|---|---|---|"]
    for model in models:
        scores = [s for j in judgements for s in j["scores"] if s["model"] == model]
        firsts = sum(1 for j in judgements if j["ranking"] and j["ranking"][0] == model)
        mine = [o for o in outputs if o["model"] == model]
        follow = sum(1 for o in mine if o["lines_follow_runs"])
        used = [u for u in usage if u["model"] == model]
        per_seg = max(1, len(mine))
        acc = statistics.mean(s["accuracy"] for s in scores) if scores else float("nan")
        sto = statistics.mean(s["story"] for s in scores) if scores else float("nan")
        clo = statistics.mean(s["closeness"] for s in scores) if scores else float("nan")
        overall = (acc + sto + clo) / 3 if scores else float("nan")
        secs = statistics.median(o["seconds"] for o in mine) if mine else float("nan")
        lines.append(f"| {model} | {acc:.1f} | {sto:.1f} | {clo:.1f} | {overall:.1f} | {firsts}/{len(judgements)} | {follow}/{len(mine)} | {secs:.0f} | {sum(u['input'] for u in used) // per_seg} / {sum(u['output'] for u in used) // per_seg} |")
    lines += ["", "Scores are 1-10 means over segments and passes. Accuracy is against the Sinhala; closeness is to the reviewer's edited English.", ""]
    # per-segment winners
    lines += ["## Winner per segment and pass", ""]
    for j in judgements:
        lines.append(f"- segment {j['segment_index']} pass {j['seed'] % 100 + 1}: {' > '.join(j['ranking'])}")
    # judge problems by model
    lines += ["", "## Problems the judge named most often", ""]
    for model in models:
        problems = [p for j in judgements for s in j["scores"] if s["model"] == model for p in s["problems"]]
        lines.append(f"### {model} ({len(problems)} problems named)")
        for p in problems[:8]:
            lines.append(f"- {p}")
        lines.append("")
    # sample
    if segments and outputs:
        seg = segments[0]
        lines += [f"## Sample: segment {seg['segment_index']}", "", "**Reviewer's edited English**", "", seg["narration_en"] or "", ""]
        for model in models:
            mine = [o for o in outputs if o["model"] == model and o["segment_index"] == seg["segment_index"]]
            if mine:
                lines += [f"**{model}**", "", mine[0]["narration"], ""]
    if failures:
        lines += ["## Failures", ""] + [f"- {f}" for f in failures]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
