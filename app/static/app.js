let currentProject = null;
let currentJob = null;
let pollTimer = null;
let inFlight = 0;  // actions running right now; a background refresh waits until this is zero

const $ = (id) => document.getElementById(id);
const request = async (url, options = {}) => {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = response.statusText;
    try { message = (await response.json()).detail || message; } catch (_) {}
    const error = new Error(message);
    error.status = response.status;                 // lets a caller tell "already running" (409) apart
    throw error;
  }
  return response.json();
};
const jsonOptions = (method, body) => ({ method, headers:{"Content-Type":"application/json"}, body:JSON.stringify(body) });
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

function parseGlossary(text) {
  const result = {};
  text.split("\n").forEach(line => {
    const [left, ...right] = line.split("=");
    if (left?.trim() && right.join("=").trim()) result[left.trim()] = right.join("=").trim();
  });
  return result;
}

/* ---------------------------------------------------------------- busy state */

function errorSlot(root) {
  return root?.classList?.contains("segment") ? root.querySelector(".segment-error") : $("global-error");
}

function showError(root, message) {
  const slot = errorSlot(root);
  if (!slot) return;
  slot.textContent = message || "";
  slot.hidden = !message;
}

/** Run one action with every button in its scope disabled, so nothing can be fired twice. */
async function run(button, action, { label = "Working…" } = {}) {
  if (button.dataset.busy === "1") return;          // a second click while the first is in flight
  const root = button.closest(".segment") || $("workspace") || document.body;
  const buttons = Array.from(root.querySelectorAll("button"));
  const restore = button.textContent;
  const previously = new Map(buttons.map(item => [item, item.disabled]));

  button.dataset.busy = "1";
  button.textContent = label;
  buttons.forEach(item => { item.disabled = true; });
  showError(root, "");
  inFlight += 1;
  try {
    await action();
  } catch (error) {
    showError(root, error.message);
  } finally {
    inFlight -= 1;
    if (button.isConnected) {                        // the card may have been re-rendered underneath us
      button.dataset.busy = "";
      button.textContent = restore;
      buttons.forEach(item => { item.disabled = previously.get(item) ?? false; });
    }
  }
}

/** True when a transcript or script box holds text the server has not been told about. */
function hasUnsavedEdits() {
  return Array.from(document.querySelectorAll(".segment textarea"))
    .some(box => box.value !== (box.dataset.initial ?? box.value));
}

/* ------------------------------------------------------------------ projects */

async function loadProjects() {
  const projects = await request("/api/projects");
  $("projects").innerHTML = projects.length ? projects.map(project => `
    <div class="project-row" data-id="${project.id}">
      <strong>${escapeHtml(project.title)}</strong><span class="pill">${escapeHtml(project.status)}</span>
    </div>`).join("") : '<p class="muted">No projects yet.</p>';
  document.querySelectorAll(".project-row").forEach(row => row.onclick = () => openProject(row.dataset.id));
}

async function openProject(id, { segment = null, all = false } = {}) {
  currentProject = await request(`/api/projects/${id}`);
  shownId = segment;
  showAll = all && !segment;
  $("segment-search").value = "";
  $("search-results").innerHTML = "";
  $("segment-finder").classList.add("hidden");
  setFinderNote("");
  rememberPlace();
  $("workspace").classList.remove("hidden");
  $("project-title").textContent = currentProject.title;
  $("project-status").textContent = currentProject.status;
  showError($("workspace"), "");
  resetTasks();
  renderSpeakingProfile(currentProject.speaking_profile);
  const latest = currentProject.jobs?.[0];
  if (latest) { currentJob = latest.id; await refreshJob(); } else { currentJob = null; $("segments").innerHTML = ""; }
  await loadExports();
  $("workspace").scrollIntoView({behavior:"smooth"});
}

function renderSpeakingProfile(profile) {
  const box = $("speaking-profile");
  if (!profile?.derived) { box.classList.add("hidden"); return; }
  const { pace, pause_style, dynamics } = profile.derived;
  const measured = profile.measured || {};
  box.classList.remove("hidden");
  box.innerHTML = `
    <strong>Detected speaking pattern</strong>
    <p>${escapeHtml(pace)} pace &middot; ${escapeHtml(pause_style)} pauses &middot; ${escapeHtml(dynamics)}
       <span class="muted">(~${measured.mean_pause_ms || 0}ms typical pause, ${measured.pauses_per_min || 0}/min)</span></p>
    ${profile.described ? `<p class="muted">${escapeHtml(profile.described)}</p>` : ""}`;
}

async function refreshJob() {
  if (!currentJob) return;
  const job = await request(`/api/jobs/${currentJob}`);
  $("progress").classList.remove("hidden");
  $("progress-bar").style.width = `${job.progress}%`;
  const warnings = (job.warnings || []).length ? ` · ${job.warnings.join(" · ")}` : "";
  $("progress-text").textContent = `${job.stage} · ${job.progress}%${job.error ? ` · ${job.error}` : ""}${warnings}`;
  const running = ["queued","running"].includes(job.status);
  $("process").disabled = running;
  $("assemble").disabled = running;
  $("export-parts").disabled = running;
  if (running) {
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshJob, 1800);
  } else {
    await loadSegments({background:true}); await loadProjects();
    await pollTasks();                               // a sentence fix or fact check may already be running
  }
}

/* ------------------------------------------------------------------ segments */

function qaBadge(segment) {
  const passed = segment.qa_status === "passed";
  const label = segment.qa?.approved ? "approved" : segment.qa_status;
  return `<span class="${passed ? "qa-pass" : "qa-review"}">${escapeHtml(label)}</span>`;
}

function issueLine(segment) {
  const qa = segment.qa || {};
  const issues = (qa.issues || []).filter(Boolean);
  if (issues.length) return `<p class="muted issues">${escapeHtml(issues.join(" · "))}</p>`;
  const overridden = (qa.overridden_issues || []).filter(Boolean);
  if (overridden.length) return `<p class="muted issues">Approved over: ${escapeHtml(overridden.join(" · "))}</p>`;
  return "";
}

function segmentCard(segment) {
  const player = segment.tts_audio_url ? `<audio controls src="${segment.tts_audio_url}"></audio>` : "";
  const head = `<div class="segment-head"><strong>Segment ${segment.segment_index}</strong>
      <span class="head-right">${segment.kind === "original" ? '<span class="pill">Recorded audio, kept as is</span>' : ""}${qaBadge(segment)}</span></div>`;

  if (segment.kind === "original") {
    return `
    <article class="segment" data-id="${segment.id}" data-kind="original">
      ${head}
      <p class="muted">${escapeHtml(segment.transcript_si || "(no speech recognised in this recording)")}</p>
      ${player}
      <div class="actions">
        ${segment.tts_audio_url
          ? `<button class="confirm"${segment.qa_status === "passed" ? " disabled" : ""}>Confirm recording</button>`
          : '<button class="regen secondary">Cut the recording again</button>'}
        <button class="to-narration secondary">Treat as narration</button>
      </div>
      ${issueLine(segment)}
      <p class="segment-error error" hidden></p>
    </article>`;
  }

  const canApprove = segment.qa_status !== "passed" && Boolean(segment.tts_audio_url);
  return `
    <article class="segment" data-id="${segment.id}" data-kind="narration">
      ${head}
      <div class="segment-grid">
        <label>Sinhala transcript<textarea class="si" data-initial="${escapeHtml(segment.transcript_si || "")}">${escapeHtml(segment.transcript_si || "")}</textarea></label>
        <label>English narration<textarea class="en" data-initial="${escapeHtml(segment.narration_en || "")}">${escapeHtml(segment.narration_en || "")}</textarea></label>
      </div>
      ${player}
      <div class="actions">
        <button class="save secondary">Save edits</button>
        <button class="regen">Regenerate this segment</button>
        ${canApprove ? '<button class="approve approve-button">Approve</button>' : ""}
        <button class="to-original secondary">Keep as recorded</button>
      </div>
      ${issueLine(segment)}
      ${sentenceFixLine(segment)}
      <div class="facts-slot">${factsBlock(segment)}</div>
      <p class="segment-error error" hidden></p>
    </article>`;
}

/** Poll a segment until the background worker has finished with it. */
async function waitForSegment(id) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await wait(1500);
    let segment = null;
    try { segment = await request(`/api/segments/${id}`); } catch (error) { if (error.status === 404) return; throw error; }
    if (segment.status !== "regenerating") return;
  }
}

/* One segment at a time: a long job drawn whole ran the browser tab out of memory. The page holds a short
   outline of every segment (for the counts and Previous/Next) and the full content of the one shown. */
let shownId = null;                                  // the segment on screen, or null before a search
/* "All segments" lists every segment's text to read, without the editing boxes, players and fact lists that
   made the whole job too heavy; Edit opens one in the single view. */
let showAll = false;

/** The outline of every segment, with the shown one in full (or every text, for the all-segments view);
    null when the job changed meanwhile. */
async function fetchSegments({ text = showAll } = {}) {
  const job = currentJob;
  const outline = await request(`/api/jobs/${job}/segment-outline${text ? "?text=true" : ""}`);
  let shown = null;
  if (shownId && outline.some(item => item.id === shownId)) shown = await request(`/api/segments/${shownId}`);
  if (job !== currentJob) return null;
  if (!shown) shownId = null;
  return overlayPending(outline.map(item => (shown && item.id === shown.id) ? shown : item));
}

async function loadSegments({ background = false } = {}) {
  if (!currentJob) return;
  // Never pull the page out from under someone who is mid-action or mid-edit.
  if (background && (inFlight > 0 || hasUnsavedEdits())) return false;
  const segments = await fetchSegments();
  if (!segments) return false;
  const shown = segments.find(item => item.id === shownId);
  $("segments").innerHTML = shown ? segmentCard(shown)
    : showAll ? segments.map(readingCard).join("")
    : (segments.length ? '<p class="muted segment-hint">Search a segment number or some words to open a segment, or show all of them.</p>' : "");
  document.querySelectorAll(".segment-read .open-segment").forEach(button =>
    button.addEventListener("click", () => showSegment(button.closest(".segment-read").dataset.id)));
  renderSummary(segments);
  renderAgentBar(segments);
  renderNavigator(segments);
  document.querySelectorAll(".segment").forEach(card => bindSegment(card));
  if (anyChecking()) scheduleTaskPoll();             // a fact check is still filling in some cards
  return true;
}

/** Show one segment; refused while the shown one has unsaved edits. */
async function showSegment(id) {
  if (id !== shownId && hasUnsavedEdits()) {
    setFinderNote("Save your edits to this segment before opening another one.", true);
    return;
  }
  shownId = id;
  showAll = false;
  rememberPlace();
  setFinderNote("");
  await loadSegments();
  $("segment-finder").scrollIntoView({block:"nearest"});
}

/** Every segment's text on one page, to read through; refused while the shown one has unsaved edits. */
async function showAllSegments() {
  if (hasUnsavedEdits()) {
    setFinderNote("Save your edits to this segment before showing all of them.", true);
    return;
  }
  const was = shownId;
  shownId = null;
  showAll = true;
  $("search-results").innerHTML = "";
  rememberPlace();
  setFinderNote("");
  await loadSegments();
  const card = was && document.querySelector(`.segment-read[data-id="${CSS.escape(was)}"]`);
  if (card) card.scrollIntoView({block:"start"});    // back from editing one: land where it was
}

function clock(ms) {
  const seconds = Math.floor((ms || 0) / 1000), h = Math.floor(seconds / 3600);
  const mm = String(Math.floor(seconds % 3600 / 60)), ss = String(seconds % 60).padStart(2, "0");
  return h ? `${h}:${mm.padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
}

/** One segment in the all-segments view: its text and fact counts, read-only, with Edit to open it. */
function readingCard(segment) {
  const text = segment.kind === "original" ? segment.transcript_si : segment.narration_en;
  const paragraphs = (text || "").split(/\n\s*\n/).map(part => part.trim()).filter(Boolean)
    .map(part => `<p>${escapeHtml(part)}</p>`).join("") || '<p class="muted">(no text)</p>';
  const items = segmentFacts(segment).items || [];
  const open = (verdict) => items.filter(item => item.verdict === verdict && !item.resolved).length;
  const notes = [[open("wrong"), "looks wrong"], [open("unverified"), "unverified"],
                 [items.filter(item => item.resolved).length, "resolved"]].filter(([count]) => count)
    .map(([count, label]) => `${count} ${label}`);
  return `
    <article class="segment-read" data-id="${escapeHtml(segment.id)}">
      <div class="segment-head"><strong>Segment ${segment.segment_index}
          <span class="muted">${clock(segment.start_ms)} – ${clock(segment.end_ms)}</span></strong>
        <span class="head-right">${segment.kind === "original" ? '<span class="pill">Recorded audio</span>' : ""}${qaBadge(segment)}
          <button type="button" class="secondary open-segment">Edit</button></span></div>
      ${paragraphs}
      ${notes.length ? `<p class="muted read-facts">Facts: ${escapeHtml(notes.join(" · "))}</p>` : ""}
    </article>`;
}

function setFinderNote(message, isError = false) {
  const note = $("finder-note");
  note.textContent = message || "";
  note.hidden = !message;
  note.classList.toggle("error", isError);
}

/** "Segment 12 of 143" and the Previous / Next buttons, by the order of the segments. */
function renderNavigator(segments) {
  const finder = $("segment-finder");
  finder.classList.toggle("hidden", !segments.length);
  const at = segments.findIndex(item => item.id === shownId);
  $("segment-position").textContent = showAll ? `All ${segments.length} segments` : at < 0 ? `${segments.length} segments`
    : `Segment ${segments[at].segment_index} · ${at + 1} of ${segments.length}`;
  $("segment-prev").disabled = at <= 0;
  $("segment-next").disabled = at < 0 || at >= segments.length - 1;
  $("segment-all").disabled = showAll;
  document.querySelectorAll("#search-results button").forEach(button =>
    button.classList.toggle("current", button.dataset.id === shownId));
}

function stepSegment(offset) {
  const at = lastSegments.findIndex(item => item.id === shownId);
  const next = lastSegments[at + offset];
  if (at >= 0 && next) return showSegment(next.id);
}

/** A number opens that segment; words list every segment holding them (one match opens directly). */
async function findSegment(text) {
  const query = text.trim();
  $("search-results").innerHTML = "";
  if (!query || !currentJob) return;
  if (/^\d+$/.test(query)) {
    const match = lastSegments.find(item => String(item.segment_index) === query);
    if (!match) {
      const indices = lastSegments.map(item => item.segment_index);
      setFinderNote(`There is no segment ${query}; they run from ${Math.min(...indices)} to ${Math.max(...indices)}.`, true);
      return;
    }
    return showSegment(match.id);
  }
  const matches = await request(`/api/jobs/${currentJob}/segment-outline?q=${encodeURIComponent(query)}`);
  if (!matches.length) { setFinderNote(`No segment contains “${query}”.`, true); return; }
  if (matches.length > 1) {
    setFinderNote(`${matches.length} segments contain “${query}”:`);
    $("search-results").innerHTML = matches.map(item => `
      <button type="button" class="secondary result" data-id="${escapeHtml(item.id)}">
        <strong>Segment ${item.segment_index}</strong> <span>${escapeHtml(item.snippet || "")}</span>
      </button>`).join("");
    $("search-results").querySelectorAll("button").forEach(button =>
      button.addEventListener("click", () => showSegment(button.dataset.id)));
  }
  await showSegment(matches[0].id);
  if (matches.length > 1) setFinderNote(`${matches.length} segments contain “${query}”:`);
}

/* The open project and segment live in the address, so a reload comes back to the same place. */
function rememberPlace() {
  const place = new URLSearchParams();
  if (currentProject) place.set("project", currentProject.id);
  if (currentProject && shownId) place.set("segment", shownId);
  else if (currentProject && showAll) place.set("view", "all");
  history.replaceState(null, "", place.toString() ? `#${place}` : location.pathname);
}

function savedPlace() {
  const place = new URLSearchParams(location.hash.slice(1));
  return { project:place.get("project"), segment:place.get("segment"), all:place.get("view") === "all" };
}

function renderSummary(segments) {
  const total = segments.length;
  const passed = segments.filter(segment => segment.qa_status === "passed").length;
  const box = $("review-summary");
  if (!box) return;
  box.classList.toggle("hidden", !total);
  box.innerHTML = total
    ? `<strong>${passed} of ${total} segments cleared.</strong> <span class="muted">${
        passed === total ? "Ready to assemble." : `${total - passed} still need a listen.`}</span>`
    : "";
}

function bindSegment(card) {
  const id = card.dataset.id;
  const after = async () => { await loadSegments(); };

  card.querySelector(".save")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    await request(`/api/segments/${id}/transcript`, jsonOptions("PATCH", {text:card.querySelector(".si").value}));
    const updated = await request(`/api/segments/${id}/script`, jsonOptions("PATCH", {text:card.querySelector(".en").value}));
    card.querySelectorAll("textarea").forEach(box => { box.dataset.initial = box.value; });
    card.querySelector(".segment-head .head-right").innerHTML = qaBadge(updated);
  }, {label:"Saving…"}));

  card.querySelector(".regen")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    // A text-only segment has no audio; the only thing to redo there is the adaptation.
    const stage = card.querySelector("audio") ? "tts" : "adaptation";
    await request(`/api/segments/${id}/regenerate`, jsonOptions("POST", {stage}));
    await waitForSegment(id);
    await after();
  }, {label:"Regenerating…"}));

  card.querySelector(".approve")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    await request(`/api/segments/${id}/approve`, {method:"POST"});
    await after();
  }, {label:"Approving…"}));

  card.querySelector(".confirm")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    await request(`/api/segments/${id}/confirm`, {method:"POST"});
    await after();
  }, {label:"Confirming…"}));

  card.querySelector(".to-original")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"original"}));
    await waitForSegment(id);
    await after();
  }, {label:"Keeping…"}));

  card.querySelector(".to-narration")?.addEventListener("click", (event) => run(event.currentTarget, async () => {
    await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"narration"}));
    await waitForSegment(id);
    await after();
  }, {label:"Voicing…"}));

  bindFacts(card);
}

/* ------------------------------------------- sentence fix and fact check agents */

// Both agents only report; nothing here gates approve, assemble or export.
let lastSegments = [];                               // the segments the cards were last drawn from
let taskTimer = null;                                // next GET /tasks
let taskPollBusy = false;
let taskJob = null;                                  // the job the task state below belongs to
let wasRunning = { sentences:false, facts:false };   // seen running (or just started) by this page
let needReload = false;                              // the sentence fix finished: redraw every card
let needFactsRefresh = false;                        // a fact check finished: redraw the facts blocks
let taskNote = "", taskNoteAt = 0;                   // "Already running" after a 409
const pendingChecks = new Map();                     // segment id -> {at, sawChecking, since}: "Check facts again" pressed

const isHttpUrl = (value) => {
  try { const url = new URL(String(value)); return url.protocol === "http:" || url.protocol === "https:" ? url : null; }
  catch (_) { return null; }
};
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const segmentFacts = (segment) => segment.facts || {};
/** Keep a just-requested check showing "Checking facts…" until the server shows it started or finished. */
function overlayPending(segments) {
  return segments.map(segment => {
    const id = String(segment.id), pending = pendingChecks.get(id);
    if (!pending) return segment;
    const facts = segmentFacts(segment);
    if (facts.state === "checking") { pending.sawChecking = true; return segment; }
    if (pending.sawChecking || facts.at !== pending.at || Date.now() - pending.since > 30000) {
      pendingChecks.delete(id);
      return segment;
    }
    return { ...segment, facts:{ ...facts, state:"checking" } };
  });
}
const anyChecking = () => lastSegments.some(segment => segmentFacts(segment).state === "checking");

function sourceLinks(sources) {
  const links = (sources || []).map(source => {
    const url = isHttpUrl(source?.url);
    if (!url) return "";                             // javascript:, data: and anything unparsable never become links
    const text = String(source.title || "").trim() || url.hostname.replace(/^www\./, "");
    return `<a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(text)}</a>`;
  }).filter(Boolean);
  return links.length ? `<span class="fact-sources">${links.join("")}</span>` : "";
}

function sentenceFixLine(segment) {
  const notes = (segment.sentence_fix?.notes || []).filter(Boolean);
  return notes.length ? `<p class="muted fix-notes">${escapeHtml(notes.join(" · "))}</p>` : "";
}

/** A wrong or unverified fact; once the user resolves it, it stays in place and turns green. */
function factItem(item, index, label) {
  const kind = item.resolved ? "resolved" : item.verdict;
  const tag = item.resolved ? `Resolved (${label.replace(/:$/, "").toLowerCase()}):` : label;
  return `
    <div class="fact fact-${kind}" data-index="${index}">
      <p class="fact-claim"><strong>${escapeHtml(tag)}</strong> ${escapeHtml(item.claim || "")}</p>
      ${item.evidence ? `<p class="fact-evidence">${escapeHtml(item.evidence)}</p>` : ""}
      ${sourceLinks(item.sources)}
      <div class="fact-actions"><button class="resolve-fact secondary" data-resolved="${item.resolved ? "false" : "true"}">${item.resolved ? "Undo" : "Resolved"}</button></div>
    </div>`;
}

function factsBlock(segment) {
  if (segment.kind !== "narration") return "";
  const facts = segmentFacts(segment);
  const checked = Boolean(facts.state);
  if (!checked && !String(segment.narration_en || "").trim()) return "";
  const items = facts.items || [];
  const byVerdict = (verdict) => items.filter(item => item.verdict === verdict);
  const correct = byVerdict("correct");
  const parts = [];
  if (facts.state === "checking") parts.push('<p class="muted facts-checking">Checking facts…</p>');
  if (facts.state === "failed") parts.push(`<p class="facts-failed">Fact check failed${facts.error ? `: ${escapeHtml(facts.error)}` : ""}</p>`);
  if (segment.facts_stale) parts.push('<p class="facts-stale">Out of date: the text changed since the check</p>');
  // The index in facts.items is what the resolve endpoint takes.
  const indexed = (verdict) => items.map((item, index) => [item, index]).filter(([item]) => item.verdict === verdict);
  indexed("wrong").forEach(([item, index]) => parts.push(factItem(item, index, "Looks wrong:")));
  indexed("unverified").forEach(([item, index]) => parts.push(factItem(item, index, "Couldn't verify:")));
  if (correct.length) {
    parts.push(`
      <details class="facts-correct">
        <summary>${plural(correct.length, "fact correct", "facts correct")}</summary>
        <ul>${correct.map(item => `<li>${escapeHtml(item.claim || "")} ${sourceLinks(item.sources)}</li>`).join("")}</ul>
      </details>`);
  }
  if (facts.state === "done" && !items.length) parts.push('<p class="muted">No checkable facts in this segment.</p>');
  const busy = facts.state === "checking" ? " disabled" : "";
  parts.push(`<div class="actions"><button class="check-facts secondary"${busy}>${checked ? "Check facts again" : "Check facts"}</button></div>`);
  return `<div class="facts">${parts.join("")}</div>`;
}

/** The line above the cards: counts across every narration segment that has been checked. */
function factSummary(segments) {
  const narration = segments.filter(segment => segment.kind === "narration");
  const checked = narration.filter(segment => segmentFacts(segment).state);
  if (!checked.length) return { text:"", tone:"" };
  const items = checked.flatMap(segment => segmentFacts(segment).items || []);
  const count = (verdict) => items.filter(item => item.verdict === verdict && !item.resolved).length;
  const wrong = count("wrong"), unverified = count("unverified"), correct = count("correct");
  const resolved = items.filter(item => item.resolved).length;
  const parts = [];
  if (wrong || unverified || resolved) {
    if (wrong) parts.push(wrong === 1 ? "1 fact looks wrong" : `${wrong} facts look wrong`);
    if (unverified) parts.push(`${unverified} couldn't verify`);
    if (resolved) parts.push(`${resolved} resolved`);
  } else if (correct) {
    parts.push(`all ${plural(correct, "fact", "facts")} correct`);
  } else if (checked.some(segment => segmentFacts(segment).state === "done")) {
    parts.push("no checkable facts found");
  }
  const checking = checked.filter(segment => segmentFacts(segment).state === "checking").length;
  const failed = checked.filter(segment => segmentFacts(segment).state === "failed").length;
  const stale = narration.filter(segment => segment.facts_stale).length;
  if (checking) parts.push(`${plural(checking, "segment", "segments")} still checking`);
  if (failed) parts.push(`${plural(failed, "segment", "segments")} failed`);
  if (stale) parts.push(`${plural(stale, "segment", "segments")} out of date`);
  const tone = wrong ? "bad" : (unverified || stale || failed) ? "unsure" : checking ? "" : "good";
  return { text:`Fact check: ${parts.join(" · ")}`, tone };
}

function renderAgentBar(segments) {
  lastSegments = segments;
  const bar = $("agent-bar");
  if (!bar) return;
  bar.classList.toggle("hidden", !segments.length);
  const { text, tone } = factSummary(segments);
  const line = $("fact-summary");
  line.textContent = text;
  line.hidden = !text;
  line.className = `fact-summary${tone ? ` ${tone}` : ""}`;
}

/** Progress for the running task(s), the fix's result, and which buttons can start a task. */
function renderTasks(tasks) {
  const sentences = tasks?.sentences || null, facts = tasks?.facts || null;
  const running = { sentences:sentences?.state === "running", facts:facts?.state === "running" };
  if (taskNote && !running.sentences && !running.facts && Date.now() - taskNoteAt > 3000) taskNote = "";
  const progress = [];
  if (taskNote) progress.push(taskNote);
  if (running.sentences) progress.push(`Fixing sentences ${sentences.done ?? 0}/${sentences.total ?? "?"}…`);
  if (sentences?.state === "failed") progress.push(`Sentence fix failed${sentences.error ? `: ${sentences.error}` : ""}`);
  if (running.facts) progress.push(`Checking facts ${facts.done ?? 0}/${facts.total ?? "?"}…`);
  if (facts?.state === "failed") progress.push(`Fact check failed${facts.error ? `: ${facts.error}` : ""}`);
  const progressLine = $("agent-progress");
  progressLine.textContent = progress.join(" · ");
  progressLine.hidden = !progress.length;

  const resultLine = $("agent-result");
  let result = "";
  if (sentences?.state === "done") {
    const changed = sentences.changed || [];
    result = changed.length ? `Moved sentence ends in segments ${changed.join(", ")}` : "No split sentences found";
  }
  resultLine.textContent = result;
  resultLine.hidden = !result;

  $("fix-sentences").disabled = running.sentences;
  $("check-facts").disabled = running.facts;
  return running;
}

function resetTasks() {
  clearTimeout(taskTimer); taskTimer = null;
  taskJob = null; wasRunning = { sentences:false, facts:false };
  needReload = false; needFactsRefresh = false; taskNote = ""; lastSegments = []; pendingChecks.clear();
  renderTasks(null);
  renderAgentBar([]);
}

function scheduleTaskPoll(ms = 2000) {
  if (!taskTimer) taskTimer = setTimeout(() => { taskTimer = null; pollTasks(); }, ms);
}

/** True when a full redraw now would yank something from the owner: a focused box or playing audio. */
function readerIsBusy() {
  const active = document.activeElement;
  if (active?.closest?.(".segment") && active.tagName === "TEXTAREA") return true;
  return Array.from(document.querySelectorAll("#segments audio")).some(audio => !audio.paused);
}

/** Redraw only the facts blocks (never a textarea or a player), then the fact summary. */
async function refreshFacts() {
  if (inFlight > 0) return false;
  const segments = await fetchSegments({ text:false });   // only the counts change here, never the texts
  if (!segments || inFlight > 0) return false;
  const cards = Array.from(document.querySelectorAll("#segments .segment"));
  const shown = segments.filter(segment => segment.id === shownId);
  const sameCards = cards.length === shown.length
    && shown.every((segment, i) => cards[i].dataset.id === String(segment.id) && cards[i].dataset.kind === segment.kind);
  if (!sameCards) return (!readerIsBusy() && await loadSegments({background:true}));
  shown.forEach((segment, i) => {
    const slot = cards[i].querySelector(".facts-slot");
    if (!slot) return;
    const html = factsBlock(segment);
    if (slot.dataset.html === html) return;          // unchanged: leave an open <details> open
    const open = slot.querySelector("details")?.open;
    slot.innerHTML = html;
    slot.dataset.html = html;
    if (open && slot.querySelector("details")) slot.querySelector("details").open = true;
    bindFacts(cards[i]);
  });
  renderAgentBar(segments);
  return true;
}

async function pollTasks() {
  if (!currentJob) return;
  if (taskPollBusy) { scheduleTaskPoll(); return; }
  taskPollBusy = true;
  clearTimeout(taskTimer); taskTimer = null;
  const job = currentJob;
  if (taskJob !== job) { taskJob = job; wasRunning = { sentences:false, facts:false }; }
  try {
    let tasks = null;
    try { tasks = await request(`/api/jobs/${job}/tasks`); } catch (_) { /* no task state: show none */ }
    if (job !== currentJob) return;
    const running = renderTasks(tasks);
    if (wasRunning.sentences && !running.sentences) needReload = true;
    if (wasRunning.facts && !running.facts) needFactsRefresh = true;
    wasRunning = running;
    try {
      if (needReload) {
        if (!readerIsBusy() && await loadSegments({background:true})) { needReload = false; needFactsRefresh = false; }
      } else if (needFactsRefresh || anyChecking()) {
        if (await refreshFacts()) needFactsRefresh = false;
      }
    } catch (_) { /* a failed refresh is retried on the next tick */ }
    if (running.sentences || running.facts || needReload || needFactsRefresh || anyChecking() || taskNote) scheduleTaskPoll();
  } finally {
    taskPollBusy = false;
  }
}

/** POST a task; a 409 means it is already going, which is fine: we just follow it. */
async function startTask(button, kind, url, label) {
  if (!currentJob) return;
  let started = false;
  await run(button, async () => {
    try {
      await request(url, {method:"POST"});
    } catch (error) {
      if (error.status !== 409) throw error;
      taskNote = "Already running"; taskNoteAt = Date.now();
    }
    started = true;
  }, {label});
  if (!started) return;
  taskJob = currentJob;
  wasRunning[kind] = true;                           // so a task that ends before the first poll still reloads
  await pollTasks();
}

function bindFacts(card) {
  card.querySelectorAll(".resolve-fact").forEach(button => button.addEventListener("click", async (event) => {
    const id = card.dataset.id;
    const target = event.currentTarget;
    const index = Number(target.closest(".fact").dataset.index);
    const segment = lastSegments.find(item => String(item.id) === id);
    const claim = segmentFacts(segment || {}).items?.[index]?.claim;
    if (claim === undefined) return;
    let updated = null;
    await run(target, async () => {
      updated = await request(`/api/segments/${id}/facts/${index}/resolved`,
        jsonOptions("PUT", { claim, resolved:target.dataset.resolved === "true" }));
    }, {label:"Saving…"});
    if (!updated) { needFactsRefresh = true; scheduleTaskPoll(); return; }   // e.g. checked again meanwhile: redraw
    lastSegments = lastSegments.map(item => String(item.id) === id ? updated : item);
    const slot = card.querySelector(".facts-slot");
    if (slot && card.isConnected) {
      const open = slot.querySelector("details")?.open;
      const html = factsBlock(updated);
      slot.innerHTML = html;
      slot.dataset.html = html;
      if (open && slot.querySelector("details")) slot.querySelector("details").open = true;
      bindFacts(card);
    }
    renderAgentBar(lastSegments);
  }));
  card.querySelector(".check-facts")?.addEventListener("click", async (event) => {
    const id = card.dataset.id;
    let started = false;
    await run(event.currentTarget, async () => {
      try { await request(`/api/segments/${id}/check-facts`, {method:"POST"}); }
      catch (error) { if (error.status !== 409) throw error; }
      started = true;
    }, {label:"Starting…"});
    if (!started) return;
    // Show "Checking facts…" now; the poll redraws the block when the answer is in.
    const previous = lastSegments.find(item => String(item.id) === id);
    pendingChecks.set(id, { at:segmentFacts(previous || {}).at, sawChecking:false, since:Date.now() });
    lastSegments = lastSegments.map(segment => String(segment.id) === id
      ? { ...segment, facts:{ ...segmentFacts(segment), state:"checking" } } : segment);
    const segment = lastSegments.find(item => String(item.id) === id);
    const slot = card.querySelector(".facts-slot");
    if (segment && slot && card.isConnected) {
      slot.innerHTML = factsBlock(segment);
      bindFacts(card);
    }
    renderAgentBar(lastSegments);
    needFactsRefresh = true;
    scheduleTaskPoll();
  });
}

/* ------------------------------------------------------------------- exports */

async function loadExports() {
  if (!currentProject) return;
  const exports = await request(`/api/projects/${currentProject.id}/export`);
  $("exports").innerHTML = Object.entries(exports).map(([name,url]) => `<a href="${url}" download>${escapeHtml(name)}</a>`).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

/* -------------------------------------------------------------- page actions */

$("create").addEventListener("click", (event) => run(event.currentTarget, async () => {
  const file = $("audio").files[0];
  if (!file) throw new Error("Choose an audio file first.");
  $("create-status").textContent = "Creating project…";
  const persona = $("persona").value.trim();
  const project = await request("/api/projects", jsonOptions("POST", {
    title:$("title").value,
    topic:$("topic").value,
    glossary:parseGlossary($("glossary").value),
    narrator_profile: persona ? {persona} : {},
  }));
  const form = new FormData(); form.append("file", file);
  $("create-status").textContent = "Uploading and validating audio…";
  await request(`/api/projects/${project.id}/upload`, {method:"POST", body:form});
  $("create-status").textContent = "Ready.";
  await loadProjects();
  await openProject(project.id);
}, {label:"Working…"}));

$("process").addEventListener("click", (event) => run(event.currentTarget, async () => {
  const result = await request(`/api/projects/${currentProject.id}/process`, jsonOptions("POST", {voice:$("voice").value, human_review_gate:true, voice_stage:!$("text-only").checked}));
  currentJob = result.job_id;
  await refreshJob();
}, {label:"Starting…"}));

$("export-parts").addEventListener("click", (event) => run(event.currentTarget, async () => {
  const files = await request(`/api/projects/${currentProject.id}/export/parts`, {method:"POST"});
  $("exports").innerHTML = Object.entries(files).map(([name,url]) => `<a href="${url}" download>${escapeHtml(name)}</a>`).join("");
}, {label:"Exporting…"}));

$("fix-sentences").addEventListener("click", (event) =>
  startTask(event.currentTarget, "sentences", `/api/jobs/${currentJob}/fix-sentences`, "Starting…"));

$("check-facts").addEventListener("click", (event) =>
  startTask(event.currentTarget, "facts", `/api/jobs/${currentJob}/check-facts`, "Starting…"));

$("assemble").addEventListener("click", (event) => run(event.currentTarget, async () => {
  await request(`/api/projects/${currentProject.id}/assemble`, {method:"POST"});
  await loadExports();
  await loadProjects();
}, {label:"Assembling…"}));

$("segment-finder").addEventListener("submit", (event) => {
  event.preventDefault();
  findSegment($("segment-search").value).catch(error => setFinderNote(error.message, true));
});
$("segment-prev").addEventListener("click", () => stepSegment(-1));
$("segment-next").addEventListener("click", () => stepSegment(1));
$("segment-all").addEventListener("click", () => showAllSegments().catch(error => setFinderNote(error.message, true)));

loadProjects()
  .then(async () => {
    const place = savedPlace();
    if (!place.project) return;
    try { await openProject(place.project, { segment:place.segment, all:place.all }); }
    catch (_) { currentProject = null; shownId = null; showAll = false; rememberPlace(); }   // a project that is gone: start fresh
  })
  .catch(error => $("projects").textContent = error.message);
