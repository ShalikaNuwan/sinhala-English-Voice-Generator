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
    throw new Error(message);
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

async function openProject(id) {
  currentProject = await request(`/api/projects/${id}`);
  $("workspace").classList.remove("hidden");
  $("project-title").textContent = currentProject.title;
  $("project-status").textContent = currentProject.status;
  showError($("workspace"), "");
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
  if (running) {
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshJob, 1800);
  } else {
    await loadSegments({background:true}); await loadProjects();
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
      <p class="segment-error error" hidden></p>
    </article>`;
}

/** Poll a segment until the background worker has finished with it. */
async function waitForSegment(id) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await wait(1500);
    const segments = await request(`/api/jobs/${currentJob}/segments`);
    const segment = segments.find(item => item.id === id);
    if (!segment || segment.status !== "regenerating") return;
  }
}

async function loadSegments({ background = false } = {}) {
  if (!currentJob) return;
  // Never pull the page out from under someone who is mid-action or mid-edit.
  if (background && (inFlight > 0 || hasUnsavedEdits())) return;
  const segments = await request(`/api/jobs/${currentJob}/segments`);
  $("segments").innerHTML = segments.map(segmentCard).join("");
  renderSummary(segments);
  document.querySelectorAll(".segment").forEach(card => bindSegment(card));
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
    await request(`/api/segments/${id}/regenerate`, jsonOptions("POST", {stage:"tts"}));
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
  const result = await request(`/api/projects/${currentProject.id}/process`, jsonOptions("POST", {voice:$("voice").value, human_review_gate:true}));
  currentJob = result.job_id;
  await refreshJob();
}, {label:"Starting…"}));

$("assemble").addEventListener("click", (event) => run(event.currentTarget, async () => {
  await request(`/api/projects/${currentProject.id}/assemble`, {method:"POST"});
  await loadExports();
  await loadProjects();
}, {label:"Assembling…"}));

loadProjects().catch(error => $("projects").textContent = error.message);
