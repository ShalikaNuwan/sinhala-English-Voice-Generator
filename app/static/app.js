let currentProject = null;
let currentJob = null;
let pollTimer = null;

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

function parseGlossary(text) {
  const result = {};
  text.split("\n").forEach(line => {
    const [left, ...right] = line.split("=");
    if (left?.trim() && right.join("=").trim()) result[left.trim()] = right.join("=").trim();
  });
  return result;
}

async function loadProjects() {
  const projects = await request("/api/projects");
  $("projects").innerHTML = projects.length ? projects.map(project => `
    <div class="project-row" data-id="${project.id}">
      <strong>${escapeHtml(project.title)}</strong><span class="pill">${project.status}</span>
    </div>`).join("") : '<p class="muted">No projects yet.</p>';
  document.querySelectorAll(".project-row").forEach(row => row.onclick = () => openProject(row.dataset.id));
}

async function openProject(id) {
  currentProject = await request(`/api/projects/${id}`);
  $("workspace").classList.remove("hidden");
  $("project-title").textContent = currentProject.title;
  $("project-status").textContent = currentProject.status;
  renderSpeakingProfile(currentProject.speaking_profile);
  const latest = currentProject.jobs?.[0];
  if (latest) { currentJob = latest.id; await refreshJob(); }
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
  if (["queued","running"].includes(job.status)) {
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshJob, 1800);
  } else {
    await loadSegments(); await loadProjects();
  }
}

function segmentCard(segment) {
  const qa = `<span class="${segment.qa_status === "passed" ? "qa-pass" : "qa-review"}">${segment.qa_status}</span>`;
  const issues = `<span class="muted">${escapeHtml((segment.qa?.issues || []).join(" · "))}</span>`;
  const player = segment.tts_audio_url ? `<audio controls src="${segment.tts_audio_url}"></audio>` : "";
  if (segment.kind === "original") {
    return `
    <article class="segment" data-id="${segment.id}" data-kind="original">
      <div class="segment-head"><strong>Segment ${segment.segment_index}</strong>
        <span class="pill">Recorded audio, kept as is</span>${qa}</div>
      <p class="muted">${escapeHtml(segment.transcript_si || "(no speech recognised in this recording)")}</p>
      ${player}
      <div class="actions">
        ${segment.tts_audio_url ? '<button class="confirm">Confirm recording</button>' : '<button class="regen secondary">Cut the recording again</button>'}
        <button class="to-narration secondary">Treat as narration</button>
        ${issues}
      </div>
    </article>`;
  }
  return `
    <article class="segment" data-id="${segment.id}" data-kind="narration">
      <div class="segment-head"><strong>Segment ${segment.segment_index}</strong>${qa}</div>
      <div class="segment-grid">
        <label>Sinhala transcript<textarea class="si">${escapeHtml(segment.transcript_si || "")}</textarea></label>
        <label>English narration<textarea class="en">${escapeHtml(segment.narration_en || "")}</textarea></label>
      </div>
      ${player}
      <div class="actions">
        <button class="save secondary">Save edits</button>
        <button class="regen">Regenerate this segment</button>
        <button class="to-original secondary">Keep as recorded</button>
        ${issues}
      </div>
    </article>`;
}

async function loadSegments() {
  if (!currentJob) return;
  const segments = await request(`/api/jobs/${currentJob}/segments`);
  $("segments").innerHTML = segments.map(segmentCard).join("");
  document.querySelectorAll(".segment").forEach(card => {
    const id = card.dataset.id;
    const later = (ms) => setTimeout(loadSegments, ms);
    card.querySelector(".save")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/transcript`, jsonOptions("PATCH", {text:card.querySelector(".si").value}));
      await request(`/api/segments/${id}/script`, jsonOptions("PATCH", {text:card.querySelector(".en").value}));
      card.querySelector(".qa-review, .qa-pass").textContent = "pending";
    });
    card.querySelector(".regen")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/regenerate`, jsonOptions("POST", {stage:"tts"}));
      card.querySelector(".regen").textContent = "Regenerating…"; later(2500);
    });
    const guarded = (action) => async () => { try { await action(); } catch (error) { alert(error.message); } };
    card.querySelector(".to-original")?.addEventListener("click", guarded(async () => {
      await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"original"})); later(1500);
    }));
    card.querySelector(".to-narration")?.addEventListener("click", guarded(async () => {
      await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"narration"}));
      card.querySelector(".to-narration").textContent = "Voicing…"; later(4000);
    }));
    card.querySelector(".confirm")?.addEventListener("click", guarded(async () => {
      await request(`/api/segments/${id}/confirm`, {method:"POST"}); await loadSegments();
    }));
  });
}

async function loadExports() {
  if (!currentProject) return;
  const exports = await request(`/api/projects/${currentProject.id}/export`);
  $("exports").innerHTML = Object.entries(exports).map(([name,url]) => `<a href="${url}" download>${name}</a>`).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

$("create").onclick = async () => {
  const file = $("audio").files[0];
  if (!file) return alert("Choose an audio file first.");
  $("create").disabled = true; $("create-status").textContent = "Creating project…";
  try {
    const project = await request("/api/projects", jsonOptions("POST", {title:$("title").value, topic:$("topic").value, glossary:parseGlossary($("glossary").value)}));
    const form = new FormData(); form.append("file", file);
    $("create-status").textContent = "Uploading and validating audio…";
    await request(`/api/projects/${project.id}/upload`, {method:"POST", body:form});
    $("create-status").textContent = "Ready."; await loadProjects(); await openProject(project.id);
  } catch (error) { alert(error.message); $("create-status").textContent = error.message; }
  finally { $("create").disabled = false; }
};

$("process").onclick = async () => {
  try {
    const result = await request(`/api/projects/${currentProject.id}/process`, jsonOptions("POST", {voice:$("voice").value, human_review_gate:true}));
    currentJob = result.job_id; refreshJob();
  } catch (error) { alert(error.message); }
};

$("assemble").onclick = async () => {
  try {
    await request(`/api/projects/${currentProject.id}/assemble`, {method:"POST"}); await loadExports(); await loadProjects();
  } catch (error) { alert(error.message); }
};

loadProjects().catch(error => $("projects").textContent = error.message);

