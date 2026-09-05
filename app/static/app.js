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
  $("progress-text").textContent = `${job.stage} · ${job.progress}%${job.error ? ` · ${job.error}` : ""}`;
  if (["queued","running"].includes(job.status)) {
    clearTimeout(pollTimer); pollTimer = setTimeout(refreshJob, 1800);
  } else {
    await loadSegments(); await loadProjects();
  }
}

async function loadSegments() {
  if (!currentJob) return;
  const segments = await request(`/api/jobs/${currentJob}/segments`);
  $("segments").innerHTML = segments.map(segment => `
    <article class="segment" data-id="${segment.id}">
      <div class="segment-head"><strong>Segment ${segment.segment_index}</strong>
        <span class="${segment.qa_status === "passed" ? "qa-pass" : "qa-review"}">${segment.qa_status}</span></div>
      <div class="segment-grid">
        <label>Sinhala transcript<textarea class="si">${escapeHtml(segment.transcript_si || "")}</textarea></label>
        <label>English narration<textarea class="en">${escapeHtml(segment.narration_en || "")}</textarea></label>
      </div>
      ${segment.tts_audio_url ? `<audio controls src="${segment.tts_audio_url}"></audio>` : ""}
      <div class="actions">
        <button class="save secondary">Save edits</button>
        <button class="regen">Regenerate this segment</button>
        <span class="muted">${escapeHtml((segment.qa?.issues || []).join(" · "))}</span>
      </div>
    </article>`).join("");
  document.querySelectorAll(".segment").forEach(card => {
    card.querySelector(".save").onclick = async () => {
      const id = card.dataset.id;
      await request(`/api/segments/${id}/transcript`, jsonOptions("PATCH", {text:card.querySelector(".si").value}));
      await request(`/api/segments/${id}/script`, jsonOptions("PATCH", {text:card.querySelector(".en").value}));
      card.querySelector(".qa-review").textContent = "pending";
    };
    card.querySelector(".regen").onclick = async () => {
      await request(`/api/segments/${card.dataset.id}/regenerate`, jsonOptions("POST", {stage:"tts"}));
      card.querySelector(".regen").textContent = "Regenerating…";
      setTimeout(loadSegments, 2500);
    };
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

