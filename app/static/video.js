/* Video tab: review a film's storyboard shot by shot, write a review an AI turns into a new version of the
   shot, approve shots, rebuild the review cut, finalize.
   Everything lives inside this closure so nothing collides with app.js (the voice studio). */
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const enc = encodeURIComponent;
  const api = async (url, options = {}) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = response.statusText;
      try {
        const detail = (await response.json()).detail;
        if (Array.isArray(detail)) message = detail.map((d) => d.msg || String(d)).join("; ");
        else if (detail) message = String(detail);
      } catch (_) {}
      throw new Error(message || `Request failed (${response.status})`);
    }
    return response.json();
  };
  const send = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  const LABELS = { actual: "Actual", "stand-in": "Stand-in", illustration: "Illustration", graphic: "Graphic", missing: "Missing" };
  const FILTERS = [["all", "All"], ["not-actual", "Not actual only"], ["illustration", "Illustration"], ["stand-in", "Stand-in"],
                   ["graphic", "Graphic"], ["not-approved", "Not approved"], ["has-review", "Has a review"]];
  const POLL_MS = 3000;
  const WORKING = new Set(["queued", "running"]);

  let film = null;            // the open film's name, or null on the films list
  let view = null;            // GET /api/video/films/{film}
  let filter = "all";
  let jobTimer = null;        // asks for the rebuild/render job
  let regenTimer = null;      // asks for the whole film view while shots are regenerating
  let refreshSeq = 0;         // only the newest film view is drawn, whatever order the answers come in
  let staleRows = false;      // a changed row was left alone because the owner is typing in it
  let source = null;          // what the player plays: "final" or "review"
  let playerKey = "";         // the src currently loaded, so a refresh never restarts playback
  let finalStamp = 0;         // bumped when a render finishes, so a re-rendered final film reloads
  const drafts = new Map();   // shot n -> review text typed but not sent yet (survives row redraws)
  const sigs = new WeakMap(); // row element -> the data it was drawn from

  const base = (name = film) => `/api/video/films/${enc(name)}`;
  const fileUrl = (path) => `${base()}/file/${String(path).split("/").map(enc).join("/")}`;
  const thumbUrl = (id) => `${base()}/thumb/${enc(id)}.jpg`;
  const clock = (t) => { t = Number(t) || 0; return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`; };
  const label = (category) => esc(LABELS[category] || category || "Unknown");
  const safeLink = (url) => /^https?:\/\//i.test(url || "") ? url : "";
  const allShots = () => view.scenes.flatMap((sc) => sc.shots);
  const findShot = (n) => allShots().find((s) => String(s.n) === String(n));
  const approved = (shot) => shot.mark?.status === "ok";
  const reviewText = (shot) => String(shot.regen?.review || "").trim();
  const working = (shot) => WORKING.has(shot.regen?.state);
  const unreviewed = () => allShots().filter((s) => !s.mark && !reviewText(s));   // what approve-unreviewed approves
  const draftOf = (shot) => drafts.has(String(shot.n)) ? drafts.get(String(shot.n)) : String(shot.regen?.review || "");

  function showError(message) { $("film-error").textContent = message || ""; $("film-error").hidden = !message; }

  function stopPolling() {
    clearTimeout(jobTimer); jobTimer = null;
    clearTimeout(regenTimer); regenTimer = null;
  }

  function unloadPlayer() {
    const player = $("player");
    player.pause();
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    playerKey = "";
  }

  /* ------------------------------------------------------------- tabs */
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    const video = tab.dataset.tab === "video";
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("voice-tab").classList.toggle("hidden", video);
    $("video-tab").classList.toggle("hidden", !video);
    if (!video) { stopPolling(); $("player").pause(); $("scenes").querySelectorAll("video").forEach((v) => v.pause()); return; }
    if (film) refresh().catch((error) => showError(error.message));    // picks the polling back up
    else loadFilms();
  }));

  /* ------------------------------------------------------------ films */
  async function loadFilms() {
    try {
      const { films } = await api("/api/video/films");
      $("films").innerHTML = films.length ? films.map((f) => `
        <div class="film-row" data-open="${esc(f.film)}">
          <strong>${esc(f.film)}</strong>
          <span class="muted">${esc(f.shots)} shots · ${esc(f.minutes)} min · ${esc(f.reviewed)} of ${esc(f.shots)} reviewed${f.not_ok ? `, ${esc(f.not_ok)} not OK` : ""}</span>
          <span class="pill">${f.final ? "final film ready" : f.animatic ? "review cut ready" : "no review cut yet"}</span>
          <button data-open="${esc(f.film)}">Open</button>
        </div>`).join("") : `<p class="muted">No films yet. Prepare one with <code>documentary.py prepare</code>.</p>`;
    } catch (error) {
      $("films").innerHTML = `<p class="error">${esc(error.message)}</p>`;
    }
  }

  $("films").addEventListener("click", (e) => {
    const row = e.target.closest("[data-open]");
    if (row) openFilm(row.dataset.open);
  });

  $("film-close").addEventListener("click", () => {
    stopPolling();
    unloadPlayer();
    film = null; view = null;
    $("scenes").innerHTML = "";                    // also stops any regenerated shot that is playing
    $("film-card").classList.add("hidden");
    $("films-card").classList.remove("hidden");
    loadFilms();
  });

  async function openFilm(name) {
    stopPolling();
    unloadPlayer();
    film = name; view = null; filter = "all"; source = null; finalStamp = 0; staleRows = false;
    drafts.clear();
    showError("");
    $("scenes").innerHTML = `<p class="muted">Loading…</p>`;
    $("films-card").classList.add("hidden");
    $("film-card").classList.remove("hidden");
    try { await refresh(); } catch (error) { showError(error.message); }
  }

  /* ----------------------------------------------------------- film page */
  async function refresh() {
    const name = film;
    if (!name) return;
    const mine = ++refreshSeq;
    const next = await api(base(name));
    if (film !== name || mine !== refreshSeq) return;   // the owner moved on, or a newer answer is coming
    view = next;
    $("film-title").textContent = view.film;
    renderPlayer();
    renderSummary();
    renderFinal();
    renderCommand();
    renderFilters();
    renderScenes();
    showJob(view.job);
    pollRegen();
  }

  function renderPlayer() {
    if (source === "final" && !view.final) source = null;
    if (source === "review" && !view.animatic) source = null;
    if (!source) source = view.animatic ? "review" : view.final ? "final" : null;   // the shots are timed to the review cut
    const both = Boolean(view.final && view.animatic);
    $("player-source").hidden = !both;
    $("player-source").innerHTML = both ? `<span class="muted">Play:</span>`
      + `<button class="${source === "review" ? "" : "secondary"}" data-source="review">Review cut</button>`
      + `<button class="${source === "final" ? "" : "secondary"}" data-source="final">Final film</button>` : "";
    const player = $("player");
    if (!source) { if (playerKey) unloadPlayer(); else player.hidden = true; return; }
    const src = source === "final" ? `${fileUrl(view.final)}?v=f${finalStamp}` : `${fileUrl("animatic.mp4")}?v=${view.animatic_version}`;
    if (src !== playerKey) {                       // only when the film, the file or its version changed
      const at = playerKey ? player.currentTime || 0 : 0;
      player.src = src;
      if (at) player.currentTime = at;             // a rebuilt cut or the other source: stay at the same moment
      playerKey = src;
    }
    player.hidden = false;
  }

  $("player-source").addEventListener("click", (e) => {
    const b = e.target.closest("[data-source]");
    if (!b || b.dataset.source === source) return;
    source = b.dataset.source;
    renderPlayer();
  });

  function renderSummary() {
    const s = view.summary;
    const shots = allShots();
    const ok = s.approved ?? shots.filter(approved).length;
    const busy = s.regenerating ?? shots.filter(working).length;
    $("film-summary").innerHTML = `<strong>Approved ${esc(ok)} of ${esc(s.shots)}</strong>`
      + esc(` · ${s.shots} shots: ${s.actual} actual · ${s["stand-in"]} stand-in · ${s.illustration} illustration · ${s.graphic} graphic`
      + (s.missing ? ` · ${s.missing} missing` : "") + (busy ? ` · ${busy} regenerating` : "")
      + (view.animatic ? "" : " — no review cut yet: rebuild it to see the frames"));
    const left = unreviewed().length;
    $("approve-unreviewed").disabled = !left;
    $("approve-unreviewed").textContent = left ? `Approve all unreviewed (${left})` : "Approve all unreviewed";
  }

  function renderFinal() {
    const box = $("film-final");
    box.hidden = !view.final;
    box.innerHTML = view.final
      ? `Final film: <a href="${esc(fileUrl(view.final))}" target="_blank" rel="noopener">${esc(view.final)}</a>` : "";
  }

  function renderCommand() {
    $("render-command").textContent = view ? `${view.render_command}${$("render-sound").checked ? " --sound" : ""}` : "";
  }
  $("render-sound").addEventListener("change", renderCommand);

  function renderFilters() {
    $("filters").innerHTML = FILTERS.map(([key, text]) =>
      `<button class="${key === filter ? "" : "secondary"}" data-filter="${key}">${text}</button>`).join("");
  }

  $("filters").addEventListener("click", (e) => {
    const b = e.target.closest("[data-filter]");
    if (!b || !view) return;
    filter = b.dataset.filter;
    renderFilters();
    renderScenes();                                // local: no need to ask the server again
  });

  /* ------------------------------------------------------------ shots */
  function visible(shot) {
    if (filter === "all") return true;
    if (filter === "not-actual") return shot.category !== "actual";
    if (filter === "not-approved") return !approved(shot);
    if (filter === "has-review") return Boolean(reviewText(shot));
    return shot.category === filter;
  }

  function sourceImage(shot) {
    if (shot.picture?.media === "clip") return thumbUrl(shot.picture.id);
    if (shot.picture?.file) return fileUrl(shot.picture.file);
    if (shot.asset && /\.(png|jpe?g|webp)$/i.test(shot.asset)) return fileUrl(shot.asset);
    return "";
  }

  /** The regenerated shot plays in place of the frame until the review cut is rebuilt with it. */
  function frameHtml(shot) {
    if (shot.frame_current) {
      return `<img loading="lazy" src="${esc(`${base()}/frame/${enc(shot.n)}.jpg?v=${view.animatic_version}`)}" alt="Frame in the review cut">`;
    }
    // A clip made before the shot's current picture was chosen (Choose another… afterwards) is out of date.
    if (shot.regen?.state === "done" && shot.regen.clip && !(shot.owner && shot.owner.at > shot.regen.at)) {
      return `<video class="regen-clip" controls preload="metadata" src="${esc(fileUrl(shot.regen.clip))}"></video>`;
    }
    return `<div class="placeholder">${view.animatic ? "Changed: rebuild the review cut to see this frame" : "No review cut yet"}</div>`;
  }

  function statusHtml(shot) {
    const r = shot.regen;
    if (!r) return "";
    if (r.state === "queued") return `<div class="regen-status working">${esc(r.step || "Queued")}</div>`;
    if (r.state === "running") return `<div class="regen-status working">${esc(r.step || "Working…")}</div>`;
    if (r.state === "failed") return `<div class="regen-status regen-failed">Could not regenerate: ${esc(r.error || "unknown error")}</div>`;
    if (r.state !== "done") return "";
    const verdict = !r.check ? ""
      : r.check.satisfied ? `<div class="regen-verdict ok">Checker: the review is satisfied.</div>`
      : `<div class="regen-verdict warn">Checker: not satisfied${r.check.problem ? ` — ${esc(r.check.problem)}` : ""}</div>`;
    return `<div class="regen-status">${r.explanation ? esc(r.explanation) : "Regenerated."}</div>${verdict}`;
  }

  function shotHtml(shot) {
    const src = sourceImage(shot);
    const p = shot.picture;
    const link = safeLink(p?.source_url);
    const details = [
      p ? `<div><strong>Picture:</strong> ${esc(p.caption)} <span class="muted">(${esc(p.found || p.source || "")}${p.licence ? `, ${esc(p.licence)}` : ""})</span>${link ? ` <a href="${esc(link)}" target="_blank" rel="noopener">source</a>` : ""}</div>` : "",
      shot.illustration ? `<div><strong>Drawn:</strong> ${esc(shot.illustration)}</div>` : "",
      shot.key_fact ? `<div><strong>Label:</strong> ${esc(String(shot.key_fact).toUpperCase())}</div>` : "",
      shot.reason ? `<div class="muted"><strong>Why:</strong> ${esc(shot.reason)}</div>` : "",
      shot.check?.problem ? `<div class="warn"><strong>Check:</strong> ${esc(shot.check.problem)}</div>` : "",
      shot.owner ? `<div class="muted">Your choice</div>` : "",
      shot.quote ? `<div class="muted">A quote card is shown over this shot</div>` : "",
      shot.mark?.status === "not_ok" && shot.mark.note ? `<div class="muted"><strong>Earlier note:</strong> ${esc(shot.mark.note)}</div>` : "",
    ].join("");
    const ok = approved(shot);
    const text = draftOf(shot);
    const busy = working(shot);
    return `
      <div class="shot ${ok ? "shot-ok" : ""}" data-n="${esc(shot.n)}">
        <div class="shot-media">${frameHtml(shot)}${src ? `<img loading="lazy" class="source" src="${esc(src)}" alt="Source picture">` : ""}</div>
        <div class="shot-body">
          <div class="shot-head">
            <button class="secondary play" data-play="${esc(shot.start)}">▶ ${clock(shot.start)}–${clock(shot.end)}</button>
            <span class="badge badge-${esc(shot.category)}">${label(shot.category)}</span>
            <span class="muted">${esc(shot.kind)}</span>
          </div>
          <p class="narration">“${esc(shot.narration)}”</p>
          ${details}
          <div class="review-box">
            <textarea class="review" rows="2" placeholder="Review: what should change? (e.g. show the house, not a street)">${esc(text)}</textarea>
            <div class="actions">
              <button data-regenerate ${!text.trim() || busy ? "disabled" : ""}>${busy ? "Regenerating…" : "Regenerate"}</button>
              <button class="${ok ? "approve-button" : "secondary"}" data-approve>${ok ? "✓ Approved" : "Approve"}</button>
              <button class="secondary" data-alternatives>Choose another…</button>
            </div>
            ${statusHtml(shot)}
          </div>
          <div class="alternatives hidden"></div>
        </div>
      </div>`;
  }

  const rowSig = (shot) => JSON.stringify([shot, view.animatic, view.animatic_version]);

  /** The owner is typing in this row (a text field has the focus): leave it alone. A focused button is not typing. */
  function typingIn(row) {
    const a = document.activeElement;
    return Boolean(a && a !== document.body && row.contains(a) && a.tagName !== "BUTTON");
  }

  function buildRow(shot, old) {
    const t = document.createElement("template");
    t.innerHTML = shotHtml(shot).trim();
    const row = t.content.firstElementChild;
    if (old) {
      // Same regenerated clip: keep the element that may be playing instead of a fresh one that starts over.
      const was = old.querySelector("video.regen-clip");
      const now = row.querySelector("video.regen-clip");
      if (was && now && was.getAttribute("src") === now.getAttribute("src")) now.replaceWith(was);
      // An open "Choose another…" panel stays open, unless a choice from it is being applied.
      const panel = old.querySelector(".alternatives");
      if (panel && !panel.classList.contains("hidden") && !panel.querySelector("button:disabled")) {
        row.querySelector(".alternatives").replaceWith(panel);
      }
    }
    sigs.set(row, rowSig(shot));
    return row;
  }

  /** Make parent's children exactly `wanted`, moving only nodes that are out of place (a moved node loses focus). */
  function syncChildren(parent, wanted) {
    const keep = new Set(wanted);
    [...parent.childNodes].forEach((c) => { if (!keep.has(c)) c.remove(); });
    wanted.forEach((node, i) => {
      const at = parent.childNodes[i];
      if (at !== node) parent.insertBefore(node, at || null);
    });
  }

  /** Draw the shot list, redrawing only rows whose data changed; never one the owner is typing in. */
  function renderScenes() {
    const box = $("scenes");
    const rows = new Map([...box.querySelectorAll(".shot")].map((r) => [r.dataset.n, r]));
    const sceneEls = new Map([...box.querySelectorAll(".scene")].map((el) => [el.dataset.scene, el]));
    const wanted = [];
    staleRows = false;
    for (const sc of view.scenes) {
      const shots = sc.shots.filter(visible);
      if (!shots.length) continue;
      let el = sceneEls.get(String(sc.id));
      if (!el) {
        el = document.createElement("div");
        el.className = "scene";
        el.dataset.scene = String(sc.id);
        el.appendChild(document.createElement("h3"));
      }
      const head = el.querySelector("h3");
      const meta = [sc.story_time, sc.look, clock(sc.start || 0)].filter(Boolean).map(esc).join(" · ");
      const headHtml = `${esc(sc.title)} <span class="muted">${meta}</span>`;
      if (sigs.get(head) !== headHtml) { head.innerHTML = headHtml; sigs.set(head, headHtml); }
      const shotEls = shots.map((shot) => {
        const old = rows.get(String(shot.n));
        if (old && sigs.get(old) === rowSig(shot)) return old;
        if (old && typingIn(old)) { staleRows = true; return old; }
        return buildRow(shot, old);
      });
      syncChildren(el, [head, ...shotEls]);
      wanted.push(el);
    }
    if (!wanted.length) {
      const none = box.querySelector(".no-shots") || Object.assign(document.createElement("p"), { className: "muted no-shots" });
      none.textContent = "No shots match this filter.";
      wanted.push(none);
    }
    syncChildren(box, wanted);
  }

  /** While shots regenerate (or a changed row waits for the owner to stop typing), ask for the film again. */
  function pollRegen() {
    clearTimeout(regenTimer); regenTimer = null;
    if (!view) return;
    const busy = view.summary.regenerating ?? allShots().filter(working).length;
    if (!busy && !staleRows) return;
    const name = film;
    regenTimer = setTimeout(async () => {
      regenTimer = null;
      if (film !== name || $("video-tab").classList.contains("hidden")) return;
      try { await refresh(); }
      catch (error) { showError(error.message); if (film === name) pollRegen(); }   // a hiccup; keep watching
    }, POLL_MS);
  }

  async function saveMark(n, status) {
    const { mark } = await api(`${base()}/shots/${enc(n)}/mark`, send("PUT", { status }));
    const shot = findShot(n);
    if (shot) shot.mark = mark;
    const shots = allShots();
    view.summary.approved = shots.filter(approved).length;
    view.summary.reviewed = shots.filter((s) => s.mark).length;
    view.summary.not_ok = shots.filter((s) => s.mark?.status === "not_ok").length;
    renderSummary();
    return shot;
  }

  $("scenes").addEventListener("click", async (e) => {
    const t = e.target.closest("button");
    const row = t?.closest(".shot");
    if (!t || !row || !view) return;
    const n = row.dataset.n;
    try {
      showError("");
      if (t.dataset.play !== undefined) {
        const player = $("player");
        if (!playerKey) { showError("No review cut yet: rebuild it to play the film."); return; }
        player.currentTime = Number(t.dataset.play);
        player.scrollIntoView({ behavior: "smooth", block: "nearest" });
        player.play().catch(() => {});
        return;
      }
      if (t.dataset.approve !== undefined) {
        const status = approved(findShot(n) || {}) ? null : "ok";
        t.disabled = true;
        await saveMark(n, status);
        renderScenes();                            // only this row changed
        return;
      }
      if (t.dataset.regenerate !== undefined) {
        const text = row.querySelector("textarea.review").value.trim();
        if (!text) return;
        if (row.contains(document.activeElement)) document.activeElement.blur();   // sent: let the row show progress
        t.disabled = true;
        t.textContent = "Sending…";
        const regen = await api(`${base()}/shots/${enc(n)}/regenerate`, send("POST", { review: text }));
        const shot = findShot(n);
        if (shot && regen && typeof regen === "object" && regen.state) shot.regen = regen;
        drafts.delete(String(n));                  // the server has the text now
        await refresh();                          // summary.regenerating > 0 starts the polling
        return;
      }
      if (t.dataset.alternatives !== undefined) { await toggleAlternatives(row, n); return; }
      if (t.dataset.choose) {
        const box = row.querySelector(".alternatives");
        box.querySelectorAll("button").forEach((b) => { b.disabled = true; });
        t.textContent = t.dataset.choose === "illustration" ? "Drawing…" : "Using…";
        await api(`${base()}/shots/${enc(n)}/choose`, send("POST", { choice: t.dataset.choose }));
        await refresh();                           // the categories and summary change with the picture
      }
    } catch (error) {
      showError(error.message);
      sigs.delete(row);                            // redraw it: its buttons were disabled for the request
      await refresh().catch(() => {});
    }
  });

  // Typed review text is kept per shot until it is sent, so a redraw never loses it.
  $("scenes").addEventListener("input", (e) => {
    if (!e.target.classList.contains("review")) return;
    const row = e.target.closest(".shot");
    const shot = findShot(row.dataset.n);
    drafts.set(String(row.dataset.n), e.target.value);
    const b = row.querySelector("[data-regenerate]");
    if (b) b.disabled = !e.target.value.trim() || Boolean(shot && working(shot));
  });

  async function toggleAlternatives(row, n) {
    const box = row.querySelector(".alternatives");
    if (!box.classList.contains("hidden")) { box.classList.add("hidden"); return; }
    box.classList.remove("hidden");
    box.innerHTML = `<p class="muted">Loading…</p>`;
    const alt = await api(`${base()}/shots/${enc(n)}/alternatives`);
    const options = alt.options.map((o) => `
      <div class="option">
        <img loading="lazy" src="${esc(o.media === "clip" ? thumbUrl(o.id) : fileUrl(o.file))}" alt="">
        <span class="badge badge-${esc(o.category)}">${label(o.category)}</span>
        <span class="muted">${o.media === "clip" ? "footage · " : ""}${esc(o.caption)} (${o.found ? `${esc(o.found)}, ` : ""}${esc(o.licence || "")})</span>
        <button data-choose="${esc(o.id)}">Use this</button>
      </div>`).join("");
    const draw = alt.illustration.allowed
      ? `<div class="option"><div class="placeholder">New illustration</div><span class="muted">An object or setting matching the words (never a person or a named place), labelled ILLUSTRATION.</span><button data-choose="illustration">Draw one</button></div>`
      : `<p class="muted">${esc(alt.illustration.why)}</p>`;
    box.innerHTML = (options || `<p class="muted">No other pictures were found for this shot.</p>`) + draw;
  }

  // A picture that cannot be loaded (a card not drawn yet, a frame ffmpeg could not grab) is hidden.
  $("scenes").addEventListener("error", (e) => {
    if (e.target.tagName === "IMG") e.target.classList.add("hidden");
  }, true);

  /* -------------------------------------------------------------- jobs */
  function showJob(job) {
    clearTimeout(jobTimer); jobTimer = null;
    const running = Boolean(job && job.state === "running");
    $("rebuild").disabled = running;
    $("render").disabled = running;
    if (!job || job.state === "none") { $("job").classList.add("hidden"); return; }
    $("job").classList.remove("hidden");
    const percent = Math.round((job.progress || 0) * 100);
    $("job-bar").style.width = `${percent}%`;
    const what = job.kind === "render" ? "Final render" : "Review cut";
    const words = { running: "running", done: "done", failed: "failed", stopped: "stopped before finishing" }[job.state] || job.state;
    $("job-text").textContent = `${what}: ${words} (${percent}%)`
      + (running && job.tail?.length ? ` — ${job.tail[job.tail.length - 1]}` : "")
      + (job.state === "done" && job.kind === "render" && view?.final ? ` — ${view.final}` : "");
    const failed = job.state === "failed" || job.state === "stopped";
    $("job-tail").hidden = !failed;
    $("job-tail").textContent = failed ? (job.tail || []).join("\n") : "";
    if (running) poll(film);
  }

  /** Ask for the job again in a few seconds; stops when the film is closed or the tab is left. */
  function poll(name) {
    jobTimer = setTimeout(async () => {
      jobTimer = null;
      if (film !== name || $("video-tab").classList.contains("hidden")) return;
      try {
        const next = await api(`${base(name)}/job`);
        if (film !== name) return;
        if (next.state === "running") { showJob(next); return; }
        if (next.kind === "render" && next.state === "done") finalStamp = Date.now();
        await refresh();                           // new review cut: frames and player reload (new version)
      } catch (error) {
        showError(error.message);
        if (film === name) poll(name);             // a hiccup; keep watching the job
      }
    }, POLL_MS);
  }

  $("rebuild").addEventListener("click", async () => {
    if (!film) return;
    try { showError(""); showJob(await api(`${base()}/rebuild`, send("POST"))); } catch (error) { showError(error.message); }
  });

  $("approve-unreviewed").addEventListener("click", async () => {
    if (!film || !view) return;
    const k = unreviewed().length;
    if (!k) return;
    if (!confirm(`Approve ${k} shot${k === 1 ? "" : "s"} that ${k === 1 ? "has" : "have"} no review and no approval yet?`)) return;
    try {
      showError("");
      $("approve-unreviewed").disabled = true;
      await api(`${base()}/approve-unreviewed`, send("POST"));
      await refresh();
    } catch (error) {
      showError(error.message);
      await refresh().catch(() => {});
    }
  });

  $("render").addEventListener("click", async () => {
    if (!film || !view) return;
    const sound = $("render-sound").checked;
    const total = allShots().length;
    const open = allShots().filter((s) => !approved(s)).length;
    const state = open ? `${open} of ${total} shot${total === 1 ? "" : "s"} ${open === 1 ? "is" : "are"} not approved yet.` : `All ${total} shots are approved.`;
    if (!confirm(`${state} Finalize the video: start the final 1080p render${sound ? " with music and effects" : ""}? It can take hours, and the review cut cannot be rebuilt while it runs.`)) return;
    try { showError(""); showJob(await api(`${base()}/render`, send("POST", { sound }))); } catch (error) { showError(error.message); }
  });
})();
