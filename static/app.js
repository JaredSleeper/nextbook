(() => {
  const MAX = 10, MIN = 3;
  const $ = (s) => document.querySelector(s);
  const KEY = "nextbook:v1";

  const EXAMPLES = {
    "Sci-fi": [126, 70, 1574, 729, 276, 400, 141, 442, 54, 215],
    "Literary fiction": [820, 268, 354, 658, 212, 1198, 146, 1005, 104, 115],
    "Fantasy": [192, 7, 39, 389, 562, 746, 167, 19, 283, 185],
    "Classics": [10, 43, 13, 4, 5, 63, 177, 172, 55, 32],
    "Nonfiction": [2205, 619, 193, 92, 208, 236, 181, 81, 385, 408],
  };

  const state = loadState();
  let lastRows = null;
  let picksSeq = 0, rowsSeq = 0;

  function loadState() {
    let s = {};
    try { s = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { s = {}; }
    return {
      seeds: s.seeds || [],
      liked: s.liked || [],
      disliked: s.disliked || [],
      read: s.read || [],
      prompt: s.prompt || "",
      known: s.known || {},
    };
  }
  function save() {
    localStorage.setItem(KEY, JSON.stringify(state));
    const ids = state.seeds.map((b) => b.id).join(",");
    history.replaceState(null, "", ids ? "#b=" + ids : location.pathname);
  }

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null) n.append(c);
    return n;
  }

  function cover(b) {
    const box = el("span", { class: "cover" });
    const ph = el("span", { class: "ph" }, b.title || "", el("small", {}, b.author || ""));
    box.append(ph);
    if (b.cover) {
      const img = el("img", { src: b.cover, alt: "", loading: "lazy" });
      img.onload = () => { if (img.naturalWidth > 2) ph.remove(); else img.remove(); };
      img.onerror = () => img.remove();
      box.append(img);
    }
    return box;
  }

  async function api(path, body) {
    const opt = body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {};
    const r = await fetch(path, opt);
    if (!r.ok) throw new Error(path + " " + r.status);
    return r.json();
  }

  const body = (extra = {}) => ({
    seeds: state.seeds.map(({ id, title, author }) => ({ id, title, author })),
    liked: state.liked, disliked: state.disliked, read: state.read, ...extra,
  });

  /* ---------- setup ---------- */
  const search = $("#search"), results = $("#search-results");
  let searchTimer = null, searchSeq = 0, hits = [], active = -1;

  search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const q = search.value.trim();
    if (q.length < 2) { results.hidden = true; return; }
    searchTimer = setTimeout(() => runSearch(q), 180);
  });
  search.addEventListener("keydown", (e) => {
    if (results.hidden) return;
    if (e.key === "ArrowDown") { active = Math.min(hits.length - 1, active + 1); paintHits(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); paintHits(); e.preventDefault(); }
    else if (e.key === "Enter") { if (hits[active]) addSeed(hits[active]); e.preventDefault(); }
    else if (e.key === "Escape") { results.hidden = true; }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search-wrap")) results.hidden = true; });

  async function runSearch(q) {
    const seq = ++searchSeq;
    const url = (src) => "/api/search?q=" + encodeURIComponent(q) + "&limit=8&source=" + src;
    const apply = (data) => {
      if (seq !== searchSeq) return;
      hits = Array.isArray(data) ? data : data.results || [];
      active = hits.length ? 0 : -1;
      paintHits();
    };
    let local = null;
    try { local = await api(url("local")); apply(local); } catch (e) { /* fall through */ }
    const n = local ? (local.results || local).length : 0;
    if (n >= 8) return;
    try { apply(await api(url("all"))); } catch (e) { /* keep local */ }
  }
  function paintHits() {
    results.replaceChildren();
    if (!hits.length) results.append(el("li", { class: "empty" }, "No matches. Try the author's name."));
    hits.forEach((b, i) => {
      const have = state.seeds.some((s) => s.id === b.id);
      const thumb = cover(b); thumb.classList.add("thumb");
      results.append(el("li", { class: i === active ? "active" : "", onmousedown: (e) => { e.preventDefault(); addSeed(b); } },
        thumb,
        el("div", {}, el("div", { class: "t" }, b.title), el("div", { class: "a" }, [b.author, b.year].filter(Boolean).join(", ") + (have ? " (added)" : "")))));
    });
    results.hidden = false;
  }

  function addSeed(b) {
    if (state.seeds.length >= MAX || state.seeds.some((s) => s.id === b.id)) return;
    state.seeds.push({ id: b.id, title: b.title, author: b.author, cover: b.cover });
    save();
    search.value = ""; results.hidden = true; search.focus();
    paintSlots();
  }
  function removeSeed(id) {
    state.seeds = state.seeds.filter((s) => s.id !== id);
    save(); paintSlots();
  }
  function paintSlots() {
    const ol = $("#slots"); ol.replaceChildren();
    for (let i = 0; i < MAX; i++) {
      const b = state.seeds[i];
      if (!b) { ol.append(el("li", { class: "slot" }, String(i + 1))); continue; }
      ol.append(el("li", { class: "slot filled", title: b.title + (b.author ? " by " + b.author : "") },
        cover(b), el("span", { class: "cap" }, b.title),
        el("button", { class: "rm", "aria-label": "Remove " + b.title, onclick: () => removeSeed(b.id) }, "\u00d7")));
    }
    const n = state.seeds.length;
    $("#go-btn").disabled = n < MIN;
    $("#count").textContent = n >= MAX ? "That's ten." : n < MIN ? `${n} of 10 added. Add at least ${MIN - n} more.` : `${n} of 10 added.`;
    search.disabled = n >= MAX;
    search.placeholder = n >= MAX ? "Remove one to add another" : "Search by title or author";
  }

  async function loadExample(name) {
    $("#count").textContent = "Loading the example list...";
    const found = await Promise.all(EXAMPLES[name].map(async (n) => {
      try { return slim(await api("/api/book/gb:" + n)); } catch (e) { return null; }
    }));
    state.seeds = found.filter(Boolean).slice(0, MAX);
    state.liked = []; state.disliked = []; state.read = []; state.prompt = "";
    save(); paintSlots(); showResults();
  }
  const ex = $("#example-links");
  for (const name of Object.keys(EXAMPLES)) ex.append(el("a", { onclick: () => loadExample(name) }, name.toLowerCase()));

  $("#go-btn").addEventListener("click", showResults);
  $("#edit-btn").addEventListener("click", showSetup);

  function showSetup() {
    $("#setup").hidden = false; $("#results").hidden = true; $("#edit-btn").hidden = true;
    paintSlots(); search.focus();
  }

  /* ---------- results ---------- */
  function showResults() {
    if (state.seeds.length < MIN) return showSetup();
    $("#setup").hidden = true; $("#results").hidden = false; $("#edit-btn").hidden = false;
    paintMine();
    $("#prompt").value = state.prompt;
    $("#clear-prompt").hidden = !state.prompt;
    $("#rows").replaceChildren(el("div", { id: "picks-row" }), el("div", { id: "std-rows" }, el("p", { class: "loading" }, "Finding books...")));
    loadRows(); loadPicks();
    window.scrollTo(0, 0);
  }
  function paintMine() {
    const m = $("#mine"); m.replaceChildren(el("span", { class: "lbl" }, "Based on"));
    for (const b of state.seeds) { const c = cover(b); c.title = b.title; c.onclick = showSetup; m.append(c); }
  }

  async function loadRows() {
    const seq = ++rowsSeq;
    try {
      const d = await api("/api/recommend", body());
      if (seq !== rowsSeq) return;
      lastRows = d.rows;
      for (const r of d.rows) for (const b of r.books) state.known[b.id] = slim(b);
      paintRows();
    } catch (e) {
      if (seq === rowsSeq) $("#std-rows").replaceChildren(el("p", { class: "err" }, "Couldn't load recommendations. Refresh to try again."));
    }
  }
  const hiddenIds = () => new Set([...state.read, ...state.disliked, ...state.seeds.map((s) => s.id)]);
  function paintRows() {
    const box = $("#std-rows"); box.replaceChildren();
    const hide = hiddenIds();
    for (const r of lastRows || []) {
      const books = r.books.filter((b) => !hide.has(b.id));
      if (books.length) box.append(row(r.title, books, false));
    }
  }

  async function loadPicks() {
    const seq = ++picksSeq;
    const prompt = state.prompt.trim();
    const host = $("#picks-row");
    host.replaceChildren(el("section", { class: "row" },
      el("h3", {}, prompt ? "For: " + prompt : "Picked for you", el("span", { class: "note" }, "Choosing from thousands of books...")),
      el("div", { class: "track" }, ...Array.from({ length: 4 }, () => el("div", { class: "skel" })))));
    try {
      const d = await api("/api/picks", body({ prompt: prompt || null }));
      if (seq !== picksSeq) return;
      for (const b of d.books) state.known[b.id] = slim(b);
      const hide = hiddenIds();
      const books = d.books.filter((b) => !hide.has(b.id));
      host.replaceChildren(row(d.title || (prompt ? "For: " + prompt : "Picked for you"), books, true,
        d.fallback ? "Our writer is busy, so these come without notes." : null));
    } catch (e) {
      if (seq === picksSeq) host.replaceChildren(el("p", { class: "err" }, "Couldn't load picks. Try again in a moment."));
    }
  }
  const slim = (b) => ({ id: b.id, title: b.title, author: b.author, cover: b.cover });

  function row(title, books, wide, note) {
    const track = el("div", { class: "track" });
    for (const b of books) track.append(card(b, wide));
    const strip = el("div", { class: "strip" },
      el("button", { class: "arrow l", "aria-label": "Scroll left", onclick: () => track.scrollBy({ left: -track.clientWidth * 0.85 }) }, "\u2039"),
      track,
      el("button", { class: "arrow r", "aria-label": "Scroll right", onclick: () => track.scrollBy({ left: track.clientWidth * 0.85 }) }, "\u203a"));
    return el("section", { class: "row" }, el("h3", {}, title, note ? el("span", { class: "note" }, note) : null), strip);
  }
  function card(b, wide) {
    const meta = el("div", {}, el("div", { class: "t" }, b.title), el("div", { class: "a" }, b.author || ""),
      wide && b.reason ? el("div", { class: "why" }, b.reason) : null);
    return el("button", { class: "card" + (wide ? " wide" : ""), "data-id": b.id, onclick: () => openBook(b) }, cover(b), meta);
  }

  $("#ask").addEventListener("submit", (e) => {
    e.preventDefault();
    state.prompt = $("#prompt").value.trim(); save();
    $("#clear-prompt").hidden = !state.prompt;
    loadPicks();
  });
  $("#clear-prompt").addEventListener("click", () => {
    $("#prompt").value = ""; state.prompt = ""; save();
    $("#clear-prompt").hidden = true; loadPicks();
  });

  /* ---------- drawer ---------- */
  const drawer = $("#drawer");
  let current = null;
  async function openBook(b) {
    current = b;
    fillDrawer(b);
    drawer.hidden = false;
    if (b.description == null && String(b.id).startsWith("gb:")) {
      try { const full = await api("/api/book/" + encodeURIComponent(b.id)); if (current === b) fillDrawer({ ...full, reason: b.reason }); } catch (e) { /* keep partial */ }
    }
  }
  function fillDrawer(b) {
    $("#d-cover").replaceChildren(cover(b));
    $("#d-title").textContent = b.title;
    $("#d-author").textContent = b.author || "";
    const bits = [];
    if (b.year) bits.push(String(b.year));
    if (b.rating) bits.push(`${Number(b.rating).toFixed(2)} avg rating`);
    if (b.ratings_count) bits.push(`${Number(b.ratings_count).toLocaleString()} ratings`);
    if (b.series) bits.push(`Series: ${b.series}`);
    $("#d-meta").textContent = bits.join(" \u00b7 ");
    $("#d-tags").textContent = (b.tags || []).join(", ");
    $("#d-reason").hidden = !b.reason; $("#d-reason").textContent = b.reason || "";
    $("#d-desc").textContent = b.description || "";
    const gr = $("#d-gr"), ol = $("#d-ol");
    gr.hidden = !b.goodreads_url; gr.href = b.goodreads_url || "#";
    ol.hidden = !b.openlibrary_url; ol.href = b.openlibrary_url || "#";
  }
  function closeDrawer() { drawer.hidden = true; current = null; }
  drawer.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeDrawer(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) closeDrawer(); });

  let refetch = null;
  drawer.querySelectorAll("[data-act]").forEach((btn) => btn.addEventListener("click", () => {
    if (!current) return;
    const id = current.id, act = btn.dataset.act;
    for (const k of ["liked", "disliked", "read"]) state[k] = state[k].filter((x) => x !== id);
    state[act].push(id);
    save(); closeDrawer();
    if (act !== "liked") document.querySelectorAll(`.card[data-id="${CSS.escape(id)}"]`).forEach((n) => n.remove());
    clearTimeout(refetch);
    refetch = setTimeout(() => { loadRows(); }, 600);
  }));

  /* ---------- boot ---------- */
  async function boot() {
    const m = location.hash.match(/b=([^&]+)/);
    if (m) {
      const ids = decodeURIComponent(m[1]).split(",").filter(Boolean).slice(0, MAX);
      const same = ids.length === state.seeds.length && ids.every((id, i) => state.seeds[i].id === id);
      if (!same) {
        const books = await Promise.all(ids.map(async (id) => {
          const local = state.seeds.find((s) => s.id === id) || state.known[id];
          if (local) return local;
          if (!id.startsWith("gb:")) return null;
          try { return slim(await api("/api/book/" + encodeURIComponent(id))); } catch (e) { return null; }
        }));
        state.seeds = books.filter(Boolean);
        state.liked = []; state.disliked = []; state.read = []; state.prompt = "";
        save();
      }
    }
    if (state.seeds.length >= MIN) showResults(); else showSetup();
  }
  boot();
})();
