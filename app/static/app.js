const els = {
  healthStatus: document.getElementById("health-status"),
  workspaceRoot: document.getElementById("workspace-root"),
  pdfCount: document.getElementById("pdf-count"),
  diagramCount: document.getElementById("diagram-count"),
  caseCount: document.getElementById("case-count"),
  chunkCount: document.getElementById("chunk-count"),
  recentCases: document.getElementById("recent-cases"),
  pdfFileList: document.getElementById("pdf-file-list"),
  diagramFileList: document.getElementById("diagram-file-list"),
  pdfSummaryCount: document.getElementById("pdf-summary-count"),
  diagramSummaryCount: document.getElementById("diagram-summary-count"),
  answerText: document.getElementById("answer-text"),
  routeUsed: document.getElementById("route-used"),
  citationsList: document.getElementById("citations-list"),
  chunksList: document.getElementById("chunks-list"),
  ingestLog: document.getElementById("ingest-log"),
  queryForm: document.getElementById("query-form"),
  ingestForm: document.getElementById("ingest-form"),
  question: document.getElementById("question"),
  refreshButton: document.getElementById("refresh-dashboard"),
};

function renderFileList(container, items) {
  if (!items.length) {
    container.innerHTML = '<p class="empty-state">No files found.</p>';
    return;
  }
  container.innerHTML = items
    .map((item) => `<div class="file-row">${item}</div>`)
    .join("");
}

function renderCards(container, cards, renderer, emptyText) {
  if (!cards.length) {
    container.innerHTML = `<p class="empty-state">${emptyText}</p>`;
    return;
  }
  container.innerHTML = cards.map(renderer).join("");
}

async function getJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function refreshDashboard() {
  els.healthStatus.textContent = "Refreshing...";
  const [health, files, summary] = await Promise.all([
    getJson("/health"),
    getJson("/workspace/files"),
    getJson("/dashboard/summary"),
  ]);

  els.healthStatus.textContent = health.status.toUpperCase();
  els.workspaceRoot.textContent = summary.workspace_root;
  els.pdfCount.textContent = summary.discovered_pdf_count;
  els.diagramCount.textContent = summary.diagram_count;
  els.caseCount.textContent = summary.ingested_case_count;
  els.chunkCount.textContent = summary.stored_chunk_count;
  els.pdfSummaryCount.textContent = files.pdf_files.length;
  els.diagramSummaryCount.textContent = files.diagram_files.length;

  renderFileList(els.pdfFileList, files.pdf_files);
  renderFileList(els.diagramFileList, files.diagram_files);
  renderCards(
    els.recentCases,
    summary.recent_cases,
    (item) => `
      <article class="case-card">
        <p class="card-title">${item.title}</p>
        <p class="card-subtext">${item.status || "status unknown"}${item.judgment_date ? ` | ${item.judgment_date}` : ""}</p>
        <p class="card-subtext muted">${item.source_file || "source file unknown"}</p>
      </article>
    `,
    "No recent cases available."
  );
}

async function runQuery(question) {
  els.answerText.textContent = "Running query...";
  els.routeUsed.textContent = "Route: working";
  const payload = await getJson("/query", {
    method: "POST",
    body: JSON.stringify({ question }),
  });

  els.answerText.textContent = payload.answer || "No answer returned.";
  els.routeUsed.textContent = `Route: ${payload.route_used || "unknown"}`;

  renderCards(
    els.citationsList,
    payload.citations || [],
    (item) => `
      <article class="citation-card">
        <p class="card-title">${item.title || "Unknown case"}</p>
        <p class="card-subtext">${item.citation}</p>
        <p class="card-subtext muted">Chunk type: ${item.chunk_type || "unknown"}</p>
      </article>
    `,
    "No citations were returned."
  );

  renderCards(
    els.chunksList,
    payload.retrieved_chunks_summary || [],
    (item) => `
      <article class="chunk-card">
        <p class="card-title">${item.title || "Unknown case"}</p>
        <p class="card-subtext">Page ${item.page_number ?? "?"} | ${item.chunk_type || "general"}</p>
        <p class="card-subtext muted">Score: ${item.score ?? "n/a"} | Chunk ${item.chunk_id}</p>
      </article>
    `,
    "No retrieved chunks were returned."
  );
}

async function runIngestion(pdfDir) {
  els.ingestLog.textContent = "Starting ingestion...";
  const payload = await getJson("/ingest/workspace", {
    method: "POST",
    body: JSON.stringify({ pdf_dir: pdfDir || null }),
  });
  els.ingestLog.textContent = JSON.stringify(payload, null, 2);
  await refreshDashboard();
}

els.refreshButton.addEventListener("click", async () => {
  try {
    await refreshDashboard();
  } catch (error) {
    els.healthStatus.textContent = "ERROR";
    els.workspaceRoot.textContent = error.message;
  }
});

els.queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question) {
    els.answerText.textContent = "Enter a legal question first.";
    return;
  }
  try {
    await runQuery(question);
  } catch (error) {
    els.answerText.textContent = `Query failed: ${error.message}`;
    els.routeUsed.textContent = "Route: error";
  }
});

els.ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const pdfDir = new FormData(els.ingestForm).get("pdf_dir");
  try {
    await runIngestion(String(pdfDir || "").trim());
  } catch (error) {
    els.ingestLog.textContent = `Ingestion failed: ${error.message}`;
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", async () => {
    els.question.value = chip.textContent.trim();
    await runQuery(els.question.value);
  });
});

refreshDashboard().catch((error) => {
  els.healthStatus.textContent = "ERROR";
  els.workspaceRoot.textContent = error.message;
});

