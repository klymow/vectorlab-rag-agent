"use strict";

const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const loadingPanel = document.querySelector("#loading");
const errorPanel = document.querySelector("#error-panel");
const errorMessage = document.querySelector("#error-message");
const resultPanel = document.querySelector("#result");
const answerStatus = document.querySelector("#answer-status");
const answerText = document.querySelector("#answer-text");
const sourcesList = document.querySelector("#sources-list");
const sourcesCount = document.querySelector("#sources-count");
const sourcesEmpty = document.querySelector("#sources-empty");
const vectorTable = document.querySelector("#vector-table");
const bm25Table = document.querySelector("#bm25-table");
const hybridTable = document.querySelector("#hybrid-table");
const contextList = document.querySelector("#context-list");

function setLoading(isLoading) {
  loadingPanel.hidden = !isLoading;
  submitButton.disabled = isLoading;
  questionInput.disabled = isLoading;
  form.setAttribute("aria-busy", String(isLoading));
}

function showError(message) {
  errorMessage.textContent = message;
  errorPanel.hidden = false;
}

function clearError() {
  errorMessage.textContent = "";
  errorPanel.hidden = true;
}

function formatNumber(value, digits) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function makeTable(columns, rows, rowClass) {
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column.label;
    headRow.append(th);
  });
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (rowClass) {
      const className = rowClass(row);
      if (className) tr.className = className;
    }
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = column.value(row);
      if (column.className) td.className = column.className;
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  return table;
}

function renderSources(sources) {
  sourcesList.replaceChildren();
  sourcesCount.textContent = `${sources.length}`;
  sourcesEmpty.hidden = sources.length !== 0;
  sources.forEach((source) => {
    const item = document.createElement("article");
    item.className = "source-item";
    const label = document.createElement("strong");
    label.className = "source-label";
    label.textContent = `[${source.label}]`;
    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = source.document_title;
    const meta = document.createElement("span");
    meta.textContent = `${source.document_id} · ${source.section} · версия ${source.version}`;
    body.append(title, meta);
    item.append(label, body);
    sourcesList.append(item);
  });
}

function renderDiagnostics(data) {
  vectorTable.replaceChildren(
    makeTable(
      [
        { label: "Rank", value: (row) => String(row.rank) },
        { label: "Similarity", value: (row) => formatNumber(row.similarity, 4) },
        { label: "Документ", value: (row) => row.document },
        { label: "Раздел", value: (row) => row.section },
        { label: "Chunk ID", value: (row) => row.chunk_id, className: "chunk-cell" },
      ],
      data.vector_diagnostics || [],
    ),
  );

  bm25Table.replaceChildren(
    makeTable(
      [
        { label: "Rank", value: (row) => String(row.rank) },
        { label: "BM25 score", value: (row) => formatNumber(row.score, 4) },
        { label: "Документ", value: (row) => row.document_title },
        { label: "Раздел", value: (row) => row.section },
        { label: "Chunk ID", value: (row) => row.chunk_id, className: "chunk-cell" },
      ],
      data.bm25_diagnostics || [],
      (row) => (row.score <= 0 ? "excluded" : ""),
    ),
  );

  hybridTable.replaceChildren(
    makeTable(
      [
        { label: "Rank", value: (row) => String(row.rank) },
        { label: "RRF", value: (row) => formatNumber(row.rrf_score, 6) },
        { label: "Vector", value: (row) => row.vector_rank == null ? "—" : String(row.vector_rank) },
        { label: "BM25", value: (row) => row.bm25_rank == null ? "—" : String(row.bm25_rank) },
        { label: "Документ", value: (row) => row.document },
        { label: "Раздел", value: (row) => row.section },
        { label: "Chunk ID", value: (row) => row.chunk_id, className: "chunk-cell" },
      ],
      data.hybrid_diagnostics || [],
    ),
  );

  contextList.replaceChildren();
  (data.final_context || []).forEach((source) => {
    const item = document.createElement("article");
    item.className = "context-item";
    const header = document.createElement("header");
    const label = document.createElement("strong");
    label.textContent = `[${source.label}]`;
    const meta = document.createElement("span");
    meta.textContent = `${source.document_title} · ${source.section} · v${source.version}`;
    const excerpt = document.createElement("p");
    const compact = source.text.replace(/\s+/g, " ").trim();
    excerpt.textContent = compact.length > 320 ? `${compact.slice(0, 319).trim()}…` : compact;
    header.append(label, meta);
    item.append(header, excerpt);
    contextList.append(item);
  });
}

function renderResult(data) {
  const sufficient = data.status === "sufficient";
  answerStatus.textContent = sufficient ? "Информация подтверждена" : "Недостаточно информации";
  answerStatus.className = `status-badge ${sufficient ? "sufficient" : "insufficient"}`;
  answerText.textContent = data.answer;
  renderSources(data.sources || []);
  renderDiagnostics(data);
  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    showError("Введите вопрос перед отправкой.");
    questionInput.focus();
    return;
  }
  clearError();
  setLoading(true);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    let payload;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    if (!response.ok) {
      throw new Error(payload?.detail || `Backend вернул HTTP ${response.status}`);
    }
    renderResult(payload);
  } catch (error) {
    showError(error instanceof Error ? error.message : "Неизвестная ошибка backend.");
  } finally {
    setLoading(false);
    questionInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion();
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.ctrlKey && !event.altKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.dataset.question || "";
    questionInput.focus();
  });
});
