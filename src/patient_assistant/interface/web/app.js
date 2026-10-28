/* UI cho AI Patient Assistant — JavaScript thuần, gọi trực tiếp HTTP API của project. */
"use strict";

const SAMPLES = [
  "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng",
  "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối",
  "tuần này tôi đau mức 3",
  "đặt lịch khoa phục hồi chức năng thứ hai ngày 5/11",
  "đặt lịch tái khám thứ sáu này buổi chiều",
  "sau khi tập gập gối có nên chườm lạnh bao lâu",
  "xem lịch hẹn của tôi",
  "tôi đau ngực khó thở",
];
const WEEKDAYS = ["Chủ Nhật", "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy"];
const ORIGIN = { user_said: "người dùng nói", user_edit: "người dùng sửa", backend: "backend", system_default: "mặc định hệ thống", ai_summary: "AI tóm tắt" };
const VALIDATION = { passed: "đạt", failed: "lỗi", missing: "thiếu", uncertain: "chưa chắc" };
const RESOLUTION = { llm_fixable: "LLM sửa", ask_user: "hỏi người dùng", block: "chặn" };
const KIND_TITLE = {
  clarify: "Agent cần hỏi thêm", confirm_report: "Bản nháp báo cáo — cần bạn xác nhận",
  choose_slot: "Chọn lịch khám", confirm_booking: "Xác nhận lịch hẹn",
};
const OUTCOME_TITLE = {
  report_saved: "Đã lưu báo cáo", appointment_booked: "Đã đặt lịch", history: "Báo cáo trước", appointments: "Lịch hẹn",
  answer: "Trả lời", cancelled: "Đã dừng", handoff: "Chuyển nhân viên xem lại", safety: "Cảnh báo an toàn", error: "Lỗi",
};
const REPORT_EDITABLE = {
  pain_level: { type: "number", min: 0, max: 10 },
  exercise_days: { type: "number", min: 0, max: 7 },
  symptom_change: { type: "select", options: ["", "better", "no_change", "worse", "worse_during_activity"] },
  triggering_activity: { type: "text" },
};

const EMPTY = document.getElementById("empty-state");
const S = { meta: null, patient: "patient-demo-001", thread: "", pending: null, card: null, turns: [], busy: false };
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ tiện ích

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function el(tag, attrs = {}, html = "") {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  node.innerHTML = html;
  return node;
}
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 4500);
}
async function api(path, options = {}) {
  const headers = { "X-Patient-Id": S.patient, ...(options.headers || {}) };
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* không phải JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}
function slotWhen(iso) {
  // Giữ đúng giờ địa phương của server (chuỗi ISO có offset), không đổi theo múi giờ trình duyệt.
  const [d, t] = iso.split("T");
  const [y, m, day] = d.split("-").map(Number);
  const weekday = WEEKDAYS[new Date(Date.UTC(y, m - 1, day)).getUTCDay()];
  return `${weekday}, ${String(day).padStart(2, "0")}/${String(m).padStart(2, "0")}/${y} · ${t.slice(0, 5)}`;
}
function deptName(code) {
  return S.meta?.departments.find((d) => d.code === code)?.name || code;
}
function scrollChat() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}
function setBusy(busy) {
  S.busy = busy;
  $("send").disabled = busy;
  $("send").textContent = busy ? "Đang xử lý…" : "Gửi";
}

// ------------------------------------------------------------------ khởi tạo

async function init() {
  try {
    S.meta = await api("/v1/meta");
  } catch (e) {
    toast(`Không kết nối được API: ${e.message}`);
    return;
  }
  const m = S.meta;
  $("badges").innerHTML = [
    `<span class="badge on">LLM: ${esc(m.llm_model)}</span>`,
    `<span class="badge ${m.stt_enabled ? "on" : "off"}">STT: ${m.stt_enabled ? esc(m.stt_model) : "tắt"}</span>`,
    `<span class="badge">Hôm nay: ${esc(slotWhen(m.now).split(" · ")[0])}</span>`,
    `<span class="badge">Khoa: ${m.departments.map((d) => esc(d.name)).join(", ")}</span>`,
  ].join("");
  $("patient").innerHTML = m.patients.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
  $("samples").innerHTML = "";
  for (const text of SAMPLES) {
    $("samples").append(el("button", { class: "chip", type: "button", title: text, onclick: () => submitText(text) }, esc(text)));
  }
  if (!m.stt_enabled) $("mic").title = "STT đang tắt (PA_STT_ENABLED=true và cài .[stt] để bật)";
  newThread();
}

function newThread() {
  S.thread = `t-${Date.now().toString(36)}`;
  S.pending = null;
  S.card = null;
  S.turns = [];
  $("thread-label").textContent = `thread ${S.thread}`;
  $("messages").innerHTML = "";
  $("messages").append(EMPTY);
  EMPTY.hidden = false;
  renderTimeline();
  for (const id of ["tab-validation", "tab-tools", "tab-state"]) $(id).innerHTML = `<p class="muted pad">Chưa có dữ liệu.</p>`;
  updateHint();
}

// ------------------------------------------------------------------ gửi / resume

function addUser(text, meta) {
  $("empty-state").hidden = true;
  const m = el("div", { class: "msg user" }, `<span class="meta">${esc(meta)}</span>${esc(text)}`);
  $("messages").append(m);
  scrollChat();
}

async function submitText(text, source = "text", flags = []) {
  text = text.trim();
  if (!text || S.busy) return;
  $("input").value = "";
  if (S.pending) {
    await resume({ action: "reply", text }, text, source === "voice" ? "bạn (giọng nói) trả lời" : "bạn trả lời");
    return;
  }
  addUser(text, source === "voice" ? "bạn · giọng nói (đã duyệt transcript)" : "bạn");
  setBusy(true);
  try {
    const res = await api(`/v1/chat/${S.thread}/messages`, { method: "POST", json: { text, input_source: source, stt_flags: flags } });
    await handleTurn(res, text);
  } catch (e) {
    toast(`Lỗi: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

async function resume(decision, label, meta = "bạn") {
  if (S.busy) return;
  addUser(label, meta);
  if (S.card) {
    S.card.classList.add("answered");
    S.card.querySelectorAll("button, input, select").forEach((b) => (b.disabled = true));
  }
  setBusy(true);
  try {
    const res = await api(`/v1/chat/${S.thread}/resume`, { method: "POST", json: decision });
    await handleTurn(res, label);
  } catch (e) {
    toast(`Lỗi: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

async function handleTurn(res, said) {
  S.turns.push({ said, steps: res.steps || [] });
  renderTimeline();
  if (res.status === "waiting_user") {
    S.pending = res.interrupt;
    S.card = renderInterrupt(res.interrupt);
  } else {
    S.pending = null;
    S.card = null;
    renderDone(res.response);
  }
  updateHint();
  scrollChat();
  await refreshState();
}

function updateHint() {
  const hint = $("pending-hint");
  if (!S.pending) {
    hint.hidden = true;
    $("input").placeholder = "Nhập yêu cầu mới, hoặc bấm 🎙 để nói…";
    return;
  }
  hint.hidden = false;
  hint.textContent = "Agent đang chờ bạn trả lời — nội dung gửi bây giờ sẽ là câu trả lời (có thể nói hoặc gõ tự do).";
  $("input").placeholder = {
    clarify: "Trả lời câu hỏi của agent…",
    choose_slot: "Ví dụ: cái thứ hai · 9 giờ · bác sĩ Bình · giờ khác buổi chiều",
    confirm_booking: "Ví dụ: xác nhận · đổi sang buổi chiều · hủy",
    confirm_report: "Ví dụ: xác nhận · hủy",
  }[S.pending.kind] || "Trả lời…";
}

// ------------------------------------------------------------------ thẻ interrupt

function explanationTable(rows, editable = false) {
  const body = rows.map((r) => {
    const conf = editable ? REPORT_EDITABLE[r.field] : null;
    let value = esc(Array.isArray(r.value) ? r.value.join(", ") : r.value ?? "—");
    if (conf) {
      const cur = r.value ?? "";
      value = conf.type === "select"
        ? `<select data-field="${r.field}">${conf.options.map((o) => `<option ${o === cur ? "selected" : ""}>${o}</option>`).join("")}</select>`
        : `<input data-field="${r.field}" type="${conf.type}" ${conf.min !== undefined ? `min="${conf.min}" max="${conf.max}"` : ""} value="${esc(cur)}">`;
    }
    return `<tr>
      <td>${esc(r.label)}</td>
      <td class="value">${value}</td>
      <td class="quote">${r.evidence ? `“${esc(r.evidence)}”` : "—"}</td>
      <td><span class="pill ${r.origin}">${esc(ORIGIN[r.origin] || r.origin)}</span></td>
      <td><span class="pill ${r.validation}">${esc(VALIDATION[r.validation] || r.validation)}</span></td>
      <td>${esc(r.explanation)}</td></tr>`;
  }).join("");
  return `<table class="explain"><thead><tr><th>Trường</th><th>Giá trị</th><th>Trích dẫn</th><th>Nguồn</th><th>Kiểm tra</th><th>Vì sao</th></tr></thead><tbody>${body}</tbody></table>`;
}

function renderInterrupt(p) {
  const card = el("div", { class: "card" });
  let text = p.message;
  if (p.kind === "choose_slot") text = p.message.split("\n").filter((l) => !/^\d+\. /.test(l) && !l.startsWith("Bạn chọn số")).join("\n");
  if (p.kind === "confirm_report" || p.kind === "confirm_booking") text = p.message.replace(/ ?Nói “xác nhận”.*$/, "");
  card.innerHTML = `<h3>${esc(KIND_TITLE[p.kind] || p.kind)}</h3><p class="text">${esc(text)}</p>`;

  if (p.kind === "choose_slot") {
    const list = el("div", { class: "slots" });
    for (const o of p.options) {
      list.append(el("button", {
        class: "slot", type: "button",
        onclick: () => resume({ action: "choose", option_no: o.option_no }, `Chọn lịch số ${o.option_no}`),
      }, `<span class="no">${o.option_no}</span><span><span class="when">${esc(slotWhen(o.slot.start_at))}</span><br>
          <span class="who">${esc(o.slot.doctor_name)} · ${esc(deptName(o.slot.department_code))} · mã ${esc(o.slot.slot_id)}</span></span>`));
    }
    card.append(list);
    const row = el("div", { class: "inline-row" });
    const input = el("input", { placeholder: "Muốn giờ/ngày khác? Ví dụ: buổi chiều, thứ năm" });
    row.append(input, el("button", {
      class: "btn", type: "button",
      onclick: () => input.value.trim() && resume({ action: "other_time", text: input.value.trim() }, `Giờ khác: ${input.value.trim()}`),
    }, "Tìm lại"));
    card.append(row);
  }
  if (p.explanations?.length) card.insertAdjacentHTML("beforeend", explanationTable(p.explanations));

  const actions = el("div", { class: "actions" });
  if (p.kind === "confirm_report" || p.kind === "confirm_booking") {
    actions.append(el("button", { class: "btn primary", type: "button", onclick: () => resume({ action: "confirm" }, "Xác nhận") }, "Xác nhận"));
  }
  if (p.kind === "confirm_report") {
    const editBtn = el("button", { class: "btn", type: "button" }, "Sửa bản nháp");
    editBtn.addEventListener("click", () => {
      const table = card.querySelector("table.explain");
      if (editBtn.dataset.mode !== "edit") {
        table.outerHTML = explanationTable(p.explanations, true);
        editBtn.dataset.mode = "edit";
        editBtn.textContent = "Gửi chỉnh sửa";
        return;
      }
      const fields = {};
      card.querySelectorAll("[data-field]").forEach((inp) => {
        const original = p.explanations.find((e) => e.field === inp.dataset.field)?.value ?? "";
        let v = inp.value;
        if (inp.type === "number") v = v === "" ? null : Number(v);
        if (v === "") v = null;
        if (String(v ?? "") !== String(original ?? "")) fields[inp.dataset.field] = v;
      });
      if (!Object.keys(fields).length) return toast("Chưa có trường nào thay đổi.");
      resume({ action: "edit", fields }, `Sửa: ${Object.entries(fields).map(([k, v]) => `${k}=${v}`).join(", ")}`);
    });
    actions.append(editBtn);
  }
  actions.append(el("button", { class: "btn danger", type: "button", onclick: () => resume({ action: "cancel" }, "Hủy") }, "Hủy"));
  card.append(actions);
  $("messages").append(card);
  return card;
}

function renderDone(r) {
  if (!r) return;
  const card = el("div", { class: `card done ${r.kind}` });
  card.innerHTML = `<h3>${esc(OUTCOME_TITLE[r.kind] || r.kind)}</h3><p class="text">${esc(r.text)}</p>` +
    (r.source_ids?.length && !r.source_ids.every((id) => r.text.includes(id)) ? `<p class="muted">Nguồn / mã bản ghi: ${r.source_ids.map((s) => `<span class="code-chip">${esc(s)}</span>`).join(" ")}</p>` : "");
  $("messages").append(card);
}

// ------------------------------------------------------------------ panel quyết định

function renderTimeline() {
  const box = $("tab-timeline");
  if (!S.turns.length) {
    box.innerHTML = `<p class="muted pad">Chưa có lượt nào. Mỗi lượt sẽ liệt kê từng node agent đã chạy, theo thứ tự, kèm lý do.</p>`;
    return;
  }
  box.innerHTML = "";
  S.turns.forEach((turn, i) => {
    const section = el("div", { class: "turn" }, `<div class="turn-head"><strong>Lượt ${i + 1}</strong><span class="said">${esc(turn.said)}</span></div>`);
    const list = el("ol", { class: "steps" });
    for (const s of turn.steps) {
      const hasIssue = /Phát hiện|lỗi|Không quy đổi|Chưa xác định/.test(s.summary) && s.kind !== "stop";
      const detail = Object.keys(s.details || {}).length ? `<pre class="json">${esc(JSON.stringify(s.details, null, 2))}</pre>` : "";
      list.append(el("li", { class: `step ${s.kind}${hasIssue ? " issue" : ""}` },
        `<details><summary><span class="title">${esc(s.title)}</span>${s.flow !== "main" ? `<span class="flow">${esc(s.flow)}</span>` : ""}
          <span class="flow">${esc(s.node)}</span><div class="summary">${esc(s.summary)}</div></summary>${detail}</details>`));
    }
    section.append(list);
    box.append(section);
  });
  box.scrollTop = box.scrollHeight;
}

function issueChip(code, resolution) {
  return `<span class="code-chip res-${resolution || ""}" title="${esc(RESOLUTION[resolution] || "")}">${esc(code)}</span>`;
}

async function refreshState() {
  let st;
  try {
    st = await api(`/v1/chat/${S.thread}/state`);
  } catch {
    return;
  }
  const limits = S.meta?.limits || {};
  // --- Kiểm tra & sửa lỗi
  const attempts = (st.attempts || []).map((a) => `<tr><td>${esc(a.flow)}</td><td>${a.attempt_no === 0 ? "lần đầu" : `sửa lần ${a.attempt_no}`}</td>
      <td>${a.passed ? '<span class="ok">đạt</span>' : '<span class="bad">không đạt</span>'}</td>
      <td>${a.issue_codes.map((c) => issueChip(c, (a.llm_fixable_codes || []).includes(c) ? "llm_fixable" : "ask_user")).join(" ") || "—"}</td></tr>`).join("");
  const issues = (st.issues || []).map((i) => `<tr><td>${esc(i.layer)}</td><td>${esc(i.field || "—")}</td><td>${issueChip(i.code, i.resolution)}</td>
      <td>${esc(i.message)}${i.suggestion ? `<br><span class="muted">Gợi ý cho LLM: ${esc(i.suggestion)}</span>` : ""}</td>
      <td class="res-${i.resolution}">${esc(RESOLUTION[i.resolution])}</td></tr>`).join("");
  const w = st.resolved_window;
  $("tab-validation").innerHTML = `
    <div class="section"><h3>Luật quyết định sau validator (code, không phải LLM)</h3>
      <div class="rule-box">Không có lỗi → đi tiếp (trình bản nháp / tìm slot).<br>
        Có lỗi <span class="res-block">chặn</span> → dừng an toàn, chuyển nhân viên.<br>
        Có lỗi <span class="res-ask_user">hỏi người dùng</span> (thiếu, ngoài phạm vi theo lời người dùng, mâu thuẫn, STT nghi sai) → hỏi lại, không để LLM đoán.<br>
        Chỉ toàn lỗi <span class="res-llm_fixable">LLM sửa</span> → gửi lỗi cụ thể cho LLM sửa, tối đa ${limits.max_corrections ?? 2} lần; hết lượt → dừng.</div></div>
    <div class="section"><h3>Bộ đếm</h3><dl class="kv">
      <dt>Số lần LLM sửa</dt><dd>${st.correction_count} / ${limits.max_corrections ?? "?"}</dd>
      <dt>Số lượt hỏi lại</dt><dd>${st.clarify_rounds} / ${limits.max_clarify_rounds ?? "?"}</dd>
      <dt>Số lần tìm slot</dt><dd>${st.search_rounds} / ${limits.max_slot_search_rounds ?? "?"}</dd>
      <dt>Thử lại khi slot bị chiếm</dt><dd>${st.conflict_retries}</dd></dl></div>
    <div class="section"><h3>Các lần trích xuất (FR-06.6)</h3>${attempts ? `<table class="grid"><thead><tr><th>Flow</th><th>Lần</th><th>Kết quả</th><th>Mã lỗi</th></tr></thead><tbody>${attempts}</tbody></table>` : '<p class="muted">Chưa có.</p>'}</div>
    <div class="section"><h3>Lỗi hiện tại</h3>${issues ? `<table class="grid"><thead><tr><th>Lớp</th><th>Trường</th><th>Mã</th><th>Mô tả</th><th>Ai xử lý</th></tr></thead><tbody>${issues}</tbody></table>` : '<p class="muted">Không có.</p>'}</div>
    <div class="section"><h3>Quy đổi ngày giờ (code)</h3>${w ? `<dl class="kv"><dt>Cụm người dùng nói</dt><dd>${esc([st.booking_request?.date_expression, st.booking_request?.time_expression].filter(Boolean).join(" · ") || "—")}</dd>
      <dt>Khoảng ngày</dt><dd>${esc(w.date_from)}${w.date_to !== w.date_from ? ` → ${esc(w.date_to)}` : ""}</dd>
      <dt>Khung giờ</dt><dd>${w.time_from ? `${esc(w.time_from.slice(0, 5))}–${esc(w.time_to.slice(0, 5))}` : "cả ngày"}</dd>
      <dt>Luật áp dụng</dt><dd>${(w.rules || []).map((r) => `<span class="code-chip">${esc(r)}</span>`).join(" ")}</dd></dl>`
      : (st.date_issues || []).length ? st.date_issues.map((i) => `${issueChip(i.code, i.resolution)} ${esc(i.message)}`).join("<br>") : '<p class="muted">Không áp dụng.</p>'}</div>
    <div class="section"><h3>Cờ STT (từ nhận dạng kém)</h3>${(st.stt_flags || []).length ? st.stt_flags.map((f) => `<span class="code-chip">${esc(f.text)} · ${Math.round(f.probability * 100)}%</span>`).join(" ") : '<p class="muted">Không có.</p>'}</div>`;

  // --- Tool
  const calls = (st.tool_calls || []).map((t, i) => `<tr><td>${i + 1}</td><td><span class="code-chip">${esc(t.tool)}</span></td>
      <td>${t.called_by === "llm" ? "LLM" : "code"}</td><td>${t.status === "ok" ? '<span class="ok">ok</span>' : `<span class="bad">${esc(t.status)}</span>`}</td>
      <td>${esc(t.error_code || "—")}</td><td>${t.latency_ms} ms</td><td>${(t.source_ids || []).map((s) => `<span class="code-chip">${esc(s)}</span>`).join(" ")}</td></tr>`).join("");
  $("tab-tools").innerHTML = `<div class="section"><div class="rule-box">Tool đọc có thể do LLM gọi (trả lời câu hỏi). Tool ghi
      (<span class="code-chip">save_patient_report</span>, <span class="code-chip">create_appointment</span>) không bao giờ được giao cho LLM —
      chỉ code gọi sau khi validator đạt và bạn xác nhận; sau đó luôn đọc lại để xác minh.</div></div>
    <div class="section">${calls ? `<table class="grid"><thead><tr><th>#</th><th>Tool</th><th>Ai gọi</th><th>Status</th><th>Lỗi</th><th>Latency</th><th>Nguồn</th></tr></thead><tbody>${calls}</tbody></table>` : '<p class="muted">Lượt này chưa gọi tool nào.</p>'}</div>`;

  // --- State
  $("tab-state").innerHTML = `<div class="section"><dl class="kv">
      <dt>Intent</dt><dd>${esc(st.intent || "—")} <span class="muted">(${esc(st.intent_source || "—")})</span></dd>
      <dt>Việc chờ làm sau</dt><dd>${esc((st.pending_intents || []).join(", ") || "—")}</dd>
      <dt>An toàn</dt><dd>${esc(st.safety)}</dd>
      <dt>Nguồn đầu vào</dt><dd>${esc(st.input_source)}</dd>
      <dt>Câu trả lời bổ sung</dt><dd>${esc((st.user_replies || []).join(" | ") || "—")}</dd>
      <dt>Kết quả</dt><dd>${st.outcome ? `${esc(st.outcome.kind)}${st.outcome.code ? ` (${esc(st.outcome.code)})` : ""}` : "chưa kết thúc"}</dd></dl></div>
    <div class="section"><h3>Toàn bộ state (JSON)</h3><pre class="json">${esc(JSON.stringify(st, null, 2))}</pre></div>`;
}

// ------------------------------------------------------------------ giọng nói

let recorder = null;
let chunks = [];

async function toggleRecording() {
  if (!S.meta?.stt_enabled) {
    toast("STT đang tắt. Đặt PA_STT_ENABLED=true trong .env và cài: pip install -e \".[stt]\"");
    return;
  }
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      $("mic").classList.remove("recording");
      $("recording").hidden = true;
      transcribe(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }), "ghi-am.webm");
    };
    recorder.start();
    $("mic").classList.add("recording");
    $("recording").hidden = false;
  } catch (e) {
    toast(`Không truy cập được micro: ${e.message}`);
  }
}

async function transcribe(blob, name) {
  const form = new FormData();
  form.append("audio", blob, name);
  setBusy(true);
  try {
    const res = await api("/v1/stt/transcribe", { method: "POST", body: form });
    renderTranscriptReview(res);
  } catch (e) {
    toast(`STT lỗi: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

function highlight(text, flags) {
  const words = new Set(flags.map((f) => f.text.toLowerCase()));
  return text.split(/(\s+)/).map((w) => (words.has(w.toLowerCase().replace(/[.,!?]/g, "")) ? `<span class="flagged" title="Nhận dạng kém">${esc(w)}</span>` : esc(w))).join("");
}

function renderTranscriptReview(res) {
  $("empty-state").hidden = true;
  const card = el("div", { class: "card transcript" });
  card.innerHTML = `<h3>Kiểm tra transcript trước khi gửi (FR-01.3)</h3>
    <p class="muted">Đoạn tô vàng là từ quan trọng (số, ngày, giờ, khoa…) mà STT không chắc. Sửa nếu sai rồi bấm Gửi.</p>
    <div class="preview">${highlight(res.text, res.flags)}</div>`;
  const area = el("textarea");
  area.value = res.text;
  card.append(area);
  const actions = el("div", { class: "actions" });
  actions.append(
    el("button", {
      class: "btn primary", type: "button",
      onclick: () => {
        card.classList.add("answered");
        card.querySelectorAll("button, textarea").forEach((b) => (b.disabled = true));
        const kept = res.flags.filter((f) => area.value.includes(f.text)); // từ đã sửa thì bỏ cờ
        submitText(area.value, "voice", kept);
      },
    }, "Gửi transcript"),
    el("button", { class: "btn", type: "button", onclick: () => card.remove() }, "Bỏ"),
  );
  card.append(actions);
  $("messages").append(card);
  scrollChat();
}

// ------------------------------------------------------------------ sự kiện

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  submitText($("input").value);
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitText($("input").value);
  }
});
$("mic").addEventListener("click", toggleRecording);
$("audio-file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!S.meta?.stt_enabled) toast("STT đang tắt — server sẽ trả lỗi 503.");
  transcribe(file, file.name);
  e.target.value = "";
});
$("new-thread").addEventListener("click", newThread);
$("patient").addEventListener("change", (e) => {
  S.patient = e.target.value;
  newThread();
});
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  for (const name of ["timeline", "validation", "tools", "state"]) $(`tab-${name}`).hidden = name !== tab.dataset.tab;
}));

init();
