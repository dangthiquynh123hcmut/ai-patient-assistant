import { useEffect, useRef, useState } from "react";

import type { AgentState, Meta, Resolution } from "../../api/types";
import { RESOLUTION_LABEL } from "../../lib/format";
import type { Turn } from "../chat/useChat";

type Tab = "timeline" | "validation" | "tools" | "state";
const TABS: [Tab, string][] = [
  ["timeline", "Diễn biến"],
  ["validation", "Kiểm tra & sửa lỗi"],
  ["tools", "Tool"],
  ["state", "State"],
];

function Chip({ code, resolution }: { code: string; resolution?: Resolution }) {
  return (
    <span className={`code-chip${resolution ? ` res-${resolution}` : ""}`} title={resolution && RESOLUTION_LABEL[resolution]}>
      {code}
    </span>
  );
}

function Empty({ text }: { text: string }) {
  return <p className="muted pad">{text}</p>;
}

function Timeline({ turns }: { turns: Turn[] }) {
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [turns]);
  if (!turns.length) {
    return <Empty text="Chưa có lượt nào. Mỗi lượt sẽ liệt kê từng node agent đã chạy, theo thứ tự, kèm lý do." />;
  }
  return (
    <div className="tab-body" ref={box}>
      {turns.map((turn, i) => (
        <div className="turn" key={i}>
          <div className="turn-head">
            <strong>Lượt {i + 1}</strong>
            <span className="said">{turn.said}</span>
          </div>
          <ol className="steps">
            {turn.steps.map((s) => {
              const issue = /Phát hiện|lỗi|Không quy đổi|Chưa xác định/.test(s.summary) && s.kind !== "stop";
              return (
                <li key={s.index} className={`step ${s.kind}${issue ? " issue" : ""}`}>
                  <details>
                    <summary>
                      <span className="title">{s.title}</span>
                      {s.flow !== "main" && <span className="flow">{s.flow}</span>}
                      <span className="flow">{s.node}</span>
                      <div className="summary">{s.summary}</div>
                    </summary>
                    {Object.keys(s.details).length > 0 && (
                      <pre className="json">{JSON.stringify(s.details, null, 2)}</pre>
                    )}
                  </details>
                </li>
              );
            })}
          </ol>
        </div>
      ))}
    </div>
  );
}

function ValidationTab({ st, limits }: { st: AgentState; limits: Meta["limits"] | undefined }) {
  const w = st.resolved_window;
  return (
    <div className="tab-body">
      <div className="section">
        <h3>Luật quyết định sau validator (code, không phải LLM)</h3>
        <div className="rule-box">
          Không có lỗi → đi tiếp (trình bản nháp / tìm slot).
          <br />
          Có lỗi <span className="res-block">chặn</span> → dừng an toàn, chuyển nhân viên.
          <br />
          Có lỗi <span className="res-ask_user">hỏi người dùng</span> (thiếu, ngoài phạm vi theo lời người dùng, mâu thuẫn,
          STT nghi sai) → hỏi lại, không để LLM đoán.
          <br />
          Chỉ toàn lỗi <span className="res-llm_fixable">LLM sửa</span> → gửi lỗi cụ thể cho LLM sửa, tối đa{" "}
          {limits?.max_corrections ?? 2} lần; hết lượt → dừng.
        </div>
      </div>
      <div className="section">
        <h3>Bộ đếm</h3>
        <dl className="kv">
          <dt>Số lần LLM sửa</dt>
          <dd>
            {st.correction_count} / {limits?.max_corrections ?? "?"}
          </dd>
          <dt>Số lượt hỏi lại</dt>
          <dd>
            {st.clarify_rounds} / {limits?.max_clarify_rounds ?? "?"}
          </dd>
          <dt>Số lần tìm slot</dt>
          <dd>
            {st.search_rounds} / {limits?.max_slot_search_rounds ?? "?"}
          </dd>
          <dt>Thử lại khi slot bị chiếm</dt>
          <dd>{st.conflict_retries}</dd>
        </dl>
      </div>
      <div className="section">
        <h3>Các lần trích xuất</h3>
        {st.attempts.length ? (
          <table className="grid">
            <thead>
              <tr>
                <th>Flow</th>
                <th>Lần</th>
                <th>Kết quả</th>
                <th>Mã lỗi</th>
              </tr>
            </thead>
            <tbody>
              {st.attempts.map((a, i) => (
                <tr key={i}>
                  <td>{a.flow}</td>
                  <td>{a.attempt_no === 0 ? "lần đầu" : `sửa lần ${a.attempt_no}`}</td>
                  <td>{a.passed ? <span className="ok">đạt</span> : <span className="bad">không đạt</span>}</td>
                  <td>
                    {a.issue_codes.length
                      ? a.issue_codes.map((c) => (
                          <Chip key={c} code={c} resolution={a.llm_fixable_codes.includes(c) ? "llm_fixable" : "ask_user"} />
                        ))
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">Chưa có.</p>
        )}
      </div>
      <div className="section">
        <h3>Lỗi hiện tại</h3>
        {st.issues.length ? (
          <table className="grid">
            <thead>
              <tr>
                <th>Lớp</th>
                <th>Trường</th>
                <th>Mã</th>
                <th>Mô tả</th>
                <th>Ai xử lý</th>
              </tr>
            </thead>
            <tbody>
              {st.issues.map((i, n) => (
                <tr key={n}>
                  <td>{i.layer}</td>
                  <td>{i.field ?? "—"}</td>
                  <td>
                    <Chip code={i.code} resolution={i.resolution} />
                  </td>
                  <td>
                    {i.message}
                    {i.suggestion && (
                      <>
                        <br />
                        <span className="muted">Gợi ý cho LLM: {i.suggestion}</span>
                      </>
                    )}
                  </td>
                  <td className={`res-${i.resolution}`}>{RESOLUTION_LABEL[i.resolution]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">Không có.</p>
        )}
      </div>
      <div className="section">
        <h3>Quy đổi ngày giờ (code)</h3>
        {w ? (
          <dl className="kv">
            <dt>Cụm người dùng nói</dt>
            <dd>
              {[st.booking_request?.date_expression, st.booking_request?.time_expression].filter(Boolean).join(" · ") ||
                "—"}
            </dd>
            <dt>Khoảng ngày</dt>
            <dd>
              {w.date_from}
              {w.date_to !== w.date_from && ` → ${w.date_to}`}
            </dd>
            <dt>Khung giờ</dt>
            <dd>{w.time_from && w.time_to ? `${w.time_from.slice(0, 5)}–${w.time_to.slice(0, 5)}` : "cả ngày"}</dd>
            <dt>Luật áp dụng</dt>
            <dd>
              {w.rules.map((r) => (
                <Chip key={r} code={r} />
              ))}
            </dd>
          </dl>
        ) : st.date_issues.length ? (
          st.date_issues.map((i, n) => (
            <div key={n}>
              <Chip code={i.code} resolution={i.resolution} /> {i.message}
            </div>
          ))
        ) : (
          <p className="muted">Không áp dụng.</p>
        )}
      </div>
      <div className="section">
        <h3>Cờ STT (từ nhận dạng kém)</h3>
        {st.stt_flags.length ? (
          st.stt_flags.map((f, i) => <Chip key={i} code={`${f.text} · ${Math.round(f.probability * 100)}%`} />)
        ) : (
          <p className="muted">Không có.</p>
        )}
      </div>
    </div>
  );
}

function ToolsTab({ st }: { st: AgentState }) {
  return (
    <div className="tab-body">
      <div className="section">
        <div className="rule-box">
          Tool đọc có thể do LLM gọi (trả lời câu hỏi). Tool ghi (<Chip code="save_patient_report" />,{" "}
          <Chip code="create_appointment" />) không bao giờ được giao cho LLM — chỉ code gọi sau khi validator đạt và bạn
          xác nhận; sau đó luôn đọc lại để xác minh.
        </div>
      </div>
      <div className="section">
        {st.tool_calls.length ? (
          <table className="grid">
            <thead>
              <tr>
                <th>#</th>
                <th>Tool</th>
                <th>Ai gọi</th>
                <th>Status</th>
                <th>Lỗi</th>
                <th>Latency</th>
                <th>Nguồn</th>
              </tr>
            </thead>
            <tbody>
              {st.tool_calls.map((t, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <Chip code={t.tool} />
                  </td>
                  <td>{t.called_by === "llm" ? "LLM" : "code"}</td>
                  <td>{t.status === "ok" ? <span className="ok">ok</span> : <span className="bad">{t.status}</span>}</td>
                  <td>{t.error_code ?? "—"}</td>
                  <td>{t.latency_ms} ms</td>
                  <td>
                    {t.source_ids.map((s) => (
                      <Chip key={s} code={s} />
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">Lượt này chưa gọi tool nào.</p>
        )}
      </div>
    </div>
  );
}

function StateTab({ st }: { st: AgentState }) {
  return (
    <div className="tab-body">
      <div className="section">
        <dl className="kv">
          <dt>Intent</dt>
          <dd>
            {st.intent ?? "—"} <span className="muted">({st.intent_source ?? "—"})</span>
          </dd>
          <dt>Việc chờ làm sau</dt>
          <dd>{st.pending_intents.join(", ") || "—"}</dd>
          <dt>An toàn</dt>
          <dd>{st.safety}</dd>
          <dt>Nguồn đầu vào</dt>
          <dd>{st.input_source}</dd>
          <dt>Câu trả lời bổ sung</dt>
          <dd>{st.user_replies.join(" | ") || "—"}</dd>
          <dt>Kết quả</dt>
          <dd>{st.outcome ? `${st.outcome.kind}${st.outcome.code ? ` (${st.outcome.code})` : ""}` : "chưa kết thúc"}</dd>
        </dl>
      </div>
      <div className="section">
        <h3>Toàn bộ state (JSON)</h3>
        <pre className="json">{JSON.stringify(st, null, 2)}</pre>
      </div>
    </div>
  );
}

/** Panel "Cách agent ra quyết định": từng node đã chạy, lỗi validator và cách xử lý, tool đã gọi, state đầy đủ. */
export function Inspector({ turns, state, limits }: { turns: Turn[]; state: AgentState | null; limits?: Meta["limits"] }) {
  const [tab, setTab] = useState<Tab>("timeline");
  return (
    <section className="panel inspector" aria-label="Cách agent ra quyết định">
      <div className="panel-head">
        <h2>Cách agent ra quyết định</h2>
        <div className="legend">
          <span>
            <i className="k code" />
            Code
          </span>
          <span>
            <i className="k llm" />
            LLM
          </span>
          <span>
            <i className="k hitl" />
            Chờ người dùng
          </span>
          <span>
            <i className="k tool" />
            Tool / backend
          </span>
          <span>
            <i className="k stop" />
            Dừng an toàn
          </span>
        </div>
      </div>
      <nav className="tabs" role="tablist">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            className={`tab${tab === id ? " active" : ""}`}
            role="tab"
            aria-selected={tab === id}
            type="button"
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "timeline" && <Timeline turns={turns} />}
      {tab !== "timeline" && !state && <Empty text="Chưa có dữ liệu." />}
      {tab === "validation" && state && <ValidationTab st={state} limits={limits} />}
      {tab === "tools" && state && <ToolsTab st={state} />}
      {tab === "state" && state && <StateTab st={state} />}
    </section>
  );
}
