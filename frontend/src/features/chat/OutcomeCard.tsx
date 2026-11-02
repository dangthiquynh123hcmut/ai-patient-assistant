import type { AgentResponse } from "../../api/types";
import { OUTCOME_TITLE } from "../../lib/format";

export function OutcomeCard({ response: r }: { response: AgentResponse }) {
  // Chỉ liệt kê mã nguồn khi câu trả lời chưa tự nhắc tới.
  const showSources = r.source_ids.length > 0 && !r.source_ids.every((id) => r.text.includes(id));
  return (
    <div className={`card done ${r.kind}`}>
      <h3>{OUTCOME_TITLE[r.kind] ?? r.kind}</h3>
      <p className="text">{r.text}</p>
      {showSources && (
        <p className="muted">
          Nguồn / mã bản ghi:{" "}
          {r.source_ids.map((s) => (
            <span key={s} className="code-chip">
              {s}
            </span>
          ))}
        </p>
      )}
    </div>
  );
}
