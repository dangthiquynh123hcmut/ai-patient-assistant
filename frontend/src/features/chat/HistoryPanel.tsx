import type { ConversationSummary } from "../../api/types";
import { MESSAGE_KIND, dateTime } from "../../lib/format";

interface Props {
  conversations: ConversationSummary[] | null;
  currentId: string;
  busy: boolean;
  onOpen: (conversationId: string) => void;
  onNew: () => void;
}

/** Danh sách hội thoại trước đây của bệnh nhân (mới nhất trước): bấm để xem lại và nói tiếp. */
export function HistoryPanel({ conversations, currentId, busy, onOpen, onNew }: Props) {
  return (
    <section className="panel history" aria-label="Lịch sử hội thoại">
      <div className="panel-head">
        <h2>Lịch sử</h2>
        <button className="btn ghost" type="button" onClick={onNew} disabled={busy}>
          + Mới
        </button>
      </div>
      <ul className="conv-list">
        {conversations === null && <li className="muted pad">Đang tải…</li>}
        {conversations?.length === 0 && <li className="muted pad">Chưa có hội thoại nào.</li>}
        {conversations?.map((c) => (
          <li key={c.conversation_id} className={c.conversation_id === currentId ? "active" : ""}>
            <button type="button" disabled={busy} onClick={() => onOpen(c.conversation_id)} title={c.title}>
              <span className="title">{c.title}</span>
              <span className="muted">
                {MESSAGE_KIND[c.last_kind ?? ""] ?? c.last_kind ?? ""} · {dateTime(c.updated_at)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
