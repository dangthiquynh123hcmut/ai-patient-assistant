import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../../api/client";
import type { ChatMessage, ConversationSummary } from "../../api/types";
import { useAuth } from "../../auth/AuthProvider";
import { errorText, useToast } from "../../components/Toast";
import { MESSAGE_KIND, dateTime } from "../../lib/format";
import { StaffTopBar } from "./StaffTopBar";

export function ConversationsPage() {
  const { meta } = useAuth();
  const toast = useToast();
  const [patientId, setPatientId] = useState("");
  const [conversations, setConversations] = useState<ConversationSummary[] | null>(null);
  const [current, setCurrent] = useState<ConversationSummary | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const box = useRef<HTMLDivElement>(null);
  const patients = meta?.patients ?? [];
  const patientName = (id: string) => patients.find((p) => p.id === id)?.label ?? id;

  const loadList = useCallback(async () => {
    try {
      setConversations(await api.staffConversations(patientId || undefined));
    } catch (err) {
      toast(errorText(err));
    }
  }, [patientId, toast]);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [messages]);

  async function open(c: ConversationSummary) {
    setCurrent(c);
    try {
      setMessages(await api.staffConversation(c.thread_id));
    } catch (err) {
      toast(errorText(err));
    }
  }

  return (
    <>
      <StaffTopBar />
      <main className="staff-layout">
        <section className="panel" aria-label="Danh sách hội thoại">
          <div className="panel-head">
            <h2>Hội thoại</h2>
            <button className="btn ghost" type="button" onClick={() => void loadList()}>
              Tải lại
            </button>
          </div>
          <div className="filter">
            <select value={patientId} onChange={(e) => setPatientId(e.target.value)} aria-label="Lọc theo bệnh nhân">
              <option value="">Tất cả bệnh nhân</option>
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <ul className="conv-list">
            {conversations === null && <li className="muted pad">Đang tải…</li>}
            {conversations?.length === 0 && <li className="muted pad">Chưa có hội thoại nào.</li>}
            {conversations?.map((c) => (
              <li key={c.thread_id} className={c.thread_id === current?.thread_id ? "active" : ""}>
                <button type="button" onClick={() => void open(c)}>
                  <span className="title">{c.title}</span>
                  <span className="muted">
                    {patientName(c.patient_id)} · {c.message_count} tin ·{" "}
                    {MESSAGE_KIND[c.last_kind ?? ""] ?? c.last_kind ?? ""} · {dateTime(c.updated_at)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel" aria-label="Nội dung hội thoại">
          <div className="panel-head">
            <h2>{current?.title ?? "Chọn một hội thoại"}</h2>
            {current && (
              <span className="muted">
                {patientName(current.patient_id)} · {current.thread_id}
              </span>
            )}
          </div>
          <div className="messages" ref={box}>
            {!current && <p className="muted pad">Chưa chọn hội thoại.</p>}
            {current &&
              messages.map((m) => (
                <div key={m.id} className={`msg ${m.role === "user" ? "user" : "agent"}`}>
                  <span className="meta">
                    {m.role === "user" ? "Bệnh nhân" : "Trợ lý"} ·{" "}
                    {m.role === "user" ? m.action : (MESSAGE_KIND[m.action] ?? m.action)}
                    {m.intent && <span className="intent">intent: {m.intent}</span>} · {dateTime(m.created_at)}
                  </span>
                  {m.text}
                </div>
              ))}
          </div>
        </section>
      </main>
    </>
  );
}
