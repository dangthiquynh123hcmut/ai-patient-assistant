import { useEffect, useRef } from "react";

import { useAuth } from "../../auth/AuthProvider";
import { TopBar } from "../../components/TopBar";
import { useToast } from "../../components/Toast";
import { slotWhen } from "../../lib/format";
import { Inspector } from "../inspector/Inspector";
import { Composer } from "./Composer";
import { InterruptCard } from "./InterruptCard";
import { OutcomeCard } from "./OutcomeCard";
import { TranscriptCard } from "./TranscriptCard";
import { useChat } from "./useChat";

const SAMPLES = [
  "tôi muốn tra cứu thông tin bệnh nhân",
  "Cho tôi đặt lịch khoa phục hồi chức năng thứ ba tuần sau buổi sáng",
  "Tuần này tôi đau mức 5 trên 10, đã tập 4 ngày, thấy đau tăng khi gập gối",
  "tuần này tôi đau mức 3",
  "đặt lịch tái khám thứ sáu này buổi chiều",
  "sau khi tập gập gối có nên chườm lạnh bao lâu",
  "xem lịch hẹn của tôi",
  "tôi đau ngực khó thở",
];

export function ChatPage() {
  const { meta } = useAuth();
  const toast = useToast();
  const chat = useChat(toast);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (list.current) list.current.scrollTop = list.current.scrollHeight;
  }, [chat.entries]);

  return (
    <>
      <TopBar
        title="Trợ lý bệnh nhân AI"
        subtitle="Dữ liệu giả lập · dự án học tập · không chẩn đoán"
        actions={
          <button className="btn ghost" type="button" onClick={chat.reset}>
            Hội thoại mới
          </button>
        }
      >
        {meta && (
          <div className="badges">
            <span className="badge on">LLM: {meta.llm_model}</span>
            <span className={`badge ${meta.stt_enabled ? "on" : "off"}`}>
              STT: {meta.stt_enabled ? meta.stt_model : "tắt"}
            </span>
            <span className="badge">Hôm nay: {slotWhen(meta.now).split(" · ")[0]}</span>
            <span className="badge">Khoa: {meta.departments.map((d) => d.name).join(", ")}</span>
          </div>
        )}
      </TopBar>

      <main className="layout">
        <section className="panel chat" aria-label="Hội thoại">
          <div className="panel-head">
            <h2>Hội thoại</h2>
            <span className="muted">thread {chat.thread}</span>
          </div>
          <div className="messages" ref={list} aria-live="polite">
            {chat.entries.length === 0 && (
              <div className="empty">
                <p className="empty-title">Bắt đầu bằng một câu nói hoặc chọn câu mẫu bên dưới.</p>
                <p className="muted">
                  Agent sẽ hỏi lại khi thiếu thông tin, đưa bản nháp để bạn xác nhận trước khi lưu, và panel bên phải cho
                  thấy từng bước nó đã quyết định thế nào.
                </p>
              </div>
            )}
            {chat.entries.map((e) => {
              switch (e.type) {
                case "user":
                  return (
                    <div key={e.id} className="msg user">
                      <span className="meta">{e.meta}</span>
                      {e.text}
                    </div>
                  );
                case "interrupt":
                  return (
                    <InterruptCard
                      key={e.id}
                      payload={e.payload}
                      answered={e.answered}
                      busy={chat.busy}
                      onDecide={(decision, label) => void chat.resume(decision, label)}
                    />
                  );
                case "outcome":
                  return <OutcomeCard key={e.id} response={e.response} />;
                case "transcript":
                  return (
                    <TranscriptCard
                      key={e.id}
                      review={e.review}
                      done={e.done}
                      onSend={(text, flags) => {
                        chat.closeTranscript(e.id, false);
                        void chat.send(text, "voice", flags);
                      }}
                      onDiscard={() => chat.closeTranscript(e.id, true)}
                    />
                  );
              }
            })}
          </div>

          <div className="samples">
            {SAMPLES.map((text) => (
              <button
                key={text}
                className="chip"
                type="button"
                title={text}
                disabled={chat.busy}
                onClick={() => void chat.send(text)}
              >
                {text}
              </button>
            ))}
          </div>

          <Composer
            pendingKind={chat.pending?.kind ?? null}
            busy={chat.busy}
            sttEnabled={meta?.stt_enabled ?? false}
            onSend={(text) => void chat.send(text)}
            onAudio={(audio, name) => void chat.transcribe(audio, name)}
          />
        </section>

        <Inspector turns={chat.turns} state={chat.agentState} limits={meta?.limits} />
      </main>
    </>
  );
}
