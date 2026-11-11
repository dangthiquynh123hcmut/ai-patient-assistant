import { useCallback, useRef, useState } from "react";

import { api } from "../../api/client";
import type {
  AgentResponse,
  AgentState,
  Decision,
  InterruptPayload,
  StepTrace,
  SttFlag,
  SttLanguage,
  TranscriptReview,
  TurnResult,
} from "../../api/types";
import { errorText } from "../../components/Toast";
import { dateTime, newConversationId } from "../../lib/format";

export type ChatEntry =
  | { type: "user"; id: number; text: string; meta: string }
  | { type: "interrupt"; id: number; payload: InterruptPayload; answered: boolean }
  | { type: "outcome"; id: number; response: AgentResponse }
  // Câu trả lời của trợ lý ở các lượt trước, đọc lại từ lịch sử (chỉ để xem, không còn nút bấm).
  | { type: "history"; id: number; kind: string; text: string; at: string };

/** Omit phân phối trên từng nhánh của union (Omit thường gộp union lại, mất các trường riêng). */
type NewEntry = ChatEntry extends infer E ? (E extends ChatEntry ? Omit<E, "id"> : never) : never;

export interface Turn {
  said: string;
  steps: StepTrace[];
}

/** Trạng thái một hội thoại với agent: tin nhắn, câu hỏi đang chờ, diễn biến từng lượt, state backend. */
export function useChat(onError: (message: string) => void) {
  const [conversationId, setConversationId] = useState(newConversationId);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [pending, setPending] = useState<InterruptPayload | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [agentState, setAgentState] = useState<AgentState | null>(null);
  const [busy, setBusy] = useState(false);
  const [version, setVersion] = useState(0); // tăng sau mỗi lượt -> danh sách lịch sử tải lại
  const nextId = useRef(0);
  const busyRef = useRef(false); // chặn gửi trùng trước khi React kịp render lại

  const push = useCallback((entry: NewEntry) => {
    setEntries((list) => [...list, { ...entry, id: nextId.current++ } as ChatEntry]);
  }, []);

  const run = useCallback(
    async (said: string, call: () => Promise<TurnResult>) => {
      busyRef.current = true;
      setBusy(true);
      try {
        const res = await call();
        setTurns((list) => [...list, { said, steps: res.steps }]);
        if (res.status === "waiting_user" && res.interrupt) {
          setPending(res.interrupt);
          push({ type: "interrupt", payload: res.interrupt, answered: false });
        } else {
          setPending(null);
          if (res.response) push({ type: "outcome", response: res.response });
        }
        setAgentState(await api.state(res.conversation_id).catch(() => null));
      } catch (err) {
        onError(`Lỗi: ${errorText(err)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
        setVersion((v) => v + 1);
      }
    },
    [onError, push],
  );

  const markAnswered = useCallback(() => {
    setEntries((list) => list.map((e) => (e.type === "interrupt" && !e.answered ? { ...e, answered: true } : e)));
  }, []);

  /** Trả lời câu hỏi đang chờ (nút bấm hoặc lời nói). */
  const resume = useCallback(
    async (decision: Decision, label: string, meta = "bạn") => {
      if (busyRef.current) return;
      push({ type: "user", text: label, meta });
      markAnswered();
      await run(label, () => api.resume(conversationId, decision));
    },
    [markAnswered, push, run, conversationId],
  );

  /** Gửi một câu: là câu trả lời nếu agent đang chờ, ngược lại là yêu cầu mới. */
  const send = useCallback(
    async (raw: string, source: "text" | "voice" = "text", flags: SttFlag[] = []) => {
      const text = raw.trim();
      if (!text || busyRef.current) return;
      if (pending) {
        await resume({ action: "reply", text }, text, source === "voice" ? "bạn (giọng nói) trả lời" : "bạn trả lời");
        return;
      }
      push({ type: "user", text, meta: source === "voice" ? "bạn · giọng nói" : "bạn" });
      await run(text, () => api.sendMessage(conversationId, text, source, flags));
    },
    [pending, push, resume, run, conversationId],
  );

  /** Ghi âm xong: chuyển thành chữ rồi gửi luôn (kèm cờ từ STT không chắc để agent hỏi lại nếu cần). */
  const transcribe = useCallback(
    async (audio: Blob, filename: string, language?: SttLanguage) => {
      if (busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      let review: TranscriptReview | null = null;
      try {
        review = await api.transcribe(audio, filename, language);
      } catch (err) {
        onError(`STT lỗi: ${errorText(err)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
      if (!review) return;
      if (!review.text.trim()) {
        onError("Không nghe rõ, bạn thử ghi âm lại nhé.");
        return;
      }
      await send(review.text, "voice", review.flags);
    },
    [onError, send],
  );

  /** Mở lại một hội thoại cũ: hiện lại tin nhắn; nếu agent còn đang chờ trả lời thì hiện câu hỏi đó để nói tiếp. */
  const open = useCallback(
    async (id: string) => {
      if (busyRef.current) return;
      busyRef.current = true;
      setBusy(true);
      try {
        const [messages, waiting] = await Promise.all([api.myConversation(id), api.pending(id)]);
        const restored: ChatEntry[] = messages.map((m, i) =>
          m.role === "user"
            ? { type: "user", id: i, text: m.text, meta: `bạn · ${dateTime(m.created_at)}` }
            : { type: "history", id: i, kind: m.action, text: m.text, at: m.created_at },
        );
        // Câu hỏi cuối đang chờ: thay bản chỉ-đọc bằng thẻ có nút để trả lời tiếp.
        const last = restored[restored.length - 1];
        if (waiting && last?.type === "history" && last.kind === waiting.kind) restored.pop();
        if (waiting) restored.push({ type: "interrupt", id: restored.length, payload: waiting, answered: false });
        nextId.current = restored.length;
        setConversationId(id);
        setEntries(restored);
        setPending(waiting);
        setTurns([]);
        setAgentState(await api.state(id).catch(() => null));
      } catch (err) {
        onError(`Không mở được hội thoại: ${errorText(err)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [onError],
  );

  const reset = useCallback(() => {
    setConversationId(newConversationId());
    setEntries([]);
    setPending(null);
    setTurns([]);
    setAgentState(null);
  }, []);

  return {
    conversationId,
    entries,
    pending,
    turns,
    agentState,
    busy,
    version,
    send,
    resume,
    transcribe,
    open,
    reset,
  };
}
