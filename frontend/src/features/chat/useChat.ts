import { useCallback, useRef, useState } from "react";

import { api } from "../../api/client";
import type {
  AgentResponse,
  AgentState,
  Decision,
  InterruptPayload,
  StepTrace,
  SttFlag,
  TranscriptReview,
  TurnResult,
} from "../../api/types";
import { errorText } from "../../components/Toast";
import { newThreadId } from "../../lib/format";

export type ChatEntry =
  | { type: "user"; id: number; text: string; meta: string }
  | { type: "interrupt"; id: number; payload: InterruptPayload; answered: boolean }
  | { type: "outcome"; id: number; response: AgentResponse }
  | { type: "transcript"; id: number; review: TranscriptReview; done: boolean };

/** Omit phân phối trên từng nhánh của union (Omit thường gộp union lại, mất các trường riêng). */
type NewEntry = ChatEntry extends infer E ? (E extends ChatEntry ? Omit<E, "id"> : never) : never;

export interface Turn {
  said: string;
  steps: StepTrace[];
}

/** Trạng thái một hội thoại với agent: tin nhắn, câu hỏi đang chờ, diễn biến từng lượt, state backend. */
export function useChat(onError: (message: string) => void) {
  const [thread, setThread] = useState(newThreadId);
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [pending, setPending] = useState<InterruptPayload | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [agentState, setAgentState] = useState<AgentState | null>(null);
  const [busy, setBusy] = useState(false);
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
        setAgentState(await api.state(res.thread_id).catch(() => null));
      } catch (err) {
        onError(`Lỗi: ${errorText(err)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
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
      await run(label, () => api.resume(thread, decision));
    },
    [markAnswered, push, run, thread],
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
      push({ type: "user", text, meta: source === "voice" ? "bạn · giọng nói (đã duyệt transcript)" : "bạn" });
      await run(text, () => api.sendMessage(thread, text, source, flags));
    },
    [pending, push, resume, run, thread],
  );

  const transcribe = useCallback(
    async (audio: Blob, filename: string) => {
      busyRef.current = true;
      setBusy(true);
      try {
        push({ type: "transcript", review: await api.transcribe(audio, filename), done: false });
      } catch (err) {
        onError(`STT lỗi: ${errorText(err)}`);
      } finally {
        busyRef.current = false;
        setBusy(false);
      }
    },
    [onError, push],
  );

  const closeTranscript = useCallback((id: number, remove: boolean) => {
    setEntries((list) =>
      remove
        ? list.filter((e) => e.id !== id)
        : list.map((e) => (e.id === id && e.type === "transcript" ? { ...e, done: true } : e)),
    );
  }, []);

  const reset = useCallback(() => {
    setThread(newThreadId());
    setEntries([]);
    setPending(null);
    setTurns([]);
    setAgentState(null);
  }, []);

  return { thread, entries, pending, turns, agentState, busy, send, resume, transcribe, closeTranscript, reset };
}
