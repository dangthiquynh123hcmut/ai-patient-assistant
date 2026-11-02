import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import type { InterruptKind } from "../../api/types";
import { errorText, useToast } from "../../components/Toast";

const PLACEHOLDER: Record<InterruptKind, string> = {
  clarify: "Trả lời câu hỏi của agent…",
  choose_slot: "Ví dụ: cái thứ hai · 9 giờ · bác sĩ Bình · giờ khác buổi chiều",
  confirm_booking: "Ví dụ: xác nhận · đổi sang buổi chiều · hủy",
  confirm_report: "Ví dụ: xác nhận · hủy",
};

interface Props {
  pendingKind: InterruptKind | null;
  busy: boolean;
  sttEnabled: boolean;
  onSend: (text: string) => void;
  onAudio: (audio: Blob, filename: string) => void;
}

export function Composer({ pendingKind, busy, sttEnabled, onSend, onAudio }: Props) {
  const toast = useToast();
  const [text, setText] = useState("");
  const [recording, setRecording] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);

  // Dừng micro nếu rời trang khi đang ghi âm.
  useEffect(() => () => recorder.current?.stream.getTracks().forEach((t) => t.stop()), []);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text);
    setText("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  async function toggleRecording() {
    if (!sttEnabled) {
      toast('STT đang tắt. Đặt PA_STT_ENABLED=true và cài: pip install -e ".[stt]"');
      return;
    }
    if (recorder.current?.state === "recording") {
      recorder.current.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        onAudio(new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "ghi-am.webm");
      };
      recorder.current = rec;
      rec.start();
      setRecording(true);
    } catch (err) {
      toast(`Không truy cập được micro: ${errorText(err)}`);
    }
  }

  return (
    <form className="composer" onSubmit={submit} autoComplete="off">
      {pendingKind && (
        <div className="pending-hint">
          Agent đang chờ bạn trả lời — nội dung gửi bây giờ sẽ là câu trả lời (có thể nói hoặc gõ tự do).
        </div>
      )}
      <div className="composer-row">
        <textarea
          rows={2}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={pendingKind ? PLACEHOLDER[pendingKind] : "Nhập yêu cầu mới, hoặc bấm 🎙 để nói…"}
          aria-label="Tin nhắn"
        />
        <div className="composer-actions">
          <button
            className={`btn icon${recording ? " recording" : ""}`}
            type="button"
            onClick={toggleRecording}
            title={sttEnabled ? "Ghi âm (STT)" : "STT đang tắt"}
            aria-label="Ghi âm"
          >
            🎙
          </button>
          <label className="btn icon" title="Tải file audio">
            ⤒
            <input
              type="file"
              accept="audio/*"
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  if (!sttEnabled) toast("STT đang tắt — server sẽ trả lỗi 503.");
                  onAudio(file, file.name);
                }
                e.target.value = "";
              }}
            />
          </label>
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Đang xử lý…" : "Gửi"}
          </button>
        </div>
      </div>
      {recording && (
        <div className="recording">
          <span className="dot" /> Đang ghi âm… bấm 🎙 lần nữa để dừng
        </div>
      )}
    </form>
  );
}
