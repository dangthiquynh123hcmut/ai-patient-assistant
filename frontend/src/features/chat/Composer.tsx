import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import type { InterruptKind, SttLanguage } from "../../api/types";
import { errorText, useToast } from "../../components/Toast";

const PLACEHOLDER: Record<InterruptKind, string> = {
  clarify: "Trả lời câu hỏi của agent…",
  choose_slot: "Ví dụ: cái thứ hai · 9 giờ · bác sĩ Bình · giờ khác buổi chiều",
  confirm_booking: "Ví dụ: xác nhận · đổi sang buổi chiều · hủy",
  confirm_report: "Ví dụ: xác nhận · hủy",
};

const LANGUAGE_LABEL: Record<SttLanguage, string> = { en: "English", vi: "Tiếng Việt", auto: "Tự nhận" };
const LANGUAGE_KEY = "pa.sttLanguage";

function storedLanguage(): SttLanguage | null {
  try {
    const v = localStorage.getItem(LANGUAGE_KEY);
    return v === "en" || v === "vi" || v === "auto" ? v : null;
  } catch {
    return null;
  }
}

interface Props {
  pendingKind: InterruptKind | null;
  busy: boolean;
  sttEnabled: boolean;
  defaultLanguage: SttLanguage;
  onSend: (text: string) => void;
  onAudio: (audio: Blob, filename: string, language: SttLanguage) => void;
}

export function Composer({ pendingKind, busy, sttEnabled, defaultLanguage, onSend, onAudio }: Props) {
  const toast = useToast();
  const [text, setText] = useState("");
  // Ngôn ngữ nói: nhớ lựa chọn của người dùng trên trình duyệt này, mặc định theo cấu hình server.
  const [language, setLanguage] = useState<SttLanguage>(() => storedLanguage() ?? defaultLanguage);

  function chooseLanguage(value: SttLanguage) {
    setLanguage(value);
    try {
      localStorage.setItem(LANGUAGE_KEY, value);
    } catch {
      /* trình duyệt chặn lưu trữ: chỉ áp dụng cho phiên này */
    }
  }
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
        onAudio(new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "ghi-am.webm", language);
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
          <select
            className="stt-language"
            value={language}
            onChange={(e) => chooseLanguage(e.target.value as SttLanguage)}
            disabled={recording}
            title="Ngôn ngữ bạn sẽ nói (chọn đúng giúp nhận dạng nhanh và chính xác hơn)"
            aria-label="Ngôn ngữ ghi âm"
          >
            {(Object.keys(LANGUAGE_LABEL) as SttLanguage[]).map((l) => (
              <option key={l} value={l}>
                {LANGUAGE_LABEL[l]}
              </option>
            ))}
          </select>
          <button
            className={`btn icon${recording ? " recording" : ""}`}
            type="button"
            onClick={toggleRecording}
            title={sttEnabled ? "Ghi âm (STT)" : "STT đang tắt"}
            aria-label="Ghi âm"
          >
            🎙
          </button>
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
