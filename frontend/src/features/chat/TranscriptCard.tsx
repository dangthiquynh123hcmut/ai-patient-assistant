import { useState } from "react";

import type { SttFlag, TranscriptReview } from "../../api/types";

interface Props {
  review: TranscriptReview;
  done: boolean;
  onSend: (text: string, flags: SttFlag[]) => void;
  onDiscard: () => void;
}

/** Tô vàng các từ quan trọng mà STT không chắc. */
function Highlighted({ text, flags }: { text: string; flags: SttFlag[] }) {
  const flagged = new Set(flags.map((f) => f.text.toLowerCase()));
  return (
    <>
      {text.split(/(\s+)/).map((word, i) =>
        flagged.has(word.toLowerCase().replace(/[.,!?]/g, "")) ? (
          <span key={i} className="flagged" title="Nhận dạng kém">
            {word}
          </span>
        ) : (
          word
        ),
      )}
    </>
  );
}

/** Người dùng duyệt/sửa transcript trước khi gửi (FR-01.3). */
export function TranscriptCard({ review, done, onSend, onDiscard }: Props) {
  const [text, setText] = useState(review.text);
  return (
    <div className={`card transcript${done ? " answered" : ""}`}>
      <h3>
        Kiểm tra transcript trước khi gửi <span className="badge">ngôn ngữ: {review.language}</span>
      </h3>
      <p className="muted">
        Đoạn tô vàng là từ quan trọng (số, ngày, giờ, khoa…) mà STT không chắc. Sửa nếu sai rồi bấm Gửi.
      </p>
      <div className="preview">
        <Highlighted text={review.text} flags={review.flags} />
      </div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} disabled={done} aria-label="Transcript" />
      <div className="actions">
        <button
          className="btn primary"
          type="button"
          disabled={done}
          // Từ người dùng đã sửa thì không còn là từ STT nghi sai -> bỏ cờ.
          onClick={() => onSend(text, review.flags.filter((f) => text.includes(f.text)))}
        >
          Gửi transcript
        </button>
        <button className="btn" type="button" disabled={done} onClick={onDiscard}>
          Bỏ
        </button>
      </div>
    </div>
  );
}
