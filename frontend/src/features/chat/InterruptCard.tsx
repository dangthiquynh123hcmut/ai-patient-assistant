import { useState } from "react";

import type { Decision, InterruptPayload } from "../../api/types";
import { useAuth } from "../../auth/AuthProvider";
import { useToast } from "../../components/Toast";
import { INTERRUPT_TITLE, slotWhen } from "../../lib/format";
import { ExplanationTable, REPORT_EDITABLE, changedFields } from "./ExplanationTable";

interface Props {
  payload: InterruptPayload;
  answered: boolean;
  busy: boolean;
  onDecide: (decision: Decision, label: string) => void;
}

/** Bỏ phần lặp lại mà thẻ đã hiển thị bằng nút (danh sách slot, câu hướng dẫn "Nói xác nhận..."). */
function cardText(p: InterruptPayload): string {
  if (p.kind === "choose_slot") {
    return p.message
      .split("\n")
      .filter((l) => !/^\d+\. /.test(l) && !l.startsWith("Bạn chọn số"))
      .join("\n");
  }
  if (p.kind === "confirm_report" || p.kind === "confirm_booking") return p.message.replace(/ ?Nói “xác nhận”.*$/, "");
  return p.message;
}

export function InterruptCard({ payload: p, answered, busy, onDecide }: Props) {
  const { meta } = useAuth();
  const toast = useToast();
  const [otherTime, setOtherTime] = useState("");
  const [edits, setEdits] = useState<Record<string, string> | null>(null);
  const disabled = answered || busy;
  const deptName = (code: string) => meta?.departments.find((d) => d.code === code)?.name ?? code;

  function startOrSubmitEdit() {
    if (edits === null) {
      const initial: Record<string, string> = {};
      for (const r of p.explanations) if (REPORT_EDITABLE[r.field]) initial[r.field] = String(r.value ?? "");
      setEdits(initial);
      return;
    }
    const fields = changedFields(p.explanations, edits);
    if (!Object.keys(fields).length) {
      toast("Chưa có trường nào thay đổi.");
      return;
    }
    const label = Object.entries(fields)
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");
    onDecide({ action: "edit", fields }, `Sửa: ${label}`);
  }

  return (
    <div className={`card${answered ? " answered" : ""}`}>
      <h3>{INTERRUPT_TITLE[p.kind] ?? p.kind}</h3>
      <p className="text">{cardText(p)}</p>

      {p.kind === "choose_slot" && (
        <>
          <div className="slots">
            {p.options.map((o) => (
              <button
                key={o.option_no}
                className="slot"
                type="button"
                disabled={disabled}
                onClick={() => onDecide({ action: "choose", option_no: o.option_no }, `Chọn lịch số ${o.option_no}`)}
              >
                <span className="no">{o.option_no}</span>
                <span>
                  <span className="when">{slotWhen(o.slot.start_at)}</span>
                  <br />
                  <span className="who">
                    {o.slot.doctor_name} · {deptName(o.slot.department_code)} · mã {o.slot.slot_id}
                  </span>
                </span>
              </button>
            ))}
          </div>
          <form
            className="inline-row"
            onSubmit={(e) => {
              e.preventDefault();
              const text = otherTime.trim();
              if (text) onDecide({ action: "other_time", text }, `Giờ khác: ${text}`);
            }}
          >
            <input
              value={otherTime}
              onChange={(e) => setOtherTime(e.target.value)}
              placeholder="Muốn giờ/ngày khác? Ví dụ: buổi chiều, thứ năm"
              disabled={disabled}
            />
            <button className="btn" type="submit" disabled={disabled}>
              Tìm lại
            </button>
          </form>
        </>
      )}

      {p.choices.length > 0 && (
        <div className="choices">
          {p.choices.map((c, i) => (
            <button
              key={c.label}
              className={`btn${i === 0 ? " suggested" : ""}`}
              type="button"
              disabled={disabled}
              onClick={() => onDecide({ action: "reply", text: c.text }, c.label)}
            >
              {c.label}
            </button>
          ))}
        </div>
      )}

      {p.explanations.length > 0 && (
        <ExplanationTable
          rows={p.explanations}
          edits={edits ?? undefined}
          onEdit={(field, value) => setEdits((cur) => ({ ...(cur ?? {}), [field]: value }))}
        />
      )}

      <div className="actions">
        {(p.kind === "confirm_report" || p.kind === "confirm_booking") && (
          <button
            className="btn primary"
            type="button"
            disabled={disabled}
            onClick={() => onDecide({ action: "confirm" }, "Xác nhận")}
          >
            Xác nhận
          </button>
        )}
        {p.kind === "confirm_report" && (
          <button className="btn" type="button" disabled={disabled} onClick={startOrSubmitEdit}>
            {edits === null ? "Sửa bản nháp" : "Gửi chỉnh sửa"}
          </button>
        )}
        <button
          className="btn danger"
          type="button"
          disabled={disabled}
          onClick={() => onDecide({ action: "cancel" }, "Hủy")}
        >
          Hủy
        </button>
      </div>
    </div>
  );
}
