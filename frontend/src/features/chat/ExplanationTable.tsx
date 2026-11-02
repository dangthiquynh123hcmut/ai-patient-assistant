import type { FieldExplanation } from "../../api/types";
import { ORIGIN_LABEL, VALIDATION_LABEL, display } from "../../lib/format";

/** Trường của bản nháp báo cáo mà người dùng được sửa trực tiếp, và kiểu ô nhập tương ứng. */
export const REPORT_EDITABLE: Record<string, { type: "number"; min: number; max: number } | { type: "text" } | { type: "select"; options: string[] }> = {
  pain_level: { type: "number", min: 0, max: 10 },
  exercise_days: { type: "number", min: 0, max: 7 },
  symptom_change: { type: "select", options: ["", "better", "no_change", "worse", "worse_during_activity"] },
  triggering_activity: { type: "text" },
};

interface Props {
  rows: FieldExplanation[];
  /** Có giá trị -> các trường sửa được hiện ô nhập; key là field, value là chuỗi đang nhập. */
  edits?: Record<string, string>;
  onEdit?: (field: string, value: string) => void;
}

export function ExplanationTable({ rows, edits, onEdit }: Props) {
  return (
    <table className="explain">
      <thead>
        <tr>
          <th>Trường</th>
          <th>Giá trị</th>
          <th>Trích dẫn</th>
          <th>Nguồn</th>
          <th>Kiểm tra</th>
          <th>Vì sao</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const conf = edits ? REPORT_EDITABLE[r.field] : undefined;
          const current = edits?.[r.field] ?? "";
          let value = <>{display(r.value)}</>;
          if (conf && onEdit) {
            value =
              conf.type === "select" ? (
                <select value={current} onChange={(e) => onEdit(r.field, e.target.value)} aria-label={r.label}>
                  {conf.options.map((o) => (
                    <option key={o} value={o}>
                      {o || "—"}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={conf.type}
                  value={current}
                  aria-label={r.label}
                  {...(conf.type === "number" ? { min: conf.min, max: conf.max } : {})}
                  onChange={(e) => onEdit(r.field, e.target.value)}
                />
              );
          }
          return (
            <tr key={r.field}>
              <td>{r.label}</td>
              <td className="value">{value}</td>
              <td className="quote">{r.evidence ? `“${r.evidence}”` : "—"}</td>
              <td>
                <span className={`pill ${r.origin}`}>{ORIGIN_LABEL[r.origin] ?? r.origin}</span>
              </td>
              <td>
                <span className={`pill ${r.validation}`}>{VALIDATION_LABEL[r.validation] ?? r.validation}</span>
              </td>
              <td>{r.explanation}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Các trường người dùng đã đổi so với bản nháp (chuỗi rỗng = xóa giá trị). */
export function changedFields(rows: FieldExplanation[], edits: Record<string, string>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const r of rows) {
    const conf = REPORT_EDITABLE[r.field];
    if (!conf || !(r.field in edits)) continue;
    const raw = edits[r.field] ?? "";
    const value = raw === "" ? null : conf.type === "number" ? Number(raw) : raw;
    if (String(value ?? "") !== String(r.value ?? "")) out[r.field] = value;
  }
  return out;
}
