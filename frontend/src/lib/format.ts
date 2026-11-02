// Nhãn hiển thị và định dạng thời gian — dùng chung cho các trang.
import type { InterruptKind, Origin, OutcomeKind, Resolution, Role, Validation } from "../api/types";

export const ROLE_LABEL: Record<Role, string> = { patient: "Bệnh nhân", staff: "Nhân viên", admin: "Quản trị" };

export const INTERRUPT_TITLE: Record<InterruptKind, string> = {
  clarify: "Agent cần hỏi thêm",
  confirm_report: "Bản nháp báo cáo — cần bạn xác nhận",
  choose_slot: "Chọn lịch khám",
  confirm_booking: "Xác nhận lịch hẹn",
};

export const OUTCOME_TITLE: Record<OutcomeKind, string> = {
  report_saved: "Đã lưu báo cáo",
  appointment_booked: "Đã đặt lịch",
  history: "Báo cáo trước",
  appointments: "Lịch hẹn",
  patient_info: "Thông tin bệnh nhân",
  answer: "Trả lời",
  chat: "Trợ lý",
  cancelled: "Đã dừng",
  handoff: "Chuyển nhân viên xem lại",
  safety: "Cảnh báo an toàn",
  error: "Lỗi",
};

/** Kind của tin nhắn trợ lý trong lịch sử (interrupt hoặc outcome) -> mô tả ngắn. */
export const MESSAGE_KIND: Record<string, string> = {
  clarify: "hỏi lại",
  confirm_report: "chờ xác nhận báo cáo",
  choose_slot: "chờ chọn lịch",
  confirm_booking: "chờ xác nhận lịch",
  report_saved: "đã lưu báo cáo",
  appointment_booked: "đã đặt lịch",
  history: "báo cáo trước",
  appointments: "lịch hẹn",
  patient_info: "thông tin bệnh nhân",
  answer: "trả lời",
  chat: "trò chuyện",
  cancelled: "đã dừng",
  handoff: "chuyển nhân viên",
  safety: "cảnh báo an toàn",
  error: "lỗi",
};

export const ORIGIN_LABEL: Record<Origin, string> = {
  user_said: "người dùng nói",
  user_edit: "người dùng sửa",
  backend: "backend",
  system_default: "mặc định hệ thống",
  ai_summary: "AI tóm tắt",
};

export const VALIDATION_LABEL: Record<Validation, string> = {
  passed: "đạt",
  failed: "lỗi",
  missing: "thiếu",
  uncertain: "chưa chắc",
};

export const RESOLUTION_LABEL: Record<Resolution, string> = {
  llm_fixable: "LLM sửa",
  ask_user: "hỏi người dùng",
  block: "chặn",
};

const WEEKDAYS = ["Chủ Nhật", "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy"];

/** "Thứ Ba, 03/11/2026 · 09:00" — giữ đúng giờ địa phương của server (chuỗi ISO có offset), không đổi theo trình duyệt. */
export function slotWhen(iso: string): string {
  const [d = "", t = ""] = iso.split("T");
  const [y = 0, m = 1, day = 1] = d.split("-").map(Number);
  const weekday = WEEKDAYS[new Date(Date.UTC(y, m - 1, day)).getUTCDay()];
  return `${weekday}, ${String(day).padStart(2, "0")}/${String(m).padStart(2, "0")}/${y} · ${t.slice(0, 5)}`;
}

export function dateTime(iso: string): string {
  return new Date(iso).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
}

/** Thread id ngẫu nhiên đủ dài để không đoán được / không trùng hội thoại của người khác. */
export function newThreadId(): string {
  const random = crypto.getRandomValues(new Uint32Array(2)).join("");
  return `t-${Date.now().toString(36)}-${random}`;
}

export function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  return Array.isArray(value) ? value.join(", ") : String(value);
}
