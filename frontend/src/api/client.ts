// Client HTTP có kiểu. Xác thực bằng cookie HttpOnly `pa_session` (cùng origin qua nginx / Vite proxy),
// nên JavaScript không bao giờ thấy token.
import type {
  AgentState,
  ChatMessage,
  ConversationSummary,
  Decision,
  InterruptPayload,
  Meta,
  NewUser,
  Registration,
  SttFlag,
  SttLanguage,
  TranscriptReview,
  TurnResult,
  User,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Gọi khi API trả 401 (phiên hết hạn / bị thu hồi) — AuthProvider đăng ký để đưa về trang đăng nhập. */
let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

function detailOf(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // Lỗi validation của FastAPI: [{loc, msg, ...}]
    return detail.map((d) => (d as { msg?: string }).msg ?? JSON.stringify(d)).join("; ");
  }
  return fallback;
}

interface RequestOptions {
  method?: "GET" | "POST" | "PATCH";
  json?: unknown;
  body?: BodyInit;
  /** Không chuyển về trang đăng nhập khi 401 (dùng cho /auth/me lúc khởi động và /auth/login). */
  allowUnauthorized?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  let body = options.body;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }
  const res = await fetch(path, { method: options.method ?? "GET", headers, body, credentials: "same-origin" });
  if (res.status === 204) return undefined as T;
  const data: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401 && !options.allowUnauthorized) onUnauthorized();
    throw new ApiError(res.status, detailOf(data, res.statusText || `HTTP ${res.status}`));
  }
  return data as T;
}

const enc = encodeURIComponent;

export const api = {
  // ---- đăng nhập
  login: (username: string, password: string) =>
    request<User>("/v1/auth/login", { method: "POST", json: { username, password }, allowUnauthorized: true }),
  register: (registration: Registration) =>
    request<User>("/v1/auth/register", { method: "POST", json: registration, allowUnauthorized: true }),
  authOptions: () => request<{ allow_registration: boolean }>("/v1/auth/options", { allowUnauthorized: true }),
  logout: () => request<void>("/v1/auth/logout", { method: "POST", allowUnauthorized: true }),
  logoutAll: () => request<void>("/v1/auth/logout-all", { method: "POST", allowUnauthorized: true }),
  me: () => request<User>("/v1/auth/me", { allowUnauthorized: true }),
  meta: () => request<Meta>("/v1/meta"),

  // ---- hội thoại của bệnh nhân
  sendMessage: (conversationId: string, text: string, inputSource: "text" | "voice" = "text", sttFlags: SttFlag[] = []) =>
    request<TurnResult>(`/v1/chat/${enc(conversationId)}/messages`, {
      method: "POST",
      json: { text, input_source: inputSource, stt_flags: sttFlags },
    }),
  resume: (conversationId: string, decision: Decision) =>
    request<TurnResult>(`/v1/chat/${enc(conversationId)}/resume`, { method: "POST", json: decision }),
  pending: (conversationId: string) =>
    request<InterruptPayload | null>(`/v1/chat/${enc(conversationId)}/pending`),
  state: (conversationId: string) => request<AgentState>(`/v1/chat/${enc(conversationId)}/state`),
  transcribe: (audio: Blob, filename: string, language?: SttLanguage) => {
    const form = new FormData();
    form.append("audio", audio, filename);
    if (language) form.append("language", language);
    return request<TranscriptReview>("/v1/stt/transcribe", { method: "POST", body: form });
  },
  myConversations: () => request<ConversationSummary[]>("/v1/me/conversations"),
  myConversation: (conversationId: string) => request<ChatMessage[]>(`/v1/me/conversations/${enc(conversationId)}`),

  // ---- nhân viên y tế
  staffConversations: (patientId?: string) =>
    request<ConversationSummary[]>(`/v1/staff/conversations${patientId ? `?patient_id=${enc(patientId)}` : ""}`),
  staffConversation: (conversationId: string) => request<ChatMessage[]>(`/v1/staff/conversations/${enc(conversationId)}`),

  // ---- quản trị
  users: () => request<User[]>("/v1/admin/users"),
  createUser: (user: NewUser) => request<User>("/v1/admin/users", { method: "POST", json: user }),
  updateUser: (id: string, patch: { active?: boolean; password?: string }) =>
    request<User>(`/v1/admin/users/${enc(id)}`, { method: "PATCH", json: patch }),
};
