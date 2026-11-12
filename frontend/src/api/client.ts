// Client HTTP có kiểu. Xác thực JWT:
// - Access token (ngắn hạn) chỉ giữ trong bộ nhớ (không localStorage) và gửi qua `Authorization: Bearer`.
// - Refresh token nằm trong cookie HttpOnly `pa_refresh` (path /v1/auth), JavaScript không đọc được. Khi API trả 401,
//   client gọi /v1/auth/refresh một lần (gộp các request đồng thời) rồi gửi lại request.
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
  TokenResponse,
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

/** Gọi khi phiên hết hạn / bị thu hồi (refresh thất bại) — AuthProvider đăng ký để đưa về trang đăng nhập. */
let onUnauthorized: () => void = () => {};
export function setUnauthorizedHandler(handler: () => void): void {
  onUnauthorized = handler;
}

let accessToken: string | null = null;
let refreshing: Promise<boolean> | null = null;

function acceptTokens(tokens: TokenResponse): User {
  accessToken = tokens.access_token;
  return tokens.user;
}

export function clearSession(): void {
  accessToken = null;
}

async function postRefresh(): Promise<Response> {
  return fetch("/v1/auth/refresh", { method: "POST", headers: { Accept: "application/json" }, credentials: "same-origin" });
}

/**
 * Lấy access token mới bằng refresh cookie. Các request bị 401 cùng lúc dùng chung một lần refresh.
 * 409: tab khác vừa xoay refresh token và cookie mới đã có trong trình duyệt -> gửi lại một lần.
 */
export function refreshSession(): Promise<boolean> {
  refreshing ??= (async () => {
    try {
      let res = await postRefresh();
      if (res.status === 409) {
        await new Promise((resolve) => setTimeout(resolve, 300));
        res = await postRefresh();
      }
      if (!res.ok) {
        accessToken = null;
        return false;
      }
      acceptTokens((await res.json()) as TokenResponse);
      return true;
    } catch {
      return false;
    } finally {
      refreshing = null;
    }
  })();
  return refreshing;
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
  /** Không refresh / không chuyển về trang đăng nhập khi 401 (dùng cho các endpoint /auth/*). */
  allowUnauthorized?: boolean;
}

async function request<T>(path: string, options: RequestOptions = {}, retried = false): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json" };
  let body = options.body;
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const res = await fetch(path, { method: options.method ?? "GET", headers, body, credentials: "same-origin" });
  if (res.status === 401 && !options.allowUnauthorized && !retried) {
    // Access token hết hạn / chưa có: đổi refresh cookie lấy token mới rồi gửi lại đúng một lần.
    // An toàn cả với POST: 401 nghĩa là server chưa xử lý request.
    if (await refreshSession()) return request<T>(path, options, true);
    onUnauthorized();
  } else if (res.status === 401 && !options.allowUnauthorized) {
    onUnauthorized();
  }
  if (res.status === 204) return undefined as T;
  const data: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, detailOf(data, res.statusText || `HTTP ${res.status}`));
  return data as T;
}

const enc = encodeURIComponent;

export const api = {
  // ---- đăng nhập
  login: async (username: string, password: string) =>
    acceptTokens(
      await request<TokenResponse>("/v1/auth/login", {
        method: "POST",
        json: { username, password },
        allowUnauthorized: true,
      }),
    ),
  register: async (registration: Registration) =>
    acceptTokens(
      await request<TokenResponse>("/v1/auth/register", { method: "POST", json: registration, allowUnauthorized: true }),
    ),
  /** Khôi phục đăng nhập khi mở / tải lại trang: dùng refresh cookie lấy access token. */
  restoreSession: refreshSession,
  authOptions: () => request<{ allow_registration: boolean }>("/v1/auth/options", { allowUnauthorized: true }),
  logout: () => request<void>("/v1/auth/logout", { method: "POST", allowUnauthorized: true }),
  logoutAll: () => request<void>("/v1/auth/logout-all", { method: "POST", allowUnauthorized: true }),
  me: () => request<User>("/v1/auth/me"),
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
