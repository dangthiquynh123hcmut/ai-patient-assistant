// Kiểu dữ liệu của HTTP API — khớp với model Pydantic phía backend
// (domain/account.py, application/agent/state.py, application/agent/trace.py, application/use_cases/*.py).

export type Role = "patient" | "staff" | "admin";

export interface User {
  id: string;
  username: string;
  display_name: string;
  role: Role;
  patient_id: string | null;
  active: boolean;
  created_at: string;
  failed_logins: number;
  locked_until: string | null;
}

export interface NewUser {
  username: string;
  display_name: string;
  role: Role;
  patient_id: string | null;
  password: string;
}

export interface Department {
  code: string;
  name: string;
  aliases: string[];
}

export interface Meta {
  user: User;
  llm_provider: "bedrock" | "offline";
  llm_model: string;
  stt_enabled: boolean;
  stt_model: string | null;
  now: string;
  patients: { id: string; label: string }[];
  departments: Department[];
  doctors: { id: string; name: string; department_code: string }[];
  limits: { max_corrections: number; max_clarify_rounds: number; max_slot_search_rounds: number };
}

// ------------------------------------------------------------------ hội thoại

export type InterruptKind = "clarify" | "confirm_report" | "choose_slot" | "confirm_booking";

export type OutcomeKind =
  | "report_saved"
  | "appointment_booked"
  | "history"
  | "appointments"
  | "patient_info"
  | "answer"
  | "chat"
  | "cancelled"
  | "handoff"
  | "safety"
  | "error";

export type Origin = "user_said" | "user_edit" | "backend" | "system_default" | "ai_summary";
export type Validation = "passed" | "failed" | "missing" | "uncertain";
export type Resolution = "llm_fixable" | "ask_user" | "block";
export type StepKind = "code" | "llm" | "hitl" | "tool" | "stop";

export interface FieldExplanation {
  field: string;
  label: string;
  value: unknown;
  evidence: string | null;
  origin: Origin;
  validation: Validation;
  explanation: string;
}

export interface Slot {
  slot_id: string;
  department_code: string;
  doctor_id: string;
  doctor_name: string;
  start_at: string;
  end_at: string;
}

export interface SlotOption {
  option_no: number;
  slot: Slot;
}

export interface Choice {
  label: string;
  text: string;
}

export interface InterruptPayload {
  kind: InterruptKind;
  message: string;
  explanations: FieldExplanation[];
  options: SlotOption[];
  choices: Choice[];
  allowed_actions: string[];
}

export interface AgentResponse {
  kind: OutcomeKind;
  text: string;
  explanations: FieldExplanation[];
  source_ids: string[];
  data: unknown;
}

export interface StepTrace {
  index: number;
  flow: string;
  node: string;
  title: string;
  kind: StepKind;
  summary: string;
  details: Record<string, unknown>;
}

export interface TurnResult {
  thread_id: string;
  status: "waiting_user" | "done";
  interrupt: InterruptPayload | null;
  response: AgentResponse | null;
  steps: StepTrace[];
}

export type Decision =
  | { action: "reply"; text: string }
  | { action: "other_time"; text: string }
  | { action: "choose"; option_no: number }
  | { action: "edit"; fields: Record<string, unknown> }
  | { action: "confirm" }
  | { action: "cancel" };

export interface SttFlag {
  text: string;
  probability: number;
  reason: string;
}

export interface TranscriptReview {
  text: string;
  words: { word: string; probability: number }[];
  flags: SttFlag[];
}

// ------------------------------------------------------------------ state của agent (panel "cách agent ra quyết định")

export interface ValidationIssue {
  field: string | null;
  layer: string;
  code: string;
  message: string;
  resolution: Resolution;
  suggestion: string | null;
}

export interface AttemptRecord {
  flow: "report" | "booking";
  attempt_no: number;
  issue_codes: string[];
  llm_fixable_codes: string[];
  passed: boolean;
}

export interface ToolCallRecord {
  tool: string;
  called_by: "code" | "llm";
  status: string;
  error_code: string | null;
  latency_ms: number;
  source_ids: string[];
}

export interface DateWindow {
  date_from: string;
  date_to: string;
  time_from: string | null;
  time_to: string | null;
  rules: string[];
}

export interface AgentState {
  intent: string | null;
  intent_source: string | null;
  pending_intents: string[];
  safety: string;
  input_source: string;
  user_replies: string[];
  issues: ValidationIssue[];
  attempts: AttemptRecord[];
  tool_calls: ToolCallRecord[];
  correction_count: number;
  clarify_rounds: number;
  search_rounds: number;
  conflict_retries: number;
  resolved_window: DateWindow | null;
  date_issues: ValidationIssue[];
  booking_request: { date_expression: string | null; time_expression: string | null } | null;
  stt_flags: SttFlag[];
  outcome: { kind: OutcomeKind; code: string | null } | null;
  [key: string]: unknown;
}

// ------------------------------------------------------------------ lịch sử (lưu ở DB)

export interface ConversationSummary {
  thread_id: string;
  patient_id: string;
  user_id: string;
  title: string;
  message_count: number;
  last_kind: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  thread_id: string;
  role: "user" | "assistant";
  action: string;
  text: string;
  intent: string | null;
  created_at: string;
}
