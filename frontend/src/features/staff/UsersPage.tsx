import { useCallback, useEffect, useState, type FormEvent } from "react";

import { api } from "../../api/client";
import type { NewUser, Role, User } from "../../api/types";
import { useAuth } from "../../auth/AuthProvider";
import { errorText, useToast } from "../../components/Toast";
import { ROLE_LABEL, patientOptionText } from "../../lib/format";
import { StaffTopBar } from "./StaffTopBar";

const EMPTY: NewUser = { username: "", display_name: "", role: "patient", patient_id: null, password: "" };

export function UsersPage() {
  const { user: me, meta } = useAuth();
  const toast = useToast();
  const patients = meta?.patients ?? [];
  const [users, setUsers] = useState<User[]>([]);
  const [form, setForm] = useState<NewUser>({ ...EMPTY, patient_id: patients[0]?.id ?? null });
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setUsers(await api.users());
    } catch (err) {
      toast(errorText(err));
    }
  }, [toast]);

  useEffect(() => {
    void load();
  }, [load]);

  async function create(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      await api.createUser({ ...form, patient_id: form.role === "patient" ? form.patient_id : null });
      setForm({ ...EMPTY, patient_id: patients[0]?.id ?? null });
      toast("Đã tạo tài khoản.");
      await load();
    } catch (err) {
      toast(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  async function update(u: User, patch: { active?: boolean; password?: string }) {
    try {
      await api.updateUser(u.id, patch);
      toast("Đã cập nhật. Các phiên đăng nhập cũ của tài khoản này đã bị đăng xuất.");
      await load();
    } catch (err) {
      toast(errorText(err));
    }
  }

  const locked = (u: User) => u.locked_until !== null && new Date(u.locked_until) > new Date();

  return (
    <>
      <StaffTopBar />
      <main className="staff-layout single">
        <section className="panel" aria-label="Tài khoản">
          <div className="panel-head">
            <h2>Tài khoản</h2>
          </div>
          <form className="staff-form" onSubmit={create} autoComplete="off">
            <label className="field">
              <span>Tên đăng nhập</span>
              <input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required />
            </label>
            <label className="field">
              <span>Tên hiển thị</span>
              <input
                value={form.display_name}
                onChange={(e) => setForm({ ...form, display_name: e.target.value })}
                required
              />
            </label>
            <label className="field">
              <span>Vai trò</span>
              <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value as Role })}>
                {(Object.keys(ROLE_LABEL) as Role[]).map((r) => (
                  <option key={r} value={r}>
                    {ROLE_LABEL[r]}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Mã bệnh nhân</span>
              <select
                value={form.patient_id ?? ""}
                disabled={form.role !== "patient"}
                onChange={(e) => setForm({ ...form, patient_id: e.target.value })}
              >
                {patients.map((p) => (
                  <option key={p.id} value={p.id}>
                    {patientOptionText(p)}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Mật khẩu (≥ 8 ký tự)</span>
              <input
                type="password"
                minLength={8}
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                autoComplete="new-password"
                required
              />
            </label>
            <button className="btn primary" type="submit" disabled={busy}>
              Tạo tài khoản
            </button>
          </form>
          <div className="tab-body">
            <table className="users-table">
              <thead>
                <tr>
                  <th>Tên đăng nhập</th>
                  <th>Tên hiển thị</th>
                  <th>Vai trò</th>
                  <th>Bệnh nhân</th>
                  <th>Trạng thái</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id} className={u.active ? "" : "inactive"}>
                    <td>{u.username}</td>
                    <td>{u.display_name}</td>
                    <td>{ROLE_LABEL[u.role]}</td>
                    <td>{u.patient_id ?? "—"}</td>
                    <td>{!u.active ? "Đã khóa" : locked(u) ? "Tạm khóa (sai mật khẩu)" : "Hoạt động"}</td>
                    <td>
                      {u.id !== me?.id && (
                        <button
                          className={`btn${u.active ? " danger" : ""}`}
                          type="button"
                          onClick={() => void update(u, { active: !u.active })}
                        >
                          {u.active ? "Khóa" : "Mở khóa"}
                        </button>
                      )}
                      <button
                        className="btn ghost"
                        type="button"
                        onClick={() => {
                          const password = window.prompt(`Mật khẩu mới cho ${u.username} (≥ 8 ký tự):`);
                          if (password) void update(u, { password });
                        }}
                      >
                        Đặt lại mật khẩu
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </>
  );
}
