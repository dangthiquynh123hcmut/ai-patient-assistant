import { useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { errorText } from "../components/Toast";
import { homeFor, useAuth } from "./AuthProvider";

// Khớp với domain/account.py (USERNAME, MIN_PASSWORD_LENGTH). Backend vẫn kiểm tra lại.
const USERNAME = "[a-z0-9._\\-]{3,32}";
const MIN_PASSWORD = 8;

export function RegisterPage() {
  const { user, register } = useAuth();
  const navigate = useNavigate();
  const [displayName, setDisplayName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={homeFor(user)} replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError("Mật khẩu nhập lại không khớp.");
      return;
    }
    setBusy(true);
    try {
      const created = await register({ username: username.trim().toLowerCase(), display_name: displayName, password });
      navigate(homeFor(created), { replace: true });
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-box panel" aria-label="Đăng ký">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            ✚
          </span>
          <div>
            <h1>Tạo tài khoản bệnh nhân</h1>
            <p className="sub">Dữ liệu giả lập · dự án học tập · không chẩn đoán</p>
          </div>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label className="field">
            <span>Họ tên</span>
            <input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              autoComplete="name"
              maxLength={128}
              required
              autoFocus
            />
          </label>
          <label className="field">
            <span>Tên đăng nhập</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              pattern={USERNAME}
              title="3–32 ký tự: chữ thường a-z, số, dấu chấm, gạch dưới, gạch ngang"
              required
            />
            <span className="hint">3–32 ký tự: chữ thường a-z, số, dấu chấm, gạch dưới, gạch ngang</span>
          </label>
          <label className="field">
            <span>Mật khẩu</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              minLength={MIN_PASSWORD}
              maxLength={128}
              required
            />
            <span className="hint">Tối thiểu {MIN_PASSWORD} ký tự</span>
          </label>
          <label className="field">
            <span>Nhập lại mật khẩu</span>
            <input
              type="password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="new-password"
              required
            />
          </label>
          {error && (
            <p className="auth-error" role="alert">
              {error}
            </p>
          )}
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Đang tạo tài khoản…" : "Đăng ký"}
          </button>
        </form>
        <p className="auth-switch">
          Đã có tài khoản? <Link to="/login">Đăng nhập</Link>
        </p>
      </section>
    </main>
  );
}
