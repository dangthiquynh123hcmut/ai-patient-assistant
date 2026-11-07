import { useEffect, useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";

import { api } from "../api/client";

import { errorText } from "../components/Toast";
import { homeFor, useAuth } from "./AuthProvider";

export function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from;
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [canRegister, setCanRegister] = useState(false);

  useEffect(() => {
    api
      .authOptions()
      .then((o) => setCanRegister(o.allow_registration))
      .catch(() => setCanRegister(false));
  }, []);

  if (user) return <Navigate to={homeFor(user)} replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const logged = await login(username, password);
      // Chỉ quay lại trang cũ nếu trang đó thuộc vai trò này.
      const home = homeFor(logged);
      navigate(from && from.startsWith(home) ? from : home, { replace: true });
    } catch (err) {
      setError(errorText(err));
      setPassword("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-box panel" aria-label="Đăng nhập">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            ✚
          </span>
          <div>
            <h1>Trợ lý bệnh nhân AI</h1>
            <p className="sub">Dữ liệu giả lập · dự án học tập · không chẩn đoán</p>
          </div>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label className="field">
            <span>Tên đăng nhập</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              autoFocus
            />
          </label>
          <label className="field">
            <span>Mật khẩu</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          {error && (
            <p className="auth-error" role="alert">
              {error}
            </p>
          )}
          <button className="btn primary" type="submit" disabled={busy}>
            {busy ? "Đang đăng nhập…" : "Đăng nhập"}
          </button>
        </form>
        {canRegister && (
          <p className="auth-switch">
            Chưa có tài khoản? <Link to="/register">Đăng ký</Link>
          </p>
        )}
        <details className="auth-demo">
          <summary>Tài khoản demo</summary>
          <p className="muted">
            Mật khẩu mặc định <code>demo1234</code> (đặt bằng <code>PA_SEED_PASSWORD</code> trước lần chạy đầu).
          </p>
          <ul className="muted">
            <li>
              <code>benhnhan1</code>, <code>benhnhan2</code>, <code>benhnhan3</code> — bệnh nhân
            </li>
            <li>
              <code>bacsi</code> — nhân viên y tế
            </li>
            <li>
              <code>admin</code> — quản trị
            </li>
          </ul>
        </details>
      </section>
    </main>
  );
}
