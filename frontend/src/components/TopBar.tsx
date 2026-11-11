import { useEffect, useRef, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthProvider";
import { ROLE_LABEL } from "../lib/format";
import { errorText, useToast } from "./Toast";

interface Props {
  title: string;
  subtitle: string;
  /** Badge, thanh điều hướng... nằm giữa. */
  children?: ReactNode;
  /** Nút thêm bên phải, trước nút đăng xuất. */
  actions?: ReactNode;
}

export function TopBar({ title, subtitle, children, actions }: Props) {
  const { user, logout, logoutAll } = useAuth();
  const navigate = useNavigate();
  const toast = useToast();
  const [leaving, setLeaving] = useState(false);
  const menu = useRef<HTMLDetailsElement>(null);

  // Đóng menu tài khoản khi bấm ra ngoài hoặc nhấn Esc.
  useEffect(() => {
    const close = (e: Event) => {
      const el = menu.current;
      if (!el?.open) return;
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !el.contains(e.target as Node)) el.open = false;
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, []);

  async function signOut(everywhere: boolean) {
    if (leaving) return;
    if (menu.current) menu.current.open = false;
    setLeaving(true);
    try {
      await (everywhere ? logoutAll() : logout());
      navigate("/login", { replace: true });
    } catch (err) {
      toast(`Chưa đăng xuất được, bạn vẫn đang đăng nhập: ${errorText(err)}`);
    } finally {
      setLeaving(false);
    }
  }

  return (
    <header className="topbar">
      <div className="brand">
        <span className="logo" aria-hidden="true">
          ✚
        </span>
        <div>
          <h1>{title}</h1>
          <p className="sub">{subtitle}</p>
        </div>
      </div>
      <div className="topbar-mid">{children}</div>
      <div className="controls">
        {user && (
          <details className="usermenu" ref={menu}>
            <summary className="userbox" title={`${user.display_name} · ${ROLE_LABEL[user.role]}`}>
              <span className="avatar" aria-hidden="true">
                {user.display_name.trim().charAt(0).toUpperCase() || "?"}
              </span>
              <span className="who">
                <span className="name">{user.display_name}</span>
                <span className="role">{ROLE_LABEL[user.role]}</span>
              </span>
              <span className="caret" aria-hidden="true">
                ▾
              </span>
            </summary>
            <div className="menu" role="menu">
              <button type="button" role="menuitem" disabled={leaving} onClick={() => void signOut(true)}>
                Đăng xuất mọi thiết bị
                <small>Thu hồi mọi phiên đăng nhập của tài khoản này</small>
              </button>
            </div>
          </details>
        )}
        {actions}
        <button className="btn ghost" type="button" disabled={leaving} onClick={() => void signOut(false)}>
          {leaving ? "Đang đăng xuất…" : "Đăng xuất"}
        </button>
      </div>
    </header>
  );
}
