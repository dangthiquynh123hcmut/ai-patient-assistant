import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthProvider";
import { ROLE_LABEL } from "../lib/format";

interface Props {
  title: string;
  subtitle: string;
  /** Badge, thanh điều hướng... nằm giữa. */
  children?: ReactNode;
  /** Nút thêm bên phải, trước nút đăng xuất. */
  actions?: ReactNode;
}

export function TopBar({ title, subtitle, children, actions }: Props) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
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
      {children}
      <div className="controls">
        {user && (
          <div className="userbox">
            <span>{user.display_name}</span>
            <span className="role">{ROLE_LABEL[user.role]}</span>
          </div>
        )}
        {actions}
        <button
          className="btn ghost"
          type="button"
          onClick={async () => {
            await logout();
            navigate("/login", { replace: true });
          }}
        >
          Đăng xuất
        </button>
      </div>
    </header>
  );
}
