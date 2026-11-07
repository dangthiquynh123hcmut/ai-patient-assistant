import type { ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import type { Role } from "./api/types";
import { homeFor, useAuth } from "./auth/AuthProvider";
import { LoginPage } from "./auth/LoginPage";
import { RegisterPage } from "./auth/RegisterPage";
import { ChatPage } from "./features/chat/ChatPage";
import { ConversationsPage } from "./features/staff/ConversationsPage";
import { UsersPage } from "./features/staff/UsersPage";

/** Chặn route theo vai trò ở phía UI. Quyền thật vẫn do API kiểm tra ở từng request. */
function RequireRole({ roles, children }: { roles: Role[]; children: ReactNode }) {
  const { user, meta } = useAuth();
  const location = useLocation();
  if (user === undefined || (user && !meta)) return <p className="loading">Đang tải…</p>;
  if (user === null) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  if (!roles.includes(user.role)) return <Navigate to={homeFor(user)} replace />;
  return <>{children}</>;
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route
        path="/"
        element={
          <RequireRole roles={["patient"]}>
            <ChatPage />
          </RequireRole>
        }
      />
      <Route
        path="/staff"
        element={
          <RequireRole roles={["staff", "admin"]}>
            <ConversationsPage />
          </RequireRole>
        }
      />
      <Route
        path="/staff/users"
        element={
          <RequireRole roles={["admin"]}>
            <UsersPage />
          </RequireRole>
        }
      />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
