import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiError, api, setUnauthorizedHandler } from "../api/client";
import type { Meta, User } from "../api/types";

interface AuthState {
  /** undefined: đang kiểm tra phiên; null: chưa đăng nhập. */
  user: User | null | undefined;
  meta: Meta | null;
  login: (username: string, password: string) => Promise<User>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null | undefined>(undefined);
  const [meta, setMeta] = useState<Meta | null>(null);

  const load = useCallback(async () => {
    try {
      const m = await api.meta();
      setMeta(m);
      setUser(m.user);
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 401)) console.error(err);
      setMeta(null);
      setUser(null);
    }
  }, []);

  useEffect(() => {
    // Phiên hết hạn / bị admin thu hồi giữa chừng -> quay về trạng thái chưa đăng nhập.
    setUnauthorizedHandler(() => {
      setUser(null);
      setMeta(null);
    });
    api
      .me()
      .then(() => load())
      .catch(() => setUser(null));
  }, [load]);

  const login = useCallback(
    async (username: string, password: string) => {
      const logged = await api.login(username, password);
      await load();
      return logged;
    },
    [load],
  );

  const logout = useCallback(async () => {
    await api.logout().catch(() => undefined);
    setUser(null);
    setMeta(null);
  }, []);

  const value = useMemo(() => ({ user, meta, login, logout }), [user, meta, login, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth phải nằm trong <AuthProvider>");
  return ctx;
}

/** Trang mặc định theo vai trò. */
export const homeFor = (user: User): string => (user.role === "patient" ? "/" : "/staff");
