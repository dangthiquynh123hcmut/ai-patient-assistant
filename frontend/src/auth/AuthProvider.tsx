import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiError, api, clearSession, setUnauthorizedHandler } from "../api/client";
import type { Meta, Registration, User } from "../api/types";

interface AuthState {
  /** undefined: đang kiểm tra phiên; null: chưa đăng nhập. */
  user: User | null | undefined;
  meta: Meta | null;
  login: (username: string, password: string) => Promise<User>;
  /** Tạo tài khoản bệnh nhân rồi đăng nhập luôn (backend trả access token và đặt cookie refresh). */
  register: (registration: Registration) => Promise<User>;
  /** Ném lỗi nếu server chưa thu hồi được phiên (vd. mất mạng) — khi đó người dùng vẫn đang đăng nhập. */
  logout: () => Promise<void>;
  /** Thu hồi mọi phiên của tài khoản này trên mọi thiết bị. */
  logoutAll: () => Promise<void>;
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
    // Phiên hết hạn / bị admin thu hồi giữa chừng (refresh thất bại) -> quay về trạng thái chưa đăng nhập.
    setUnauthorizedHandler(() => {
      clearSession();
      setUser(null);
      setMeta(null);
    });
    // Access token chỉ nằm trong bộ nhớ: tải lại trang thì lấy lại bằng refresh cookie.
    api
      .restoreSession()
      .then((ok) => (ok ? load() : setUser(null)))
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

  const register = useCallback(
    async (registration: Registration) => {
      const created = await api.register(registration);
      await load();
      return created;
    },
    [load],
  );

  /**
   * Chỉ coi là đã đăng xuất khi server xác nhận đã thu hồi phiên (hoặc phiên vốn đã hết hạn — 401).
   * Lỗi mạng / server thì ném lỗi ra để UI báo, vì phiên phía server vẫn còn hiệu lực.
   */
  const endSession = useCallback(async (call: () => Promise<void>) => {
    try {
      await call();
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 401)) throw err;
    }
    clearSession();
    setUser(null);
    setMeta(null);
  }, []);

  const logout = useCallback(() => endSession(api.logout), [endSession]);
  const logoutAll = useCallback(() => endSession(api.logoutAll), [endSession]);

  const value = useMemo(
    () => ({ user, meta, login, register, logout, logoutAll }),
    [user, meta, login, register, logout, logoutAll],
  );
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth phải nằm trong <AuthProvider>");
  return ctx;
}

/** Trang mặc định theo vai trò. */
export const homeFor = (user: User): string => (user.role === "patient" ? "/" : "/staff");
