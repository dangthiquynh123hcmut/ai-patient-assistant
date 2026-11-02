import { NavLink } from "react-router-dom";

import { useAuth } from "../../auth/AuthProvider";
import { TopBar } from "../../components/TopBar";

export function StaffTopBar() {
  const { user } = useAuth();
  return (
    <TopBar title="Trợ lý bệnh nhân AI · Nhân viên" subtitle="Xem câu hỏi và hội thoại của bệnh nhân (dữ liệu giả lập)">
      <nav className="top-nav">
        <NavLink to="/staff" end>
          Hội thoại
        </NavLink>
        {user?.role === "admin" && <NavLink to="/staff/users">Tài khoản</NavLink>}
      </nav>
    </TopBar>
  );
}
