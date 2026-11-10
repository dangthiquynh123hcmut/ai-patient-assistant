import re
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator

_PHONE = re.compile(r"^0\d{9,10}$")


def normalize_phone(value: str | None) -> str | None:
    """Số điện thoại Việt Nam: bỏ khoảng trắng/dấu chấm/gạch, đổi +84 thành 0. Rỗng -> None."""
    if value is None:
        return None
    digits = re.sub(r"[\s.\-()]", "", value)
    if digits.startswith("+84"):
        digits = "0" + digits[3:]
    if not digits:
        return None
    if not _PHONE.match(digits):
        raise ValueError("Số điện thoại phải gồm 10–11 chữ số, bắt đầu bằng 0 (hoặc +84)")
    return digits


class PatientProfile(BaseModel):
    """Hồ sơ tối thiểu của bệnh nhân (dữ liệu giả lập). Họ tên có thể trùng: mã `id` mới là định danh."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    date_of_birth: date | None = None
    phone: str | None = None

    _phone = field_validator("phone")(normalize_phone)
