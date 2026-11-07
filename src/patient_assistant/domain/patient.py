
from pydantic import BaseModel, ConfigDict


class PatientProfile(BaseModel):
    """Hồ sơ tối thiểu của bệnh nhân (dữ liệu giả lập)."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
