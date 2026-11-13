"""FastAPI: lớp mỏng chuyển HTTP <-> use case. Không chứa nghiệp vụ.

Chỉ phục vụ API (`/v1/...`, `/health`). Frontend React được build riêng (`frontend/`) và do nginx phục vụ,
nginx proxy `/v1/` về đây — cùng origin nên cookie refresh token dùng được mà không cần CORS.

Dependency injection (dependency-injector, như JESCO-AI-BE): router khai báo phụ thuộc bằng
`Depends(Provide[Container.x])` + `@inject`; `create_app` wire container vào các module router.

Xác thực JWT:
- Access token (ngắn hạn) trả trong body, client giữ trong bộ nhớ và gửi `Authorization: Bearer <token>`.
- Refresh token nằm trong cookie HttpOnly `pa_refresh` — xem `routers/auth_router.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from patient_assistant.config import Settings
from patient_assistant.container import Container, build_container
from patient_assistant.interface.api import deps
from patient_assistant.interface.api.routers import auth_router, chat_router, me_router, staff_router

ROUTERS = (auth_router, chat_router, me_router, staff_router)
# Module có `Provide[...]` cần wire
WIRED_MODULES = (deps, *ROUTERS)


def wire(app: FastAPI, container: Container) -> None:
    """Gắn container vào app: các `Provide[Container.x]` trong router trỏ tới provider của container này."""
    container.wire(modules=list(WIRED_MODULES))
    app.state.container = container


def create_app(
    container: Container | None = None,
    container_factory: Callable[[], AbstractAsyncContextManager[Container]] | None = None,
) -> FastAPI:
    """`container`: đã lắp sẵn (test) — dùng ngay, lifespan không tạo container.
    Nếu không, lifespan tạo container bằng `container_factory` (mặc định `build_container(Settings())`).
    """
    factory = container_factory or (lambda: build_container(Settings()))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is not None:
            yield
            return
        async with factory() as built:
            wire(app, built)
            yield

    app = FastAPI(title="AI Patient Assistant", version="0.1.0", lifespan=lifespan)
    for module in ROUTERS:
        app.include_router(module.router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if container is not None:
        wire(app, container)
    return app


def app() -> FastAPI:
    """Entry point cho uvicorn: `uvicorn patient_assistant.interface.api.app:app --factory`."""
    return create_app()
