"""Phase 11 — the worker's side of the control plane.

Registers a :class:`~clinic_agent.core.worker.Worker` with the session router, heartbeats its
load, and polls for the rooms it has been assigned. Deliberately thin: all placement decisions
live in the router (``session_router/registry.py``), and a worker that started making its own
would be a second, disagreeing scheduler.

Two properties matter more than features here:

* **Heartbeats are how death is detected.** The router expires a worker that goes quiet and
  re-dispatches its rooms, so the heartbeat interval has to stay comfortably inside the router's
  timeout. Failing to heartbeat is not an error path to swallow — it is the signal.
* **Losing the router must not drop live calls.** If the control plane is unreachable, calls
  already in progress keep running; the worker just cannot be given new ones. Tearing down
  active sessions because a *coordinator* is down would convert a control-plane outage into a
  data-plane outage, which is exactly backwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
from collections.abc import Callable

import httpx
from loguru import logger

from .worker import Worker

# Comfortably inside the router's 15 s expiry, so a single dropped request never looks like death.
DEFAULT_HEARTBEAT_SECS = 4.0


class RouterClient:
    """Keeps one worker registered with the session router."""

    def __init__(
        self,
        worker: Worker,
        router_url: str,
        *,
        worker_id: str | None = None,
        heartbeat_secs: float = DEFAULT_HEARTBEAT_SECS,
        on_assigned: Callable[[str], None] | None = None,
    ) -> None:
        self.worker = worker
        self.router_url = router_url.rstrip("/")
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.heartbeat_secs = heartbeat_secs
        self._on_assigned = on_assigned
        self._client = httpx.AsyncClient(base_url=self.router_url, timeout=5.0)
        self._task: asyncio.Task | None = None
        self._known_rooms: set[str] = set()
        self.connected = False

    async def register(self) -> bool:
        try:
            resp = await self._client.post(
                "/workers/register",
                json={"worker_id": self.worker_id, "capacity": self.worker.capacity},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f"[router] registration failed: {exc}")
            self.connected = False
            return False
        self.connected = True
        logger.info(
            f"[router] registered {self.worker_id} (capacity={self.worker.capacity}) "
            f"with {self.router_url}"
        )
        return True

    async def start(self) -> None:
        await self.register()
        self._task = asyncio.create_task(self._heartbeat_loop(), name="router-heartbeat")

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.heartbeat_secs)
                await self._beat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the control plane must not end calls
                logger.warning(f"[router] heartbeat error: {exc}")
                self.connected = False

    async def _beat(self) -> None:
        stats = self.worker.stats
        try:
            resp = await self._client.post(
                "/workers/heartbeat",
                json={
                    "worker_id": self.worker_id,
                    "active": stats.active,
                    "capacity": stats.capacity,
                    "draining": not self.worker.can_accept() and stats.active >= stats.capacity,
                },
            )
        except httpx.HTTPError as exc:
            # In-flight calls are untouched: a coordinator outage must not become a call outage.
            logger.warning(f"[router] unreachable ({exc}); {stats.active} call(s) continue")
            self.connected = False
            return

        if resp.status_code == 404:
            # The router restarted, or expired us while we were busy. Re-announce rather than
            # sit there healthy and invisible.
            logger.info("[router] not recognized — re-registering")
            await self.register()
            return

        self.connected = True
        await self._sync_assignments()

    async def _sync_assignments(self) -> None:
        """Start a session for any room the router has given us that we are not running."""
        try:
            resp = await self._client.get(f"/assignments/{self.worker_id}")
            resp.raise_for_status()
        except httpx.HTTPError:
            return

        rooms = set(resp.json().get("rooms", []))
        for room in sorted(rooms - self._known_rooms):
            accepted = await self.worker.accept(room)
            if accepted:
                logger.info(f"[router] assigned room {room}")
                if self._on_assigned is not None:
                    self._on_assigned(room)
            else:
                logger.warning(f"[router] could not accept {room} — worker full or draining")
        self._known_rooms = rooms

    async def drain(self) -> None:
        """Tell the router to stop sending work here, so a deploy sheds cleanly."""
        with contextlib.suppress(httpx.HTTPError):
            await self._client.post("/workers/drain", json={"worker_id": self.worker_id})

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._client.aclose()
