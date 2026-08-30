"""Phase 10 — tool execution adapter.

Runs each :class:`~clinic_agent.core.actions.InvokeTool` against the scheduling API and turns
the outcome into a :class:`~clinic_agent.core.events.ToolCompleted` event. The HTTP work itself
is ``scheduling_tools.execute_tool`` — shared verbatim with the Pipecat path, so the two
engines make identical requests and write identical PHI-minimized logs.

Calls run as independent tasks, which gives parallel execution for free when the model
requests several at once. That matters more than it looks: the reducer will not resume the
turn until the last result lands, so serializing two 300 ms lookups would add 300 ms of dead
air to the caller's turn.

A failed tool is a normal event, never an exception — the model sees ``{"ok": false, ...}``,
apologizes, and recovers. The per-tool latency budget that speaks a filler line on a slow tool
is Phase 13; here the 10 s client timeout still applies.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from ...metrics import LatencyCollector
from ...scheduling_tools import SchedulingClient, execute_tool
from .. import events as ev

EmitFn = Callable[[ev.Event], None]


class ToolExecutor:
    """Executes scheduling-API tool calls for one call."""

    def __init__(
        self,
        client: SchedulingClient,
        emit: EmitFn,
        collector: LatencyCollector | None = None,
    ) -> None:
        self._client = client
        self._emit = emit
        self._collector = collector
        self._tasks: dict[str, asyncio.Task] = {}
        self._memory_task: asyncio.Task | None = None

    def invoke(self, tool_call_id: str, name: str, arguments: dict[str, Any]) -> None:
        """Start a tool call. Returns immediately; the result arrives as an event."""
        if tool_call_id in self._tasks:
            return
        self._tasks[tool_call_id] = asyncio.create_task(
            self._run(tool_call_id, name, arguments), name=f"tool-{name}-{tool_call_id}"
        )

    async def _run(self, tool_call_id: str, name: str, arguments: dict[str, Any]) -> None:
        try:
            result, latency_ms, http_status = await execute_tool(
                self._client, name, arguments, self._collector
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool crash must not end the call
            logger.error(f"[tools] {name} raised: {exc}")
            result, latency_ms, http_status = (
                {"ok": False, "error": f"the scheduling system failed ({exc})"},
                0.0,
                None,
            )
        finally:
            self._tasks.pop(tool_call_id, None)

        self._emit(
            ev.ToolCompleted(
                t=time.monotonic(),
                tool_call_id=tool_call_id,
                name=name,
                result=result,
                ok=bool(result.get("ok")),
                latency_ms=latency_ms,
                http_status=http_status,
            )
        )

    def load_caller_memory(self, phone: str) -> None:
        """Look up caller memory by ANI, off the critical path (Phase 13).

        Deliberately not awaited anywhere: the greeting is already being spoken when this
        starts, and the result arrives as a ``CallerMemoryLoaded`` event whenever it arrives.
        A failure is not surfaced to the caller — an unrecognised returning caller is a
        slightly colder greeting, while a greeting that waits on a database is dead air.
        """
        if self._memory_task is not None:
            return
        self._memory_task = asyncio.create_task(self._load_memory(phone), name="caller-memory")

    async def _load_memory(self, phone: str) -> None:
        t0 = time.monotonic()
        try:
            result = await self._client.caller_memory(phone=phone)
        except Exception as exc:  # noqa: BLE001 - never let memory take down a call
            logger.warning(f"[memory] lookup failed: {exc}")
            result = {"ok": False, "known": False, "upcoming_appointments": 0}
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(
            f"[memory] caller lookup in {latency_ms:.0f} ms → "
            f"known={result.get('known')} upcoming={result.get('upcoming_appointments')}"
        )
        self._emit(
            ev.CallerMemoryLoaded(
                t=time.monotonic(),
                known=bool(result.get("known")),
                upcoming_appointments=int(result.get("upcoming_appointments") or 0),
            )
        )

    async def aclose(self) -> None:
        if self._memory_task is not None and not self._memory_task.done():
            self._memory_task.cancel()
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        await self._client.aclose()
