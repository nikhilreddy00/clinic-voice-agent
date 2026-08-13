"""Phase 11 — worker registry and room assignment policy.

The control-plane decision layer, kept free of I/O and of any clock of its own: every method
that cares about time takes ``now``. That is the same discipline the Phase-10 reducer follows,
and for the same reason — worker death, heartbeat expiry, and assignment races are all timing
behavior, and timing behavior you cannot drive deterministically is timing behavior you cannot
test.

The model is pull-free and deliberately simple: workers announce themselves and heartbeat their
own load; the router picks the least-loaded worker with room to spare. It does not try to be
clever. At the scale a clinic fleet actually operates, the interesting failures are not
suboptimal packing — they are *losing a worker mid-call* and *accepting a call nobody has
capacity for*, and both of those are about honesty rather than optimization.

**On re-dispatch.** When a worker dies its rooms are orphaned. Reassigning the room gets the
caller a working agent again, but with a *fresh* session — the conversation state died with the
worker. This registry reports that plainly (``orphaned``) rather than implying continuity it
cannot deliver. Real continuity is reachable: every call records a complete event trace, and
``recorder.replay()`` reconstructs ``CallState`` from it, so a receiving worker could rebuild
the conversation if traces lived somewhere shared. That is Phase 15's work, not a claim to make
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# A worker that has not heartbeated in this long is presumed dead and its rooms are orphaned.
# Generous relative to a heartbeat interval of a few seconds: declaring a busy worker dead and
# re-dispatching its live calls is far more damaging than noticing a real death a bit late.
DEFAULT_HEARTBEAT_TIMEOUT = 15.0


class Rejection(str, Enum):
    """Why an assignment could not be made. Each maps to a different caller experience."""

    NO_CAPACITY = "no_capacity"      # fleet is full — overflow to human/voicemail
    NO_WORKERS = "no_workers"        # nothing registered — hard outage
    ALREADY_ASSIGNED = "already_assigned"  # duplicate webhook; the existing assignment stands


@dataclass
class WorkerRecord:
    worker_id: str
    capacity: int
    active: int = 0
    draining: bool = False
    last_seen: float = 0.0
    url: str = ""
    rooms: set[str] = field(default_factory=set)

    @property
    def free(self) -> int:
        return max(0, self.capacity - max(self.active, len(self.rooms)))

    @property
    def load(self) -> float:
        used = max(self.active, len(self.rooms))
        return used / self.capacity if self.capacity else 1.0

    def alive(self, now: float, timeout: float) -> bool:
        return (now - self.last_seen) <= timeout

    def as_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "capacity": self.capacity,
            "active": self.active,
            "free": self.free,
            "load": round(self.load, 3),
            "draining": self.draining,
            "rooms": sorted(self.rooms),
            "last_seen": self.last_seen,
        }


@dataclass
class Assignment:
    room_name: str
    worker_id: str
    assigned_at: float
    call_id: str = ""


class WorkerRegistry:
    """Tracks workers and which room each call is running on."""

    def __init__(self, *, heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT) -> None:
        self.heartbeat_timeout = heartbeat_timeout
        self._workers: dict[str, WorkerRecord] = {}
        self._assignments: dict[str, Assignment] = {}
        self.orphaned: list[str] = []          # rooms whose worker died, awaiting re-dispatch
        self.rejections: dict[str, int] = {}   # Rejection value -> count

    # --- worker lifecycle ---------------------------------------------------------------

    def register(self, worker_id: str, capacity: int, now: float, *, url: str = "") -> WorkerRecord:
        record = self._workers.get(worker_id)
        if record is None:
            record = WorkerRecord(worker_id=worker_id, capacity=capacity, url=url)
            self._workers[worker_id] = record
        record.capacity = capacity
        record.url = url or record.url
        record.last_seen = now
        record.draining = False
        return record

    def heartbeat(
        self, worker_id: str, now: float, *, active: int | None = None,
        capacity: int | None = None, draining: bool | None = None,
    ) -> WorkerRecord | None:
        record = self._workers.get(worker_id)
        if record is None:
            return None
        record.last_seen = now
        if active is not None:
            record.active = active
        if capacity is not None:
            record.capacity = capacity
        if draining is not None:
            record.draining = draining
        return record

    def deregister(self, worker_id: str) -> list[str]:
        """Remove a worker deliberately (a clean shutdown). Returns the rooms it was carrying."""
        record = self._workers.pop(worker_id, None)
        if record is None:
            return []
        return self._orphan_rooms(record)

    def expire(self, now: float) -> list[str]:
        """Drop workers past the heartbeat timeout. Returns newly orphaned rooms."""
        orphaned: list[str] = []
        for worker_id, record in list(self._workers.items()):
            if not record.alive(now, self.heartbeat_timeout):
                self._workers.pop(worker_id)
                orphaned.extend(self._orphan_rooms(record))
        return orphaned

    def _orphan_rooms(self, record: WorkerRecord) -> list[str]:
        rooms = sorted(record.rooms)
        for room in rooms:
            self._assignments.pop(room, None)
        record.rooms.clear()
        self.orphaned.extend(rooms)
        return rooms

    # --- assignment ---------------------------------------------------------------------

    def candidates(self, now: float) -> list[WorkerRecord]:
        """Live, non-draining workers with free capacity, least-loaded first."""
        available = [
            w
            for w in self._workers.values()
            if w.alive(now, self.heartbeat_timeout) and not w.draining and w.free > 0
        ]
        # Tie-break on worker_id so assignment is deterministic and therefore testable.
        return sorted(available, key=lambda w: (w.load, w.worker_id))

    def assign(self, room_name: str, now: float, *, call_id: str = "") -> Assignment | Rejection:
        """Place ``room_name`` on the least-loaded worker, or say why it could not be placed."""
        existing = self._assignments.get(room_name)
        if existing is not None:
            # LiveKit can deliver a webhook more than once; the first assignment wins so a
            # retry never starts a second agent in the same room.
            self._count(Rejection.ALREADY_ASSIGNED)
            return Rejection.ALREADY_ASSIGNED

        live = [w for w in self._workers.values() if w.alive(now, self.heartbeat_timeout)]
        if not live:
            self._count(Rejection.NO_WORKERS)
            return Rejection.NO_WORKERS

        options = self.candidates(now)
        if not options:
            self._count(Rejection.NO_CAPACITY)
            return Rejection.NO_CAPACITY

        worker = options[0]
        assignment = Assignment(
            room_name=room_name, worker_id=worker.worker_id, assigned_at=now, call_id=call_id
        )
        worker.rooms.add(room_name)
        self._assignments[room_name] = assignment
        if room_name in self.orphaned:
            self.orphaned.remove(room_name)
        return assignment

    def release(self, room_name: str) -> Assignment | None:
        """The call ended. Frees the slot on its worker."""
        assignment = self._assignments.pop(room_name, None)
        if assignment is None:
            return None
        record = self._workers.get(assignment.worker_id)
        if record is not None:
            record.rooms.discard(room_name)
        if room_name in self.orphaned:
            self.orphaned.remove(room_name)
        return assignment

    def redispatch(self, now: float) -> list[tuple[str, Assignment | Rejection]]:
        """Try to place every orphaned room again. Returns each room's outcome.

        A successful re-dispatch gives the caller a live agent in the same room, NOT their
        conversation back — that died with the worker (see the module docstring).
        """
        results: list[tuple[str, Assignment | Rejection]] = []
        for room in list(self.orphaned):
            results.append((room, self.assign(room, now)))
        return results

    # --- introspection ------------------------------------------------------------------

    def _count(self, reason: Rejection) -> None:
        self.rejections[reason.value] = self.rejections.get(reason.value, 0) + 1

    def assignment_for(self, room_name: str) -> Assignment | None:
        return self._assignments.get(room_name)

    def snapshot(self, now: float) -> dict:
        live = [w for w in self._workers.values() if w.alive(now, self.heartbeat_timeout)]
        return {
            "workers": [w.as_dict() for w in sorted(self._workers.values(), key=lambda w: w.worker_id)],
            "live_workers": len(live),
            "fleet_capacity": sum(w.capacity for w in live),
            "fleet_active": sum(max(w.active, len(w.rooms)) for w in live),
            "fleet_free": sum(w.free for w in live if not w.draining),
            "assignments": len(self._assignments),
            "orphaned": sorted(self.orphaned),
            "rejections": dict(self.rejections),
        }
