"""Stateful bookkeeping for frozen repeated-compaction evaluation.

The evaluator records the actual output of each compaction as the next input
state. It never reconstructs a later state from the original full memory, which
makes ancestry and repeated-compression claims auditable. Reader scoring remains
an injected callback owned by the host experiment driver.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class CompactionEvent:
    step: int
    source_ids: List[str]
    output_ids: List[str]
    input_slots: int
    output_slots: int
    mode: str
    scores: Dict[str, Any] = field(default_factory=dict)


class RepeatedCompactionLedger:
    """Keep the live state and immutable event history for horizon evaluation."""

    def __init__(self, state: Any, slot_count: int, state_id: str = "initial"):
        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        self.state = state
        self.slot_count = int(slot_count)
        self.state_id = str(state_id)
        self.events: List[CompactionEvent] = []
        self.ancestry: Dict[str, List[str]] = {self.state_id: []}

    def append_write(self, state: Any, slot_count: int, state_id: str):
        """Advance to the actual post-write state before any compaction."""
        if slot_count < 1:
            raise ValueError("slot_count must be positive")
        self.state = state
        self.slot_count = int(slot_count)
        self.state_id = str(state_id)
        self.ancestry.setdefault(self.state_id, [self.state_id])

    def compact(
        self,
        operator: Callable[[Any], Any],
        output_slots: int,
        output_id: str,
        mode: str,
        scores: Optional[Dict[str, Any]] = None,
    ) -> CompactionEvent:
        """Run one operator on the live state and make its output the next state."""
        if output_slots < 1 or output_slots > self.slot_count:
            raise ValueError("output_slots must be in [1, current slot_count]")
        previous_id = self.state_id
        previous_state = self.state
        next_state = operator(previous_state)
        if next_state is None:
            raise ValueError("compaction operator returned None")
        event = CompactionEvent(
            step=len(self.events) + 1,
            source_ids=[previous_id],
            output_ids=[str(output_id)],
            input_slots=self.slot_count,
            output_slots=int(output_slots),
            mode=str(mode),
            scores=dict(scores or {}),
        )
        self.state = next_state
        self.slot_count = int(output_slots)
        self.state_id = str(output_id)
        self.events.append(event)
        self.ancestry[self.state_id] = self.ancestry.get(previous_id, []) + [previous_id]
        return event

    def report(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "slot_count": self.slot_count,
            "compaction_count": len(self.events),
            "events": [event.__dict__ for event in self.events],
            "ancestry": self.ancestry,
        }


def evaluate_horizon(
    initial_state: Any,
    initial_slots: int,
    writes: List[Dict[str, Any]],
    compact_fn: Callable[[Any, Dict[str, Any]], Any],
) -> Dict[str, Any]:
    """Replay writes and compactions from the previous real output state."""
    ledger = RepeatedCompactionLedger(initial_state, initial_slots)
    for write in writes:
        ledger.append_write(write["state"], int(write["slots"]), str(write["id"]))
        if write.get("compact"):
            spec = write["compact"]
            ledger.compact(
                lambda state, spec=spec: compact_fn(state, spec),
                int(spec["output_slots"]),
                str(spec["output_id"]),
                str(spec.get("mode", "unknown")),
                spec.get("scores"),
            )
    return ledger.report()
