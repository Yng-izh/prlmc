"""CSR and VR, the two metrics this project defines itself.

Section 6.1 defines, over N episodes with completion indicator s_i and violation
indicator v_i,

    CSR = (1/N) sum s_i (1 - v_i)
    VR  = (1/N) sum v_i

and states two things about how they may be reported:

  * CSR is a project-defined constrained success rate. It must not be labelled an
    official AppWorld metric unless the same evaluator is used.
  * VR uses a per-episode denominator so that an agent cannot dilute its
    violation rate by taking extra ineffective steps. The per-step rate is a
    diagnostic only.

That second rule is a real measurement hazard, so `violation_rate` counts
episodes rather than steps, and `step_violation_rate` is a separate function
whose docstring says it is diagnostic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class EpisodeOutcome:
    """One episode's outcome for the constrained metrics.

    `violations` holds the individual constraint checks so the trajectory-level
    and terminal-state checks of section 4.2 stay distinguishable. `completed` is
    the goal indicator; it is never inferred from a text claim of success.
    """

    episode_id: str = ""
    completed: bool = False
    violations: List[str] = field(default_factory=list)
    steps: int = 0
    terminal_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def violated(self) -> bool:
        return bool(self.violations) or bool(self.terminal_violations)


def constrained_success_rate(outcomes: Sequence[EpisodeOutcome]) -> float:
    """CSR = mean of s_i * (1 - v_i). Returns 0.0 for an empty input."""
    if not outcomes:
        return 0.0
    total = sum(1.0 for o in outcomes if o.completed and not o.violated)
    return total / len(outcomes)


def violation_rate(outcomes: Sequence[EpisodeOutcome]) -> float:
    """VR = fraction of episodes with at least one violation.

    Episode-denominated by design: an agent that takes many ineffective steps
    cannot push this number down (section 6.1).
    """
    if not outcomes:
        return 0.0
    return sum(1.0 for o in outcomes if o.violated) / len(outcomes)


def goal_completion_rate(outcomes: Sequence[EpisodeOutcome]) -> float:
    """Fraction of episodes that completed the goal, ignoring violations."""
    if not outcomes:
        return 0.0
    return sum(1.0 for o in outcomes if o.completed) / len(outcomes)


def step_violation_rate(outcomes: Sequence[EpisodeOutcome]) -> float:
    """Violations per step. DIAGNOSTIC ONLY -- do not report as the headline VR.

    Included because the note calls for it as a diagnostic, and separated because
    its denominator makes it trivially gameable by taking extra steps.
    """
    steps = sum(o.steps for o in outcomes)
    if steps <= 0:
        return 0.0
    return sum(len(o.violations) for o in outcomes) / float(steps)


def violation_breakdown(outcomes: Sequence[EpisodeOutcome]) -> Dict[str, int]:
    """Count each distinct violation label so a single constraint cannot hide."""
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        for label in list(outcome.violations) + list(outcome.terminal_violations):
            counts[label] = counts.get(label, 0) + 1
    return counts


def summarise(outcomes: Sequence[EpisodeOutcome]) -> Dict[str, Any]:
    """All constrained metrics together, with the diagnostic clearly labelled."""
    return {
        "n_episodes": len(outcomes),
        "CSR": constrained_success_rate(outcomes),
        "VR": violation_rate(outcomes),
        "goal_completion_rate": goal_completion_rate(outcomes),
        "step_violation_rate_diagnostic": step_violation_rate(outcomes),
        "violation_breakdown": violation_breakdown(outcomes),
        "note": "CSR and VR are project-defined (see section 6.1); not official benchmark metrics",
    }


# ---------------------------------------------------------------------------
# Metric-splitting helpers
#
# Section 2 and section 4 require MemoryAgentBench results to be reported by
# capability rather than as one aggregate, so that a regression in selective
# forgetting cannot be hidden by a gain in retrieval.
# ---------------------------------------------------------------------------

CAPABILITIES = (
    "accurate_retrieval",
    "test_time_learning",
    "long_range_understanding",
    "conflict_resolution",
    "selective_forgetting",
)


def capability_of(dataset_name: str) -> str:
    """Map a MemoryAgentBench dataset name to its capability bucket."""
    lowered = dataset_name.lower()
    if "eventqa" in lowered:
        return "accurate_retrieval"
    if lowered.startswith("icl_") or "test_time_learning" in lowered:
        return "test_time_learning"
    if "detective" in lowered or "long_range" in lowered:
        return "long_range_understanding"
    if "factconsolidation" in lowered or "conflict" in lowered:
        return "conflict_resolution"
    if "forget" in lowered or "ruler" in lowered:
        return "selective_forgetting"
    return "unclassified"


def split_by_capability(records: Iterable[Dict[str, Any]], key: str = "sub_dataset") -> Dict[str, List[Dict[str, Any]]]:
    """Group per-query records by capability instead of returning one average."""
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CAPABILITIES}
    grouped["unclassified"] = []
    for record in records:
        grouped[capability_of(str(record.get(key, "")))].append(record)
    return grouped
