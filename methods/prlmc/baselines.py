"""Ablation modes B0-B8 and their guarantee table.

Section 5.1 separates the controls that the conclusion actually rests on. The
critical chain is B6 -> B7 -> B8 -> Ours; a win over B1 or B5 alone does not
support the claim that residual writing and adaptive compaction each contribute,
because B5 lacks the Anchor and B1 lacks everything.

The modes below are the subset this codebase can express with one backbone. B0-B4
are context-management baselines that the benchmark already implements through
its long-context and RAG agents, so they are named here for reference and mapped
to the existing agent names rather than reimplemented.

`GUARANTEES` records what each mode does and does not establish. It exists so that
a later results table cannot quietly upgrade a weaker control into a stronger
claim.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AblationMode:
    """One row of the control table."""

    code: str
    name: str
    isolated_question: str
    anchor: bool
    projection: str
    compaction: str
    implemented_here: bool
    maps_to_existing_agent: Optional[str] = None


MODES: Dict[str, AblationMode] = {
    "B0": AblationMode(
        code="B0", name="sufficient_context_reference",
        isolated_question="What is the backbone's basic task ability?",
        anchor=False, projection="none", compaction="none",
        implemented_here=False, maps_to_existing_agent="Long_context_agent_gpt-4o-mini",
    ),
    "B1": AblationMode(
        code="B1", name="recent_window_no_anchor",
        isolated_question="How hard is the long-horizon goal-loss problem?",
        anchor=False, projection="none", compaction="none",
        implemented_here=False, maps_to_existing_agent="Long_context_agent_gpt-4o-mini",
    ),
    "B2": AblationMode(
        code="B2", name="replay_goal_and_constraints_plus_window",
        isolated_question="Is turning the initial instruction into latent memory worth it at all?",
        anchor=True, projection="none", compaction="none",
        implemented_here=False, maps_to_existing_agent="Long_context_agent_gpt-4o-mini",
    ),
    "B3": AblationMode(
        code="B3", name="replay_goal_plus_periodic_text_summary",
        isolated_question="How does explicit textual working-memory compression compare?",
        anchor=True, projection="none", compaction="text",
        implemented_here=False,
    ),
    "B4": AblationMode(
        code="B4", name="replay_goal_plus_explicit_rag",
        isolated_question="How does retrievable history compare, at its own index cost?",
        anchor=True, projection="none", compaction="retrieval",
        implemented_here=False, maps_to_existing_agent="RAG_Agents_rag_bm25",
    ),
    "B5": AblationMode(
        code="B5", name="full_latent_rewrite",
        isolated_question="Baseline: rewrite the whole state every step, no Anchor.",
        anchor=False, projection="none", compaction="none",
        implemented_here=True,
    ),
    "B6": AblationMode(
        code="B6", name="anchor_plus_working_memory_full_rewrite",
        isolated_question="Is the benefit only from permanently keeping the initial instruction?",
        anchor=True, projection="none", compaction="none",
        implemented_here=True,
    ),
    "B7": AblationMode(
        code="B7", name="anchor_plus_additive_gated_residual",
        isolated_question="What does residual writing contribute on its own?",
        anchor=True, projection="none", compaction="none",
        implemented_here=True,
    ),
    "B8": AblationMode(
        code="B8", name="anchor_plus_residual_plus_fixed_period_compact",
        isolated_question="What does the compaction *timing* contribute?",
        anchor=True, projection="none", compaction="periodic",
        implemented_here=True,
    ),
    "OURS": AblationMode(
        code="Ours", name="anchor_plus_residual_plus_capacity_aware_functional_compact",
        isolated_question="Does the joint method work?",
        anchor=True, projection="none", compaction="capacity_aware_functional",
        implemented_here=True,
    ),
}

# Structural ablations from section 5.3, expressed as config overrides on top of a
# mode. Kept separate from MODES because they are not additional control arms;
# they probe one mechanism at a time.
STRUCTURAL_ABLATIONS: Dict[str, Dict[str, object]] = {
    "anchor_text_replay": {"anchor_source": "explicit"},
    "no_anchor": {"anchor_source": "none"},
    "projection_off": {"projection": "none"},
    "projection_hard": {"projection": "hard"},
    "projection_soft": {"projection": "soft"},
    "trigger_periodic": {"trigger": "periodic"},
    "trigger_occupancy_only": {"trigger": "occupancy"},
    "trigger_occupancy_novelty": {"trigger": "occupancy_novelty"},
    "trigger_with_rank": {"trigger": "occupancy_novelty_rank"},
    "compact_geometric": {"compact_mode": "svd_tokens"},
    "compact_weighted": {"compact_mode": "weighted_svd_tokens"},
    "compact_learned": {"compact_mode": "learned"},
    "compact_recent_blocks": {"compact_mode": "recent_blocks"},
    "compact_block_select": {"compact_mode": "select_blocks"},
    "no_compaction": {"compact_mode": "none"},
    "gate_off": {"gate_mode": "ones"},
    "novelty_gate": {"gate_mode": "novelty"},
}

# What each implemented mode does and does not establish. Section 6.2 lists these
# inference rules; encoding them next to the modes keeps a result table honest.
GUARANTEES: Dict[str, Dict[str, str]] = {
    "B5": {
        "establishes": "Lower bound for latent rewrite without protection.",
        "does_not_establish": "Anything about residual writing or compaction.",
    },
    "B6": {
        "establishes": "Whether an Anchor alone explains the gain.",
        "does_not_establish": "That residual writing helps; B6 has no residual path.",
    },
    "B7": {
        "establishes": "Residual writing's contribution over B6.",
        "does_not_establish": "That compaction timing matters.",
    },
    "B8": {
        "establishes": "Whether fixed-period compaction is enough.",
        "does_not_establish": "That the trigger signal matters; timing is fixed.",
    },
    "OURS": {
        "establishes": "The joint method against B6/B7/B8 at comparable cost.",
        "does_not_establish": "Superiority over text replay (B2-B4) unless those are run.",
    },
}


def get_mode(code: str) -> AblationMode:
    """Look up a mode, accepting case-insensitive aliases."""
    key = code.strip().upper()
    if key in MODES:
        return MODES[key]
    raise KeyError(f"unknown ablation mode: {code}; known: {sorted(MODES)}")


def config_overrides_for_mode(code: str) -> Dict[str, object]:
    """Translate a mode code into PRLMCConfig overrides.

    The three implemented control arms differ from Ours by exactly one mechanism
    each, which is what makes the B6 -> B7 -> B8 -> Ours chain interpretable.
    """
    key = code.strip().upper()

    if key == "B5":
        return {
            "mode": "full_rewrite",
            "anchor_source": "none",
            "compact_mode": "none",
            "projection": "none",
            "gate_mode": "ones",
        }
    if key == "B6":
        return {
            "mode": "anchor_rewrite",
            "anchor_source": "first_chunk",
            "compact_mode": "none",
            "projection": "none",
            "gate_mode": "ones",
        }
    if key == "B7":
        return {
            "mode": "ours",
            "anchor_source": "first_chunk",
            "compact_mode": "none",
            "projection": "none",
            "gate_mode": "novelty",
        }
    if key == "B8":
        return {
            "mode": "ours",
            "anchor_source": "first_chunk",
            "compact_mode": "svd_tokens",
            "trigger": "periodic",
            "projection": "none",
            "gate_mode": "novelty",
        }
    if key == "OURS":
        return {
            "mode": "ours",
            "anchor_source": "first_chunk",
            "compact_mode": "svd_tokens",
            "trigger": "occupancy_novelty_rank",
            "projection": "none",
            "gate_mode": "novelty",
        }

    raise KeyError(f"mode {code} is not implemented by this adapter; see MODES for its mapping")


def apply_structural_ablation(overrides: Dict[str, object], ablation: str) -> Dict[str, object]:
    """Fold a section 5.3 structural ablation into an override mapping."""
    if ablation not in STRUCTURAL_ABLATIONS:
        raise KeyError(f"unknown structural ablation: {ablation}; known: {sorted(STRUCTURAL_ABLATIONS)}")
    merged = dict(overrides)
    merged.update(STRUCTURAL_ABLATIONS[ablation])
    return merged


def implemented_modes() -> List[str]:
    return [code for code, mode in MODES.items() if mode.implemented_here]
