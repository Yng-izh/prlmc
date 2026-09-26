"""Occupancy, effective rank, energy rank, and the compact trigger.

Section 3.3 of the note insists on keeping three capacity quantities apart:

  1. actual storage occupancy        -- hard capacity, slots or bytes
  2. representational redundancy     -- effective rank, energy rank
  3. incoming novelty pressure       -- the EMA of the feature-novelty proxy

It also states the trigger explicitly:

    compact if K_t / K_max >= tau_K and rho_bar_t >= tau_N for W consecutive updates

with two further rules that are easy to miss and are therefore implemented as
hard branches rather than as thresholds:

  * If allocation has already exceeded the budget, compact immediately. Waiting
    for the novelty EMA to qualify would let the store run over budget, which the
    note forbids.
  * Enter/exit hysteresis must prevent chatter. The exit thresholds are separate
    config fields and are validated to sit below the entry ones.

All rank functions here take the *memory activations*, never an orthonormal
basis: the note points out that the singular values of a basis are all 1 and
therefore carry no information.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from methods.prlmc.config import PRLMCConfig


def effective_rank(singular_values: torch.Tensor) -> float:
    """exp(-sum p_i log p_i) with p_i = sigma_i^2 / sum sigma_j^2.

    The note fixes the convention 0*log(0) = 0 and defines the result as 0 when
    every singular value is zero. Using torch.where rather than an epsilon inside
    the logarithm honours that convention exactly instead of nudging it.
    """
    sigma = singular_values.detach().to(torch.float64).flatten()
    if sigma.numel() == 0:
        return 0.0
    energy = sigma.pow(2)
    total = energy.sum()
    if float(total) <= 0.0:
        return 0.0
    p = energy / total
    log_p = torch.where(p > 0, torch.log(p), torch.zeros_like(p))
    entropy = -(p * log_p).sum()
    return float(torch.exp(entropy))


def energy_rank(singular_values: torch.Tensor, eta: float = 0.95) -> int:
    """Smallest k whose top-k energy share reaches eta. 0 when energy is zero."""
    sigma = singular_values.detach().to(torch.float64).flatten()
    if sigma.numel() == 0:
        return 0
    energy = sigma.pow(2)
    total = energy.sum()
    if float(total) <= 0.0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    reached = torch.nonzero(cumulative >= eta, as_tuple=False)
    if reached.numel() == 0:
        return int(sigma.numel())
    return int(reached[0].item()) + 1


def occupancy_ratio(used_slots: int, budget_slots: int) -> float:
    """Hard-capacity occupancy. Budget <= 0 is treated as unusable, not as full."""
    if budget_slots <= 0:
        return 0.0
    return float(used_slots) / float(budget_slots)


@dataclass
class CapacitySnapshot:
    """One observation of the three capacity signals."""

    used_slots: int = 0
    budget_slots: int = 0
    occupancy: float = 0.0
    effective_rank: float = 0.0
    energy_rank: int = 0
    rank_max: int = 0
    novelty_ema: float = 0.0
    over_budget: bool = False

    @property
    def rank_utilisation(self) -> float:
        """effective rank as a fraction of the statistical budget."""
        if self.rank_max <= 0:
            return 0.0
        return self.effective_rank / float(self.rank_max)

    def to_dict(self) -> Dict[str, float]:
        return {
            "used_slots": self.used_slots,
            "budget_slots": self.budget_slots,
            "occupancy": self.occupancy,
            "effective_rank": self.effective_rank,
            "energy_rank": self.energy_rank,
            "rank_max": self.rank_max,
            "rank_utilisation": self.rank_utilisation,
            "novelty_ema": self.novelty_ema,
            "over_budget": self.over_budget,
        }


class CompactTrigger:
    """Decide when to consolidate, with hysteresis and an immediate hard branch.

    The object is stateful because the trigger requires W *consecutive* updates
    above both thresholds. Any update that drops below either threshold resets
    the run, which is what makes W meaningful.
    """

    def __init__(self, config: PRLMCConfig):
        self.cfg = config
        self.consecutive = 0
        self.armed = False
        self.update_count = 0

    def reset(self) -> None:
        self.consecutive = 0
        self.armed = False
        self.update_count = 0

    def observe(self, snapshot: CapacitySnapshot) -> bool:
        """Feed one capacity observation and return whether to compact now."""
        self.update_count += 1

        # Over budget: the note requires an immediate response, not a threshold
        # wait. This bypasses hysteresis entirely.
        if snapshot.over_budget:
            self.consecutive = 0
            self.armed = True
            return True

        trigger = self.cfg.trigger

        if trigger == "periodic":
            should = self.update_count % max(1, self.cfg.compact_period) == 0
            self.consecutive = 0
            return should

        if trigger == "none":
            return False

        above_k = snapshot.occupancy >= self.cfg.tau_k
        above_n = snapshot.novelty_ema >= self.cfg.tau_n

        if trigger == "occupancy":
            above_n = True
        elif trigger not in ("occupancy_novelty", "occupancy_novelty_rank"):
            raise ValueError(f"unknown trigger: {trigger}")

        # `occupancy_novelty_rank` differs from `occupancy_novelty` only in that
        # the rank statistics are *computed* (layout._observe_capacity gates the
        # spectrum on the trigger name). They are not a veto here.
        #
        # The note calls k_eta/r_max and r_eff/r_max an auxiliary signal: a ratio
        # near 1 means the representation is hard to reduce by low-rank
        # approximation, and a low ratio means there may be redundancy to squeeze
        # out. Neither is proof of functional fidelity, so neither may decide
        # whether consolidation happens. Vetoing on `energy_rank >= rank_max`
        # blocked 494 of 500 observed writes and made the trigger unreachable.

        if above_k and above_n:
            self.consecutive += 1
        else:
            self.consecutive = 0

        if self.consecutive >= max(1, self.cfg.trigger_consecutive):
            self.armed = True
            self.consecutive = 0
            return True

        return False

    def on_compacted(self) -> None:
        """Clear the armed flag and the consecutive run after a compaction."""
        self.armed = False
        self.consecutive = 0


def summarise_trigger(config: PRLMCConfig) -> Dict[str, object]:
    """Echo the trigger configuration into the stats record."""
    return {
        "trigger": config.trigger,
        "tau_k": config.tau_k,
        "tau_n": config.tau_n,
        "tau_k_exit": config.tau_k_exit,
        "tau_n_exit": config.tau_n_exit,
        "trigger_consecutive": config.trigger_consecutive,
        "compact_period": config.compact_period,
        "rank_eta": config.rank_eta,
        "rank_max": config.rank_max,
    }
