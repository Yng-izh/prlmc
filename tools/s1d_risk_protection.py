"""Fixed write-time risk proxy; no benchmark QA inputs are accepted."""
import math

PROTECTION_COUNT = 20
CE_CEILING = 2.0
PROTECTION_WEIGHT = 1.0


def select_risk_facts(kept_records, encoded, d0_ce):
    """Largest finite D0 CE within the existing learnable-fact ceiling.

    CE is a risk proxy, not an observed correctness boundary. Eligibility uses
    no Full CE, QA answer, QA correctness, or post-training score.
    """
    candidates = {}
    for record in kept_records:
        key = record['key']
        ce = d0_ce.get(key, float('nan'))
        if key in encoded and math.isfinite(ce) and 0 <= ce <= CE_CEILING:
            candidates[key] = ce
    return sorted(candidates, key=lambda k: (-candidates[k], k))[:PROTECTION_COUNT]


def risk_protocol():
    return dict(
        name='C_risk_v1', version=1, weight=PROTECTION_WEIGHT,
        max_facts=PROTECTION_COUNT, d0_ce_ceiling=CE_CEILING,
        selection='kept latest-owner facts; finite 0<=D0 CE<=2; descending D0 CE then key; top20',
        objective='sum tail-4 CE + 1.0 * mean ReLU(kept fact answer CE - D0 CE)',
        tail_gradient_reduction='sum, unchanged from C',
        protection_gradient_reduction='mean over selected kept facts',
        qa_used_for_selection=False, qa_used_for_training=False,
        risk_is_proxy=True, weight_scan=False, checkpoint_selection='final step60 only',
        weak_gate=dict(window=[3, 17510], min_net=5, max_harmed=4),
        strong_gate=dict(window=[3, 16370], min_net=9, max_harmed=1))
