"""Probe-resampling stability and empirical calibration gates.

Resampling frequencies are perturbation diagnostics, NOT generalization CIs.
Calibration correctness is measured by greedy generation, not teacher forcing.
"""
import itertools
import math
import random


def normalized_rows(rows):
    if not rows or not rows[0]:
        raise ValueError('nonempty probe/block matrix required')
    width = len(rows[0])
    result = []
    for row in rows:
        if len(row) != width or any(not math.isfinite(x) or x < 0 for x in row):
            raise ValueError('finite rectangular nonnegative matrix required')
        peak = max(row)
        scaled = [x / peak if peak else 0.0 for x in row]
        total = sum(scaled)
        result.append([x / total if total else 0.0 for x in scaled])
    return result


def stability_select(rows, budget, *, seed=0, repeats=128):
    matrix = normalized_rows(rows)
    width = len(matrix[0])
    if type(budget) is not int or not 0 < budget < width:
        raise ValueError('strict positive reduced budget required')
    if type(repeats) is not int or repeats < 2:
        raise ValueError('at least two resamples required')
    rng = random.Random(seed)
    counts = [0] * width
    totals = [sum(row[j] for row in matrix) / len(matrix) for j in range(width)]
    # Seeded random priorities avoid treating temporal position as fact value.
    priority = list(range(width))
    rng.shuffle(priority)
    tie = {j: index for index, j in enumerate(priority)}
    selections = []
    for _ in range(repeats):
        sampled = [matrix[rng.randrange(len(matrix))] for _ in matrix]
        scores = [sum(row[j] for row in sampled) for j in range(width)]
        chosen = sorted(sorted(range(width), key=lambda j: (-scores[j], tie[j]))[:budget])
        selections.append(chosen)
        for j in chosen:
            counts[j] += 1
    chosen = sorted(sorted(range(width), key=lambda j: (-counts[j], -totals[j], tie[j]))[:budget])
    frequencies = [count / repeats for count in counts]
    overlap = sum(len(set(chosen) & set(s)) / budget for s in selections) / repeats
    return dict(kept=chosen, frequencies=frequencies, mean_overlap=overlap,
                resamples=repeats, probes=len(matrix), seed=seed,
                interpretation='resampling stability, not a reader-risk confidence bound')


def calibration_gate(native, candidate, *, alpha=0.05):
    if len(native) != len(candidate) or not native:
        raise ValueError('aligned nonempty generation outcomes required')
    if any(type(x) is not bool for x in list(native) + list(candidate)):
        raise ValueError('bool correctness outcomes required')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    rescued = sum(not n and c for n, c in zip(native, candidate))
    harmed = sum(n and not c for n, c in zip(native, candidate))
    n = rescued + harmed
    # Operational screening statistic only: questions share one memory pool.
    p = sum(math.comb(n, k) for k in range(rescued, n + 1)) / 2**n if n else 1.0
    accept = rescued > harmed and p <= alpha
    return dict(accept=accept, rescued=rescued, harmed=harmed, n=len(native),
                screening_p=p, alpha=alpha,
                reason='calibration_gain_screen_passed' if accept else 'native_fallback',
                statistical_guarantee=False)


def episode_summary(episodes, arm, *, seed=0, bootstraps=10000):
    if not episodes or type(bootstraps) is not int or bootstraps < 2:
        raise ValueError('episodes and at least two bootstrap draws required')
    differences, delta_counts, sizes = [], [], []
    rescued = harmed = base_correct = candidate_correct = total = 0
    for episode in episodes:
        records = episode['evaluation']
        n = len(records)
        if not n:
            raise ValueError('empty evaluation episode')
        b = sum(r['arms']['select']['em'] for r in records)
        c = sum(r['arms'][arm]['em'] for r in records)
        rescued += sum(not r['arms']['select']['em'] and r['arms'][arm]['em'] for r in records)
        harmed += sum(r['arms']['select']['em'] and not r['arms'][arm]['em'] for r in records)
        base_correct += b
        candidate_correct += c
        total += n
        differences.append((c - b) / n)
        delta_counts.append(c - b)
        sizes.append(n)
    rng = random.Random(seed)
    count = len(differences)
    values = sorted(sum(differences[rng.randrange(count)] for _ in range(count)) / count
                    for _ in range(bootstraps))
    observed = abs(sum(differences))
    if count <= 20:
        if len(set(sizes)) == 1:
            threshold = abs(sum(delta_counts))
            p = sum(abs(sum(s * d for s, d in zip(signs, delta_counts))) >= threshold
                    for signs in itertools.product((-1, 1), repeat=count)) / 2**count
        else:
            p = sum(abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-12
                    for signs in itertools.product((-1, 1), repeat=count)) / 2**count
        p_kind = 'exact_episode_sign_flip'
    else:
        draws = 100000
        p = (1 + sum(abs(sum(rng.choice((-1, 1)) * d for d in differences)) >= observed - 1e-12
                     for _ in range(draws))) / (draws + 1)
        p_kind = 'monte_carlo_episode_sign_flip'
    return dict(episodes=count, questions=total, select_correct=base_correct,
                candidate_correct=candidate_correct, rescued=rescued, harmed=harmed,
                macro_em_delta=sum(differences) / count,
                micro_em_delta=(candidate_correct - base_correct) / total,
                episode_bootstrap_ci95=[values[int(.025 * (bootstraps - 1))],
                                        values[int(.975 * (bootstraps - 1))]],
                sign_flip_p=p, sign_flip_method=p_kind,
                episode_deltas=differences, positive=sum(d > 0 for d in differences),
                negative=sum(d < 0 for d in differences), zero=sum(d == 0 for d in differences),
                inference_scope='controlled episodes conditional on source corpus and generator')
