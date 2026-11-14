#!/usr/bin/env python3
"""Evidence chain: four tables, read from finished eval reports, standard library only.

    python tools/evidence_chain.py --eval reports/cross_block/eval_v13_lr.json \
        [reports/cross_block/eval_v17.json ...] --out reports/cross_block/evidence_chain.md

Nothing here runs a model, trains anything or recomputes a prediction: every number is
read out of the `samples` of the reports named on the command line. That is the point --
the chain has to be checkable against the artefacts themselves, not against a second
harness that could disagree with them.

The four tables
---------------
1. THREE-ARM SEPARATION. Trained / NativeGraft / RandomControl / Counterfactual: questions
   correct, accuracy, mean official QA F1, plus the pre-stated bars --
   `Trained >= NativeGraft + 4 questions`, and both controls (random replacement of the
   candidate content, counterfactual donor content) falling back to NativeGraft or below.
2. McNEMAR. The paired 2x2 of Trained against each control: `b` = rows only Trained
   answers, `c` = rows only the control answers, `net = b - c`, and the exact two-sided
   binomial p-value over the discordant pairs. The four target lines (Harmed <= 2,
   Rescued >= 13, net >= +11, p < 0.05) are each marked PASS / FAIL.
3. EVIDENCE POSITION. Rows split by where the supporting facts are (inside the dropped
   blocks, inside the retained blocks, not in the window at all, or not located) and by
   whether the uncompressed Full arm already answers them (the protected group); rescues
   and harms are attributed to those groups rather than to the run as a whole.
4. METADATA CLOSURE. Capacity 2,304, the graft share, the merged / absorbed counters, and
   whether the slot-age fields survived the pipeline -- each one checked, each one printed
   as MISSING when the report does not carry it.

What this tool refuses to do
----------------------------
Invent a number. Arms are resolved by an alias list (`--arm ROLE=NAME` pins one
explicitly) and the resolved arm name is printed next to every cell; when no alias is
present in a report the cell says `MISSING` and the reason is printed under the table.
An arm that exists but is missing on some rows is counted over the rows it has, and the
row count is printed so a thinner denominator is visible. No silent skipping, no
substitution of one arm for another.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

MISSING = "MISSING"
# The closure the round's budget promises (spec §0 / §6): 2,304 = 1,620 core + 684 graft.
TARGET_SLOTS = 2304
TARGET_CORE = 1620
TARGET_GRAFT = 684
# Roles, with the names finished reports actually use. First match wins and is printed.
ROLES = (
    ("Trained", ("V17", "V13Plus", "V16", "V15", "V14", "V13", "Trained", "Ours")),
    # `Graft` is the native graft arm (V10's operator: 1,944 carriers + 360 grafts).
    # `V11Evict` is NOT it -- that is the ordinary full-pool calibration arm, and listing
    # it here turned a report that only has V11Evict into a "native graft baseline".
    ("NativeGraft", ("NativeGraft", "Graft")),
    ("RandomControl", ("RandomControl", "V17Random", "RandomGraft", "RandGraft",
                       "RandomControlGraft", "Random")),
    ("Counterfactual", ("Counterfactual", "V17Donor", "V13PlusDonor", "V16Donor",
                        "V15Donor", "V14Donor", "V13Donor", "V11EvictDonor", "Donor")),
)
# The pre-stated bars of the evidence chain (spec §6). Not tuned after the fact.
BARS = dict(rescued=13, harmed=2, net=11, alpha=0.05, over_native=4)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def load_report(path):
    """The report, or a MISSING verdict -- a missing file is data, not an exception."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"cannot read {path}: {error}"
    if not isinstance(payload, dict) or "samples" not in payload:
        return None, f"{path} carries no `samples`"
    return payload, None


def arm_entry(sample, arm):
    return ((sample.get("arms") or {}).get(arm)) if arm else None


def arm_correct(sample, arm, threshold):
    """Correct / not, from the report's own verdict, else from its qa_f1."""
    entry = arm_entry(sample, arm)
    if not entry:
        return None
    value = entry.get("correct")
    if isinstance(value, bool):
        return value
    f1 = entry.get("qa_f1")
    if f1 is None:
        return None
    return bool(float(f1) >= float(threshold))


def arm_f1(sample, arm):
    entry = arm_entry(sample, arm)
    if not entry or entry.get("qa_f1") is None:
        return None
    return float(entry["qa_f1"])


def threshold_of(payload, samples):
    """The F1 threshold the run used, read from the report rather than assumed."""
    for sample in samples:
        for entry in (sample.get("arms") or {}).values():
            if entry.get("f1_threshold") is not None:
                return float(entry["f1_threshold"])
    protocol = payload.get("protocol") or {}
    if protocol.get("f1_threshold") is not None:
        return float(protocol["f1_threshold"])
    return 0.5


def resolve_roles(payload, samples, overrides):
    """Role -> arm name, resolved by alias; `overrides` pins a name explicitly."""
    present = sorted({arm for sample in samples for arm in (sample.get("arms") or {})})
    resolved, unresolved = {}, {}
    for role, aliases in ROLES:
        if role in overrides:
            name = overrides[role]
            if name not in present:
                unresolved[role] = (f"--arm pinned {name}, which this report does not "
                                    f"carry (arms: {', '.join(present) or 'none'})")
                resolved[role] = None
            else:
                resolved[role] = name
            continue
        match = next((alias for alias in aliases if alias in present), None)
        resolved[role] = match
        if match is None:
            unresolved[role] = ("no alias of " + " / ".join(aliases)
                                + " appears in this report")
    return resolved, unresolved, present


def _stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=statistics.mean(values), min=min(values),
                max=max(values))


def _fmt(value, digits=1, suffix=""):
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _verdict(ok):
    return ("MISSING" if ok is None else ("PASS" if ok else "FAIL"))


# ---------------------------------------------------------------------------
# table 1: three-arm separation
# ---------------------------------------------------------------------------
def table_separation(samples, resolved, unresolved, threshold):
    """"Does the trained arm beat the native graft, and do the controls fall back?"""
    rows, notes = [], []
    counts = {}
    for role, _ in ROLES:
        arm = resolved.get(role)
        if arm is None:
            rows.append(f"| {role} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | "
                        f"{MISSING} |")
            notes.append(f"{role}: {unresolved.get(role, MISSING)}")
            counts[role] = None
            continue
        values = [arm_correct(sample, arm, threshold) for sample in samples
                  if arm_entry(sample, arm)]
        values = [value for value in values if value is not None]
        f1s = [arm_f1(sample, arm) for sample in samples if arm_entry(sample, arm)]
        correct = int(sum(1 for value in values if value))
        counts[role] = dict(correct=correct, n=len(values))
        rows.append(f"| {role} | {arm} | {len(values)} | {correct} | "
                    f"{100.0 * correct / max(len(values), 1):.1f}% | "
                    f"{_fmt(_stats(f1s).get('mean'), 4)} |")

    def pair(role):
        return counts.get(role), counts.get("NativeGraft")

    trained, native = pair("Trained")
    gate_over_native = None
    if trained and native:
        gate_over_native = (trained["correct"] - native["correct"]) >= BARS["over_native"]
        notes.append(f"Trained − NativeGraft = {trained['correct'] - native['correct']} "
                     f"questions (bar: >= +{BARS['over_native']})")
    elif trained is None or native is None:
        notes.append("Trained − NativeGraft: " + MISSING + " (an arm is absent)")

    control_gates = {}
    for role in ("RandomControl", "Counterfactual"):
        control, _ = pair(role)
        if control is None or native is None:
            control_gates[role] = None
            continue
        control_gates[role] = control["correct"] <= native["correct"]
        notes.append(f"{role} − NativeGraft = {control['correct'] - native['correct']} "
                     f"questions (bar: <= 0, the control must fall back)")

    lines = [
        "| Role | Arm in the report | Rows | Correct | Accuracy | Mean F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ] + rows + [
        "",
        "| Criterion | Verdict | Detail |",
        "| --- | --- | --- |",
        f"| Trained >= NativeGraft + {BARS['over_native']} questions | "
        f"{_verdict(gate_over_native)} | "
        + (MISSING if trained is None or native is None else
           f"{trained['correct']} vs {native['correct']}") + " |",
        f"| RandomControl <= NativeGraft | {_verdict(control_gates['RandomControl'])} | "
        + (MISSING if counts.get("RandomControl") is None or native is None else
           f"{counts['RandomControl']['correct']} vs {native['correct']}") + " |",
        f"| Counterfactual <= NativeGraft | {_verdict(control_gates['Counterfactual'])} | "
        + (MISSING if counts.get("Counterfactual") is None or native is None else
           f"{counts['Counterfactual']['correct']} vs {native['correct']}") + " |",
        "",
        "Mean F1 is the mean official HotpotQA QA F1 over the same rows; a row is correct "
        "when its F1 >= the report's threshold.",
    ]
    return lines, notes, counts


# ---------------------------------------------------------------------------
# table 2: McNemar
# ---------------------------------------------------------------------------
def _mcnemar_p(b, c):
    """Exact two-sided McNemar p-value over the discordant pairs.

    Only the `b + c` rows where the two arms disagree carry any information. Under the
    null the split of those rows between the two sides is Binomial(b + c, 1/2), so

        p = min(1, 2 * sum_{k=0}^{min(b, c)} C(b + c, k) / 2^(b + c))

    Two-sided, exact, no continuity correction and no normal approximation (with b + c in
    the tens the chi-square version is optimistic). n = 0 gives p = 1: no discordant pair,
    no evidence either way.
    """
    b, c = int(b), int(c)
    n = b + c
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / float(2 ** n))


def mcnemar(samples, arm_a, arm_b, threshold):
    """The paired 2x2 of `arm_a` against `arm_b` over the rows both are present on."""
    b = c = same = paired = 0
    for sample in samples:
        left = arm_correct(sample, arm_a, threshold)
        right = arm_correct(sample, arm_b, threshold)
        if left is None or right is None:
            continue
        paired += 1
        if left and not right:
            b += 1
        elif right and not left:
            c += 1
        elif left and right:
            same += 1
    return dict(paired=paired, both_correct=same, rescued=b, harmed=c,
                discordant=b + c, net=b - c, p=_mcnemar_p(b, c))


def table_mcnemar(samples, resolved, threshold):
    """Trained against every control, with the four target lines marked one by one."""
    trained = resolved.get("Trained")
    lines = [
        "| Arm A (treatment) | Arm B (control) | Paired rows | Both correct | Rescued (A only) "
        "| Harmed (B only) | Net | Exact two-sided p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    verdicts, notes = [], []
    if trained is None:
        notes.append(f"Trained arm {MISSING}: no paired test can be run")
        lines.append(f"| {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | "
                     f"{MISSING} | {MISSING} | {MISSING} |")
        return lines, notes, verdicts

    for role in ("NativeGraft", "RandomControl", "Counterfactual"):
        control = resolved.get(role)
        if control is None:
            notes.append(f"{role} {MISSING}: the report carries no such arm, so "
                         f"Trained vs {role} is not reported")
            lines.append(f"| {trained} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | "
                         f"{MISSING} | {MISSING} | {MISSING} |")
            continue
        result = mcnemar(samples, trained, control, threshold)
        lines.append(f"| {trained} | {control} | {result['paired']} | "
                     f"{result['both_correct']} | {result['rescued']} | "
                     f"{result['harmed']} | {result['net']:+d} | {result['p']:.4g} |")
        verdicts.append(dict(
            control=role, arm=control, **result,
            harmed_ok=result["harmed"] <= BARS["harmed"],
            rescued_ok=result["rescued"] >= BARS["rescued"],
            net_ok=result["net"] >= BARS["net"],
            p_ok=result["p"] < BARS["alpha"]))

    lines += [
        "",
        "p-value: exact two-sided binomial test on the discordant pairs at p = 1/2, "
        "`p = min(1, 2 * P(X <= min(b, c)))`, X ~ Binomial(b + c, 1/2). Only the rows "
        "where the two arms disagree are used; `net = rescued - harmed`.",
    ]
    if verdicts:
        lines += [
            "",
            f"| Pair | Harmed <= {BARS['harmed']} | Rescued >= {BARS['rescued']} | "
            f"Net >= +{BARS['net']} | p < {BARS['alpha']} | All four |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in verdicts:
            all_four = (item["harmed_ok"] and item["rescued_ok"] and item["net_ok"]
                        and item["p_ok"])
            lines.append(
                f"| {trained} vs {item['arm']} | {_verdict(item['harmed_ok'])} "
                f"({item['harmed']}) | {_verdict(item['rescued_ok'])} "
                f"({item['rescued']}) | {_verdict(item['net_ok'])} ({item['net']:+d}) | "
                f"{_verdict(item['p_ok'])} ({item['p']:.4g}) | {_verdict(all_four)} |")
        primary = verdicts[0]
        notes.append(
            f"the four target lines belong to the primary pair "
            f"({trained} vs {primary['arm']}): harmed {primary['harmed']}, rescued "
            f"{primary['rescued']}, net {primary['net']:+d}, p {primary['p']:.4g}")
    return lines, notes, verdicts


# ---------------------------------------------------------------------------
# table 3: evidence position
# ---------------------------------------------------------------------------
GROUPS = ("dropped", "kept", "outside", "unlocated")


def evidence_group(sample, window_blocks):
    """Where this row's supporting facts are: dropped blocks, retained blocks, nowhere."""
    evidence = sample.get("evidence") or {}
    blocks = set(int(block) for block in (evidence.get("blocks") or []))
    if not evidence.get("located") or not blocks:
        return "unlocated"
    inside = blocks
    if window_blocks is not None:
        # Evidence beyond the written window cannot be read at all; a row whose facts are
        # partly inside is judged by the part that made it into the pool.
        inside = set(block for block in blocks if block < int(window_blocks))
        if not inside:
            return "outside"
    dropped = set(int(block) for block in (sample.get("dropped_blocks") or []))
    return "dropped" if inside <= dropped else "kept"


def table_evidence(samples, resolved, payload, threshold):
    """Rescues and harms attributed to where the evidence actually sits."""
    protocol = payload.get("protocol") or {}
    window_blocks = protocol.get("window_blocks")
    trained = resolved.get("Trained")
    native = resolved.get("NativeGraft")
    full = next((arm for arm in ("Full",) if any(arm_entry(s, arm) for s in samples)), None)

    buckets = {group: dict(rows=0) for group in GROUPS}
    for sample in samples:
        group = evidence_group(sample, window_blocks)
        buckets[group]["rows"] += 1
    protected = dict(rows=0)
    for sample in samples:
        if full and arm_correct(sample, full, threshold):
            protected["rows"] += 1

    def attribute(pool):
        entry = dict(rows=len(pool))
        if trained and native:
            entry["rescued"] = int(sum(
                1 for sample in pool
                if arm_correct(sample, trained, threshold) is True
                and arm_correct(sample, native, threshold) is False))
            entry["harmed"] = int(sum(
                1 for sample in pool
                if arm_correct(sample, native, threshold) is True
                and arm_correct(sample, trained, threshold) is False))
        if trained:
            entry["trained_correct"] = int(sum(
                1 for sample in pool if arm_correct(sample, trained, threshold)))
        if native:
            entry["native_correct"] = int(sum(
                1 for sample in pool if arm_correct(sample, native, threshold)))
        if full:
            entry["full_correct"] = int(sum(
                1 for sample in pool if arm_correct(sample, full, threshold)))
        return entry

    groups = {}
    for group in GROUPS:
        groups[group] = attribute([sample for sample in samples
                                   if evidence_group(sample, window_blocks) == group])
    groups["protected (Full correct)"] = attribute(
        [sample for sample in samples if full and arm_correct(sample, full, threshold)])

    lines = [
        f"| Evidence position | Rows | {trained or MISSING} correct | "
        f"{native or MISSING} correct | Rescued by Trained | Harmed by Trained | "
        f"Full correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = dict(dropped="evidence ONLY in blocks that were dropped",
                  kept="evidence in retained (carrier) blocks",
                  outside="evidence beyond the written window",
                  unlocated="supporting facts not located")
    for group in GROUPS + ("protected (Full correct)",):
        entry = groups[group]
        label = labels.get(group, group)
        lines.append(
            f"| {group}: {label} | {entry['rows']} | "
            f"{_fmt(entry.get('trained_correct'))} | {_fmt(entry.get('native_correct'))} | "
            f"{_fmt(entry.get('rescued'))} | {_fmt(entry.get('harmed'))} | "
            f"{_fmt(entry.get('full_correct'))} |")
    rescued_total = sum(groups[group].get("rescued") or 0 for group in GROUPS)
    harmed_total = sum(groups[group].get("harmed") or 0 for group in GROUPS)
    protected_entry = groups["protected (Full correct)"]
    lines += [
        "",
        f"Attribution over the four position groups (each row in exactly one): "
        f"{_fmt(rescued_total)} rescues, {_fmt(harmed_total)} harms. The protected group "
        f"is reported separately -- {_fmt(protected_entry.get('rescued'))} rescues and "
        f"{_fmt(protected_entry.get('harmed'))} harms on the rows Full already answers -- "
        f"because those rows also sit in a position group and must not be added twice.",
        "`dropped` = every supporting-fact block is one of the blocks the window builder "
        "discarded; `kept` = at least one supporting-fact block survived as a carrier. "
        "Evidence beyond the 19 written blocks ('outside') cannot be read at all, and is "
        "kept separate from 'unlocated' (no supporting facts were found for the row).",
    ]
    notes = []
    if not trained or not native:
        notes.append(f"Rescued/Harmed need both Trained and NativeGraft: {MISSING} cells "
                     f"above are arms this report does not carry")
    if not full:
        notes.append(f"protected group {MISSING}: the report carries no Full arm")
    return lines, notes, groups


# ---------------------------------------------------------------------------
# table 4: metadata closure
# ---------------------------------------------------------------------------
def table_metadata(samples, payload):
    """Capacity, graft share, merge / absorb counters, slot-age fields: checked, not assumed."""
    protocol = payload.get("protocol") or {}
    lines = [
        "| Check | Expected | Observed | Verdict |",
        "| --- | --- | --- | --- |",
    ]
    notes = []

    def row(name, expected, observed, ok):
        lines.append(f"| {name} | {expected} | {observed} | {_verdict(ok)} |")

    budget = protocol.get("budget")
    row("protocol.budget", TARGET_SLOTS,
        _fmt(budget) if budget is not None else MISSING,
        (int(budget) == TARGET_SLOTS) if budget is not None else None)

    capacities = [int(sample["v11"]["capacity"]) for sample in samples
                  if isinstance(sample.get("v11"), dict) and "capacity" in sample["v11"]]
    observed = (f"{_stats(capacities)['min']:.0f}..{_stats(capacities)['max']:.0f} "
                f"over {len(capacities)} rows" if capacities else MISSING)
    row("per-row capacity (sample.v11.capacity)", TARGET_SLOTS, observed,
        all(capacity == TARGET_SLOTS for capacity in capacities) if capacities else None)

    splits = [(int(sample["v11"]["core_total"]), int(sample["v11"]["graft_total"]))
              for sample in samples
              if isinstance(sample.get("v11"), dict)
              and "core_total" in sample["v11"] and "graft_total" in sample["v11"]]
    observed = ("; ".join(sorted({f"{core}+{graft}" for core, graft in splits}))
                if splits else MISSING)
    row("per-row split (core + graft)", f"{TARGET_CORE} + {TARGET_GRAFT}", observed,
        all(core == TARGET_CORE and graft == TARGET_GRAFT for core, graft in splits)
        if splits else None)

    if splits:
        shares = [graft / float(core + graft) for core, graft in splits]
        share = _stats(shares)
        row("graft share of the window",
            f"{100.0 * TARGET_GRAFT / TARGET_SLOTS:.2f}%",
            f"{100.0 * share['mean']:.2f}% (min {100.0 * share['min']:.2f}%, "
            f"max {100.0 * share['max']:.2f}%)",
            abs(share["mean"] - TARGET_GRAFT / float(TARGET_SLOTS)) < 1e-9)
    else:
        row("graft share of the window", f"{100.0 * TARGET_GRAFT / TARGET_SLOTS:.2f}%",
            MISSING, None)

    fields = ("merged_pairs", "avoided_evictions", "blocks_with_no_merge",
              "mean_cluster_size", "folded_slots")
    for field in fields:
        values = [sample["v11"][field] for sample in samples
                  if isinstance(sample.get("v11"), dict) and field in sample["v11"]]
        if not values:
            row(f"v11.{field}", "present on every row", MISSING, None)
            notes.append(f"v11.{field} {MISSING}: the merge counters are not carried in "
                         f"these reports")
            continue
        stats = _stats(values)
        row(f"v11.{field}", "present on every row",
            f"mean {stats['mean']:.3f}, min {stats['min']:.3f}, max {stats['max']:.3f} "
            f"(n={stats['n']}/{len(samples)})",
            stats["n"] == len(samples))

    absorbed = ("absorbed_pairs", "absorbed_mean_alpha", "absorbed_mean_cos")
    for field in absorbed:
        values = [sample["v11"][field] for sample in samples
                  if isinstance(sample.get("v11"), dict) and field in sample["v11"]]
        if not values:
            row(f"v11.{field}", "present when absorption is on", MISSING, None)
            continue
        stats = _stats(values)
        row(f"v11.{field}", "present when absorption is on",
            f"mean {stats['mean']:.3f}, max {stats['max']:.3f} (n={stats['n']})",
            stats["n"] == len(samples))
    if not any(isinstance(sample.get("v11"), dict) and "absorbed_pairs" in sample["v11"]
               for sample in samples):
        notes.append("absorbed_pairs / absorbed_mean_alpha / absorbed_mean_cos are "
                     f"{MISSING}: absorption (--block-merge-cos) was off in this run, or "
                     "the telemetry was not carried into the report -- the two cannot be "
                     "told apart from the report alone")

    ages = [sample.get("carrier_ages") for sample in samples]
    present = [value for value in ages if isinstance(value, list)]
    target_blocks = protocol.get("target_blocks")
    lengths = sorted({len(value) for value in present})
    row("slot age (sample.carrier_ages)",
        f"one integer per carrier ({target_blocks} when protocol.target_blocks is set)",
        (f"present on {len(present)}/{len(samples)} rows, length(s) {lengths}"
         if present else MISSING),
        (len(present) == len(samples)
         and (target_blocks is None or lengths == [int(target_blocks)]))
        if present else None)
    if not present:
        notes.append("carrier_ages " + MISSING + ": no per-carrier age is carried, so a "
                     "slot's lifetime cannot be audited from this report")

    slot_level = [sample for sample in samples
                  if isinstance(sample.get("slot_age"), list)
                  or isinstance(sample.get("delta_position"), list)]
    row("per-slot age (slot_age / delta_position)",
        "one value per slot (2,304)",
        (f"present on {len(slot_level)}/{len(samples)} rows" if slot_level else MISSING),
        (len(slot_level) == len(samples)) if slot_level else None)
    if not slot_level:
        notes.append("per-slot age (slot_age / delta_position) " + MISSING
                     + ": `meta['delta_position']` exists in the window metadata but is "
                       "not exported per slot, so slot-level ageing is unverifiable here")

    carried = sorted(key for key in (samples[0] if samples else {})
                     if key in ("v11", "evict", "evict_donor", "dropped_blocks",
                                "selected_blocks", "carrier_ages", "evidence"))
    lines += [
        "",
        f"Telemetry blocks present in the samples: {', '.join(carried) or MISSING}.",
        "Every row above is read from the report's own `protocol` / `sample.v11` fields; "
        "a MISSING cell means the field was not carried, and no value was inferred for it.",
    ]
    return lines, notes


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def analyse(path, overrides):
    """The four tables for one report, as markdown lines (a list of sections)."""
    payload, error = load_report(path)
    if payload is None:
        return [dict(title=f"{path}", lines=[f"{MISSING} -- {error}"], notes=[])]
    samples = [sample for sample in payload["samples"] if isinstance(sample, dict)]
    graded = [sample for sample in samples if sample.get("status", "graded") == "graded"]
    if not graded:
        return [dict(title=str(path),
                     lines=[f"{MISSING} -- the report holds no graded sample"], notes=[])]
    threshold = threshold_of(payload, graded)
    resolved, unresolved, present = resolve_roles(payload, graded, overrides)
    header = [
        f"schema: {payload.get('schema', MISSING)} · status: "
        f"{payload.get('status', MISSING)} · graded rows: {len(graded)} · "
        f"F1 threshold: {threshold}",
        f"arms in the report: {', '.join(present) or MISSING}",
        "role resolution: " + "; ".join(
            f"{role} -> {resolved[role] or MISSING}" for role, _ in ROLES),
    ]
    sections = [dict(title=f"report: {path}", lines=header, notes=[])]

    lines, notes, _ = table_separation(graded, resolved, unresolved, threshold)
    sections.append(dict(title="1. three-arm separation", lines=lines, notes=notes))

    lines, notes, _ = table_mcnemar(graded, resolved, threshold)
    sections.append(dict(title="2. McNemar (paired, exact two-sided)", lines=lines,
                         notes=notes))

    lines, notes, _ = table_evidence(graded, resolved, payload, threshold)
    sections.append(dict(title="3. evidence position", lines=lines, notes=notes))

    lines, notes = table_metadata(graded, payload)
    sections.append(dict(title="4. metadata closure", lines=lines, notes=notes))
    return sections


def render(path, sections):
    out = [f"## {path}", ""]
    for section in sections:
        out += [f"### {section['title']}", ""] + section["lines"]
        if section["notes"]:
            out += ["", "Notes: " + "; ".join(section["notes"])]
        out.append("")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", dest="eval", nargs="+", required=True,
                        help="one or more finished eval reports (each analysed on its own "
                             "row set: two runs are never pooled into one paired test)")
    parser.add_argument("--out", default=None, help="write the four tables here as markdown")
    parser.add_argument("--arm", dest="arm", action="append", default=[],
                        help="pin a role to an arm name, e.g. --arm Trained=V17 "
                             "(repeatable); otherwise the alias list decides and prints "
                             "which name it used")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.arm:
        if "=" not in item:
            raise SystemExit(f"--arm expects ROLE=NAME, got {item!r}")
        role, name = item.split("=", 1)
        role, name = role.strip(), name.strip()
        if role not in dict(ROLES):
            raise SystemExit(f"--arm: unknown role {role!r}, expected one of "
                             f"{[role for role, _ in ROLES]}")
        overrides[role] = name

    written, text = [], ["# Evidence chain", "",
                         "Read from finished reports; nothing here retrains or re-reads a "
                         "model. A cell says MISSING when the report does not carry it.", ""]
    for path in args.eval:
        sections = analyse(path, overrides)
        lines = render(path, sections)
        text += lines
        written.append((path, len(sections)))
        print("\n".join(lines), flush=True)

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(text) + "\n", encoding="utf-8")
        print(f"\nwrote {target} ({len(written)} report(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
