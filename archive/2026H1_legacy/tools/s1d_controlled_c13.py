#!/usr/bin/env python3
"""Frozen C13 on controlled development episodes; sealed tests have no entry.

Each episode gets its own write-time basis and step60 alpha. Frozen means the
algorithm/hyperparameters, not reusing row3 basis/alpha on a different memory.
Evaluation probes are constructed only after the checkpoint has been saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVELOPMENT_IDS = (0, 2, 4, 8, 14, 15, 17, 18)
PROTOCOL = dict(name='S1d-controlled-C13-development-v1', rank=13, seed=0,
    steps=60, measure_every=10, lr=.003, weight_decay=.01, budget=1.,
    fact_max=40, fact_full_ce_max=2., fact_delta_threshold=.10, qa_max_tokens=256,
    tail_span=4, tail_max=40, basis='D0 damaged latest-owner fact answer gradients; SVD13',
    objective='sum of original C per-line mean tail4 CE', protection=False,
    arms=['Full', 'Selection', 'D0', 'C'], window_blocks=19, target_blocks=9,
    anchor_lines=30, block_lines=20, evaluation_questions=64,
    prompt='fixed B2 FEWSHOT', stop_strings=['\n'], max_new_tokens=32,
    primary='episode paired C-Selection generation EM', secondary='C-D0 and R/H',
    scope='controlled cross-entity generalization; shared corpus/templates',
    supervision='write-time statements and fact answers, not evaluation QA',
    evaluation_qa_used_for_training=False, sealed_test_opened=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def development_episodes(plan, wanted):
    """Inspect partition labels only for test rows; never materialize test probes."""
    if plan.get('status') != 'completed':
        raise ValueError('completed plan required')
    dev = [e for e in plan['episodes'] if e['partition'] == 'development']
    if tuple(e['episode'] for e in dev) != DEVELOPMENT_IDS:
        raise ValueError('frozen development IDs changed')
    if sum(e['partition'] == 'test' for e in plan['episodes']) != 16:
        raise ValueError('frozen sealed split changed')
    if not wanted or not set(wanted).issubset(DEVELOPMENT_IDS) or len(set(wanted)) != len(wanted):
        raise ValueError('only unique development IDs are allowed')
    return [e for e in dev if e['episode'] in wanted]


def write_inputs(episode):
    """The training boundary deliberately accepts only lines, not QA or facts metadata."""
    lines = list(episode['lines'])
    if len(lines) != 410 or hashlib.sha256('\n'.join(lines).encode()).hexdigest() != episode['context_sha256']:
        raise ValueError('episode context integrity failure')
    raw_lines = [re.sub(r'^\d+\.\s+', '', line).strip() for line in lines]
    chunks = [lines[:30]] + [lines[30 + j*20:50 + j*20] for j in range(19)]
    return raw_lines, chunks


def source_manifest():
    files = ['tools/s1d_controlled_c13.py', 'tools/s1d_c_generation.py',
             'tools/s1d_experiment1.py', 'tools/s1c_wash.py', 'tools/readout_probe.py',
             'tools/e6a_overfit.py', 'tools/native_qa_compare.py',
             'configs/agent_conf/prlmc_e6.yaml']
    files += [str(p.relative_to(ROOT)) for folder in ('methods/prlmc', 'methods/memoryllm_modeling')
              for p in (ROOT / folder).rglob('*.py')]
    return {p: sha(ROOT / p) for p in sorted(set(files))}


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    os.environ['S0B_GPU_SET'] = '1'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    import torch.nn.functional as F
    import yaml
    import e6a_overfit as E
    from torch.nn.utils import parametrize
    from transformers import AutoTokenizer
    from s1d_experiment1 import (orth, reader_evaluation, memory_digest, select_lines_v2,
                                 select_fact_cohorts, QAParallel)
    from s1_local_protected import disable_reader_dropout
    from s1c_wash import make_arm_displacement
    from s1d_c_generation import evaluate_four_arms
    from native_qa_compare import atomic_json
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.consolidate import CarrierResidualCompactor, _select_block_indices
    from methods.prlmc.probes import (_pool_in_place, _compacted_mask, Probe,
                                      differentiable_memory, active_mask_override)

    plan_path = Path(args.plan)
    if any(plan_path.with_suffix(s).exists() for s in ('.test_opened.json', '.s1d_test_opened.json')):
        raise ValueError('sealed test already opened')
    chosen = development_episodes(json.loads(plan_path.read_text()), [args.episode])
    episode = chosen[0]
    lines, chunks = write_inputs(episode)
    out = Path(args.out)
    alpha_path = out.with_suffix('.alpha.pt')
    gen_path = out.with_suffix('.generation.json')
    manifest = source_manifest()
    carrier_sha = sha(args.carrier_ckpt)
    cache_manifest = Path(args.model_path) / 'fp16_cache_manifest.json'
    identity = dict(plan_sha256=sha(plan_path), episode=args.episode,
        context_sha256=episode['context_sha256'], source_manifest=manifest,
        carrier_sha256=carrier_sha, reader_manifest_sha256=sha(cache_manifest),
        protocol=PROTOCOL, smoke=args.smoke)
    if out.exists():
        previous = json.loads(out.read_text())
        if previous.get('status') == 'complete' or not args.resume:
            raise ValueError('report exists; completed reports cannot be overwritten')
        if previous['identity'] != identity or not alpha_path.exists():
            raise ValueError('resume requires identical provenance and saved frozen checkpoint')
    elif args.resume:
        raise ValueError('no interrupted report to resume')
    elif alpha_path.exists() or gen_path.exists():
        raise ValueError('checkpoint/generation output exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    args.lr, args.budget, args.qa_max_tokens = .003, 1., 256
    MAX_SELFSUP_LINES, MASK_SPAN = 40, 4
    parallel = QAParallel()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    started = time.monotonic()
    config = dict(yaml.safe_load((ROOT / 'configs/agent_conf/prlmc_e6.yaml').read_text())['prlmc'])
    config.update(device='cuda:0', dtype='float16', active_memory_mask=True, trigger='none',
                  mode='ours', write_mode='append', gate_mode='ones', tau_alloc=-1.,
                  projection='none', basis_mode='none', functional_control='none',
                  compact_probe=False, stats_dir=None)
    cfg = PRLMCConfig.from_dict(config)
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
                                     attn_implementation='sdpa').to('cuda:0')
    model.initialized.zero_()
    model.requires_grad_(False)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    manager = ProtectedResidualMemory(model, AutoTokenizer.from_pretrained(args.model_path), cfg)
    disable_reader_dropout(model)
    state = torch.load(args.carrier_ckpt, map_location='cuda:0', weights_only=True)
    window_used, target_used = 19 * manager.block, 9 * manager.block
    line_cache = {args.episode: lines}
    report = dict(schema_version='s1d_controlled_c13/1', status='running', identity=identity,
                  stage='development', episode=args.episode, evaluation_opened=False)
    atomic_json(out, report)

    # The following kernels are verbatim copies of the reviewed original C.
    # CPU AST regression checks prevent accidental objective/injection changes.

    def base():
        arm = CarrierResidualCompactor(
            num_layers=manager.num_layers, hidden=manager.hidden,
            block=manager.block, rank=int(cfg.carrier_rank),
            gamma_max=float(cfg.carrier_gamma_max)).to(
                device="cuda:0", dtype=torch.float32)
        arm.load_state_dict(state["state_dict"])
        for x in arm.parameters():
            x.requires_grad_(False)
        return arm

    def candidate(arm):
        source = manager._memory[:, manager.total - manager.used:manager.total, :].to("cuda:0").clone()
        weights = manager._window_importance(manager.total - manager.used, window_used)
        return arm(source, weights, target=target_used, block=manager.block,
                   goal_direction=manager.goal_direction, select_weights=manager.select_weights)

    @torch.no_grad()
    def answer_ce(encoded, questions, arm=None):
        """Teacher-forced answer-token CE used by the protected oracle."""
        losses = {}
        pool, mask_m = (None, None) if arm is None else E.candidate_pool(manager, arm, "cuda:0")
        context = differentiable_memory(model, pool) if pool is not None else nullcontext()
        guard = active_mask_override(model, mask_m) if mask_m is not None else nullcontext()
        with reader_evaluation(model), context, guard:
            for question in parallel.shard(questions):
                input_ids, stem_len = encoded[question]
                labels = input_ids.clone()
                labels[:, :stem_len] = -100
                logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
                losses[question] = float(F.cross_entropy(
                    logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                    labels[:, 1:].reshape(-1), ignore_index=-100))
                del logits, labels
        del pool, mask_m
        torch.cuda.empty_cache()
        return parallel.merge(losses, questions)

    def supervision_lines(row, start, line_to_block, dropped):
        """Stage-1 v2 supervision pool: up to 40 valid dropped-block lines."""
        lines = line_cache[row]
        primary, aux = select_lines_v2(lines, line_to_block, dropped, E)
        chosen, skipped = [], 0
        for line_idx in primary + aux:
            if len(chosen) >= MAX_SELFSUP_LINES:
                break
            text = lines[line_idx].rstrip(".")
            tokens = manager.tokenizer(text, return_tensors="pt",
                                       add_special_tokens=False).input_ids
            if tokens.shape[1] < 2 * MASK_SPAN + 2:
                skipped += 1
                continue
            chosen.append((line_idx, tokens))
        return chosen, skipped

    def fact_records(row, indices, line_to_block, dropped_blocks=None):
        """Build answer-masked targets from latest raw fact statements.

        These targets are derived only from the write-time fact lines.  They do
        not call the benchmark QA renderer, so evaluation questions cannot leak
        into either the new basis or the new training loss.
        """
        lines = line_cache[row]
        latest = {}
        for line_idx, text in enumerate(lines):
            found = E.first_match(text)
            if found is not None:
                slot, subject, answer, question = found
                latest[(slot, subject)] = (line_idx, answer, question)
        records = []
        for line_idx in sorted(indices):
            block = line_to_block.get(line_idx)
            if dropped_blocks is not None and block not in dropped_blocks:
                continue
            found = E.first_match(lines[line_idx])
            if found is None:
                continue
            slot, subject, answer, question = found
            owner = latest.get((slot, subject))
            if owner is None or owner[0] != line_idx:
                continue
            raw = lines[line_idx].rstrip(".").strip()
            answer = answer.strip()
            if not answer or not raw.endswith(answer):
                continue
            prefix = raw[:-len(answer)].rstrip()
            key = "fact:%d:%d:%s" % (line_idx, slot, "|".join(subject))
            records.append({"key": key, "line_idx": line_idx,
                            "block": block, "prefix": prefix,
                            "answer": answer, "question_format": question})
        return records

    def encode_fact_records(records):
        encoded = {}
        for record in records:
            prefix, answer = record["prefix"], record["answer"]
            text = "%s %s" % (prefix, answer)
            result = manager.tokenizer(text, return_tensors="pt", add_special_tokens=False,
                                       return_offsets_mapping=True)
            tokens = result.input_ids
            offsets = result.offset_mapping[0].tolist()
            answer_char = len(prefix) + 1
            stem_len = next((i for i, (_lo, hi) in enumerate(offsets)
                             if hi > answer_char), len(offsets))
            if any(lo < len(prefix) for lo, hi in offsets[stem_len:] if hi > answer_char):
                raise RuntimeError("fact answer token crosses prefix boundary")
            width = min(args.qa_max_tokens, tokens.shape[1])
            offset = max(0, tokens.shape[1] - width)
            stem_len -= offset
            if stem_len <= 0 or stem_len >= width:
                raise RuntimeError("fact token budget removes answer or prefix: " + record["key"])
            encoded[record["key"]] = (tokens[:, offset:offset + width].to("cuda:0"), stem_len)
        return encoded

    def fact_basis(arm, encoded, keys, mask):
        """SVD basis of D0 answer-span gradients for damaged facts."""
        if not keys:
            raise RuntimeError("no damaged fact targets for new basis")
        arm.down.weight.requires_grad_(True)
        rows = []
        try:
            for key in keys:
                input_ids, stem_len = encoded[key]
                local_cand = candidate(arm)
                with torch.autograd.graph.save_on_cpu(pin_memory=True), _pool_in_place(manager, "cuda:0", local_cand,
                                    window_used, target_used) as injection:
                    pool = injection.reinject()
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=input_ids, use_cache=False,
                                       return_dict=True).logits
                        loss = F.cross_entropy(
                            logits[:, stem_len - 1:-1, :].float().reshape(-1, logits.shape[-1]),
                            input_ids[:, stem_len:].reshape(-1))
                        grad = torch.autograd.grad(loss, arm.down.weight,
                                                   retain_graph=False)[0]
                    rows.append(grad.detach().float().reshape(-1).cpu())
                del local_cand, pool, logits, loss, grad
                torch.cuda.empty_cache()
        finally:
            arm.down.weight.requires_grad_(False)
        fact_rows = torch.stack(rows).double()
        basis = orth(fact_rows)
        provenance = "fact_answer_gradients"
        if basis.shape[1] < 13:
            raise RuntimeError("fact basis rank %d < 13" % basis.shape[1])
        return basis[:, :13].contiguous(), fact_rows, provenance

    def train_arm(arm, displacement, optimizer, sup, mask,
                  protection=None):
        """Old summed tail CE, with candidate binding alive during recomputation."""
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for line_idx, tokens in sup:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                cand = candidate(arm)
                with _pool_in_place(manager, "cuda:0", cand, window_used,
                                    target_used) as injection:
                    pool = injection.reinject()
                    input_ids = tokens.to("cuda:0")
                    labels = input_ids.clone()
                    labels[:, :-MASK_SPAN] = -100
                    with differentiable_memory(model, pool), active_mask_override(model, mask):
                        logits = model(input_ids=input_ids, use_cache=False,
                                       return_dict=True).logits
                        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                               labels[:, 1:].reshape(-1))
                        loss.backward()
                        total += float(loss.detach())
                del cand, pool, loss, logits
        protect_penalty = 0.0
        if protection is not None:
            protect_penalty = backward_fact_protection(arm, *protection, mask)
        if not torch.isfinite(displacement.value.grad).all():
            raise RuntimeError("nonfinite tail gradient")
        optimizer.step()
        displacement.clamp(args.budget)
        torch.cuda.empty_cache()
        if protection is not None:
            return {"loss": total + protect_penalty,
                    "tail_loss_mean": total / max(1, len(sup)),
                    "tail_loss_sum": total, "protect_penalty": protect_penalty}
        return total / max(1, len(sup))

    try:
        writes = []
        with torch.no_grad():
            for index, chunk in enumerate(chunks):
                text = '\n'.join(chunk)
                count = len(manager.tokenizer(text, add_special_tokens=False).input_ids)
                if count > 512:
                    raise ValueError('controlled write would be truncated')
                record = manager.initialize_anchor(text, max_tokens=512) if index == 0 else manager.write(text, max_tokens=512)
                if record.get('dropped') or record.get('evicted_blocks') or record.get('compacted'):
                    raise ValueError('controlled write integrity failure')
                writes.append(dict(record, input_tokens=count))
            if manager.used != window_used or manager.block != 256:
                raise ValueError('nineteen native 256-slot blocks required')
            begin = manager.total - manager.used
            kept = _select_block_indices(manager._memory[:, begin:begin+window_used], target_used,
                manager.block, manager._window_importance(begin, window_used),
                manager.goal_direction, manager.select_weights).tolist()
        mask = _compacted_mask(manager, target_used, 'cuda:0')
        expected_slots = manager.layout.k_anchor + target_used
        if int(mask.sum()) != expected_slots:
            raise ValueError('compressed active budget mismatch')
        source_hash = memory_digest(model.memory)
        dropped = sorted(set(range(19)) - set(kept))
        indices = list(range(30, 410))
        line_to_block = {i: (i-30)//20 for i in indices}
        d0_arm = base()
        sup, skipped = supervision_lines(args.episode, 30, line_to_block, dropped)
        if len(sup) < 10:
            raise ValueError('insufficient tail training pool; no fallback')
        report.update(writes=writes, source_memory_sha256=source_hash, kept_blocks=kept,
            dropped_blocks=dropped, active_slots=expected_slots,
            tail_line_indices=[i for i, _ in sup], skipped_short_tail_lines=skipped,
            tail_token_sha256=hashlib.sha256(b''.join(t.numpy().tobytes() for _,t in sup)).hexdigest())
        atomic_json(out, report)
        if args.resume:
            saved = torch.load(alpha_path, map_location='cpu', weights_only=False)
            if saved['identity'] != identity or saved['source_memory_sha256'] != source_hash:
                raise ValueError('frozen checkpoint provenance/memory mismatch')
            basis = saved['basis']
            report['training'] = saved['training']
        else:
            discovery_started = time.monotonic()
            records = fact_records(args.episode, indices, line_to_block)
            encoded = encode_fact_records(records)
            keys = [r['key'] for r in records]
            full_ce = answer_ce(encoded, keys)
            d0_ce = answer_ce(encoded, keys, d0_arm)
            dropped_records = [r for r in records if r['block'] in dropped]
            kept_records = [r for r in records if r['block'] in kept]
            thresholded, basis_keys, _, _ = select_fact_cohorts(dropped_records, kept_records,
                encoded, full_ce, d0_ce, 40, 2., .10)
            report['fact_discovery'] = dict(records=records, full_ce=full_ce, d0_ce=d0_ce,
                thresholded_keys=thresholded, basis_keys=basis_keys,
                seconds=time.monotonic()-discovery_started)
            atomic_json(out, report)
            print('[controlled] episode=%d facts=%d damaged=%d basis_rows=%d tail=%d' %
                  (args.episode, len(records), len(thresholded), len(basis_keys), len(sup)), flush=True)
            basis_started = time.monotonic()
            basis, rows, _ = fact_basis(d0_arm, encoded, basis_keys, mask)
            torch.save(dict(identity=identity, rows=rows, basis=basis, basis_keys=basis_keys),
                       out.with_suffix('.basis.pt'))
            report['basis_seconds'] = time.monotonic()-basis_started
            del rows, encoded
        if basis.shape != (d0_arm.down.weight.numel(), 13) or not torch.isfinite(basis).all():
            raise ValueError('invalid C13 basis')
        arm = base()
        displacement = make_arm_displacement(arm.down.weight, basis, torch)
        parametrize.register_parametrization(arm.down, 'weight', displacement)
        if not torch.equal(arm.down.weight, d0_arm.down.weight):
            raise ValueError('identity-at-init weight mismatch')
        # Check functional equality using write-time facts only, before QA opens.
        audit_record = fact_records(args.episode, indices, line_to_block)[:1]
        audit_encoded = encode_fact_records(audit_record)
        audit_key = [r['key'] for r in audit_record]
        if not audit_key or answer_ce(audit_encoded, audit_key, arm) != answer_ce(audit_encoded, audit_key, d0_arm):
            raise ValueError('identity-at-init reader mismatch')
        if any(p.requires_grad for p in model.parameters()) or [p for p in arm.parameters() if p.requires_grad] != [displacement.value]:
            raise ValueError('only thirteen alpha values may train')
        report['initial_identity_verified'] = True
        frozen_basis = displacement.basis.detach().clone()
        if args.resume:
            with torch.no_grad():
                displacement.value.copy_(saved['alpha'].to(displacement.value))
        else:
            optimizer = torch.optim.AdamW([displacement.value], lr=.003, weight_decay=.01)
            history = []
            training_started = time.monotonic()
            steps = 1 if args.smoke else 60
            for step in range(1, steps+1):
                loss = train_arm(arm, displacement, optimizer, sup, mask)
                if not torch.equal(displacement.basis, frozen_basis):
                    raise ValueError('frozen basis changed')
                if step == 1 or step % 10 == 0 or step == steps:
                    history.append(dict(step=step, tail_loss_mean=loss,
                        alpha=displacement.value.detach().cpu().tolist(),
                        alpha_norm=float(displacement.value.detach().norm()),
                        elapsed_seconds=time.monotonic()-training_started))
                    report['training'] = dict(history=history, steps_completed=step,
                        seconds=time.monotonic()-training_started)
                    atomic_json(out, report)
                    print('[controlled] episode=%d step=%d/%d tail_ce=%.6f alpha_norm=%.6f seconds=%.1f' %
                          (args.episode,step,steps,loss,history[-1]['alpha_norm'],history[-1]['elapsed_seconds']),flush=True)
            saved = dict(identity=identity, source_memory_sha256=source_hash,
                basis=displacement.basis.detach().cpu(), alpha=displacement.value.detach().cpu(),
                training=report['training'])
            temporary = alpha_path.with_suffix('.pt.tmp')
            torch.save(saved, temporary)
            temporary.replace(alpha_path)
        displacement.value.requires_grad_(False)
        if not torch.isfinite(displacement.value).all() or float(displacement.value.norm()) > 1.000001:
            raise ValueError('invalid frozen alpha')
        if memory_digest(model.memory) != source_hash:
            raise ValueError('training altered source memory')
        report.update(checkpoint_sha256=sha(alpha_path), checkpoint_frozen=True)
        atomic_json(out, report)
        # Evaluation questions/answers first enter the reader after freeze.
        evaluation = episode['splits']['evaluation']
        if len(evaluation) != 64 or len({q['question'] for q in evaluation}) != 64:
            raise ValueError('exactly 64 distinct paired questions required')
        probes = [Probe(question=q['question'], answers=[q['answer']], split='development') for q in evaluation]
        select_arm = base()
        with torch.no_grad():
            select_arm.up.weight.zero_()
        args.generation_smoke = args.smoke
        generation_args = argparse.Namespace(out=str(gen_path), generation_smoke=args.smoke)
        report['evaluation_opened'] = True
        atomic_json(out, report)
        result = evaluate_four_arms(args=generation_args, manager=manager, model=model, probes=probes,
            arms=dict(Full=None, Selection=select_arm, D0=d0_arm, C=arm),
            candidate_pool=lambda a: E.candidate_pool(manager, a, 'cuda:0'),
            window_report=dict(window=[args.episode,30], baselines={}), reference=None,
            covered=[(30+j*20,50+j*20,j,None) for j in range(19)], lines=episode['lines'],
            checkpoint_sha=sha(alpha_path), carrier_sha=carrier_sha,
            driver_sha=manifest['tools/s1d_controlled_c13.py'], loaded_tf=None,
            generation_context=dict(metadata=dict(episode=args.episode, stage='development'),
                protocol=dict(qa_origin='64 held-out controlled evaluation probes',
                    checkpoint_selection='fixed step1 smoke' if args.smoke else 'fixed step60',
                    scope=PROTOCOL['scope'])))
        atomic_json(gen_path, result)
        report.update(status='complete', generation_file=str(gen_path), generation_sha256=sha(gen_path),
            summary=result['summary'], source_memory_unchanged=True,
            costs=dict(elapsed_seconds=time.monotonic()-started,
                generation_seconds=result['elapsed_seconds'],
                basis_seconds=report.get('basis_seconds'),
                fact_discovery_seconds=report.get('fact_discovery',{}).get('seconds'),
                training_seconds=report['training']['seconds'],
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                explicit_basis_fp32_bytes=displacement.basis.numel()*4,
                alpha_fp32_bytes=13*4),
            runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), physical_device=args.cuda_device))
        atomic_json(out, report)
        print('FINAL', json.dumps(report['summary']), flush=True)
    except Exception as error:
        report.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        atomic_json(out, report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default='runs/sic_paper_controlled_plan.json')
    p.add_argument('--episode', type=int, choices=DEVELOPMENT_IDS, required=True)
    p.add_argument('--cuda-device', type=int, required=True)
    p.add_argument('--model-path', default='/dev/shm/s1d_reader_fp16')
    p.add_argument('--carrier-ckpt', default='runs/e6d_interleave_ckpt.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true', help='load identical frozen checkpoint after interrupted generation')
    run(p.parse_args())


if __name__ == '__main__':
    main()
