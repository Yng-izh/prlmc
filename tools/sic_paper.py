"""SIC-P1 controlled fact-stream experiment, no runtime production changes.

Development, freeze and test are separate commands. Test validates the full
protocol/code/data freeze. Gold is never concatenated into generation queries.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from native_qa_compare import atomic_json
from sic_stability import calibration_gate, episode_summary, stability_select

PROTOCOL = dict(name='SIC-P1', setting='controlled_entity_disjoint_versioned_fact_streams',
                scope='synthetic rearrangement/reassignment of source facts; not native benchmark score',
                window_blocks=19, target_blocks=9, block=256, seed=20261012,
                discovery=32, calibration=32, evaluation=64, bootstrap_repeats=128,
                calibration_alpha=.05, primary='gated_stability', reference='select',
                arms=['select', 'raw', 'eligible_normalized', 'stability', 'gated_stability', 'recent', 'random'],
                generation=dict(max_new_tokens=16, greedy=True, stop_strings=['\n'], batch_size=1),
                primary_metric='all_question_official_style_generation_em',
                structural_metric='specified_text_support_coverage_gap_to_oracle',
                acceptance_gates=['ungated_sic_em_not_below_prespecified_nonoracle_baselines',
                                 'ungated_sic_support_oracle_gap_smaller_than_native'],
                oracle_gap_tolerance=None,
                fallback_scope='native fallback is operational protection, never evidence of SIC superiority',
                inference='episode sign flip under symmetry assumption + episode bootstrap95',
                oracle_scope='support oracle is diagnostic target, not reader EM upper bound',
                deployment='labels derived from available pre-compression source text via fixed templates',
                future_query_answers_used_for_selection=False,
                zero_pool_eligibility='secondary operational ablation, not causal necessity',
                test_policy='one prespecified primary comparison; no reselection from test')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def code_manifest():
    paths = ['tools/sic_paper.py', 'tools/sic_paper_data.py', 'tools/sic_stability.py',
             'tools/sic_data_inventory.py', 'tools/e6a_overfit.py', 'tools/sic_probe.py',
             'tools/readout_probe.py', 'tools/native_qa_compare.py',
             'methods/prlmc/sic.py', 'methods/prlmc/layout.py', 'methods/prlmc/probes.py',
             'methods/prlmc/consolidate.py', 'methods/prlmc/config.py',
             'configs/agent_conf/prlmc_e6.yaml', 'methods/memoryllm_modeling/modeling_memoryllm.py']
    paths += [str(p.relative_to(ROOT)) for folder in ('methods/prlmc', 'methods/memoryllm_modeling')
              for p in (ROOT / folder).rglob('*.py')]
    return {p: sha(ROOT / p) for p in sorted(set(paths))}


def load_json(path):
    return json.loads(Path(path).read_text())


def support_oracle(items, budget=9, width=19):
    best = []
    best_score = -1
    support = [int(item['block']) for item in items]
    for candidate in __import__('itertools').combinations(range(width), budget):
        kept = set(candidate)
        score = sum(block in kept for block in support)
        if score > best_score:
            best_score, best = score, list(candidate)
    return best, best_score / len(items) if items else 0.0


def support_coverage(items, kept):
    selected = set(kept)
    return sum(int(item['block']) in selected for item in items) / len(items) if items else 0.0

def validate_plan(plan):
    if plan.get('status') != 'completed':
        raise ValueError('completed data plan required')
    from sic_paper_data import validate
    validate(plan)
    episodes = [dict(e, id=e['episode'], split=e['partition'], **e['splits'])
                for e in plan['episodes']]
    if len(episodes) != 24:
        raise ValueError('exactly 24 controlled episodes required')
    if sum(e['split'] == 'development' for e in episodes) != 8:
        raise ValueError('eight development episodes required')
    if sum(e['split'] == 'test' for e in episodes) != 16:
        raise ValueError('sixteen test episodes required')
    hashes = set()
    for e in episodes:
        digest = hashlib.sha256('\n'.join(e['lines']).encode()).hexdigest()
        if digest != e['context_sha256'] or digest in hashes or len(e['lines']) != 410:
            raise ValueError('context identity/length violation')
        hashes.add(digest)
        for split, count in (('discovery', 32), ('calibration', 32), ('evaluation', 64)):
            if len(e[split]) != count:
                raise ValueError('probe count violation')
    return episodes


def summarize(results):
    summary = {name: episode_summary(results, name, seed=PROTOCOL['seed'])
               for name in PROTOCOL['arms']}
    def rate(arm):
        return sum(sum(r['arms'][arm]['em'] for r in ep['evaluation']) / len(ep['evaluation'])
                   for ep in results) / len(results)
    competitors = [name for name in PROTOCOL['arms'] if name not in ('stability', 'gated_stability')]
    sic_gap = sum(ep['structural']['oracle_coverage'] - ep['structural']['stability_coverage']
                  for ep in results) / len(results)
    native_gap = sum(ep['structural']['oracle_coverage'] - ep['structural']['select_coverage']
                     for ep in results) / len(results)
    summary['development_gates'] = dict(
        ungated_sic_macro_em=rate('stability'),
        differences_vs_baselines={name: rate('stability') - rate(name) for name in competitors},
        observed_em_not_below_baselines=all(rate('stability') >= rate(name) for name in competitors),
        ungated_sic_mean_support_oracle_gap=sic_gap,
        native_mean_support_oracle_gap=native_gap,
        support_gap_improved=sic_gap < native_gap,
        fallback_episode_count=sum(not ep['gate']['accept'] for ep in results),
        interpretation='development screening only; passing observed gates is not a confidence guarantee')
    return summary


def freeze(args):
    data = load_json(args.plan)
    episodes = validate_plan(data)
    report = load_json(args.development)
    expected = {e['id'] for e in episodes if e['split'] == 'development'}
    if (report.get('status') != 'completed' or report['stage'] != 'development'
            or report.get('complete_prespecified_stage') is not True
            or len(report['episodes']) != 8
            or {e['id'] for e in report['episodes']} != expected
            or report['plan_sha256'] != sha(args.plan)
            or report['protocol'] != PROTOCOL or report['code_manifest'] != code_manifest()):
        raise ValueError('complete matching development report required for test freeze')
    for episode in report['episodes']:
        if len(episode['evaluation']) != 64 or episode['status'] != 'completed':
            raise ValueError('incomplete development episode')
    decision = dict(stage='freeze', frozen_at=datetime.now(timezone.utc).isoformat(),
                    plan_sha256=sha(args.plan), development_sha256=sha(args.development),
                    code_manifest=code_manifest(), protocol=PROTOCOL,
                    selected='gated_stability', selection='prespecified before development, no arm tuning',
                    model_manifest=report['model_manifest'],
                    development_summary=summarize(report['episodes']),
                    test_episodes=sorted(e['id'] for e in episodes if e['split'] == 'test'))
    atomic_json(args.out, decision)
    print(json.dumps(dict(status='frozen', selected=decision['selected'], sha256=sha(args.out))))


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import yaml
    import e6a_overfit as E
    import sic_probe as S
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import Probe, active_mask_override, differentiable_memory
    from methods.prlmc.consolidate import _select_block_indices
    from methods.prlmc.sic import block_salience, normalize_probe_scores, salience_select
    from readout_probe import official_score

    data = load_json(args.plan)
    episodes = validate_plan(data)
    manifest = code_manifest()
    if args.stage == 'test':
        frozen = load_json(args.freeze)
        if (frozen['stage'] != 'freeze' or frozen['plan_sha256'] != sha(args.plan)
                or frozen['code_manifest'] != manifest or frozen['protocol'] != PROTOCOL
                or frozen['selected'] != 'gated_stability'):
            raise ValueError('test freeze mismatch')
        if args.limit_episodes is not None or args.only_ids:
            raise ValueError('test subsets are prohibited')
    chosen = [e for e in episodes if e['split'] == args.stage]
    if args.only_ids:
        wanted = {int(x) for x in args.only_ids.split(',')}
        chosen = [e for e in chosen if e['id'] in wanted]
    if args.limit_episodes is not None:
        chosen = chosen[:args.limit_episodes]
    if not chosen:
        raise ValueError('empty stage')
    if Path(args.out).exists():
        raise ValueError('output already exists; keep original artifacts')
    seed = PROTOCOL['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    weight_files = sorted(Path(E.MODEL).glob('*.safetensors'))
    if not weight_files:
        weight_files = sorted(Path(E.MODEL).glob('pytorch_model*.bin'))
    if not weight_files:
        raise ValueError('model shard files missing')
    weight_manifest = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                       for p in sorted(Path(E.MODEL).iterdir()) if p.is_file()}
    if Path(E.CONFIG).resolve() != (ROOT / 'configs/agent_conf/prlmc_e6.yaml').resolve():
        raise ValueError('actual configuration differs from code manifest path')
    if args.stage == 'test' and weight_manifest != frozen['model_manifest']:
        raise ValueError('model/tokenizer manifest changed')
    started = time.monotonic()
    model = MemoryLLM.from_pretrained(E.MODEL, torch_dtype=torch.float16,
                                     attn_implementation='sdpa').to('cuda:0')
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if hasattr(module, 'attention_dropout'):
            module.attention_dropout = 0.
    model.eval()
    if (not hasattr(model, 'memory_active_mask') or model.memory_active_mask.dtype != torch.bool
            or model.memory_active_mask.numel() != model.memory.shape[1]):
        raise ValueError('patched active memory mask is required')
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL)
    raw = yaml.safe_load(Path(E.CONFIG).read_text())
    cfg = dict(raw['prlmc'])
    cfg.update(device='cuda:0', active_memory_mask=True, trigger='none',
               mode='ours', write_mode='append', gate_mode='ones', tau_alloc=-1.,
               projection='none', basis_mode='none', functional_control='none',
               compact_probe=False, stats_dir=None)
    calls = {'forward': 0}
    def count_forward(_module, _inputs):
        calls['forward'] += 1
    model.register_forward_pre_hook(count_forward)
    manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(cfg))
    report = dict(status='running', stage=args.stage, plan_sha256=sha(args.plan),
                  code_manifest=manifest, model_manifest=weight_manifest, protocol=PROTOCOL,
                  frozen_sha256=sha(args.freeze) if args.stage == 'test' else None,
                  opened_at=datetime.now(timezone.utc).isoformat(), episodes=[],
                  runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(0), deterministic_warn_only=True,
                               warning_capture='stderr of managed job; warn_only is not a determinism guarantee'))
    if args.stage == 'test':
        ledger = Path(args.plan).with_suffix('.test_opened.json')
        fd = os.open(ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(dict(plan_sha256=sha(args.plan), freeze_sha256=sha(args.freeze),
                           opened_at=datetime.now(timezone.utc).isoformat(), out=str(Path(args.out).resolve())), handle)
            handle.flush()
            os.fsync(handle.fileno())
    atomic_json(args.out, report)

    def prompt(item):
        return E.query_stem(item['question'])

    def generation(items, pool=None, mask=None):
        before = time.monotonic()
        output = {}
        guard = differentiable_memory(model, pool) if pool is not None else nullcontext()
        active = active_mask_override(model, mask) if mask is not None else nullcontext()
        model.eval()
        with guard, active, torch.inference_mode():
            for item in items:
                ids = tokenizer(prompt(item), return_tensors='pt', add_special_tokens=False).input_ids.to('cuda:0')
                if ids.shape[1] >= 512:
                    raise ValueError('query exceeds fixed budget')
                generated = model.generate(input_ids=ids, do_sample=False, use_cache=True,
                                           max_new_tokens=PROTOCOL['generation']['max_new_tokens'],
                                           stop_strings=['\n'], tokenizer=tokenizer,
                                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                text = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
                em, sub = official_score(text, [item['answer']])
                output[item['question']] = dict(prediction=text, em=bool(em), sub=bool(sub),
                                               generated_tokens=int(generated.shape[1] - ids.shape[1]))
        return output, time.monotonic() - before

    mask_audit = {}
    def score_kept(blocks, items):
        with torch.no_grad():
            pool, mask = S.pool_from_blocks(manager, blocks)
        if (mask is None or mask.dtype != torch.bool or mask.numel() != manager.total
                or int(mask.sum()) != manager.layout.k_anchor + 9 * manager.block):
            raise ValueError('compressed active-slot budget mismatch')
        try:
            if not mask_audit:
                ids = tokenizer(E.query_stem(current_discovery_question), return_tensors='pt',
                                add_special_tokens=False).input_ids.to('cuda:0')
                altered = pool.clone()
                altered[:, ~mask, :] = 0
                with torch.inference_mode(), active_mask_override(model, mask):
                    with differentiable_memory(model, pool):
                        before_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, -1].float()
                    with differentiable_memory(model, altered):
                        after_logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, -1].float()
                difference = float((before_logits - after_logits).abs().max())
                if not torch.isfinite(after_logits).all() or difference > 1e-3:
                    raise ValueError('masked memory perturbation changed reader logits')
                mask_audit.update(active_slots=int(mask.sum()), inactive_slots=int((~mask).sum()),
                                  max_logit_delta=difference, passed=True)
                del altered, before_logits, after_logits
            return generation(items, pool, mask)
        finally:
            del pool, mask
            torch.cuda.empty_cache()

    try:
        for ep in chosen:
            ep_started = time.monotonic()
            mask_audit.clear()
            current_discovery_question = ep['discovery'][0]['question']
            torch.cuda.reset_peak_memory_stats()
            forward_before = calls['forward']
            # initialize_anchor calls reset(), zeroing the model pool and mask state.
            manager = ProtectedResidualMemory(model, tokenizer, PRLMCConfig.from_dict(cfg))
            torch.manual_seed(seed + ep['id'])
            writes = []
            chunks = [ep['lines'][:30]] + [ep['lines'][30 + j*20:50 + j*20] for j in range(19)]
            with torch.no_grad():
                for j, chunk in enumerate(chunks):
                    text = '\n'.join(chunk)
                    token_count = len(tokenizer(text, add_special_tokens=False).input_ids)
                    if token_count > 512:
                        raise ValueError('write truncation would occur for episode %s chunk %d (%d)' % (ep['id'], j, token_count))
                    write = manager.initialize_anchor(text, max_tokens=512) if j == 0 else manager.write(text, max_tokens=512)
                    writes.append(dict(write, input_tokens=token_count))
                    if write.get('dropped') or write.get('evicted_blocks') or write.get('compacted'):
                        raise ValueError('write integrity failure')
                if manager.used != 19 * manager.block:
                    raise ValueError('working memory must contain exactly nineteen native blocks')
                begin = manager.total - manager.used
                # Hash tensors without a fp32 expansion of the full memory pool.
                memory_hash = hashlib.sha256(model.memory.detach().cpu().numpy().tobytes()).hexdigest()
                source = manager._memory[:, begin:begin + 19 * manager.block]
                native = _select_block_indices(source, 9 * manager.block, manager.block,
                                              manager._window_importance(begin, 19 * manager.block),
                                              manager.goal_direction, manager.select_weights).tolist()
                del source
            discovery = ep['discovery']
            full_d, time_full_d = generation(discovery)
            zero = torch.zeros_like(manager._memory)
            zero_d, time_zero_d = generation(discovery, zero)
            del zero
            eligible_indices = [i for i, item in enumerate(discovery)
                                if full_d[item['question']]['em'] and not zero_d[item['question']]['em']]
            probes = [Probe(question=x['question'], answers=[x['answer']], prompt=prompt(x)) for x in discovery]
            matrix = []
            backward_started = time.monotonic()
            model.train()
            memory_parameter = model.memory
            memory_cpu = memory_parameter.detach().cpu()
            memory_parameter.data = memory_cpu
            prefix = memory_cpu[:, :begin, :].to('cuda:0')
            window = memory_cpu[:, begin:begin + 19 * manager.block, :].to('cuda:0').requires_grad_(True)
            suffix = memory_cpu[:, begin + 19 * manager.block:, :].to('cuda:0')
            pool = torch.cat((prefix, window, suffix), dim=1)
            gradient_transfers = 0
            try:
                for probe in probes:
                    scalar = S.probe_loss_scalar(manager, probe, pool, None, 512)
                    if scalar is None:
                        raise ValueError('invalid discovery gradient')
                    grad = torch.autograd.grad(scalar, window)[0]
                    scores = torch.stack([block_salience(grad[:, j*manager.block:(j+1)*manager.block], manager.block)[0]
                                          for j in range(19)])
                    matrix.append(scores.detach().cpu())
                    gradient_transfers += 1
                    del grad, scores, scalar
                    torch.cuda.empty_cache()
            finally:
                del pool, prefix, suffix, window
                memory_parameter.data = memory_cpu.to('cuda:0')
                del memory_cpu
                torch.cuda.empty_cache()
            gradient_seconds = time.monotonic() - backward_started
            model.eval()
            energy = torch.stack(matrix)
            del matrix
            kept = dict(select=native, raw=salience_select(energy.sum(0), 9), recent=list(range(10, 19)))
            rng = random.Random(seed + ep['id'])
            kept['random'] = sorted(rng.sample(range(19), 9))
            if len(eligible_indices) >= 4:
                eligible_energy = energy[eligible_indices]
                kept['eligible_normalized'] = salience_select(normalize_probe_scores(eligible_energy).sum(0), 9)
                stable = stability_select(eligible_energy.tolist(), 9, seed=seed + ep['id'], repeats=128)
                kept['stability'] = stable['kept']
            else:
                stable = dict(status='insufficient_eligible_discovery', probes=len(eligible_indices))
                kept['eligible_normalized'] = kept['stability'] = list(native)
            del energy
            calibration = ep['calibration']
            native_cal, native_cal_time = score_kept(native, calibration)
            if kept['stability'] == native:
                candidate_cal, candidate_cal_time = native_cal, 0.
            else:
                candidate_cal, candidate_cal_time = score_kept(kept['stability'], calibration)
            gate = calibration_gate([native_cal[x['question']]['em'] for x in calibration],
                                    [candidate_cal[x['question']]['em'] for x in calibration])
            kept['gated_stability'] = list(kept['stability'] if gate['accept'] else native)
            # Evaluation is opened only after all selections and the calibration gate are fixed.
            evaluation = ep['evaluation']
            full_eval, time_full_eval = generation(evaluation)
            zero = torch.zeros_like(manager._memory)
            zero_eval, time_zero_eval = generation(evaluation, zero)
            del zero
            outcomes, timings, cache = {}, {}, {}
            oracle_blocks, oracle_cov = support_oracle(evaluation)
            oracle_outcomes, oracle_time = score_kept(oracle_blocks, evaluation)
            for name in PROTOCOL['arms']:
                key = tuple(kept[name])
                if key not in cache:
                    cache[key] = score_kept(kept[name], evaluation)
                outcomes[name], timings[name] = cache[key]
            records = []
            for item in evaluation:
                q = item['question']
                records.append(dict(item, full=full_eval[q], zero_pool=zero_eval[q],
                                    eligible=full_eval[q]['em'] and not zero_eval[q]['em'],
                                    support_block=int(item['block']),
                                    arms={name: outcome[q] for name, outcome in outcomes.items()},
                                    support_oracle=oracle_outcomes[q]))
            result = dict(id=ep['id'], status='completed', context_sha256=ep['context_sha256'],
                          pool_sha256=memory_hash, mask_audit=dict(mask_audit),
                          writes=writes, kept=kept, stability=stable, gate=gate,
                          discovery=[dict(x, full=full_d[x['question']], zero_pool=zero_d[x['question']]) for x in discovery],
                          calibration=[dict(x, native=native_cal[x['question']], candidate=candidate_cal[x['question']]) for x in calibration],
                          evaluation=records, eligible_discovery=len(eligible_indices),
                           structural=dict(oracle_blocks=oracle_blocks, oracle_coverage=oracle_cov,
                                           gated_coverage=support_coverage(evaluation, kept['gated_stability']),
                                           select_coverage=support_coverage(evaluation, native),
                                           stability_coverage=support_coverage(evaluation, kept['stability']),
                                           gated_oracle_gap=oracle_cov - support_coverage(evaluation, kept['gated_stability'])),
                          costs=dict(reader_backwards=len(discovery), gradient_window_backwards=gradient_transfers,
                                      pool_offload='cpu_source_full_geometry_gpu_forward', gradient_seconds=gradient_seconds,
                                     writing_forward_calls=len(chunks), candidate_sets=len(cache),
                                     calibration_seconds=native_cal_time + candidate_cal_time,
                                     generation_seconds=dict(full_discovery=time_full_d, zero_discovery=time_zero_d,
                                                             full_evaluation=time_full_eval, zero_evaluation=time_zero_eval,
                                                             candidates=timings),
                                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                     peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                     elapsed_seconds=time.monotonic() - ep_started),
                          slots=dict(window=19*256, target=9*256, released=10*256,
                                     total_pool_unchanged=True, active_mask_compaction=True))
            report['episodes'].append(result)
            atomic_json(args.out, report)
            print('EPISODE', ep['id'], 'generation EM', {name: sum(v['em'] for v in outcomes[name].values()) for name in outcomes},
                  'eligible discovery', len(eligible_indices), 'gate', gate['reason'],
                  'seconds', round(result['costs']['elapsed_seconds'], 1), flush=True)
        report['status'] = 'completed'
        report['summary'] = summarize(report['episodes'])
        report['complete_prespecified_stage'] = len(chosen) == (8 if args.stage == 'development' else 16)
        report['runtime']['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.out, report)
        print('FINAL', json.dumps(report['summary']['gated_stability']), flush=True)
    except Exception as error:
        report.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        atomic_json(args.out, report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['development', 'freeze', 'test'], required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--development')
    p.add_argument('--freeze')
    p.add_argument('--cuda-device', type=int, default=9)
    p.add_argument('--limit-episodes', type=int)
    p.add_argument('--only-ids')
    args = p.parse_args()
    if args.stage == 'freeze':
        if not args.development:
            p.error('--development required for freeze')
        freeze(args)
    else:
        if args.stage == 'test' and not args.freeze:
            p.error('--freeze required for test')
        run(args)


if __name__ == '__main__':
    main()
