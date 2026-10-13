"""Frozen C checkpoint restoration and fixed four-arm B2 generation."""
import hashlib
import json
from pathlib import Path
import time
from contextlib import nullcontext

from readout_probe import FEWSHOT, official_score
from native_qa_compare import atomic_json


def paired_summary(maps):
    questions = list(maps['Full'])
    if any(set(m) != set(questions) for m in maps.values()):
        raise ValueError('unpaired question universes')
    full, select, d0, c = [maps[k] for k in ('Full', 'Selection', 'D0', 'C')]
    lost = [q for q in questions if full[q] and not select[q]]
    rescue_target = [q for q in lost if not d0[q]]
    rescued = [q for q in rescue_target if c[q]]
    harmed = [q for q in questions if d0[q] and not c[q]]
    result = dict(n=len(questions),
        arms={k: dict(correct=sum(m.values()), em=sum(m.values())/len(questions)) for k,m in maps.items()},
        compression_lost=lost, rescue_d0_wrong=rescue_target, rescued=rescued, harmed=harmed,
        R=len(rescued), H=len(harmed), net_target=len(rescued)-len(harmed))
    for key, baseline in (('Selection', select), ('D0', d0)):
        gains = [q for q in questions if c[q] and not baseline[q]]
        losses = [q for q in questions if baseline[q] and not c[q]]
        result['C_vs_'+key] = dict(gained=gains, harmed=losses, net=len(gains)-len(losses),
                                  delta_em=(len(gains)-len(losses))/len(questions))
    return result


def load_verified_c(args, reference, window_report, arm_entry, displacement):
    import torch
    from s1d_experiment1 import sha256_file
    if reference['status'] != 'complete' or reference['protocol']['exp1_arm'] != 'C':
        raise ValueError('reference must be completed C')
    rw = reference['windows'][0]; ra = rw['arms']['C']
    for k in ('window', 'sets', 'baselines', 'n_supervision_lines'):
        if window_report[k] != rw[k]:
            raise ValueError('C reproduction drift: '+k)
    if arm_entry['basis_sha256'] != ra['basis_sha256']:
        raise ValueError('C basis hash mismatch')
    saved = torch.load(args.c_checkpoint, map_location='cpu', weights_only=False)
    if saved['window'] != rw['window'] or saved['arm'] != 'C':
        raise ValueError('checkpoint window/arm mismatch')
    if saved['checkpoint_sha256'] != sha256_file(args.carrier_ckpt):
        raise ValueError('carrier hash mismatch')
    if saved['driver_sha256'] != reference['driver_sha256']:
        raise ValueError('training driver provenance mismatch')
    if not torch.equal(saved['basis'], displacement.basis.detach().cpu()):
        raise ValueError('basis checkpoint differs')
    if (saved['alpha'].shape != (13,) or not torch.isfinite(saved['alpha']).all()
            or saved['alpha'].norm() > 1.000001):
        raise ValueError('invalid alpha')
    expected = torch.tensor(ra['final']['alpha'], dtype=saved['alpha'].dtype)
    if not torch.equal(saved['alpha'], expected):
        raise ValueError('not the published final checkpoint')
    with torch.no_grad():
        displacement.value.copy_(saved['alpha'].to(displacement.value))
    displacement.value.requires_grad_(False)


def evaluate_four_arms(*, args, manager, model, probes, arms, candidate_pool,
                       window_report, reference, covered, lines, checkpoint_sha,
                       carrier_sha, driver_sha, loaded_tf, generation_context=None):
    import torch
    from methods.prlmc.probes import differentiable_memory, active_mask_override
    from s1d_experiment1 import reader_evaluation, memory_digest
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('reader must be frozen')
    selected = probes[:2] if args.generation_smoke else probes
    original_parameter = model.memory
    original_mask = model._buffers.get('memory_active_mask')
    source_hash = memory_digest(model.memory)
    started = time.monotonic()
    result = dict(schema_version='s1d_c_generation/1', status='running',
        window=window_report['window'], smoke=args.generation_smoke,
        protocol=dict(prompt='fixed B2 FEWSHOT', prompt_sha256=hashlib.sha256(FEWSHOT.encode()).hexdigest(),
            stop_strings=['\n'], max_new_tokens=32, do_sample=False, attention_scale=1.0,
            scorer='readout_probe.official_score: official two-tier EM',
            qa_origin='S1d development fact-derived probes, not native official question set',
            checkpoint_selection='published C step60', training=False, seed=0,
            compressed_budget_slots=manager.layout.k_anchor+manager.used-19*manager.block+9*manager.block),
        checkpoint_sha256=checkpoint_sha, carrier_sha256=carrier_sha, driver_sha256=driver_sha,
        scorer_sha256=hashlib.sha256(Path(__file__).with_name('readout_probe.py').read_bytes()).hexdigest(),
        generation_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_memory_sha256=source_hash,
        context_lines_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest(),
        covered=covered, reference_tf=dict(baselines=window_report['baselines'], C=loaded_tf),
        arm_results={}, pairs=[dict(question=q.question, answers=list(q.answers)) for q in selected])
    if generation_context is not None:
        result.update(generation_context['metadata'])
        result['protocol'].update(generation_context['protocol'])
    # Native generation can update model.memory. Bind a disposable full copy
    # during each arm; restore it after EVERY question if its version changes.
    for name, arm in arms.items():
        with reader_evaluation(model, verify_memory=True):
            if arm is None:
                pool = model.memory.detach().clone()
                mask = original_mask.detach().clone() if original_mask is not None else None
            else:
                pool, mask = candidate_pool(arm)
            template = pool.detach().clone()
            active_slots = int(mask.sum()) if mask is not None else manager.layout.k_anchor+manager.used
            if arm is not None and active_slots != result['protocol']['compressed_budget_slots']:
                raise RuntimeError('compressed active-slot budget mismatch')
            with differentiable_memory(model, pool), active_mask_override(model, mask):
                if hasattr(model, 'memory_attn_scale') and not torch.all(model.memory_attn_scale == 1):
                    raise RuntimeError('reader attention scale differs from frozen B2')
                predictions, correct, substring = {}, {}, {}
                for index, q in enumerate(selected):
                    prompt = FEWSHOT.format(question=q.question)
                    ids = manager.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda:0')
                    version = model.memory._version
                    with torch.inference_mode():
                        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                            do_sample=False, max_new_tokens=32, stop_strings=['\n'],
                            tokenizer=manager.tokenizer, use_cache=True,
                            pad_token_id=manager.tokenizer.pad_token_id if manager.tokenizer.pad_token_id is not None
                            else manager.tokenizer.eos_token_id)
                    pred = manager.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True)
                    if model.memory is not pool:
                        raise RuntimeError('generation replaced memory binding')
                    # Tensor .data assignment may not increment _version.
                    # Compare values as well, then reset before the next query.
                    if model.memory._version != version or not torch.equal(pool, template):
                        with torch.no_grad():
                            pool.copy_(template)
                    em, sub = official_score(pred, q.answers)
                    predictions[q.question] = pred; correct[q.question] = bool(em); substring[q.question] = bool(sub)
                    result['pairs'][index][name] = dict(prediction=pred, em=bool(em), substring_em=bool(sub))
                    if index % 20 == 0:
                        print('[s1d-gen] %s %d/%d'%(name,index+1,len(selected)),flush=True)
                        atomic_json(args.out,result)
                if not torch.equal(pool, template):
                    raise RuntimeError('candidate pool changed during generation')
            result['arm_results'][name] = dict(predictions=predictions, correct=correct,
                substring_correct=substring, active_slots=active_slots,
                pool_sha256=memory_digest(template))
            del pool, template, mask
        if model.memory is not original_parameter or model._buffers.get('memory_active_mask') is not original_mask:
            raise RuntimeError('memory/mask restoration failed')
        if memory_digest(model.memory) != source_hash:
            raise RuntimeError('source pool changed between arms')
        atomic_json(args.out,result)
        torch.cuda.empty_cache()
    maps = {k:v['correct'] for k,v in result['arm_results'].items()}
    result['summary'] = paired_summary(maps)
    if reference is not None:
        tf_recovered = reference['windows'][0]['arms']['C']['final']['new_recovery_questions']
        result['tf_to_generation'] = dict(
            tf_recovered_questions=tf_recovered,
            C_generation_correct=[q for q in tf_recovered if maps['C'].get(q,False)],
            newly_correct_vs_D0=[q for q in tf_recovered if maps['C'].get(q,False) and not maps['D0'].get(q,False)])
    result.update(status='complete',elapsed_seconds=time.monotonic()-started,
                  source_memory_unchanged=True, reader_frozen=True)
    return result
