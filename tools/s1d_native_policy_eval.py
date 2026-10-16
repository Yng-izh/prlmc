#!/usr/bin/env python3
"""Native MemoryLLM write/drop policy on the fixed controlled development split.

NativeDefault uses the exported 50*256 pool. NativeBudget changes capacity only
(to 11*256=2816), retaining the exact native injection/drop implementation.
Both use the same raw write chunks and B2 readout as frozen shared E6d.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from s1d_shared_compactor_eval import (DEVELOPMENT_IDS, choose_development,
    validate_write_plan, sha, atomic_json)

ARMS = {'NativeBudget': 11, 'NativeDefault': 50}
NATIVE_METHODS = ('inject_memory', 'drop_memory', 'update_memory_with_delta_memory',
                  'cat_memory_and_hiddens')
PROTOCOL = dict(name='s1d-native-write-drop-v1', arms=ARMS, native_block_tokens=256,
    native_drop_per_layer=True, training=False, seed=0, write_chunks=[30]+[20]*19,
    write_input='identical raw numbered fact chunks to shared E6d evaluation',
    write_max_tokens=512, injection='MemoryLLM.inject_memory(update_memory=True)',
    first_write='cold start; native repeat fills pool',
    drop='native random per-layer slot removal, preserved order, then append',
    anchor_protection=False, compactor=False, active_mask=False,
    prompt='fixed B2 FEWSHOT', stop_strings=['\n'], max_new_tokens=32,
    scorer='readout_probe.official_score', evaluation_questions=64,
    scope='native write/drop with controlled inputs and fixed B2 readout; not a full official benchmark run',
    test_policy='development only; sealed test remains unopened')


def native_method_source(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MemoryLLM')
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    return {name: ast.dump(methods[name], include_attributes=False) for name in NATIVE_METHODS}


def verify_native_policy(current, original):
    current_methods = native_method_source(current)
    if current_methods != native_method_source(original):
        raise ValueError('native write/drop methods differ from original backup')
    return dict(original_sha256=sha(original), current_sha256=sha(current),
                unchanged_native_methods=list(NATIVE_METHODS),
                method_sha256={name: hashlib.sha256(src.encode()).hexdigest()
                               for name, src in current_methods.items()})


def source_manifest():
    files = ['tools/s1d_native_policy_eval.py', 'tools/run_s1d_native_policy_eval.py',
             'tools/s1d_shared_compactor_eval.py', 'tools/s1d_experiment1.py',
             'tools/readout_probe.py', 'tools/native_qa_compare.py',
             'methods/prlmc/probes.py', 'methods/memoryllm_modeling/modeling_memoryllm.py']
    return {name: sha(ROOT/name) for name in files}


def set_native_capacity(model, blocks):
    import torch
    if model.num_tokens != 256 or not model.drop_memory_per_layer:
        raise ValueError('unexpected native block size or drop policy')
    model.num_blocks = int(blocks)
    model.config.num_blocks = int(blocks)
    model.config.num_memory_tokens = blocks * model.num_tokens
    # First injection replaces this placeholder by repeating the native delta.
    memory = model.memory
    model.memory = torch.nn.Parameter(torch.empty(
        model.L, blocks * model.num_tokens, model.d, device=memory.device, dtype=memory.dtype),
        requires_grad=False)
    model.initialized.zero_()
    # None follows the original unpatched SDPA branch (no memory bias).
    model._buffers['memory_active_mask'] = None
    if hasattr(model, 'memory_attn_scale'):
        model.memory_attn_scale.fill_(1.)


def native_write(model, tokenizer, chunks, on_progress=None):
    import torch
    records = []
    with torch.no_grad():
        for index, chunk in enumerate(chunks):
            text = '\n'.join(chunk)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not 16 <= len(ids) <= 512:
                raise ValueError('native chunk would be skipped or truncated')
            cold = not bool(model.initialized.item())
            started = time.monotonic()
            delta = model.inject_memory(torch.tensor([ids], device=model.memory.device), update_memory=True)
            width = int(delta.shape[-2])
            if width != 256 or model.memory.shape[1] != model.num_blocks * 256:
                raise ValueError('native delta/pool width changed')
            if model.memory_active_mask is not None:
                raise ValueError('native memory unexpectedly masked')
            records.append(dict(chunk=index, lines=len(chunk), input_tokens=len(ids),
                native_delta_slots=width, first_write=cold,
                randomly_removed_slots_per_layer=0 if cold else width,
                seconds=time.monotonic()-started))
            del delta
            if on_progress:
                on_progress(records)
            print('[native] write %d/%d' % (index+1, len(chunks)), flush=True)
    return records


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    os.environ['S0B_GPU_SET'] = '1'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from s1d_shared_compactor_eval import _generate
    from s1d_experiment1 import memory_digest
    from readout_probe import FEWSHOT
    out = Path(args.out)
    if out.exists():
        raise ValueError('refuse to overwrite existing native report')
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    episode = choose_development(plan, args.episode)
    _, chunks = validate_write_plan(episode)
    modeling = ROOT/'methods/memoryllm_modeling/modeling_memoryllm.py'
    audit = verify_native_policy(modeling, modeling.with_suffix('.py.bak'))
    reference = json.loads(Path(args.reference_prefix+'.ep%d.json'%args.episode).read_text())
    model_manifest = sha(Path(args.model_path)/'fp16_cache_manifest.json')
    if (reference.get('status') != 'complete' or reference['identity']['plan_sha256'] != sha(plan_path)
            or reference['identity']['context_sha256'] != episode['context_sha256']
            or reference['identity']['model_manifest_sha256'] != model_manifest):
        raise ValueError('shared reference missing or input/model provenance differs')
    report = dict(schema_version='s1d_native_policy_eval/1', status='running',
        identity=dict(episode=args.episode, context_sha256=episode['context_sha256'],
            plan_sha256=sha(plan_path), model_manifest_sha256=model_manifest,
            reference_sha256=sha(args.reference_prefix+'.ep%d.json'%args.episode),
            source_manifest=source_manifest(), protocol=PROTOCOL,
            prompt_sha256=hashlib.sha256(FEWSHOT.encode()).hexdigest()),
        native_policy_audit=audit, smoke=args.smoke, stage='development', test_opened=False,
        training_performed=False, evaluation_opened=False, writes={}, predictions={}, arm_metadata={})
    atomic_json(out, report)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.monotonic()
    model = MemoryLLM.from_pretrained(args.model_path, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    if model.num_blocks != 50 or model.num_tokens != 256:
        raise ValueError('exported default native capacity changed')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    manager = SimpleNamespace(tokenizer=tokenizer)
    evaluation = episode['splits']['evaluation']
    if len(evaluation) != 64 or len({q['question'] for q in evaluation}) != 64:
        raise ValueError('expected exactly 64 evaluation questions')
    if args.smoke:
        evaluation = evaluation[:2]
    for name, blocks in ARMS.items():
        print('[native] arm', name, 'episode', args.episode, flush=True)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        set_native_capacity(model, blocks)
        torch.cuda.empty_cache()
        def save_writes(records):
            report['writes'][name] = records
            report['progress'] = dict(arm=name, phase='writing', chunks=len(records))
            atomic_json(out, report)
        native_write(model, tokenizer, chunks, save_writes)
        before = memory_digest(model.memory)
        report['arm_metadata'][name] = dict(active_slots=blocks*256, physical_slots=blocks*256,
            source_memory_sha256=before, no_anchor=True, native_drop_per_layer=model.drop_memory_per_layer)
        report['evaluation_opened'] = True
        def save_predictions(values):
            report['predictions'][name] = values
            report['progress'] = dict(arm=name, phase='generation', questions=len(values), total=len(evaluation))
            atomic_json(out, report)
        _generate(model, manager, evaluation, None, None, save_predictions)
        if memory_digest(model.memory) != before:
            raise ValueError('generation changed native source memory')
        report['arm_metadata'][name]['source_memory_unchanged'] = True
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('reader unexpectedly trainable')
        torch.cuda.empty_cache()
    n = len(evaluation)
    report['summary'] = {name: dict(correct=sum(row['em'] for row in rows.values()),
        n=n, em=sum(row['em'] for row in rows.values())/n) for name, rows in report['predictions'].items()}
    report.update(status='complete', generation_questions=[q['question'] for q in evaluation],
        costs=dict(elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()),
        runtime=dict(gpu=torch.cuda.get_device_name(0), physical_gpu=args.cuda_device,
                     torch=torch.__version__))
    atomic_json(out, report)
    print('FINAL', json.dumps(report['summary']), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, choices=DEVELOPMENT_IDS, required=True)
    p.add_argument('--cuda-device', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--plan', default='runs/sic_paper_controlled_plan.json')
    p.add_argument('--reference-prefix', default='runs/s1d_shared_e6d_dev_gpu59_v1')
    p.add_argument('--model-path', default='/dev/shm/s1d_reader_fp16')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        out = Path(args.out)
        if out.exists():
            report = json.loads(out.read_text())
            if report.get('status') == 'running':
                report.update(status='failed', failure=dict(type=type(exc).__name__, message=str(exc)))
                atomic_json(out, report)
        raise

if __name__ == '__main__':
    main()
