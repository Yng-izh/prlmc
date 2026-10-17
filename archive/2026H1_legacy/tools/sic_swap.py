"""Discovery-only finite one-for-one swap approximation to native support oracle."""
import argparse, hashlib, itertools, json, os, random, time
from pathlib import Path


def save(path, obj):
    p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(p)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','source','choice','oracle','out'): ap.add_argument('--'+n,required=True)
    ap.add_argument('--row',type=int,default=0); ap.add_argument('--max-candidates',type=int,default=8); ap.add_argument('--cuda-device',default='cuda:0')
    a=ap.parse_args(); pp=Path(a.plan); sp=Path(a.source); cp=Path(a.choice); op=Path(a.oracle)
    plan_bytes=pp.read_bytes(); source_bytes=sp.read_bytes(); plan=json.loads(plan_bytes); source=json.loads(source_bytes); choice=json.loads(cp.read_text()); oracle=json.loads(op.read_text())
    ph=hashlib.sha256(plan_bytes).hexdigest(); sh=hashlib.sha256(source_bytes).hexdigest()
    if source.get('plan_sha256')!=ph or choice.get('source_sha256')!=source.get('source_sha256'):
        raise ValueError('plan/source/choice hash mismatch')
    p=next(x for x in plan['plans'] if x['row']==a.row); w=next(x for x in source['windows'] if x['row']==a.row)
    base=choice['perrow_kept'][str(a.row)]; oracle_row=next(x for x in oracle['rows'] if x['row']==a.row)
    if base!=w['kept'][choice['selected']]: raise ValueError('choice kept differs from source')
    if len(base)!=9: raise ValueError('base budget')
    import torch
    import e6a_overfit as E
    import sic_probe as S
    import yaml
    from transformers import AutoTokenizer
    from methods.memoryllm_modeling.modeling_memoryllm import MemoryLLM
    from methods.prlmc.config import PRLMCConfig
    from methods.prlmc.layout import ProtectedResidualMemory
    from methods.prlmc.probes import Probe
    from methods.prlmc.consolidate import _select_block_indices
    device=a.cuda_device
    random.seed(plan['seed']); torch.manual_seed(plan['seed']); torch.cuda.manual_seed_all(plan['seed'])
    torch.use_deterministic_algorithms(True,warn_only=True); torch.backends.cudnn.benchmark=False
    cfg=PRLMCConfig.from_dict(dict(yaml.safe_load(Path(E.CONFIG).read_text())['prlmc'],device=device,active_memory_mask=True))
    model=MemoryLLM.from_pretrained(E.MODEL,torch_dtype=torch.float16,attn_implementation='sdpa').to(device)
    for q in model.parameters(): q.requires_grad_(False)
    model.config.use_cache=False
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout): m.p=0.
        if hasattr(m,'attention_dropout'): m.attention_dropout=0.
    model.eval(); manager=ProtectedResidualMemory(model,AutoTokenizer.from_pretrained(E.MODEL),cfg)
    lines=E.read_fact_lines(a.row)
    if hashlib.sha256('\n'.join(lines).encode()).hexdigest()!=p['context_sha256']: raise ValueError('context changed')
    if E.build_window(manager,lines,19*manager.block,30)!=[(60+20*j,80+20*j,256*j,256) for j in range(19)]: raise ValueError('window layout')
    begin=manager.total-manager.used; mem=manager._memory[:,begin:begin+19*manager.block]
    native=_select_block_indices(mem,9*manager.block,manager.block,manager._window_importance(begin,19*manager.block),manager.goal_direction,manager.select_weights).tolist(); del mem
    if native!=w['kept']['select']: raise ValueError('native mismatch')
    discovery=p['discovery']
    def make_probes(items):
        return [Probe(question=x['question'],answers=[x['answer']],prompt=E.query_stem(x['question'])) for x in items]
    discovery_probes=make_probes(discovery)
    full=E.score_uncompacted(manager,discovery_probes,256)['correct']
    no_mem=E.score_no_memory(manager,discovery_probes,256)['correct']
    eligible=[i for i,x in enumerate(discovery) if full[x['question']] and not no_mem[x['question']] ]
    if not eligible: raise ValueError('no eligible discovery probes')
    probes=make_probes([discovery[i] for i in eligible])
    energy=torch.tensor(w['energy'],dtype=torch.float32)

    dropped=sorted(set(range(19))-set(base)); all_swaps=[]
    for out,inn in itertools.product(base,dropped):
        blocks=sorted((set(base)-{out})|{inn})
        # Gradient energy predicts which finite swaps are worth measuring.
        gain=float(energy[eligible,inn].sum()-energy[eligible,out].sum())
        all_swaps.append((gain,blocks,out,inn))
    all_swaps.sort(key=lambda z:(-z[0],z[2],z[3])); candidates=all_swaps[:a.max_candidates]
    def score(blocks):
        with torch.no_grad():
            pool,mask=S.pool_from_blocks(manager,blocks)
            try: r=E.score_probes(manager,probes,pool,256,mask)
            finally: del pool,mask; torch.cuda.empty_cache()
        return dict(kept=blocks,gold_loss=r['gold_loss'],correct=sum(r['correct'].values()),n=r['n'],outcomes=r['correct'])
    started=time.monotonic(); base_score=score(base); results=[dict(base_score,kind='base')]
    for _,blocks,out,inn in candidates:
        r=score(blocks); r.update(kind='swap',out=out,inn=inn,predicted_gain=next(x[0] for x in candidates if x[1]==blocks)); results.append(r); print('swap',out,inn,r['gold_loss'],flush=True)
    chosen=min(results,key=lambda r:(r['gold_loss'],r['kept']))
    save(a.out,dict(experiment='SIC-swap',row=a.row,plan_sha256=ph,source_sha256=sh,oracle_aware=False,discovery_only=True,eligible_discovery=len(eligible),base=base,base_score=base_score,candidates=results,selected=chosen,oracle_support_kept=oracle_row['kept'],oracle_support_gap=len(set(chosen['kept'])^set(oracle_row['kept']))//2,elapsed_seconds=time.monotonic()-started,confirmation_used=False))

if __name__=='__main__': main()
