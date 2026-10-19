#!/usr/bin/env python3
"""Split frozen evaluation by whole context/stream; preserve one sealed protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_public_retention import sha
from prepare_natural_merge import validate_plan
from verify_natural_merge import verify,summarize


def merge_reports(reports,plan):
    import copy
    count=len(reports)
    if not reports or {r["sharding"]["index"] for r in reports}!=set(range(count)):
        raise ValueError("missing/duplicate sample shards")
    identity=("schema","experiment","partition","diagnostic_only","protocol","plan_sha256","plan_content_sha256",
        "checkpoint_sha256","sources","model","suite_sha256","freeze_sha256","resources",
        "shared_state_sha256_before","shared_state_sha256_after","training_performed","optimizer_created")
    for report in reports:
        verify(report,plan)
        if report["sharding"]["count"]!=count or any(report.get(k)!=reports[0].get(k) for k in identity):
            raise ValueError("mixed shard identity/protocol/checkpoint")
    combined=copy.deepcopy(reports[0])
    combined["episodes"]=sorted([e for r in reports for e in r["episodes"]],key=lambda e:e["episode_index"])
    if [e["episode_index"] for e in combined["episodes"]]!=list(range(len(plan["episodes"]))):
        raise ValueError("shards do not cover each original context exactly once")
    combined["sharding"]=dict(index=0,count=1,episode_indices=list(range(len(plan["episodes"]))))
    combined["worker_shards"]=count
    combined["elapsed_seconds"]=max(r["elapsed_seconds"] for r in reports)
    combined["summary"]=summarize(combined)
    verify(combined,plan)
    return combined


def main():
    from native_qa_compare import atomic_json
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","checkpoint","devices","out"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--freeze");p.add_argument("--suite")
    p.add_argument("--e2-result");p.add_argument("--e2-plan")
    p.add_argument("--legacy-diagnostic",action="store_true")
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());validate_plan(plan)
    devices=[int(d) for d in a.devices.split(",")]
    if len(set(devices))!=len(devices) or min(devices)<0 or len(devices)>len(plan["episodes"]):
        raise ValueError("unique GPUs and at least one whole episode per shard required")
    # Refuse occupied devices; no automatic termination of other jobs.
    probe=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                         check=True,capture_output=True,text=True)
    free={int(line.split(",")[0]):int(line.split(",")[1]) for line in probe.stdout.splitlines()}
    if any(free.get(d,0)<28000 for d in devices):
        raise ValueError("every selected GPU needs >=28000MiB free; choose genuinely free cards")
    prefix=Path(a.out)
    paths=[Path(str(prefix)+".shard"+str(i)+".json") for i in range(len(devices))]
    fleet_path=Path(str(prefix)+".fleet.json")
    cert_path=Path(str(prefix)+".certificate.json")
    if any(x.exists() for x in [prefix,fleet_path,cert_path]+paths):
        raise ValueError("parallel output exists")
    prefix.parent.mkdir(parents=True,exist_ok=True)
    fleet=dict(schema="natural_merge_fleet/1",status="running",plan_sha256=sha(a.plan),workers=[])
    jobs=[]
    try:
        for i,device in enumerate(devices):
            cmd=[sys.executable,str(ROOT/"tools/natural_merge_eval.py"),"--plan",a.plan,"--checkpoint",a.checkpoint,
                 "--model-path",a.model_path,"--cuda-device",str(device),"--out",str(paths[i]),
                 "--shard-index",str(i),"--shard-count",str(len(devices))]
            for name in ("freeze","suite","e2_result","e2_plan"):
                if getattr(a,name):cmd += ["--"+name.replace("_","-"),getattr(a,name)]
            if a.legacy_diagnostic:cmd.append("--legacy-diagnostic")
            log=Path(str(prefix)+".shard"+str(i)+".log").open("x")
            job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            jobs.append((job,log));fleet["workers"].append(dict(shard=i,device=device,pid=job.pid,out=str(paths[i]),command=cmd))
            atomic_json(fleet_path,fleet)
        while any(job.poll() is None for job,_ in jobs):
            for entry,(job,_) in zip(fleet["workers"],jobs):entry["exit_code"]=job.poll()
            atomic_json(fleet_path,fleet);time.sleep(5)
        exits=[job.returncode for job,_ in jobs]
        if any(code!=0 for code in exits):raise RuntimeError("worker failed: "+repr(exits))
        from transformers import AutoTokenizer
        from prepare_public_retention import tokenizer_manifest
        if tokenizer_manifest(a.model_path)!=plan["tokenizer"]:
            raise ValueError("aggregate scoring tokenizer mismatch")
        tokenizer=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True)
        shard_reports=[json.loads(path.read_text()) for path in paths]
        for result in shard_reports:
            if result["plan_sha256"]!=sha(a.plan) or result["checkpoint_sha256"]!=sha(a.checkpoint):
                raise ValueError("worker used a different plan/checkpoint file")
            verify(result,plan,tokenizer)
        combined=merge_reports(shard_reports,plan)
        atomic_json(prefix,combined)
        certificate=verify(combined,plan,tokenizer);certificate["result_sha256"]=sha(prefix)
        atomic_json(cert_path,certificate)
        fleet.update(status="complete",exit_codes=exits,result=str(prefix),certificate=str(cert_path))
    except Exception as exc:
        fleet.update(status="failed",failure=str(exc))
        # Stop only worker processes started by this launcher.
        import signal
        for job,_ in jobs:
            if job.poll() is None:os.killpg(job.pid,signal.SIGTERM)
        raise
    finally:
        for _,log in jobs:log.close()
        atomic_json(fleet_path,fleet)

if __name__=="__main__":main()
