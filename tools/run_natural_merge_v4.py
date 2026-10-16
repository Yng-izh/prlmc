#!/usr/bin/env python3
"""User-started Dsup/Dnative queue: smoke -> training -> training-fit diagnosis -> dev E1.

One variant per GPU, then queue remaining variants. Never starts formal E1/E2/E3
or stops unrelated jobs. Exclusive output directory refuses duplicate launches.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_public_retention import sha
from prepare_natural_merge import validate_plan
from prepare_natural_merge_v4 import validate_v4_plan
from native_qa_compare import atomic_json


def devices_with_room(devices):
    probe=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
        check=True,capture_output=True,text=True)
    free={int(line.split(",")[0]):int(line.split(",")[1]) for line in probe.stdout.splitlines()}
    return [d for d in devices if free.get(d,0)>=32000]


def validate_scopes(train,dev):
    validate_v4_plan(train); validate_plan(dev)
    if dev["experiment"]!="E1" or dev["partition"]!="development":
        raise ValueError("only E1 development may run in the Dsup/Dnative queue")
    if train["tokenizer"]!=dev["tokenizer"] or train["base_plan"]["protocol"]!=dev["protocol"]:
        raise ValueError("train/development tokenizer/protocol differs")
    for field in ("context_sha256","article"):
        training={c[field] for e in train["episodes"] for c in e["contexts"]}
        development={c[field] for e in dev["episodes"] for c in e["contexts"]}
        if training&development: raise ValueError("train/development leakage: "+field)


def command(args,out,variant,device,stage):
    shared=["--model-path",args.model_path,"--cuda-device",str(device)]
    if stage in ("smoke","train"):
        cmd=[sys.executable,str(ROOT/"tools/natural_merge_v4_train.py"),"--plan",args.train_plan,
            "--variant",variant,"--out",str(out/(variant+"."+stage+".json")),
            "--checkpoint-out",str(out/(variant+"."+stage+".pt"))]+shared
        if stage=="smoke": cmd += ["--smoke-updates","1"]
        return cmd
    if stage=="fit":
        return [sys.executable,str(ROOT/"tools/natural_merge_v4_training_fit.py"),"--plan",args.train_plan,
            "--checkpoint",str(out/(variant+".train.pt")),"--out",str(out/(variant+".training_fit.json"))]+shared
    return [sys.executable,str(ROOT/"tools/natural_merge_eval.py"),"--plan",args.dev_plan,
        "--checkpoint",str(out/(variant+".train.pt")),"--out",str(out/(variant+".e1_development.json"))]+shared


def check_stage(out,variant,stage):
    path=out/(variant+"."+({"eval":"e1_development","fit":"training_fit"}.get(stage,stage))+".json")
    report=json.loads(path.read_text())
    if report["status"]!="complete": raise ValueError("stage has no complete result: "+str(path))
    if stage=="smoke":
        if report["evaluation_eligible"] or len(report["updates"])!=1:
            raise ValueError("smoke incorrectly marked eligible")
    elif stage=="train":
        if not report["evaluation_eligible"] or len(report["updates"])!=64:
            raise ValueError("training incomplete/ineligible")
        if report["checkpoint_sha256"]!=sha(out/(variant+".train.pt")):
            raise ValueError("training checkpoint hash mismatch")
    elif stage=="fit":
        if not report.get("training_fit_only") or report.get("evaluation_eligible"):
            raise ValueError("training-fit diagnosis incorrectly labeled as evaluation")
    elif report.get("audit",{}).get("status")!="passed":
        raise ValueError("development verification did not pass")


def run(args):
    train=json.loads(Path(args.train_plan).read_text()); dev=json.loads(Path(args.dev_plan).read_text())
    validate_scopes(train,dev)
    devices=[int(d) for d in args.devices.split(",")]
    if not devices or len(set(devices))!=len(devices) or min(devices)<0:
        raise ValueError("unique nonnegative physical GPU indices required")
    variants=args.variants.split(",")
    if not variants or len(set(variants))!=len(variants) or any(v not in ("Dsup","Dnative") for v in variants):
        raise ValueError("unique variants chosen from Dsup,Dnative required")
    if devices_with_room(devices)!=devices:
        raise ValueError("selected GPUs each need >=32000MiB free; no tasks will be stopped")
    args.train_plan=str(Path(args.train_plan).resolve()); args.dev_plan=str(Path(args.dev_plan).resolve())
    args.model_path=str(Path(args.model_path).resolve()); out=Path(args.out_dir).resolve()
    locks=[]; jobs={}
    def interrupted(signum,frame):
        raise KeyboardInterrupt("launcher received signal "+str(signum))
    previous_term=signal.signal(signal.SIGTERM,interrupted)
    try:
        for d in devices:
            handle=open("/tmp/prlmc-natural-merge-gpu-"+str(d)+".lock","a")
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close(); raise ValueError("another v4 launcher holds GPU "+str(d))
            locks.append(handle)
            for prefix in ("prlmc-natural-merge-v2-gpu-",):
                legacy=open("/tmp/"+prefix+str(d)+".lock","a")
                try: fcntl.flock(legacy,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    legacy.close(); raise ValueError("an existing launcher holds GPU "+str(d))
                locks.append(legacy)
        out.mkdir(parents=True,exist_ok=False)
        state=dict(schema="natural_merge_v4_queue/1",status="running",launcher_pid=os.getpid(),
            train_plan_sha256=sha(args.train_plan),dev_plan_sha256=sha(args.dev_plan),
            devices=devices,variants=variants,workers=[],formal_evaluation_started=False)
        state_path=out/"launcher.json"; atomic_json(state_path,state)
        pending=list(variants)
        def launch(variant,device,stage):
            if device not in devices_with_room([device]):
                raise ValueError("GPU became occupied before "+variant+" "+stage)
            cmd=command(args,out,variant,device,stage)
            log=(out/(variant+"."+stage+".log")).open("x")
            job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
            entry=dict(variant=variant,device=device,stage=stage,pid=job.pid,command=cmd,
                log=str(out/(variant+"."+stage+".log")),started=time.time(),exit_code=None)
            state["workers"].append(entry); jobs[device]=(job,log,entry)
            atomic_json(state_path,state); print(json.dumps(entry),flush=True)
        while pending or jobs:
            for device in devices:
                if device not in jobs and pending: launch(pending.pop(0),device,"smoke")
            for device,(job,log,entry) in list(jobs.items()):
                code=job.poll()
                if code is None: continue
                entry.update(exit_code=code,ended=time.time()); log.close(); del jobs[device]
                atomic_json(state_path,state)
                if code!=0: raise RuntimeError(entry["variant"]+" "+entry["stage"]+" failed, see "+entry["log"])
                check_stage(out,entry["variant"],entry["stage"])
                if entry["stage"]!="eval":
                    # Wait only for this terminated worker's CUDA allocations to
                    # disappear; fail if another job has taken the released card.
                    for _ in range(5):
                        if device in devices_with_room([device]): break
                        time.sleep(1)
                    launch(entry["variant"],device,{"smoke":"train","train":"fit","fit":"eval"}[entry["stage"]])
            time.sleep(2)
        if set(variants)=={"Dsup","Dnative"}:
            from summarize_natural_merge_v4 import summarize_variants
            results={v:json.loads((out/(v+".e1_development.json")).read_text()) for v in variants}
            summary=summarize_variants(results,dev)
            summary["training_fit"]={v:json.loads((out/(v+".training_fit.json")).read_text())["summary"] for v in variants}
            summary["result_files"]={v:dict(path=str(out/(v+".e1_development.json")),
                sha256=sha(out/(v+".e1_development.json"))) for v in variants}
            atomic_json(out/"comparison.json",summary)
        state.update(status="complete",ended=time.time()); atomic_json(state_path,state)
    except BaseException as exc:
        # Only terminate children launched by this queue, never unrelated jobs.
        for job,log,entry in jobs.values():
            if job.poll() is None: os.killpg(job.pid,signal.SIGTERM)
        for job,log,entry in jobs.values():
            try: job.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(job.pid,signal.SIGKILL); job.wait()
            entry.update(exit_code=job.returncode,ended=time.time()); log.close()
        if "state" in locals():
            state.update(status="failed",failure=dict(type=type(exc).__name__,message=str(exc)))
            atomic_json(state_path,state)
        raise
    finally:
        for handle in locks: handle.close()
        signal.signal(signal.SIGTERM,previous_term)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-plan",required=True); p.add_argument("--dev-plan",required=True)
    p.add_argument("--devices",required=True); p.add_argument("--out-dir",required=True)
    p.add_argument("--model-path",default="/dev/shm/s1d_reader_fp16")
    p.add_argument("--variants",default="Dsup,Dnative")
    run(p.parse_args())


if __name__=="__main__": main()
