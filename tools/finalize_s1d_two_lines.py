#!/usr/bin/env python3
"""Wait for four reports plus independent official audit, then write summary."""
import argparse
import json
from pathlib import Path
import time
from native_qa_compare import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='s1d_two_lines_v2')
    args=p.parse_args();runs=ROOT/'runs'
    summarypath=runs/(args.prefix+'.summary.json');auditpath=runs/(args.prefix+'.official_audit.json')
    def read(path):return json.loads(path.read_text()) if path.exists() else {}
    while True:
        summary=read(summarypath);audit=read(auditpath)
        if summary.get('status') in ('complete','failed') and audit.get('status') in ('complete','failed'):break
        time.sleep(20)
    summary['official_audit']=audit
    summary['status']='complete' if summary['status']=='complete' and audit['status']=='complete' else 'failed'
    cachepaths=[runs/'s1d_two_lines_v1.v100-1.cache_verification.json',
                runs/'s1d_two_lines_v2.v100-2.cache_verification.json']
    caches=[read(p) for p in cachepaths]
    summary['model_caches_verified']=all(c.get('valid') for c in caches) and len({c.get('manifest_sha256') for c in caches})==1
    summary['model_cache_certificates']=[str(p) for p in cachepaths]
    summary['input_assets']=read(runs/(args.prefix+'.asset_manifest.json'))
    if not summary['model_caches_verified'] or not summary['input_assets'].get('valid'):
        summary['status']='failed'
    atomic_json(runs/(args.prefix+'.final_summary.json'),summary)
    lines=['# S1d 四卡两线最终汇总','',
           '仅两个已见开发窗口、单seed；生成问题为事实模板构造，按官方EM规则计分，不是原生官方题集成绩。',
           '', '状态：`'+summary['status']+'`。', '',
           '| 方法任务 | R | H | net | 当前推进门槛 |','|---|---:|---:|---:|---|']
    for name,result in summary.get('method',{}).items():
        lines.append('| %s | %s | %s | %s | %s |'%(name,result['R'],result['H'],result['net'],
                    '通过' if result['gate_passed'] else '未通过'))
    lines+=['','| 生成任务 | Full EM | Selection EM | D0 EM | C EM | C−D0净正确 | C−Selection净正确 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,result in summary.get('application',{}).items():
        s=result['summary'];a=s['arms'];n=s['n']
        lines.append('| %s | %d/%d | %d/%d | %d/%d | %d/%d | %+d | %+d |'%(name,
            a['Full']['correct'],n,a['Selection']['correct'],n,a['D0']['correct'],n,a['C']['correct'],n,
            s['C_vs_D0']['net'],s['C_vs_Selection']['net']))
    lines+=['','方法推进门槛：弱net≥5且H≤4；强net≥7并单独检查损伤。方法门槛'+
            ('通过' if summary.get('method_gates_passed') else '未全部通过')+'。',
            '官方源码逐题复算状态：`'+audit['status']+'`。',
            'M1为接管的训练进程，退出码未直接取得；其完整报告与冻结basis证书是完成依据。',
            '生成线测旧C，不能作为新RiskProtect的生成效果证明。所有checkpoint使用固定step60，不挑最佳中途值。',
            '16个测试episode保持封存；不得依据应用成绩扫描prompt、λ或删选测试子集。',
            '', '完整逐题结果、命令、真实退出码（新任务）、哈希、smoke/正式证书位于对应runs报告。']
    (ROOT/'experiments'/'S1d-two-lines-v2-results.md').write_text('\n'.join(lines)+'\n')
    print('final summary written',summary['status'],flush=True)

if __name__=='__main__':main()
