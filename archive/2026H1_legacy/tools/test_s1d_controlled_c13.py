"""CPU checks for sealed split, write-only boundary and original C kernels."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import tempfile

from s1d_controlled_c13 import DEVELOPMENT_IDS, PROTOCOL, development_episodes, write_inputs

HERE = Path(__file__).parent


class NoSealedAccess(dict):
    def __getitem__(self, key):
        if key != 'partition':
            raise AssertionError('sealed row content was accessed')
        return super().__getitem__(key)


class Contract(unittest.TestCase):
    def setUp(self):
        self.plan = dict(status='completed', episodes=[dict(episode=i, partition='development') for i in DEVELOPMENT_IDS]
                         + [NoSealedAccess(partition='test') for _ in range(16)])

    def test_development_only(self):
        chosen = development_episodes(self.plan, list(DEVELOPMENT_IDS))
        self.assertEqual([e['episode'] for e in chosen], list(DEVELOPMENT_IDS))
        for ids in ([1], [0,1], [0,0], []):
            with self.assertRaises(ValueError):
                development_episodes(self.plan, ids)

    def test_write_boundary_does_not_access_qa(self):
        lines = ['%d. fact %d' % (i+1,i) for i in range(410)]
        class WriteOnly(dict):
            def __getitem__(self, key):
                if key not in ('lines','context_sha256'):
                    raise AssertionError('training accessed question/answer metadata')
                return super().__getitem__(key)
        episode = WriteOnly(lines=lines, context_sha256=hashlib.sha256('\n'.join(lines).encode()).hexdigest())
        raw, chunks = write_inputs(episode)
        self.assertEqual(len(chunks),20)
        self.assertEqual(len(chunks[0]),30)
        self.assertEqual([len(c) for c in chunks[1:]],[20]*19)
        self.assertEqual(raw[0],'fact 0')
        self.assertEqual(sum(chunks, []),lines)

    def test_protocol_is_fixed(self):
        self.assertEqual((PROTOCOL['rank'],PROTOCOL['steps'],PROTOCOL['lr'],PROTOCOL['budget']),(13,60,.003,1.))
        self.assertEqual(PROTOCOL['arms'],['Full','Selection','D0','C'])
        self.assertFalse(PROTOCOL['evaluation_qa_used_for_training'])

    def test_original_c_kernels_unchanged(self):
        old=ast.parse((HERE/'s1d_experiment1.py').read_text())
        new=ast.parse((HERE/'s1d_controlled_c13.py').read_text())
        original=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        runtime=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        for name in ('base','candidate','answer_ce','supervision_lines','fact_records','encode_fact_records','fact_basis','train_arm'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)

    def test_evaluation_is_after_checkpoint_freeze(self):
        source=(HERE/'s1d_controlled_c13.py').read_text()
        self.assertLess(source.index('temporary.replace(alpha_path)'),source.index("evaluation = episode['splits']['evaluation']"))
        self.assertNotIn("choices=['development', 'test']",source)

    def test_episode_aggregation_rescores_and_rejects_tampering(self):
        from run_s1d_controlled_dev import aggregate
        from s1d_c_generation import paired_summary
        from s1d_controlled_c13 import source_manifest,sha
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for episode in DEVELOPMENT_IDS:
                path=Path(folder)/('ep%d.json'%episode)
                alpha=path.with_suffix('.alpha.pt');alpha.write_bytes(b'frozen-checkpoint')
                generation_path=path.with_suffix('.generation.json')
                maps={name:{'q%d'%i:(name in ('Full','C') or i%2==0) for i in range(64)}
                      for name in PROTOCOL['arms']}
                summary=paired_summary(maps)
                pairs=[dict(question='q%d'%i,answers=['Paris'],**{
                    name:dict(prediction='Paris' if maps[name]['q%d'%i] else 'Rome',em=maps[name]['q%d'%i])
                    for name in maps}) for i in range(64)]
                generation=dict(status='complete',episode=episode,source_memory_unchanged=True,
                    pairs=pairs,checkpoint_sha256=sha(alpha),summary=summary,
                    arm_results={name:dict(active_slots=2816) for name in PROTOCOL['arms']})
                generation_path.write_text(json.dumps(generation))
                report=dict(status='complete',episode=episode,identity=dict(protocol=PROTOCOL,
                    smoke=False,episode=episode,context_sha256='context%d'%episode,
                    source_manifest=source_manifest()),training=dict(steps_completed=60),
                    checkpoint_frozen=True,checkpoint_sha256=sha(alpha),active_slots=2816,
                    generation_file=str(generation_path),generation_sha256=sha(generation_path),
                    summary=summary,costs={})
                path.write_text(json.dumps(report));paths.append(path)
            result=aggregate(paths)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['positive_vs_Selection'],8)
            self.assertEqual(result['paired']['Selection']['ci95'],[.5,.5])
            generation['pairs'][0]['C']['prediction']='Rome'
            generation_path.write_text(json.dumps(generation))
            report['generation_sha256']=sha(generation_path)
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'scoring mismatch'):
                aggregate(paths)

    def test_failed_episodes_are_not_dropped_for_inference(self):
        from run_s1d_controlled_dev import aggregate
        with tempfile.TemporaryDirectory() as folder:
            paths=[Path(folder)/('missing-ep%d.json'%i) for i in DEVELOPMENT_IDS]
            result=aggregate(paths)
            self.assertEqual(len(result['failures']),8)
            self.assertNotIn('paired',result)


if __name__=='__main__':
    unittest.main()
