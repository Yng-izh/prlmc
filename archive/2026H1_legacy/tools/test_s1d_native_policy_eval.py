"""Exercise the actual upstream native write/drop functions on small tensors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from s1d_native_policy_eval import NATIVE_METHODS, native_method_source, set_native_capacity, PROTOCOL

ROOT=Path(__file__).resolve().parents[1]

class NativePolicyRuntime(unittest.TestCase):
    def runtime_class(self):
        import torch
        path=ROOT/'methods/memoryllm_modeling/modeling_memoryllm.py'
        tree=ast.parse(path.read_text())
        source=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemoryLLM')
        methods=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('drop_memory','update_memory_with_delta_memory')]
        cls=ast.ClassDef(name='Native',bases=[ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='nn',ctx=ast.Load())],keywords=[],body=methods,decorator_list=[])
        # Replace the base by torch.nn.Module.
        cls.bases=[ast.Attribute(value=cls.bases[0],attr='Module',ctx=ast.Load())]
        module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
        env={'torch':torch}; exec(compile(module,str(path),'exec'),env)
        return env['Native']

    def test_native_first_write_and_random_append_for_both_budgets(self):
        import torch
        cls=self.runtime_class()
        for blocks in (11,50):
            model=cls(); model.L=2; model.d=3; model.num_tokens=256
            model.num_blocks=50; model.drop_memory_per_layer=True
            model.config=SimpleNamespace()
            model.memory=torch.nn.Parameter(torch.zeros(2,12800,3),requires_grad=False)
            model.register_buffer('initialized',torch.tensor(0,dtype=torch.uint8))
            model.register_buffer('memory_active_mask',torch.ones(12800,dtype=torch.bool))
            set_native_capacity(model,blocks)
            self.assertIsNone(model.memory_active_mask)
            delta=torch.arange(2*256*3,dtype=torch.float32).reshape(1,2,256,3)
            model.update_memory_with_delta_memory(delta)
            self.assertTrue(torch.equal(model.memory,delta[0].repeat(1,blocks,1)))
            model.memory.data=torch.arange(2*blocks*256*3,dtype=torch.float32).reshape(2,blocks*256,3)
            source=model.memory.detach().clone(); appended=torch.full((1,2,256,3),-2.)
            torch.manual_seed(0)
            indices=[torch.randperm(blocks*256)[:(blocks-1)*256].sort()[0] for _ in range(2)]
            torch.manual_seed(0)
            model.update_memory_with_delta_memory(appended)
            for layer in range(2):
                self.assertTrue(torch.equal(model.memory[layer,:-256],source[layer,indices[layer]]))
                self.assertTrue(torch.equal(model.memory[layer,-256:],appended[0,layer]))
            self.assertFalse(torch.equal(indices[0],indices[1]))
            self.assertEqual(model.memory.shape,(2,blocks*256,3))

    def test_native_methods_identical_to_original_and_no_manager_in_runtime(self):
        current=ROOT/'methods/memoryllm_modeling/modeling_memoryllm.py'
        self.assertEqual(native_method_source(current),native_method_source(current.with_suffix('.py.bak')))
        tree=ast.parse((ROOT/'tools/s1d_native_policy_eval.py').read_text())
        forbidden={'ProtectedResidualMemory','CarrierResidualCompactor','AdamW','backward','zero_grad'}
        found={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}|{n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
        self.assertFalse(found & forbidden)
        self.assertEqual(PROTOCOL['seed'],0)
        self.assertEqual(PROTOCOL['evaluation_questions'],64)

if __name__=='__main__': unittest.main()
