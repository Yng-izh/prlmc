import json
from pathlib import Path
import tempfile
from s1d_controlled_contract import claim_sealed_test,file_sha,episode_bootstrap


def test_single_open():
    with tempfile.TemporaryDirectory() as d:
        plan=Path(d)/'plan.json';plan.write_text('{}')
        frozen=Path(d)/'frozen.json';manifest=dict(method='C',code='sha')
        frozen.write_text(json.dumps(dict(status='frozen',plan_sha256=file_sha(plan),
              manifest=manifest,all_development_complete=True,test_scored=False)))
        try:claim_sealed_test(plan,frozen,dict(method='changed'))
        except ValueError:pass
        else:raise AssertionError('freeze mismatch accepted')
        assert not plan.with_suffix('.s1d_test_opened.json').exists()
        ledger=claim_sealed_test(plan,frozen,manifest);assert ledger.exists()
        try:claim_sealed_test(plan,frozen,manifest)
        except FileExistsError:pass
        else:raise AssertionError('test reopened')


def test_bootstrap():
    rows=[dict(episode=i,status='complete',C_em=.7,D0_em=.5,Selection_em=.6) for i in range(16)]
    result=episode_bootstrap(rows,'D0')
    assert result['valid'] and abs(result['mean_delta']-.2)<1e-8
    assert all(abs(x-.2)<1e-8 for x in result['ci95'])
    assert result==episode_bootstrap(rows,'D0')
    rows[-1]['status']='failed'
    assert episode_bootstrap(rows,'D0')['valid'] is False

if __name__=='__main__':
    for f in (test_single_open,test_bootstrap):f();print('PASS '+f.__name__)
