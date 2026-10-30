"""CPU-only NC01 audit. Reuses frozen VF01 numerical criteria, never imports torch."""
import sys
from pathlib import Path
import loop_kv_nc01_contract as c
np=c.np

def verify(root,check_summary=True):
    root=Path(root)
    c.require(c.read(root/'loop_kv_nc01_inputs.json')==c.PLAN,'NC01 input contract changed')
    capsule=c.read(root/'capsule.json')
    for n,h in capsule.items():
        p=Path(n);c.require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
        c.require(c.file_sha(root/p)==h,'Source digest '+n)
    frozen=c.read(Path(__file__).with_name('nc01_frozen_dependencies.json'))
    for n,h in frozen.items():c.require(capsule.get(n)==h and c.file_sha(root/n)==h,'Frozen VF01 dependency '+n)
    env=c.read(root/'environment.json')
    c.require(env['model_loaded'] is False and env['torch']=='2.11.0+cu128' and env['numpy']=='2.1.3','Environment mismatch')
    c.require(env['kernel']['reference']['baseline']['triton']=='3.6.0','Triton version')
    c.require(c.read(root/'execution_settings.json')==c.SETTINGS,'Precision settings')
    controls=c.rows(root/'controls.jsonl');c.plans(controls,c.SPECS,'controls')
    old=c.rows(root/'old_layouts.jsonl');c.plans(old,[dict(fixture=s['id']) for s in c.SPECS],'old layouts')
    attempts=c.rows(root/'attempts.jsonl');c.plans(attempts,c.ATTEMPTS,'attempts')
    inputs={s['id']:c.base.fixture(s) for s in c.SPECS}
    reports=[];old_reports=[];negative=[];complete=True;unchanged=True
    for row in controls:
        a=inputs[row['id']];z=c.load(c.file(root,row))
        c.require(all(n in z and c.eq(z[n],x) for n,x in a.items()),'Changed control fixture')
        c.require(row['input_before']==c.arrays_hash(a),'Control initial fingerprint')
        stable=row['input_after']==row['input_before'];unchanged=unchanged and stable
        c.check_layout(row['v_layout'],a['v_tail'].shape);c.require(row['v_layout']['contiguous'],'Control must be contiguous')
        rep=c.numerics(z,a['q'])
        c.require(set(row['statuses']) in (set(c.ARMS),set(c.ARMS)|{'direct'}),'Control statuses inventory')
        all_finite=all(v=='FINITE' for v in row['statuses'].values())
        full=rep['complete'] and all_finite and all('direct_'+key in z for key in ('output','v','score','context'))
        complete=complete and full
        reports.append(dict(id=row['id'],complete=full,unchanged=stable,numerics=rep,route_pass=all(rep['route'].values()),numeric_pass=c.numeric_ok(rep)))
    for i,row in enumerate(old):
        a=inputs[row['fixture']];z=c.load(c.file(root,row))
        c.require(set(z)=={'original','old_view'} and c.eq(z['original'],a['v_tail']),'Old-layout evidence')
        c.check_layout(row['original_layout'],z['original'].shape);c.check_layout(row['old_layout'],z['old_view'].shape)
        same=c.eq(z['original'],z['old_view']);c.require(row['values_equal']==same,'Old-layout value claim')
        old_reports.append(dict(fixture=row['fixture'],values_equal=same,contiguous=row['old_layout']['contiguous'],prediction_pass=same and row['old_layout']['contiguous']==c.PLAN['expected_old_contiguous'][i]))
    for row in attempts:
        a=inputs[row['fixture']];z=c.load(c.file(root,row))
        c.require(c.eq(z['original'],a['v_tail']),'Negative source fixture')
        for key,array in [('original_layout','original'),('view_layout','view_before'),('backing_layout','backing_before')]:c.check_layout(row[key],z[array].shape)
        ready=c.construction_checks(z['original'],z['view_before'],z['backing_before'],row['original_layout'],row['view_layout'],row['backing_layout'],row['shared_storage'])
        c.require(ready==row['construction_checks'],'Construction checks disagree with evidence')
        valid=all(ready.values())
        c.require(row['construction']==('READY' if valid else 'FAILED'),'Construction status')
        if not valid:
            c.require(row['target_called'] is False and row['result']=='NOT_RUN' and row['input_unchanged'] is None and row['finite_launch_delta'] is None,'Unconstructed input must not be scored as rejection')
            complete=False;negative.append(dict(id=row['id'],construction='FAILED',result='NOT_RUN',rejection_pass=None,unchanged=None));continue
        c.check_layout(row['view_layout'],a['v_tail'].shape,True)
        c.require(row['backing_layout']['contiguous'],'Backing must be contiguous')
        c.require(row['view_layout']['stride']==row['backing_layout']['stride'][:-1]+[2],'View/backing stride relationship')
        c.require(row['target_called'] is True,'Constructed input not attempted')
        c.require(row['input_before']==c.arrays_hash(a),'Negative initial logical fingerprint')
        c.require(row['result'] in ('REJECTED','ACCEPTED','ERROR'),'Unknown target result')
        if row['result']=='ERROR':
            complete=False;negative.append(dict(id=row['id'],construction='READY',result='ERROR',rejection_pass=None,unchanged=None));continue
        if row['result']=='ACCEPTED':c.require('unexpected_output' in z,'Accepted input output missing')
        else:c.require('unexpected_output' not in z,'Rejected input has an output')
        stable=row['input_before']==row['input_after'] and c.eq(z['view_before'],z['view_after']) and c.eq(z['backing_before'],z['backing_after'])
        c.require(type(row['input_unchanged'])==bool and row['input_unchanged']==stable,'Mutation report mismatch')
        c.require(type(row['finite_launch_delta'])==int and row['finite_launch_delta']>=0,'Invalid launch count')
        unchanged=unchanged and stable
        passed=row['result']=='REJECTED' and row['stage']=='structural' and row['error_type']=='IntegrityError' and row['finite_launch_delta']==0
        negative.append(dict(id=row['id'],construction='READY',result=row['result'],stage=row['stage'],finite_launch_delta=row['finite_launch_delta'],rejection_pass=passed,unchanged=stable))
    gates=dict(normal_route=all(r['route_pass'] for r in reports),normal_numeric=all(r['numeric_pass'] for r in reports),
        construction=all(r['construction']=='READY' for r in negative),negative_rejection=all(r['rejection_pass'] is True for r in negative),
        input_unchanged=unchanged,old_layout_prediction=all(r['prediction_pass'] for r in old_reports))
    result=dict(experiment='LOOP-KV-NC01',version=c.PLAN['version'],status='PASS' if complete and all(gates.values()) else 'FAIL' if complete else 'INCONCLUSIVE',
        complete=complete,planned=c.PLAN['planned'],observed=dict(control_fixtures=len(controls),control_arm_outputs=sum(sum(r['numerics']['metrics'][arm]['complete'] for arm in c.ARMS) for r in reports),negative_attempts=len(attempts),old_layout_witnesses=len(old)),
        gates=gates,controls=reports,old_layouts=old_reports,attempts=negative,model_loaded=False,timing_comparison=False,parent_h1_rewritten=False)
    if check_summary:c.base.fc.equal(result,c.read(root/'summary.json'),'NC01 summary')
    return result
if __name__=='__main__':
    r=verify(sys.argv[1]);print('NC01_OFFLINE_VERIFIED',r['status'])
