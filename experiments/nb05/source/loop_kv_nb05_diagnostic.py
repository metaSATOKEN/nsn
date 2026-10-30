"""Unchanged NB03 audit applied to captured model Q/KV."""
import numpy as np
import loop_kv_nb05_numeric as c

def diagnostic(z,a):
    cc,tt=c.structure(a,True);expected=set()
    for arm in c.ARMS:
        expected.update(arm+'_'+s for s in ('output','normal','k','v'))
        for s in ('output','normal','k','v'):
            x=z[arm+'_'+s];shape=(1,1,48,128) if s in ('output','normal') else (1,8,tt,128)
            c.require(x.dtype==np.uint16 and x.shape==shape and np.isfinite(c.from_bits(x)).all(),'Saved BF16 '+arm+s)
    expected.update(('GQA_FUSED_context','GQA_FUSED_score'))
    c.require(set(z)==expected,'Diagnostic field set')
    ctx=z['GQA_FUSED_context'];score=z['GQA_FUSED_score']
    c.require(ctx.dtype==np.float32 and ctx.shape==(1,48,1,128) and np.isfinite(ctx).all(),'Context format')
    c.require(score.dtype==np.float32 and score.shape==(1,48,tt) and np.isfinite(score).all(),'Score format')
    normal={arm:np.array_equal(z[arm+'_normal'],z[arm+'_output']) for arm in c.ARMS}
    restored={side:np.array_equal(z['TILE_SDPA_'+side],z['GQA_FUSED_'+side]) for side in ('k','v')}
    restoration={side:c.k_metrics(z['GQA_FUSED_'+side],z['TORCH_SDPA_'+side]) for side in ('k','v')}
    # CPU calculation is a numerical audit of the frozen NSN path, not a replacement for GPU evidence.
    reference_cpu={side:c.k_metrics(z['TORCH_SDPA_'+side],c.cpu_restore({n[len(side)+1:]:x for n,x in a.items() if n.startswith(side+'_')})) for side in ('k','v')}
    s,absprod=c.gold(np.repeat(z['GQA_FUSED_k'],6,axis=1),a['q'])
    score_result=c.score_metrics(score,s,absprod)
    ctx_result=c.context_metrics(score,np.repeat(z['GQA_FUSED_v'],6,axis=1),ctx)
    rounding=np.array_equal(c.bf16_bits(ctx).transpose(0,2,1,3),z['GQA_FUSED_output'])
    sdpa={arm:c.sdpa_metrics(z[arm+'_output'],z[arm+'_k'],z[arm+'_v'],a['q']) for arm in c.ARMS[:2]}
    pairwise={arm:c.difference(c.from_bits(z['GQA_FUSED_output']),c.from_bits(z[arm+'_output'])) for arm in c.ARMS[:2]}
    passed=all(normal.values()) and all(restored.values()) and all(m['pass_budget'] for m in list(restoration.values())+list(reference_cpu.values())+list(sdpa.values())+[score_result,ctx_result]) and rounding
    return dict(pass_all=bool(passed),trace_normal_equal=normal,tile_fused_restore_equal=restored,restoration=restoration,torch_cpu_restoration=reference_cpu,score=score_result,context=ctx_result,output_rounding_equal=bool(rounding),sdpa=sdpa,fused_vs_sdpa=pairwise)
