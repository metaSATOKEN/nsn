"""NB03 actual-array capture and replay, outside all primary timings."""
import loop_kv_nb05_kernel as k
import loop_kv_nb05_numeric as c
w=k.w

def array(x,bits=False):
    x=x.detach().contiguous()
    if bits:x=x.view(w.torch.uint16)
    return x.cpu().numpy().copy()

def input_arrays(data):
    g,kt,q,vg,vt=data;out={'q':array(q,True),'k_tail':array(kt,True),'v_tail':array(vt,True)}
    for side,group in [('k',g),('v',vg)]:
        if group is not None:
            out[side+'_codes']=array(group['codes']);out[side+'_headers']=array(group['headers'])
            out.update({side+'_meta'+str(i):array(x) for i,x in enumerate(group['metas'])})
    return out

def diagnostic(data):
    saved={}
    for arm in c.ARMS:
        out,ctx,score,v,key=k.pipeline(arm,data,trace=True)
        saved.update({arm+'_output':array(out,True),arm+'_k':array(key,True),arm+'_v':array(v,True)})
        if ctx is not None:saved[arm+'_context']=array(ctx)
        if score is not None:saved[arm+'_score']=array(score)
        del out,ctx,score,v,key
        plain=k.pipeline(arm,data);saved[arm+'_normal']=array(plain,True);del plain
    return saved

