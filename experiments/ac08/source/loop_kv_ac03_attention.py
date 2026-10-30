"""AC03 decode-only common connection. Import has no GPU/model side effects."""
from collections import Counter
import loop_kv_qk01_kernel as qk
import loop_kv_af01_kernel as fused
w=qk.w;ref=qk.ref

def make_cache(base):
    parent=ref.make_cache(base)
    class ConnectionCache(parent):
        ac01_connection=True
        def __init__(self,r,bits=3,arm='PACKED_SDPA'):
            super().__init__(r,bits);self.arm=arm;self._defer_k=False
            self.connection_calls=Counter();self.skipped_k=Counter();self.capture=None
        def materialize(self,slot,kv):
            if kv=='K' and self._defer_k:
                self.skipped_k[slot]+=1
                return None
            return super().materialize(slot,kv)
        def append_for_attention(self,k,v,slot,cache_kwargs):
            w.require(not self._defer_k and self.restore_observer is None,'Nested/labeled cache update')
            self._defer_k=True
            try:return super().update(k,v,slot,cache_kwargs)
            finally:self._defer_k=False
    return ConnectionCache

def check_decode(module,q,cache,slot,position,mask):
    t=w.torch;total=cache.get_seq_length(slot)
    w.require(not module.training and module.num_key_value_groups==1 and module.sliding_window is None,'AC03 requires eval, no GQA/sliding window')
    w.require(tuple(q.shape)==(1,w.HEADS,1,128) and q.dtype==t.bfloat16,'Decode query shape/dtype')
    w.require(position is not None and position.numel()==1 and int(position.item())==total-1,'Decode must append at current last position')
    w.require(mask is not None and tuple(mask.shape)==(1,1,1,total) and mask.is_floating_point() and bool((mask==0).all()),'AC03 requires explicit all-allowed additive decode mask')
    w.require(module.scaling==128**-.5,'Attention scaling changed')
    return total

def finish(score,v):
    t=w.torch
    probability=t.softmax(score,dim=-1,dtype=t.float32)
    context=t.matmul(probability.unsqueeze(-2),v.float()).contiguous()
    if not bool(t.isfinite(probability).all() and t.isfinite(context).all()):raise w.NumericalFailure('Nonfinite explicit attention')
    output=context.to(t.bfloat16).transpose(1,2).contiguous()
    return output,probability,context

def install(model,module):
    """Patch instances of the pinned class, restoring every original bound method on exit."""
    import types
    changed=[]
    for layer in model.model.layers:
        attn=layer.self_attn;original=attn.forward
        def forward(self,hidden_states,position_embeddings,attention_mask,past_key_value=None,cache_position=None,current_ut=0,_original=original,**kwargs):
            if not getattr(past_key_value,'ac01_connection',False) or hidden_states.shape[1]!=1:
                return _original(hidden_states,position_embeddings,attention_mask,past_key_value=past_key_value,cache_position=cache_position,current_ut=current_ut,**kwargs)
            cache=past_key_value
            input_shape=hidden_states.shape[:-1]
            hidden_shape=(*input_shape,-1,self.head_dim)
            query_states=self.q_proj(hidden_states).view(hidden_shape).transpose(1,2)
            key_states=self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
            value_states=self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
            cos,sin=position_embeddings
            query_states,key_states=module.apply_rotary_pos_emb(query_states,key_states,cos,sin)
            slot=current_ut*self.config.num_hidden_layers+self.layer_idx
            _,value_states=cache.append_for_attention(key_states,value_states,slot,{'sin':sin,'cos':cos,'cache_position':cache_position})
            query_states=query_states.contiguous()
            total=check_decode(self,query_states,cache,slot,cache_position,attention_mask)
            tail=cache.key_cache[slot];g=cache.packed.get((slot,'K'))
            # Same common shape/finite checks for all three arms. Representation-specific
            # checks in qk.validate remain part of the measured implementation cost.
            qk.validate(g,tail,query_states)
            fused.validate_v(value_states,query_states,total)
            if cache.arm in ('NATIVE','PACKED_SDPA'):
                key_states=qk.reference_k(g,tail)
                attn_output,_=module.ALL_ATTENTION_FUNCTIONS['sdpa'](self,query_states,key_states,value_states,attention_mask,dropout=0.0,scaling=self.scaling,sliding_window=None,**kwargs)
            elif cache.arm=='PACKED_FUSED':
                attn_output,_,_=fused.launch(g,tail,query_states,value_states,full=True)
            else:raise w.IntegrityError('Unknown AC03 connected arm')
            if not bool(w.torch.isfinite(attn_output).all()):raise w.NumericalFailure('Nonfinite attention output')
            cache.connection_calls[slot]+=1
            if cache.capture is not None:cache.capture(slot,total,g,tail,query_states,value_states,attn_output)
            attn_output=attn_output.reshape(*input_shape,-1).contiguous()
            attn_output=self.o_proj(attn_output)
            return attn_output,None
        changed.append((attn,original));attn.forward=types.MethodType(forward,attn)
    def restore():
        for attn,original in changed:attn.forward=original
    return restore
