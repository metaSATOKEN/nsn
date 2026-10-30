"""Nanbeige official projection/norm/RoPE + AC08 deferred packed restoration.

The original bound forward handles prefill. Only mask-free single-query decode
uses the NB03 component. No model or Transformers imports at module import time.
"""
from collections import Counter
from types import MethodType
import inspect
import loop_kv_nb05_contract as c
import loop_kv_nb05_kernel as k
from loop_kv_nb05_packed import make_packed_cache

def make_cache(base):
    parent=make_packed_cache(base)
    class ConnectionCache(parent):
        def __init__(self,arm):
            c.require(arm in ('OFFICIAL',*c.SERIES),'Unknown model arm')
            super().__init__(2,None if arm in ('OFFICIAL','NATIVE') else 3)
            self.arm=arm;self._defer=False;self.connections=Counter()
            self.skipped_k=Counter();self.skipped_v=Counter();self.capture=None
        def materialize(self,slot,kv):
            if self._defer:
                (self.skipped_k if kv=='K' else self.skipped_v)[slot]+=1
                return None
            return super().materialize(slot,kv)
        def append_for_attention(self,key,value,slot,kwargs):
            c.require(not self._defer and self.restore_observer is None,'Recursive/decorated restore')
            self._defer=True
            try:super().update(key,value,slot,kwargs)
            finally:self._defer=False
    return ConnectionCache

def decode(self,module,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
           output_attentions=False,use_cache=False,cache_position=None,**kwargs):
    loop_idx=kwargs.pop('loop_idx',0)
    override=kwargs.pop('loop_cache_layer_idx',None)
    for name in ('loop_share_kv_cache','loop_share_kv_repeat_idx','depth_attention_kv_cache'):
        c.require(kwargs.pop(name,None) is None,'Unsupported attention feature '+name)
    c.require(not kwargs,'Unexpected attention keywords')
    bsz,q_len,_=hidden_states.size()
    c.require((bsz,q_len)==(1,1) and use_cache and not output_attentions and not self.training,'Single-query eval/cache only')
    c.require(attention_mask is None,'Only official mask-free single-query decode is supported')
    c.require((self.num_heads,self.num_key_value_heads,self.head_dim,self.num_key_value_groups)==(48,8,128,6),'GQA dimensions')
    slot=module._get_loop_cache_layer_idx(self.layer_idx,loop_idx,self.config.num_hidden_layers,override)
    c.require(type(loop_idx) is int and loop_idx in (0,1) and slot==self.layer_idx+loop_idx*22 and 0<=slot<44,'Loop/cache slot identity')
    # Same ordered operations as the pinned official NanbeigeSdpaAttention.forward.
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    if self.q_layernorm is not None:
        query_states = self.q_layernorm(query_states)
    if self.k_layernorm is not None:
        key_states = self.k_layernorm(key_states)
    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = module.apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {'sin':sin,'cos':cos,'cache_position':cache_position}
    past_key_value.append_for_attention(key_states,value_states,slot,cache_kwargs)
    data=(past_key_value.packed.get((slot,'K')),past_key_value.key_cache[slot],query_states.contiguous(),
          past_key_value.packed.get((slot,'V')),past_key_value.value_cache[slot])
    arm='GQA_FUSED' if past_key_value.arm=='GQA_FUSED' else 'TORCH_SDPA'
    attn_output=k.pipeline(arm,data)
    past_key_value.connections[slot]+=1
    if past_key_value.capture is not None:past_key_value.capture(slot,data,attn_output)
    attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output,None,past_key_value

def install(model):
    originals=[]
    for layer in model.model.layers:
        attn=layer.self_attn;original=attn.forward;module=inspect.getmodule(attn.__class__)
        originals.append((attn,original))
        def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                    output_attentions=False,use_cache=False,cache_position=None,_original=original,_module=module,**kwargs):
            if hidden_states.shape[1]!=1 or getattr(past_key_value,'arm',None)=='OFFICIAL':
                return _original(hidden_states,attention_mask=attention_mask,position_ids=position_ids,
                    past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,
                    cache_position=cache_position,**kwargs)
            c.require(hasattr(past_key_value,'append_for_attention'),'NB05 cache required')
            return decode(self,_module,hidden_states,attention_mask,position_ids,past_key_value,
                          output_attentions,use_cache,cache_position,**kwargs)
        attn.forward=MethodType(forward,attn)
    def restore():
        for attn,original in originals:attn.forward=original
    return restore
