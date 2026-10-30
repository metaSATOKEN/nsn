"""PK01 cache adapted only for Transformers 4.45.1 constructor/clear; 22 layers, 8 KV heads."""
from collections import Counter
from loop_kv_nb06_codec import *

def make_packed_cache(base):
    class PackedCache(base):
        def __init__(self,r,bits=3):
            super().__init__()
            w.require(bits in (None,3),'PK01 supports native/3bit only')
            self.r,self.bits=r,bits;self.packed={};self.compressed={};self.calls=Counter()
            self.encoded_blocks=Counter();self.restore_observer=None
        def get_seq_length(self,layer_idx=0):
            i=0 if layer_idx is None else layer_idx
            if i<0 or i>=len(self.key_cache) or self.key_cache[i] is None:return 0
            return self.compressed.get(i,0)+self.key_cache[i].shape[-2]
        def materialize(self,slot,kv):
            tail=(self.key_cache if kv=='K' else self.value_cache)[slot]
            g=self.packed.get((slot,kv))
            if g is None:return tail
            return w.torch.cat((restore_group(g,tail.dtype),tail),-2)
        def update(self,key,value,layer_idx,cache_kwargs=None):
            t=w.torch;i=layer_idx;old=self.get_seq_length(i)
            w.require(0<=i<w.LAYERS*self.r and key.shape==value.shape and tuple(key.shape[:2])==(1,w.HEADS) and key.shape[-1]==w.DIM,'Packed KV coordinates')
            pos=(cache_kwargs or {}).get('cache_position')
            if pos is not None:w.require(t.equal(pos,t.arange(old,old+key.shape[-2],device=pos.device)),'Packed append position')
            self.calls[i]+=1
            if self.compressed.get(i,0)==0:
                # Keep exactly the official native route until the first compression.
                return super().update(key,value,i,cache_kwargs)
            self.key_cache[i]=t.cat((self.key_cache[i],key),-2)
            self.value_cache[i]=t.cat((self.value_cache[i],value),-2)
            self._seen_tokens=self.get_seq_length(i)
            k=self.materialize(i,'K');v=self.materialize(i,'V')
            if self.restore_observer is not None:self.restore_observer(i,k,v)
            return k,v  # caller's temporary; never retained as a full float shadow
        def compress_to(self,target):
            if self.bits is None:return
            start=self.compressed.get(0,0)
            w.require(target>=start and (target-start)%64==0,'Block overlap/ordering')
            if target==start:return
            n=target-start
            for slot in range(w.LAYERS*self.r):
                w.require(self.compressed.get(slot,0)==start and self.get_seq_length(slot)>=target+64,'Tail/compressed divergence')
                for kv,data in [('K',self.key_cache),('V',self.value_cache)]:
                    old=data[slot]
                    new=encode_group(old[:,:,:n,:].reshape(1,w.HEADS,n//64,64,w.DIM),start)
                    self.packed[(slot,kv)]=append_group(self.packed.get((slot,kv)),new)
                    # clone is necessary: a view would retain the original large storage.
                    data[slot]=old[:,:,n:,:].contiguous().clone()
                self.compressed[slot]=target;self.encoded_blocks[slot]+=n//64
        def post_prefill(self):
            self.compress_to(max(0,((self.get_seq_length()-64)//64)*64))
        def post_decode(self):
            if self.bits is not None:
                current=self.compressed.get(0,0)
                if self.get_seq_length()-current>=128:self.compress_to(current+64)
        def reorder_cache(self,*args,**kwargs):
            raise w.IntegrityError('PK01 beam reorder not supported')
        def clear(self):
            self.key_cache.clear();self.value_cache.clear();self._seen_tokens=0;self.packed.clear();self.compressed.clear();self.calls.clear();self.encoded_blocks.clear()
    return PackedCache
