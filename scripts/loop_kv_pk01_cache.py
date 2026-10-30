"""PK01 real GPU-resident NSN3 representation. No model/network on import."""
import struct
from collections import Counter
import loop_kv_pilot as w


def pack3(q):
    t=w.torch
    w.require(q.dtype==t.uint8 and bool((q<=7).all()), '3bit code range/dtype')
    n=q.numel();a=q.contiguous().reshape(-1)
    if n%8:a=t.cat((a,t.zeros(8-n%8,dtype=t.uint8,device=a.device)))
    a=a.reshape(-1,8)
    b0=a[:,0] | (a[:,1]<<3) | (a[:,2]<<6)
    b1=(a[:,2]>>2) | (a[:,3]<<1) | (a[:,4]<<4) | (a[:,5]<<7)
    b2=(a[:,5]>>1) | (a[:,6]<<2) | (a[:,7]<<5)
    return t.stack((b0,b1,b2),-1).reshape(-1)[:(n*3+7)//8].clone()


def unpack3(payload,shape):
    t=w.torch;n=1
    for x in shape:n*=x
    w.require(payload.dtype==t.uint8 and payload.numel()==(n*3+7)//8,'3bit payload shape')
    a=payload.reshape(-1)
    if a.numel()%3:a=t.cat((a,t.zeros(3-a.numel()%3,dtype=t.uint8,device=a.device)))
    a=a.reshape(-1,3);x,y,z=a[:,0],a[:,1],a[:,2]
    vals=(x&7,(x>>3)&7,((x>>6)|(y<<2))&7,(y>>1)&7,(y>>4)&7,
          ((y>>7)|(z<<1))&7,(z>>2)&7,(z>>5)&7)
    return t.stack(vals,-1).reshape(-1)[:n].reshape(shape).contiguous()


def encode_group(x,start):
    """x is [1,H,nblocks,64,128]; keep no view into x or uncompressed q."""
    t=w.torch;q,metas=w.nsn_encode(x,3);b,h,n,_,_=q.shape
    packed=pack3(q).reshape(b,h,n,64*128*3//8)
    header=b''.join(struct.pack('<QHBB',start+i*64,64,3,0) for _ in range(b*h) for i in range(n))
    return dict(codes=packed,metas=tuple(a.contiguous().clone() for a in metas),
                headers=t.tensor(list(header),device=x.device,dtype=t.uint8).reshape(b,h,n,12))


def restore_group(g,dtype):
    b,h,n,_=g['codes'].shape
    return w.nsn_decode(unpack3(g['codes'],(b,h,n,64,128)),g['metas'],dtype).reshape(b,h,n*64,128)


def append_group(old,new):
    if old is None:return new
    t=w.torch
    return dict(codes=t.cat((old['codes'],new['codes']),2),
                metas=tuple(t.cat((a,b),2) for a,b in zip(old['metas'],new['metas'])),
                headers=t.cat((old['headers'],new['headers']),2))


def tensor_inventory(cache):
    tensors=[]
    for kv,data in [('K',cache.key_cache),('V',cache.value_cache)]:
        for slot,x in enumerate(data):
            if x is not None:tensors.append(('native_tail' if hasattr(cache,'packed') and cache.bits else 'float_kv',x))
            g=getattr(cache,'packed',{}).get((slot,kv))
            if g:
                tensors.extend([('codes',g['codes']),('headers',g['headers'])])
                tensors.extend(('metadata',v) for v in g['metas'])
    components=Counter();seen={};devices=set()
    for name,t in tensors:
        components[name]+=t.numel()*t.element_size();st=t.untyped_storage()
        seen[(str(t.device),st.data_ptr())]=st.nbytes();devices.add(str(t.device))
    return dict(component_bytes=dict(components),tensor_bytes=sum(components.values()),
                unique_storage_bytes=sum(seen.values()),tensor_count=len(tensors),storage_count=len(seen),
                devices=sorted(devices),compressed_tokens=cache.compressed.get(0,0) if hasattr(cache,'compressed') else 0)


def full_pair(cache,slot):
    if hasattr(cache,'materialize'):return cache.materialize(slot,'K'),cache.materialize(slot,'V')
    return cache.key_cache[slot],cache.value_cache[slot]


def make_packed_cache(base):
    class PackedCache(base):
        def __init__(self,r,bits=3):
            super().__init__(max_cache_size=w.LAYERS*r)
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
            super().clear();self.packed.clear();self.compressed.clear();self.calls.clear();self.encoded_blocks.clear()
    return PackedCache


def validate(cache,length,calls,bits):
    slots=set(range(w.LAYERS*cache.r))
    w.require(set(cache.calls)==slots and set(cache.calls.values())=={calls},'Missing/duplicate updates')
    target=max(0,((length-64)//64)*64) if bits else 0
    for i in slots:
        w.require(cache.get_seq_length(i)==length and cache.compressed.get(i,0)==target,'Length/compression inventory')
        w.require(cache.encoded_blocks.get(i,0)==target//64,'Re-encoding count')
        if hasattr(cache,'packed') and bits:
            for kv,data in [('K',cache.key_cache),('V',cache.value_cache)]:
                tail=data[i];w.require(tail.shape[-2]==length-target,'Tail length')
                w.require(tail.untyped_storage().nbytes()==tail.numel()*tail.element_size(),'Tail retains float shadow')
                g=cache.packed.get((i,kv))
                if target:
                    w.require(g['codes'].dtype==w.torch.uint8 and g['codes'].shape[2]==target//64,'Packed inventory')
                    expect=b''.join(struct.pack('<QHBB',n*64,64,3,0) for _ in range(w.HEADS) for n in range(target//64))
                    w.require(bytes(g['headers'].cpu().reshape(-1).tolist())==expect,'Header inventory')
                    w.require(all(x.dtype==w.torch.float16 for x in g['metas']),'Metadata precision')
    inv=tensor_inventory(cache)
    one=w.planned_bytes((1,w.HEADS,length,w.DIM),target,3 if bits else None)
    expected=sum(one.values())*2*len(slots) if hasattr(cache,'packed') and bits else 2*len(slots)*w.HEADS*length*w.DIM*2
    # Native prefill can be a view with official stride; all finished trajectories have appended tokens.
    w.require(inv['tensor_bytes']==expected,'Actual tensor byte accounting')
    if hasattr(cache,'packed') and bits:w.require(inv['unique_storage_bytes']==expected,'Packed unique storage byte accounting')
    return inv


class FixtureBase:
    def __init__(self,max_cache_size):self.key_cache=[];self.value_cache=[]
    def get_seq_length(self,layer_idx=0):
        return 0 if layer_idx>=len(self.key_cache) else self.key_cache[layer_idx].shape[-2]
    def update(self,k,v,i,cache_kwargs=None):
        while len(self.key_cache)<=i:self.key_cache.append(None);self.value_cache.append(None)
        for data,x in [(self.key_cache,k),(self.value_cache,v)]:
            data[i]=x if data[i] is None else w.torch.cat((data[i],x),-2)
        return self.key_cache[i],self.value_cache[i]
    def clear(self):self.key_cache=[];self.value_cache=[]


def _fixtures(device):
    t=w.torch;np=w.np;checks=[]
    for n in [1,2,3,7,8,9,63,64,65,8192]:
        a=((np.arange(n)*5+3)%8).astype(np.uint8)
        q=t.from_numpy(a).to(device);p=pack3(q)
        w.require(p.cpu().numpy().tobytes()==w.pack_codes(a,3),'Independent packed stream')
        w.require(t.equal(q,unpack3(p,q.shape)),'Packed roundtrip')
    checks.append(dict(test='pack3_independent_bytes',device=device,status='PASS'))
    gen=t.Generator(device='cpu').manual_seed(w.SEED)
    samples=[t.zeros(1,2,2,64,128),t.ones(1,2,2,64,128),t.randn(1,2,2,128,64,generator=gen).transpose(-1,-2)]
    for a in samples:
        x=a.to(device=device,dtype=t.bfloat16);g=encode_group(x,0)
        y=restore_group(g,x.dtype).reshape(x.shape)
        w.require(t.equal(y,w.nsn_qdq(x,3)),'Group reconstructed values')
    checks.append(dict(test='group_codec_values',device=device,status='PASS'))
    old=(w.LAYERS,w.HEADS)
    try:
        w.LAYERS,w.HEADS=2,2;r=2
        fake=w.make_adapter_class(FixtureBase)(r,3,'DECODE_ONLINE');real=make_packed_cache(FixtureBase)(r,3)
        x=t.randn(4,2,1023,128,generator=gen).to(device=device,dtype=t.bfloat16)
        for end in [768]+list(range(769,1024)):
            start=0 if end==768 else end-1
            for s in range(4):
                k=x[s:s+1,:,start:end,:];v=(k.float()*.75).to(t.bfloat16)
                pos={'cache_position':t.arange(start,end,device=device)}
                fk,fv=fake.update(k,v,s,pos);rk,rv=real.update(k,v,s,pos)
                # Check every returned attention input, not merely final stored values.
                w.require(t.equal(fk,rk) and t.equal(fv,rv),'Returned attention values differ')
            if end==768:fake.post_prefill();real.post_prefill()
            else:fake.post_decode();real.post_decode()
            if end in (768,831,832,895,896,959,960,1023):
                for s in range(4):
                    for a,b in zip(full_pair(fake,s),full_pair(real,s)):
                        w.require(t.equal(a,b),'Stored values/tail differ')
        inv=validate(real,1023,256,3);validate(fake,1023,256,3)
        w.require(inv['unique_storage_bytes']<tensor_inventory(fake)['unique_storage_bytes'],'Fixture resident reduction')
        checks.append(dict(test='multi_slot_lifecycle',device=device,status='PASS',**inv))
        real.clear();w.require(not real.packed and not real.key_cache,'Clear retains storage')
    finally:w.LAYERS,w.HEADS=old
    # All real slot coordinates, without loading the model.
    many=make_packed_cache(FixtureBase)(8,None)
    for slot in range(w.LAYERS*8):
        z=t.full((1,w.HEADS,1,128),float(slot),device=device,dtype=t.bfloat16)
        many.update(z,z.clone(),slot,{'cache_position':t.tensor([0],device=device)})
    w.require(set(many.calls)==set(range(w.LAYERS*8)) and all(many.get_seq_length(i)==1 for i in many.calls),'Full slot coordinates')
    checks.append(dict(test='full_slot_coordinates',device=device,status='PASS',slots=w.LAYERS*8))
    return checks


def fixtures(device):
    previous=w.torch.get_num_threads()
    try:
        if device=='cpu':w.torch.set_num_threads(1)
        return _fixtures(device)
    finally:
        if device=='cpu':w.torch.set_num_threads(previous)
