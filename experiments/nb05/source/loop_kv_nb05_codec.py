"""PK01 codec function bodies copied verbatim; Nanbeige dimensions are bound by primitives."""
import struct
from collections import Counter
import loop_kv_nb05_primitives as w

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
