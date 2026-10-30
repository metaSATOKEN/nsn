"""Small evidence helpers; actual model loading remains in the copied NB01 worker."""
import base64,json
from collections import Counter
import loop_kv_nb06_contract as c
import loop_kv_nb06_primitives as w
import loop_kv_nb06_codec as codec

def bits(t):
    return t.detach().contiguous().view(w.torch.uint16).cpu().numpy().copy()

def initialize(t,np):w.initialize(t,np)

def prefix_source(cache):
    return {f'{kv}_{s:02d}':bits(data[s][:,:,:64,:]) for s in range(44) for kv,data in [('K',cache.key_cache),('V',cache.value_cache)]}

def snapshot(cache,length,keep_raw=False,expected_device='cuda'):
    t=w.torch;np=w.np;states={};raw={}
    c.require(len(cache.key_cache)==len(cache.value_cache)==44,'Cache slot count')
    for s in range(44):
        pair=codec.full_pair(cache,s)
        for kv,x in zip(('K','V'),pair):
            c.require(list(x.shape)==c.expected_shape(length) and x.dtype==t.bfloat16 and x.device.type==expected_device,'Restored cache structure')
            a=bits(x);finite=bool(np.isfinite(c.bf32(a)).all());c.require(finite,'Nonfinite restored KV')
            key=f'{kv}_{s:02d}'
            states[key]={'shape':list(a.shape),'dtype':'bfloat16','finite':True,'sha256':c.tensor_hash(a),
                        'prefix_sha256':c.tensor_hash(a[:,:,:64,:])}
            if keep_raw:raw[key]=a
    return states,raw

def packed_arrays(cache):
    arrays={}
    for s in range(44):
        for kv,data in [('K',cache.key_cache),('V',cache.value_cache)]:
            key=f'{kv}_{s:02d}';g=cache.packed[(s,kv)]
            for field in ['codes','headers']:arrays[key+'_'+field]=g[field].detach().cpu().numpy().copy()
            for i,x in enumerate(g['metas']):arrays[key+f'_meta{i}']=x.detach().cpu().numpy().copy()
            arrays[key+'_tail']=bits(data[s])
    return arrays

def persistent_inventory(cache):
    # Check every persistent tensor attribute, not just the expected key/value containers.
    expected=codec.tensor_inventory(cache);seen_objects=set();storage={};components=Counter();tensor_count=0
    def walk(obj):
        nonlocal tensor_count
        if isinstance(obj,w.torch.Tensor):
            tensor_count+=1;st=obj.untyped_storage();storage[(str(obj.device),st.data_ptr())]=st.nbytes()
            return
        if isinstance(obj,(dict,list,tuple)):
            if id(obj) in seen_objects:return
            seen_objects.add(id(obj))
            for x in (obj.values() if isinstance(obj,dict) else obj):walk(x)
    walk(vars(cache))
    return {**expected,'all_attribute_tensor_count':tensor_count,'all_attribute_unique_storage_bytes':sum(storage.values()),
        'unexpected_persistent_tensor':tensor_count!=expected['tensor_count'] or sum(storage.values())!=expected['unique_storage_bytes'],
        'slot_lengths':[cache.get_seq_length(i) for i in range(44)],
        'compressed_by_slot':[getattr(cache,'compressed',{}).get(i,0) for i in range(44)],
        'encoded_blocks_by_slot':[getattr(cache,'encoded_blocks',{}).get(i,0) for i in range(44)]}

def codec_fixtures(root,device='cuda'):
    """Finite mismatches are returned; nonfinite/unsafe structures raise. No model forward."""
    t=w.torch;np=w.np;checks=[]
    for n in [1,2,3,7,8,9,63,64,65,8192]:
        a=((np.arange(n)*5+3)%8).astype(np.uint8);q=t.from_numpy(a).to(device);p=codec.pack3(q)
        checks.append({'id':f'pack_{n}','pass':p.cpu().numpy().tobytes()==w.pack_codes(a,3) and bool(t.equal(q,codec.unpack3(p,q.shape)))})
    gen=t.Generator(device='cpu').manual_seed(c.MODEL['seed'])
    samples=[('zero',t.zeros(1,2,1,64,128)),('constant',t.ones(1,2,1,64,128)),
             ('strided',t.randn(1,2,1,128,64,generator=gen).transpose(-1,-2))]
    path=c.run_root(root)/'nb06_codec_fixture.json';c.require(c.file_digest(path)==c.PLAN['codec_fixture_sha256'],'Codec fixture bytes')
    fixture=c.read(path)
    for name,entry in fixture['blocks'].items():
        raw=base64.b64decode(entry['data_base64'],validate=True);c.require(c.sha(raw)==entry['sha256'],'Real fixture bits')
        a=np.frombuffer(raw,dtype='<u2').copy().reshape(entry['shape'])
        samples.append((name,t.from_numpy(a).view(t.bfloat16).reshape(1,8,1,64,128)))
    for name,a in samples:
        x=a.to(device=device,dtype=t.bfloat16);q,meta=w.nsn_encode(x,3);g=codec.encode_group(x,0)
        y=codec.restore_group(g,t.bfloat16).reshape(x.shape);ref=w.nsn_decode(q,meta,t.bfloat16)
        same=bool(t.equal(y,ref)) and bool(t.equal(q,codec.unpack3(g['codes'],q.shape))) and all(bool(t.equal(a,b)) for a,b in zip(meta,g['metas']))
        checks.append({'id':'group_'+name,'pass':same,'shape':list(x.shape),'noncontiguous_input':not x.is_contiguous()})
    return {'device':device,'checks':checks,'pass':all(x['pass'] for x in checks),'model_forward_calls':0}
