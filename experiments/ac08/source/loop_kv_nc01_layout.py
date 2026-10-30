"""Small fixture transformation. The SAME functions run on GPU and CPU test doubles."""
def old_view(original):
    return original.transpose(-1,-2).contiguous().transpose(-1,-2)

def noncontiguous_view(torch,original):
    backing=torch.zeros((*original.shape[:-1],original.shape[-1]*2),device=original.device,dtype=original.dtype)
    view=backing[...,::2]
    view.copy_(original)
    return view,backing

def layout(x):
    return dict(shape=list(x.shape),stride=list(x.stride()),contiguous=bool(x.is_contiguous()),dtype=str(x.dtype),device=str(x.device),storage_offset=int(x.storage_offset()))
