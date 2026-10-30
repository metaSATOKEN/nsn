"""CPU float64 scorer. Model output and original experiments are unchanged."""
def nll(logits,target):
    import torch
    if logits.device.type!='cpu':raise ValueError('PD01 scorer requires saved CPU logits')
    if logits.ndim!=1 or not 0<=target<len(logits):raise ValueError('Score shape/target')
    if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite logits')
    return float(-torch.log_softmax(logits.to(torch.float64),-1)[target])
