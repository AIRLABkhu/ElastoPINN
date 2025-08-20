import gc, torch

def collect():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
