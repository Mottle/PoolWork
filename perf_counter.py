from time import perf_counter

def measure_time(fn, cuda_sync=True):
    def wrapper(*args, **kwargs):
        start = get_time_sync()
        result = fn(*args, **kwargs)
        end = get_time_sync()
        return result, end - start
    return wrapper

def cuda_sync():
    import torch
    torch.cuda.synchronize()

def get_time_sync(use_cuda_sync=True):
    if use_cuda_sync:
        cuda_sync()
    return perf_counter()