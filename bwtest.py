#!/usr/bin/env python
"""Host->Device PCIe transfer bandwidth = the ceiling for any streaming loader."""
import torch, time

def bench(pin):
    x = torch.empty(256*1024*1024, dtype=torch.float32, pin_memory=pin)  # 1 GB
    g = torch.empty_like(x, device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(8):
        g.copy_(x, non_blocking=pin)
    torch.cuda.synchronize()
    dt = time.time() - t
    return 8.0 / dt

for pin in (False, True):
    print(f"H2D {'pinned' if pin else 'pageable'}: {bench(pin):.1f} GB/s")
