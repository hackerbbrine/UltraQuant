import numpy as np, qlab
from qlab import load_WH, rht_rows, q_e8_fast, irht_rows, deq_iq
for nm in ['blk.16.attn_v.weight', 'blk.16.attn_k.weight']:
    W, H = load_WH(nm); sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
    Arot = rht_rows(W*sH)
    Wq = deq_iq(r'models\8b\iq2xxs.gguf', nm)[:W.shape[0], :W.shape[1]]
    iq = float(np.sum(((W-Wq)*sH)**2)/np.sum((W*sH)**2))
    print(f'{nm}: IQ2={iq:.4f}')
    for bpw in (3.0, 4.0, 5.0):
        rec = irht_rows(q_e8_fast(Arot, bpw, calib_key=f'g{bpw}'), W.shape[1])
        Wh = rec / sH
        mine = float(np.sum(((W-Wh)*sH)**2)/np.sum((W*sH)**2))
        print(f'    my E8 @{bpw}bpw = {mine:.4f}  {"WIN" if mine<iq else "lose"}')
