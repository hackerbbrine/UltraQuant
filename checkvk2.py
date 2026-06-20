import numpy as np, qlab
from qlab import load_WH, rht_rows, q_e8_fast, irht_rows, deq_iq, e8_decode, _entropy_bpw
for nm in ['blk.16.attn_v.weight', 'blk.16.attn_k.weight']:
    W, H = load_WH(nm); sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
    Arot = rht_rows(W*sH)
    Wq = deq_iq(r'models\8b\iq2xxs.gguf', nm)[:W.shape[0], :W.shape[1]]
    iq = float(np.sum(((W-Wq)*sH)**2)/np.sum((W*sH)**2))
    print(f'{nm}: IQ2={iq:.4f}')
    for step in (0.30, 0.22, 0.16, 0.11):
        rec = irht_rows(q_e8_fast(Arot, fixed_step=step), W.shape[1]); Wh = rec / sH
        mine = float(np.sum(((W-Wh)*sH)**2)/np.sum((W*sH)**2))
        # rough effective bpw via entropy on full (saturates >2.36 but gives a floor)
        g = Arot.ravel()[:Arot.size//256*256].reshape(-1, 256)
        rms = np.sqrt((g**2).mean(1, keepdims=True))+1e-12; gn = (g/rms).reshape(-1, 8)
        bpw = _entropy_bpw(e8_decode(gn[:400000]/step))
        print(f'    step={step:.2f}: WNMSE={mine:.4f} (entropy-bpw>={bpw:.2f})  {"WIN" if mine<iq else "lose"}')
