import numpy as np, qlab
from qlab import load_WH, rht_rows, q_e8_fast, e8_decode, _entropy_bpw, deq_iq
# reproduce the build's GLOBAL step: calibrated on blk.0.ffn_down (first tensor)
W, H = load_WH('blk.0.ffn_down.weight'); sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
_ = q_e8_fast(rht_rows(W*sH), 2.0, calib_key='global'); step = qlab._E8_SCALE['global']
print('global step =', round(step, 3))
names = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
for short in names:
    nm = f'blk.16.{short}.weight'
    W, H = load_WH(nm); sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
    Arot = rht_rows(W*sH)
    g = Arot.ravel()[:Arot.size//256*256].reshape(-1, 256)
    rms = np.sqrt((g**2).mean(1, keepdims=True)) + 1e-12
    gn = (g/rms).reshape(-1, 8); n = min(800000, gn.shape[0])
    bpw = _entropy_bpw(e8_decode(gn[:n]/step))
    # weighted-NMSE of my method vs IQ2 on this tensor (under global step)
    rec = qlab.irht_rows(q_e8_fast(Arot, 2.0, calib_key='global'), W.shape[1])
    Wh = rec / sH
    mine = float(np.sum(((W-Wh)*sH)**2)/np.sum((W*sH)**2))
    Wq = deq_iq(r'models\8b\iq2xxs.gguf', nm)[:W.shape[0], :W.shape[1]]
    iq = float(np.sum(((W-Wq)*sH)**2)/np.sum((W*sH)**2))
    print(f'  {short:12s} bpw={bpw:.2f}  myWNMSE={mine:.4f}  IQ2={iq:.4f}  {"WIN" if mine<iq else "lose"}')
