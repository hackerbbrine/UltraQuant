import numpy as np, time
from qlab import load_WH, rht_rows, deq_iq, q_e8_fast
from trellis import trellis_quant

E8 = {'blk.16.attn_q.weight': 0.0762, 'blk.16.ffn_gate.weight': 0.0781}
for nm in ['blk.16.attn_q.weight', 'blk.16.ffn_gate.weight']:
    W, H = load_WH(nm); sH = np.sqrt(np.maximum(H, H.max()*1e-3))[None, :]
    A = W*sH; Arot = rht_rows(A)
    den = np.sum(Arot**2)
    Wq = deq_iq(r'models\8b\iq2xxs.gguf', nm)[:W.shape[0], :W.shape[1]]
    iq = float(np.sum(((W-Wq)*sH)**2)/np.sum((W*sH)**2))
    print(f'\n{nm}: IQ2={iq:.4f}  E8(entropy2.0)={E8[nm]:.4f}', flush=True)
    for L, cr in [(10, 512), (12, 128)]:
        t = time.time()
        Ah = trellis_quant(Arot, L=L, R=2, group=256, chunk_rows=cr)
        wn = float(np.sum((Arot-Ah)**2)/den); dt = time.time()-t
        tag = 'WIN' if wn < min(iq, E8[nm]) else ('beats-IQ2' if wn < iq else 'lose')
        print(f'  trellis L={L} R=2 (fixed 2.06bpw): WNMSE={wn:.4f}  [{tag}]  ({dt:.0f}s)', flush=True)
