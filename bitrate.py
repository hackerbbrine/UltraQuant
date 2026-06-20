import numpy as np, qlab
from qlab import load_WH, rht_rows, q_e8_fast, e8_decode, _entropy_bpw
# estimate avg bpw of the e8mix model: bulk @ entropy-2bpw, V/K @ step0.22 (~theoretical R-D bpw)
# calibrate global bulk step
W,H=load_WH('blk.0.ffn_down.weight'); sH=np.sqrt(np.maximum(H,H.max()*1e-3))[None,:]
_=q_e8_fast(rht_rows(W*sH),2.0,calib_key='global'); step=qlab._E8_SCALE['global']

# param counts (Llama-3.1-8B, 32 layers)
P = {'attn_q':4096*4096,'attn_k':1024*4096,'attn_v':1024*4096,'attn_output':4096*4096,
     'ffn_gate':14336*4096,'ffn_up':14336*4096,'ffn_down':4096*14336}
bulk = ['attn_q','attn_output','ffn_gate','ffn_up','ffn_down']; vk=['attn_k','attn_v']

# measure bulk entropy-bpw (avg over a sample layer) and V/K R-D bpw
def wnmse_to_bpw(wn):  # Gaussian R-D: R = 0.5 log2(1/D) per dim (D = normalized distortion)
    return max(0.0, 0.5*np.log2(1.0/max(wn,1e-9)))

tot_bits=0.0; tot_p=0.0
for short in P:
    nm=f'blk.16.{short}.weight'; W,H=load_WH(nm); sH=np.sqrt(np.maximum(H,H.max()*1e-3))[None,:]; Arot=rht_rows(W*sH)
    if short in vk:
        rec=qlab.irht_rows(q_e8_fast(Arot,fixed_step=0.22),W.shape[1]); Wh=rec/sH
        wn=float(np.sum(((W-Wh)*sH)**2)/np.sum((W*sH)**2)); bpw=wnmse_to_bpw(wn)
    else:
        g=Arot.ravel()[:Arot.size//256*256].reshape(-1,256); rms=np.sqrt((g**2).mean(1,keepdims=True))+1e-12
        gn=(g/rms).reshape(-1,8); bpw=_entropy_bpw(e8_decode(gn[:600000]/step))
    p=P[short]*32; tot_bits+=bpw*p; tot_p+=p
    print(f'  {short:12s} bpw~{bpw:.2f}  ({100*p/ (sum(P.values())*32):.1f}% of linear params)')
print(f'\nE8MIX avg linear bpw ~= {tot_bits/tot_p:.3f}   (IQ2_XXS linear ~= 2.06)')
