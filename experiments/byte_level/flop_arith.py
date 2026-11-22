"""Analytic params and FLOPs for FusionDance-Byte designs (computed, no model built, no GPU).
Mirrors gogeta/model.py flops_per_token: fwd = 2*matmul_params + attention/GDN terms; train = 3*fwd.
Run: python experiments/byte_level/flop_arith.py   (pure python)"""
def swiglu(d, h): return 2*h*d + h*d
def attn_blk(d, h): return 4*d*d + swiglu(d, h) + 2*d
def gdn_blk(d, H, h, conv=4):
    hd = d // H
    return 5*d*d + 2*d*H + 3*d*conv + 2*H + hd + swiglu(d, h) + 2*d
D, H, MLP, V = 384, 6, 1024, 8192
E1 = 3*attn_blk(D, MLP) + 3*gdn_blk(D, H, MLP)          # SWA,GDN,GDN,SWA,GDN,attn
print(f"E1 body params          {E1:,}  (E1 total {E1 + V*D + D:,}; embedding {V*D:,} = {V*D/(E1+V*D+D):.1%})")
def fwd_body(T, win, d=D, Hn=H):
    mm = 2*(E1)  # matmul params ~ all body params (norm/conv/gate-vectors are <1%)
    att = 2*4*d*min(win, T/2) + 4*d*(T/2) + 3*4*Hn*(d//Hn)**2
    return mm, att
# BPE E1 (check against 91 MFLOP/token in notes)
mm, att = fwd_body(1024, 256); head = 2*D*V
e1 = 3*(mm + att + head); print(f"E1 BPE T=1024: train FLOP/token {e1/1e6:.1f} M (registry: 91M)")
TOK_BYTES = 3.7347  # bytes/token on 4,000 val docs (seq_stats.json)
E1_TOKENS = 2e8; E1_FLOPS = e1*E1_TOKENS
# Byte E1: vocab 256 tied, same body, T=4096
for win in (256, 1024):
    mm, att = fwd_body(4096, win); head = 2*D*256
    b = 3*(mm + att + head)
    print(f"Byte-E1 T=4096 win={win}: params {E1 + 256*D + D:,}; train FLOP/byte {b/1e6:.1f}M; per BPE-token-equiv {b*TOK_BYTES/1e6:.0f}M = {b*TOK_BYTES/e1:.2f}x E1;"
          f" matched-FLOPs bytes {E1_FLOPS/b/1e6:.0f}M ({E1_FLOPS/b/TOK_BYTES/1e6:.0f}M tok-equiv, {E1_FLOPS/b/TOK_BYTES/E1_TOKENS:.0%}); matched-bytes FLOPs {E1_TOKENS*TOK_BYTES*b:.3e} vs E1 {E1_FLOPS:.3e}")
# BLT-lite: P=4 fixed patches, local d=192 (3 heads, mlp 512): enc 1 GDN, dec GDN+SWA(32)
dl, Hl, ml, P = 192, 3, 512, 4
enc = gdn_blk(dl, Hl, ml); dec = gdn_blk(dl, Hl, ml) + attn_blk(dl, ml)
pin = P*dl*D; up = D*dl; emb = 256*dl
loc = emb + enc + pin + up + dec
print(f"BLT-lite local params: emb {emb:,} enc {enc:,} patch-in {pin:,} up {up:,} dec {dec:,} = {loc:,}; total {loc+E1+D:,} ({(loc+E1+D)/(E1+V*D+D)-1:+.1%} vs E1)")
core_fwd_patch = 2*E1 + fwd_body(1024, 256)[1]       # core on 1024 patches
loc_fwd_byte = 2*enc + 2*dec + 4*dl*32 + 2*dl*256 + 4*Hl*(dl//Hl)**2*2
loc_fwd_byte += (2*pin + 2*up)/P
fwd_byte = core_fwd_patch/P + loc_fwd_byte
bl = 3*fwd_byte
print(f"BLT-lite: core {core_fwd_patch/P/1e6:.2f}M + local {loc_fwd_byte/1e6:.2f}M fwd/byte; train {bl/1e6:.1f}M/byte = {bl*TOK_BYTES/1e6:.0f}M per tok-equiv = {bl*TOK_BYTES/e1:.2f}x E1")
# layer-equivalents per patch for the bandwidth model (positions x layers x width/384)
print("BLT-lite bandwidth proxy (layer-equiv per patch, width-weighted):", f"core 6 + local {3*P*dl/D:.1f} = {6+3*P*dl/D:.1f} vs E1 6")
