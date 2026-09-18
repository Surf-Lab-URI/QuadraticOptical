"""Are the newly-admitted shallow vectors physically plausible?

Baseline-accepted vectors are the reference. For each relaxed profile we ask of
the vectors it ADDS (accepted now, withheld before):
  1. continuity  - does the new shallow profile continue the baseline profile,
                   or jump? Compared at the shallowest baseline-covered band.
  2. dispersion  - are they coherent (tight spread within a depth band) or
                   scatter, relative to baseline vectors at the same depth?
  3. IR consistency - does the near-surface profile extrapolate toward the
                   independently measured IR surface velocity, or overshoot?
  4. sign sanity - downstream-positive u, and |u| below a physical ceiling.
"""
import json, sys, glob
import numpy as np
from pathlib import Path
Q=Path('/media/surflab/LC_Working24/LC/FabMarcNovDec2014/data/Longitudinal/PIVdt10ms_IRlas1_8hz/ExpLCL_1_03/Results_Surflab/quadratic_optical')
DX=5.650454946380008e-05; DT=0.01
PAIRS=[80,100,115,125,140]
def load(root,n):
    d=root/('ExpLCL_1_03_%d'%n)
    if not (d/'results.npz').exists(): return None
    r=np.load(d/'results.npz',allow_pickle=True)
    ir=None
    p=d/'surface_ir_reference.json'
    if p.exists():
        j=json.loads(p.read_text())
        if j.get('available'): ir=j.get('velocity_m_per_s')
    return dict(u=r['disp'][:,0]*DX/DT, w=-r['disp'][:,1]*DX/DT,
                d=r['depth'], a=r['accepted'], q=r['query'], ir=ir)
BANDS=[(5,7),(7,9),(9,12),(12,16),(16,22),(22,30)]
for prof in ('mild','aggressive'):
    root=Q/('accept_'+prof)
    print('='*78); print('PROFILE:',prof); print('='*78)
    print('%-6s %8s %7s  %-28s  %-22s'%('pair','IR u','band','median u (m/s)  base / new','scatter IQR base / new'))
    for n in PAIRS:
        b=load(Q/'analysis_L2',n); m=load(root,n)
        if b is None or m is None: continue
        new = m['a'] & ~b['a']
        first=True
        for lo,hi in BANDS:
            sb=b['a']&(b['d']>=lo)&(b['d']<hi)
            sn=new&(m['d']>=lo)&(m['d']<hi)
            if sn.sum()<3 and sb.sum()<3: continue
            f=lambda v,s: np.median(v[s]) if s.sum() else np.nan
            iqr=lambda v,s: (np.percentile(v[s],75)-np.percentile(v[s],25)) if s.sum()>=4 else np.nan
            print('%-6s %8s %5d-%-2d  %8.4f / %-8s (+%4d)   %6.4f / %-8s'%(
                ('%d'%n) if first else '', ('%.4f'%b['ir']) if (first and b['ir']) else '',
                lo,hi, f(b['u'],sb), ('%.4f'%f(m['u'],sn)) if sn.sum() else '   -   ', sn.sum(),
                iqr(b['u'],sb), ('%.4f'%iqr(m['u'],sn)) if sn.sum()>=4 else '  -  '))
            first=False
        # physical sanity on everything newly admitted
        if new.any():
            un=m['u'][new]; wn=m['w'][new]
            print('%-6s   new n=%d  u range [%.3f, %.3f]  |w| p95 %.3f  negative-u %.0f%%  |u|>0.35 %.0f%%'%(
                '',new.sum(),un.min(),un.max(),np.percentile(np.abs(wn),95),
                100*(un<0).mean(),100*(np.abs(un)>.35).mean()))
        print()
