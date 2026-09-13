"""Independent scalar audit of held-out PIV comparisons, never importing producers."""
from pathlib import Path
import bisect,hashlib,json,math
import h5py
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'image_only_cm'

def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(a,b,tol=1e-12):
    a=np.asarray(a);b=np.asarray(b)
    assert a.shape==b.shape
    assert np.array_equal(np.isfinite(a),np.isfinite(b))
    finite=np.isfinite(a)&np.isfinite(b)
    e=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
    assert e<=tol,(e,tol)
    return e

def source_ok(q,mask,surface):
    x,y=q
    if not(np.isfinite(x) and np.isfinite(y) and 0<=x<=mask.shape[1]-1 and 0<=y<=mask.shape[0]-1):return False
    i=int(math.floor(x));j=int(math.floor(y));tx=x-i;ty=y-j
    a=0.
    for di,dj,w in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]:
        if w>0:a+=w*bool(mask[j+dj,i+di])
    si=min(i+1,len(surface)-1)
    s=(1-tx)*surface[i]+tx*surface[si]
    return bool(a>.99 and y>=s)

def corners(q,x,y):
    qx,qy=q
    if not (np.isfinite(qx) and np.isfinite(qy) and x[0]<=qx<=x[-1] and y[0]<=qy<=y[-1]):return []
    i=min(bisect.bisect_right(x,qx)-1,len(x)-2);j=min(bisect.bisect_right(y,qy)-1,len(y)-2)
    tx=(qx-x[i])/(x[i+1]-x[i]);ty=(qy-y[j])/(y[j+1]-y[j])
    return [(j+dj,i+di,w) for di,dj,w in [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)] if w>0]

def run(pair):
    folder=BASE/str(pair);r=read(folder/'results.npz');a=read(folder/'inputs.npz')
    z=read(HERE/f'pair_{pair}_integral_comparison.npz');g=read(HERE/f'pair_{pair}_gradient_comparison.npz')
    paths=[folder/(k+'.npz') for k in ['inputs','results','main','affine','large_window','margin14','reverse','ptv_tracks','integration_profile','plot_samples']]
    before={str(p):sha(p) for p in paths}
    prior=json.loads((HERE/f'pair_{pair}_integral_comparison.json').read_text())
    for p,h in prior['immutable_image_only_hashes_before'].items():assert sha(p)==h
    with h5py.File(f'data/ExpLCL_1_03_{pair}_PIV.mat','r') as f:
        c=f['compVel'];x=(c['xPIV'][:].ravel()-1).tolist();y=(c['zPIV'][:].ravel()-1).tolist()
        dx=c['delta_x'][:].T;dy=-c['delta_z'][:].T
    d=np.stack([dx,dy],axis=2);mask=a['availability_a'];surf=a['surface_a'];DX=float(a['DX']);DT=float(a['DT'])
    node_cache={};grad_cache={}
    def node(j,i):
        if (j,i) not in node_cache:node_cache[j,i]=bool(np.isfinite(d[j,i]).all() and source_ok((x[i],y[j]),mask,surf))
        return node_cache[j,i]
    def piv_sample(q):
        cc=corners(q,x,y)
        if not cc or not source_ok(q,mask,surf):return np.full(2,np.nan),False
        out=np.zeros(2)
        for j,i,w in cc:
            if not node(j,i):return np.full(2,np.nan),False
            out+=w*d[j,i]
        return out,True
    def gradient_node(j,i,c):
        if (j,i,c) not in grad_cache:
            loc=[(j,i+k) if c==0 else (j+k,i) for k in range(-2,3)]
            ok=all(0<=jj<len(y) and 0<=ii<len(x) and node(jj,ii) for jj,ii in loc)
            value=(d[loc[-1]][c]-d[loc[0]][c])/16 if ok else np.nan
            grad_cache[j,i,c]=(value,ok)
        return grad_cache[j,i,c]
    def gradient_sample(q,c):
        cc=corners(q,x,y)
        if not cc or not source_ok(q,mask,surf):return np.nan,False
        out=0.
        for j,i,w in cc:
            v,ok=gradient_node(j,i,c)
            if not ok:return np.nan,False
            out+=w*v
        return out,True
    # Independently sample all x values on selected depth rows, including sparse
    # first coverage, band boundary, and the four requested representative depths.
    h=z['depth_m'];rows=np.unique([int(np.argmin(abs(h-hh))) for hh in [.0007,.0009,.001,20*DX,.002,.005,.01]])
    nv=[];ng=[];oldv=[];oldg=[]
    for j in rows:
        for i,qx in enumerate(z['horizontal_sample_x_px']):
            v,ok=piv_sample((qx,float(z['sample_query_y_px'][j,i])))
            nv.append(v);ng.append(ok);oldv.append(z['piv_sample_displacement_px'][j,i]);oldg.append(z['piv_sample_available'][j,i])
    sample_error=compare(nv,oldv);assert np.array_equal(ng,oldg)
    q_error=width_error=mean_error=rms_error=0.
    for j in range(len(h)):
        Qof=Qpiv=W=E2=0.;count=0
        for k,width in enumerate(z['horizontal_interval_width_m']):
            use=all(bool(z['of_sample_accepted'][j,i]) and np.isfinite(z['of_sample_u_m_per_s'][j,i]) and bool(z['piv_sample_available'][j,i]) and np.isfinite(z['piv_sample_u_m_per_s'][j,i]) for i in [2*k,2*k+1,2*k+2])
            assert use==z['matched_interval_mask'][j,k]
            if use:
                oo=z['of_sample_u_m_per_s'][j,2*k:2*k+3];pp=z['piv_sample_u_m_per_s'][j,2*k:2*k+3]
                Qof+=width*(oo[0]+4*oo[1]+oo[2])/6
                Qpiv+=width*(pp[0]+4*pp[1]+pp[2])/6
                de=(oo-pp)**2;E2+=width*(de[0]+4*de[1]+de[2])/6
                W+=width;count+=1
        if not count:Qof=Qpiv=np.nan
        q_error=max(q_error,compare([Qof,Qpiv],[z['of_matched_integral_m2_per_s'][j],z['piv_matched_integral_m2_per_s'][j]]))
        width_error=max(width_error,compare(W,z['matched_width_m'][j]))
        mean_error=max(mean_error,compare([Qof/W,Qpiv/W] if W else [np.nan,np.nan],[z['of_matched_mean_u_m_per_s'][j],z['piv_matched_mean_u_m_per_s'][j]]))
        rms_error=max(rms_error,compare(math.sqrt(E2/W) if W else np.nan,z['matched_rms_difference_m_per_s'][j]))
    band=(h>=20*DX-1e-15)&(h<=.01+1e-15)
    independent_common=np.array([all(z['matched_interval_mask'][j,k] for j in np.flatnonzero(band)) for k in range(len(z['horizontal_interval_width_m']))])
    assert np.array_equal(independent_common,z['common_interval_mask'])
    assert not independent_common.any()
    assert np.isnan(z['of_common_integral_m2_per_s']).all() and np.isnan(z['piv_common_integral_m2_per_s']).all()
    assert not z['piv_on_of_original_common_available'].any()
    # Independent +/-8 Cartesian gradients: native source has +/-4 intermediates.
    rng=np.random.default_rng(812)
    selected=np.unique(np.r_[rng.choice(len(g['query']),min(2000,len(g['query'])),replace=False),np.flatnonzero(g['common_gradient_valid'].any(axis=1))[::50]])
    gv=[];gok=[]
    for q in g['query'][selected]:
        vv=[];kk=[]
        for c in range(2):v,ok=gradient_sample(q,c);vv.append(v);kk.append(ok)
        gv.append(vv);gok.append(kk)
    grad_error=compare(gv,g['piv_gradient_per_pair'][selected]);assert np.array_equal(gok,g['piv_gradient_valid'][selected])
    analytic=np.stack([read(folder/'plot_samples.npz')['gradient'][:,0,0],read(folder/'plot_samples.npz')['gradient'][:,1,1]],axis=1)
    compare(analytic,g['of_gradient_per_pair'])
    lookup={tuple(q):i for i,q in enumerate(r['query'])};fd=np.full((len(r['query']),2),np.nan);fg=np.zeros(fd.shape,bool)
    pg=np.full(fd.shape,np.nan);pvalid=np.zeros(fd.shape,bool)
    for i,q in enumerate(r['query']):
        for c in range(2):
            offset=np.zeros(2);offset[c]=8
            im=lookup.get(tuple(q-offset));ip=lookup.get(tuple(q+offset))
            if im is not None and ip is not None and all(r['accepted'][k] and np.isfinite(r['disp'][k]).all() for k in [im,i,ip]):
                fd[i,c]=(r['disp'][ip,c]-r['disp'][im,c])/16;fg[i,c]=True
            pg[i,c],pvalid[i,c]=gradient_sample(q,c)
    of_fd_error=compare(fd,g['matched_of_gradient_per_pair']);piv_fd_error=compare(pg,g['matched_piv_gradient_per_pair'])
    assert np.array_equal(fg,g['matched_of_valid']);assert np.array_equal(pvalid,g['matched_piv_valid']);assert np.array_equal(fg&pvalid,g['matched_valid'])
    compare(np.where(g['matched_valid'],(fd-pg)/DT,np.nan),g['matched_difference_assumed_per_s'])
    compare(np.where(g['common_gradient_valid'],(g['of_gradient_per_pair']-g['piv_gradient_per_pair'])/DT,np.nan),g['gradient_difference_assumed_per_s'])
    after={str(p):sha(p) for p in paths};assert before==after
    representative=[]
    for hmm in [1,2,5,10]:
        j=int(np.argmin(abs(h*1000-hmm)))
        representative.append(dict(depth_mm=float(h[j]*1000),coverage_percent=float(z['matched_coverage_fraction'][j]*100),Q_OF_cm2_s=float(z['of_matched_integral_m2_per_s'][j]*1e4),Q_PIV_cm2_s=float(z['piv_matched_integral_m2_per_s'][j]*1e4),mean_OF_cm_s=float(z['of_matched_mean_u_m_per_s'][j]*100),mean_PIV_cm_s=float(z['piv_matched_mean_u_m_per_s'][j]*100),rms_difference_cm_s=float(z['matched_rms_difference_m_per_s'][j]*100),matched_fraction_retained_if_finite_dcor=float(z['finite_dcor_retained_fraction_of_matched_width'][j])))
    out=dict(pair=pair,passed=True,comparison_code_imported=False,native_loader_imported=False,image_only_sources_unchanged=True,
        hashes=before,independent_PIV_sample_queries=len(nv),native_sample_displacement_max_error_px=sample_error,native_sample_flags_exact=True,
        all_depth_interval_masks_exact=True,loop_Simpson_integral_max_error_m2_s=q_error,loop_covered_width_max_error_m=width_error,loop_mean_max_error_m_s=mean_error,loop_RMS_max_error_m_s=rms_error,
        joint_fixed_support_empty_verified=True,original_OF_common_not_fully_PIV_supported_verified=True,
        independent_rectified_gradient_queries=len(selected),native_gradient_5node_and_interpolation_max_error_per_pair=grad_error,native_gradient_flags_exact=True,
        matched_report_grid_queries=len(r['query']),OF_matched_FD_max_error_per_pair=of_fd_error,PIV_matched_FD_max_error_per_pair=piv_fd_error,all_matched_FD_flags_exact=True,
        vertical_sign_and_inverse_DT_export_verified=True,representative_depths=representative)
    (HERE/f'pair_{pair}_independent_comparison_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='hashes'},indent=2),flush=True)

if __name__=='__main__':
    for pair in [80,100]:run(pair)
