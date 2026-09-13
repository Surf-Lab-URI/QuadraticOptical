"""Compare frozen image-only predictions with held-out native supplied PIV."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json, hashlib
import numpy as np
import h5py
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/piv_comparison_top_cm'
OUT=ROOT/'outputs/piv_comparison_top_cm'
OF='#d54b28'
PIV='#2363b1'
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white','svg.fonttype':'none'})

def read(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(a,b):
    d=a-b
    if not len(d):return {'count':0}
    return dict(count=len(d),bias_cm_per_s=np.mean(d,axis=0).tolist(),
        component_rmse_cm_per_s=np.sqrt(np.mean(d*d,axis=0)).tolist(),
        median_vector_difference_cm_per_s=float(np.median(np.linalg.norm(d,axis=1))),
        vector_rmse_cm_per_s=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))))

def prepare(pair):
    folder=ROOT/'work/image_only_cm'/str(pair)
    r=read(folder/'results.npz')
    with np.load(folder/'inputs.npz') as f:
        raw=f['rawA'];surface=f['surface_a'];avail=f['availability_a']
    path=Path('data/ExpLCL_1_03_%s_PIV.mat'%pair)
    with h5py.File(path,'r') as f:
        x=f['compVel/xPIV'][...].ravel()-1;y=f['compVel/zPIV'][...].ravel()-1
        px=f['compVel/delta_x'][...].T;py=-f['compVel/delta_z'][...].T
        cor=f['compVel/dcor'][...].T
        assert float(f['compVel/DX'][0,0])==float(r['DX'])
        assert float(f['compVel/DT'][0,0])==float(r['DT'])
    q=r['query'];ix=np.searchsorted(x,q[:,0]);iy=np.searchsorted(y,q[:,1])
    inside=(ix<len(x))&(iy<len(y))
    assert np.array_equal(x[ix[inside]],q[inside,0]) and np.array_equal(y[iy[inside]],q[inside,1])
    p=np.full((len(q),2),np.nan);pcor=np.full(len(q),np.nan)
    p[inside]=np.c_[px[iy[inside],ix[inside]],py[iy[inside],ix[inside]]]
    pcor[inside]=cor[iy[inside],ix[inside]]
    valid=np.isfinite(p).all(axis=1)&avail[q[:,1].astype(int),q[:,0].astype(int)]&(r['depth']>=0)
    joint=valid&r['accepted'];factor=float(r['DX'])/float(r['DT'])*100
    ov=r['disp']*factor;pv=p*factor;ov[:,1]*=-1;pv[:,1]*=-1
    bands=[]
    for lo,hi in zip(np.arange(10.),np.arange(1.,11.)):
        use=joint&(r['depth']*float(r['DX'])*1000>=lo)&(r['depth']*float(r['DX'])*1000<hi+1e-12)
        bands.append(dict(depth_mm=[lo,hi],**stats(ov[use],pv[use])))
    metadata=dict(pair=pair,predictions_unchanged=True,piv_used_for_prediction=False,
        result_sha256=sha(folder/'results.npz'),native_mat_sha256=sha(path),
        native_fields=['compVel/xPIV','compVel/zPIV','compVel/delta_x','compVel/delta_z','compVel/dcor','compVel/DX','compVel/DT'],
        coordinate_convention='MAT axes minus 1; transpose native arrays; image dy=-delta_z; w=delta_z*DX/DT',
        exact_native_node_lookup=True,interpolation_for_quiver=False,
        piv_gate='Both displacement components finite and source location retained in raw A. No correlation threshold or target-position gate.',
        tested_grid_points=len(q),accepted_image_only=int(r['accepted'].sum()),
        available_piv=int(valid.sum()),joint_count=int(joint.sum()),
        piv_finite_without_dcor=int((valid&~np.isfinite(pcor)).sum()),
        overall=stats(ov[joint],pv[joint]),by_depth_mm=bands,
        units_note='Assume DX metres/pixel and DT seconds; no explicit source unit labels.',
        comparison_note='Differences measure agreement with supplied PIV, not error against ground truth.')
    data=dict(pair=pair,query_xy_zero_based_px=q,depth_mm=r['depth']*float(r['DX'])*1000,
        x_cm=(q[:,0]+.5)*float(r['DX'])*100,
        z_up_cm=(np.median(surface)-q[:,1])*float(r['DX'])*100,
        image_only_accepted=r['accepted'],piv_available=valid,joint_available=joint,
        image_only_u_cm_per_s=np.where(r['accepted'],ov[:,0],np.nan),
        image_only_w_cm_per_s=np.where(r['accepted'],ov[:,1],np.nan),
        piv_u_cm_per_s=np.where(valid,pv[:,0],np.nan),piv_w_cm_per_s=np.where(valid,pv[:,1],np.nan),
        supplied_dcor=pcor,DX=r['DX'],DT=r['DT'])
    dest=OUT/('pair_'+str(pair));dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'velocity_comparison.npz',**data)
    savemat(str(dest/'velocity_comparison.mat'),data,long_field_names=True,do_compression=True)
    (dest/'velocity_comparison_metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return r,raw,surface,p,valid,metadata

def draw(ax,pair,items,gain=4,min_depth=0):
    r,raw,surface,p,valid,meta=items
    dx=float(r['DX'])*100;dt=float(r['DT']);sr=float(np.median(surface))
    q=r['query'];d=r['disp']
    # Identical, deterministic display thinning for both methods, including unmatched sites.
    thin=((q[:,0].astype(int)-7)%32==0)&((q[:,1].astype(int)-7)%16==0)&(r['depth']*dx>=min_depth)
    low=int(np.floor(surface.min()))-4;high=int(np.ceil(surface.max()+1/dx))+4
    crop=raw[low:high].copy();yy=np.arange(low,high)[:,None]
    crop[(yy<surface[None,:]+min_depth/dx)|(yy>surface[None,:]+1/dx)]=np.nan
    ax.imshow(crop,cmap='gray_r',vmin=0,vmax=180,alpha=.25,interpolation='nearest',aspect='equal',
        extent=[0,2048*dx,(sr-(high-.5))*dx,(sr-(low-.5))*dx])
    x=(np.arange(2048)+.5)*dx
    ax.plot(x,(sr-surface)*dx-min_depth,color='#087c87',lw=.9)
    ax.plot(x,(sr-surface)*dx-1,color='#b66d13',lw=.8)
    ax.plot(x,(sr-surface-12)*dx,color='.55',ls=':',lw=.7)
    endpoints=[]
    for dis,ok,col,width,alpha in [(p,valid,PIV,.0010,.78),(d,r['accepted'],OF,.00067,.95)]:
        use=ok&thin
        endpoints.append(np.c_[(q[use,0]+.5+gain*dis[use,0])*dx,(sr-q[use,1]-gain*dis[use,1])*dx])
        Q=ax.quiver((q[use,0]+.5)*dx,(sr-q[use,1])*dx,dis[use,0]*dx,-dis[use,1]*dx,
            color=col,alpha=alpha,angles='xy',scale_units='xy',scale=1/gain,width=width,
            headwidth=3.1,headlength=4.2,minlength=.2,pivot='tail')
    speed=5 if gain==4 else 1
    ax.quiverkey(Q,.86,1.16,speed*dt,'%g cm/s'%speed,coordinates='axes',labelpos='E',color=OF)
    endpoints=np.concatenate(endpoints)
    xlim=[min(0,float(endpoints[:,0].min())-.01),max(2048*dx+(.3 if gain>4 else .1),float(endpoints[:,0].max())+.01)]
    ylim=[min((sr-surface.max())*dx-1.03,float(endpoints[:,1].min())-.01),max((sr-surface.min())*dx-min_depth+(.07 if gain>4 else .03),float(endpoints[:,1].max())+.01)]
    ax.set_xlim(*xlim);ax.set_ylim(*ylim)
    ax.set_xlabel('Horizontal position x (cm)');ax.set_ylabel('Height z (cm)')
    ax.set_facecolor('#f6f7f8')
    band='top 1 cm' if min_depth==0 else '0.5–1 cm below local surface'
    ax.set_title('Pair %d · %s · arrows %g× pair displacement'%(pair,band,gain),loc='left',pad=22)
    return dict(image_only_shown=int((r['accepted']&thin).sum()),piv_shown=int((valid&thin).sum()),xlim_cm=xlim,ylim_cm=ylim,arrow_gain=gain,min_depth_cm=min_depth)

def save(fig,path):
    fig.savefig(str(path)+'.png',dpi=200,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight');plt.close(fig)

def main():
    data={p:prepare(p) for p in [80,100]};drawn={}
    legend=[Line2D([0],[0],color=OF,lw=2,label='Image-only optical flow'),Line2D([0],[0],color=PIV,lw=2,label='Supplied PIV')]
    for gain,depth,name,height in [(4,0,'quiver_overlay',5.2),(20,.5,'quiver_overlay_deeper_half',4.1)]:
        fig,axs=plt.subplots(2,1,figsize=(17,height))
        for pair,ax in zip([80,100],axs):drawn[str(pair)+'_'+name]=draw(ax,pair,data[pair],gain,depth)
        xbounds=[min(ax.get_xlim()[0] for ax in axs),max(ax.get_xlim()[1] for ax in axs)]
        ybounds=[min(ax.get_ylim()[0] for ax in axs),max(ax.get_ylim()[1] for ax in axs)]
        for pair,ax in zip([80,100],axs):
            ax.set_xlim(*xbounds);ax.set_ylim(*ybounds)
            drawn[str(pair)+'_'+name].update(xlim_cm=xbounds,ylim_cm=ybounds)
        axs[0].set_xlabel('')
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,1.015),ncol=2,frameon=False)
        fig.text(.12,.041,'Both methods share arrow origins and scale. Grid is thinned equally; missing estimates are omitted. PIV is used only for comparison.',fontsize=9)
        fig.text(.12,.015,'Cyan = local upper depth boundary; orange = 1 cm depth; dotted gray = retained-image boundary. Physical units and surface offset are assumed.',fontsize=8)
        fig.subplots_adjust(left=.085,right=.97,bottom=.18,top=.85,hspace=.69)
        save(fig,OUT/name)
    (OUT/'quiver_display_metadata.json').write_text(json.dumps(drawn,indent=2)+'\n')
    print(json.dumps({p:data[p][-1]['overall'] for p in data},indent=2))

if __name__=='__main__':main()
