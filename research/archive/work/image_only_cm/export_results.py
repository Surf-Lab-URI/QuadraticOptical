"""User-facing figures and data for fresh image-only top-centimetre results."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,copy,argparse
import numpy as np
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from finalize_fields import ConservativeEvaluator

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/image_only_top_cm'
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white',
                     'savefig.facecolor':'white','svg.fonttype':'none'})

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def save(fig,path):
    fig.savefig(str(path)+'.png',dpi=200,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight')
    plt.close(fig)

def rectified(pair):
    folder=HERE/str(pair);ev=ConservativeEvaluator(folder)
    x=np.arange(7,2048,8.);h=np.arange(0,float(ev.max_depth_px)+1e-6,4.)
    if h[-1]<ev.max_depth_px:h=np.r_[h,ev.max_depth_px]
    xx,hh=np.meshgrid(x,h);yy=hh+np.interp(xx,np.arange(2048),ev.inputs['surface_a'])
    r=ev.evaluate_conservative(np.c_[xx.ravel(),yy.ravel()])
    r.update(x_axis_px=x,depth_axis_px=h,DX=np.array(ev.DX),DT=np.array(ev.DT))
    sample_path=folder/'plot_samples.npz'
    if sample_path.exists():
        previous=read(sample_path)
        if set(previous)!=set(r) or any(not np.allclose(previous[k],r[k],rtol=0,atol=0,equal_nan=True) for k in r):
            raise ValueError('Previously audited plot samples do not match fresh evaluation.')
        return previous
    np.savez_compressed(sample_path,**r)
    return r

def quiver(ax,r,raw,surface,pair,gain=4,detail=False,min_depth_cm=0):
    dx=float(r['DX'])*100;dt=float(r['DT']);sref=float(np.median(surface))
    q=r['query'];d=r['disp'];ok=r['accepted']
    # Keep actual Cartesian vector angles. The curved requested boundary is drawn.
    # Do not draw dy as the vertical velocity in flattened surface coordinates.
    if detail:
        thin=((q[:,0].astype(int)-7)%16==0)
    else:
        thin=((q[:,0].astype(int)-7)%32==0)&((q[:,1].astype(int)-7)%16==0)
    use=ok&thin&(r['depth']*dx>=min_depth_cm)
    low=int(np.floor(surface.min()))-4;high=int(np.ceil(surface.max()+.01/(dx/100)))+4
    crop=raw[low:high].copy()
    yy=np.arange(low,high)[:,None]
    crop[(yy<surface[None]+min_depth_cm/dx)|(yy>surface[None]+.01/(dx/100))]=np.nan
    ax.imshow(crop,cmap='gray_r',vmin=0,vmax=180,alpha=.42,interpolation='nearest',
        extent=[0,2048*dx,(sref-(high-.5))*dx,(sref-(low-.5))*dx],aspect='equal')
    x=(np.arange(2048)+.5)*dx
    ax.plot(x,(sref-surface)*dx-min_depth_cm,c='#087c87',lw=1,label='Inferred local surface' if min_depth_cm==0 else '0.5 cm below local surface')
    ax.plot(x,(sref-surface)*dx-1,c='#b66d13',lw=.8,label='1 cm below local surface')
    ax.plot(x,(sref-surface-12)*dx,c='#888888',lw=.7,ls=':',label='Approximate retained-image boundary')
    Q=ax.quiver((q[use,0]+.5)*dx,(sref-q[use,1])*dx,d[use,0]*dx,-d[use,1]*dx,
        color='#c43d31',angles='xy',scale_units='xy',scale=1/gain,
        width=.0007,headwidth=3.2,headlength=4.2,minlength=.25,pivot='tail')
    key_speed=5 if gain==4 else 1
    ax.quiverkey(Q,.82,1.15,key_speed*dt,'%g cm/s'%key_speed,coordinates='axes',labelpos='E',color='#c43d31')
    # A small shared display margin keeps magnified slow-motion arrow tips visible.
    ax.set_xlim(0,2048*dx+(.25 if gain>4 else .05))
    ax.axvline(2048*dx,color='.65',lw=.6,ls=':')
    ax.set_ylim((sref-surface.max())*dx-1.03,(sref-surface.min())*dx-min_depth_cm+(.065 if gain>4 else .025))
    ax.set_xlabel('Horizontal position x (cm)');ax.set_ylabel('Height z (cm)')
    band='top 1 cm' if min_depth_cm==0 else '0.5–1 cm local depth'
    ax.set_title('Pair %d · image-only velocity · %s · arrows %g× pair displacement'%(pair,band,gain),loc='left',pad=24)
    ax.set_facecolor('#f6f7f8');ax.tick_params(direction='out')
    return dict(drawn_vectors=int(use.sum()),accepted_vectors=int(ok.sum()),arrow_displacement_gain=gain,
                z_reference_image_y=sref,min_depth_cm=min_depth_cm,quiver_components='dx*DX and -dy*DX, actual Cartesian axes')

def profile(fig,axes,p,pair,color):
    h=p['depth_m']*100
    q=p['covered_segment_integral_assumed_m2_per_s']*1e4
    common=p['common_domain_integral_assumed_m2_per_s']*1e4
    axes[0].plot(q,h,color=color,lw=1.8,label='Accepted segments')
    axes[0].fill_betweenx(h,p['covered_model_sensitivity_min_assumed_m2_per_s']*1e4,
        p['covered_model_sensitivity_max_assumed_m2_per_s']*1e4,color=color,alpha=.2,label='Model sensitivity')
    axes[0].plot(common,h,color='#1c324f',lw=1.5,ls='--',label='Fixed common domain')
    axes[0].fill_betweenx(h,p['common_model_sensitivity_min_assumed_m2_per_s']*1e4,
        p['common_model_sensitivity_max_assumed_m2_per_s']*1e4,color='#1c324f',alpha=.12)
    axes[0].axvline(0,color='.6',lw=.7);axes[0].set_xlabel(r'Horizontal integral $\int u\,dx$ (cm$^2$/s)')
    axes[0].set_ylabel('Depth below local surface (cm)');axes[0].legend(fontsize=8,loc='lower right')
    axes[1].plot(p['coverage_fraction']*100,h,color=color,lw=1.8,label='Accepted width')
    band=p['common_depth_band_actual_m']*100;frac=float(p['common_domain_fraction'])*100
    axes[1].plot([frac,frac],band,color='#1c324f',ls='--',label='Common width %.1f%%'%frac)
    axes[1].set_xlim(0,100);axes[1].set_xlabel('Horizontal coverage (%)');axes[1].legend(fontsize=8,loc='lower left')
    axes[2].plot(p['covered_width_mean_u_assumed_m_per_s']*100,h,color=color,lw=1.8,label='Accepted width')
    axes[2].plot(p['common_domain_mean_u_assumed_m_per_s']*100,h,color='#1c324f',lw=1.5,ls='--',label='Fixed common domain')
    axes[2].axvline(0,color='.6',lw=.7);axes[2].set_xlabel('Mean horizontal velocity (cm/s)')
    axes[2].legend(fontsize=8,loc='lower right')
    for ax in axes:
        ax.set_ylim(1,0);ax.grid(alpha=.2);ax.axhspan(0,12*float(p['DX'])*100,color='.85',alpha=.6)
        ax.tick_params(labelbottom=True)
    axes[0].set_title('Pair %d · depth profile'%pair,loc='left',pad=12)

def main():
    OUT.mkdir(parents=True,exist_ok=True);results={};profiles={};samples={};raw={};surfaces={};stats={}
    for pair in [80,100]:
        folder=HERE/str(pair);dest=OUT/('pair_'+str(pair));dest.mkdir(exist_ok=True)
        r=read(folder/'results.npz');p=read(folder/'integration_profile.npz');s=rectified(pair)
        with np.load(folder/'inputs.npz') as z:raw[pair]=z['rawA'];surfaces[pair]=z['surface_a']
        results[pair]=r;profiles[pair]=p;samples[pair]=s
        # Reusable data carry masks; additionally supply directly masked physical arrays.
        data=dict(r)
        data.update(x_assumed_cm=(r['query'][:,0]+.5)*float(r['DX'])*100,
            z_up_assumed_cm=(float(np.median(surfaces[pair]))-r['query'][:,1])*float(r['DX'])*100,
            z_reference_image_y_px=np.array(float(np.median(surfaces[pair]))),
            u_assumed_cm_per_s=np.where(r['accepted'],r['disp'][:,0]*float(r['DX'])/float(r['DT'])*100,np.nan),
            w_up_assumed_cm_per_s=np.where(r['accepted'],-r['disp'][:,1]*float(r['DX'])/float(r['DT'])*100,np.nan),
            du_dx_assumed_per_s=np.where(r['gradient_accepted_xx'],r['gradient'][:,0,0]/float(r['DT']),np.nan),
            dw_dz_assumed_per_s=np.where(r['gradient_accepted_yy'],r['gradient'][:,1,1]/float(r['DT']),np.nan),
            depth_assumed_cm=r['depth']*float(r['DX'])*100)
        np.savez_compressed(dest/'velocity_gradients.npz',**data)
        savemat(str(dest/'velocity_gradients.mat'),data,long_field_names=True,do_compression=True)
        np.savez_compressed(dest/'horizontal_integral.npz',**p)
        savemat(str(dest/'horizontal_integral.mat'),p,long_field_names=True,do_compression=True)
        # CSV is plain numeric export, not a spreadsheet model.
        columns=['depth_cm','covered_integral_cm2_per_s','coverage_percent','covered_width_cm','covered_mean_u_cm_per_s','common_integral_cm2_per_s','common_width_cm','full_width_integral_cm2_per_s','model_min_cm2_per_s','model_max_cm2_per_s']
        table=np.c_[p['depth_m']*100,p['covered_segment_integral_assumed_m2_per_s']*1e4,
            p['coverage_fraction']*100,p['covered_width_m']*100,p['covered_width_mean_u_assumed_m_per_s']*100,
            p['common_domain_integral_assumed_m2_per_s']*1e4,np.full(len(p['depth_m']),float(p['common_domain_width_m'])*100),
            p['full_width_integral_assumed_m2_per_s']*1e4,p['covered_model_sensitivity_min_assumed_m2_per_s']*1e4,
            p['covered_model_sensitivity_max_assumed_m2_per_s']*1e4]
        np.savetxt(dest/'horizontal_integral.csv',table,delimiter=',',header=','.join(columns),comments='')
        fig,ax=plt.subplots(figsize=(16,3.7));stats[str(pair)]=quiver(ax,r,raw[pair],surfaces[pair],pair)
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.53),ncol=3,frameon=False,fontsize=8)
        fig.text(.12,.03,'Arrows are thinned for clarity; estimates failing conservative screens are omitted. Units assume DX in metres/pixel and DT in seconds.',fontsize=9)
        fig.subplots_adjust(bottom=.30,top=.76);save(fig,dest/'quiver')
        fig,axes=plt.subplots(1,3,figsize=(13,6),sharey=True)
        profile(fig,axes,p,pair,'#167a91' if pair==80 else '#bd5527')
        fig.text(.075,.030,'Accepted segments only; gaps are not filled. Shading = model sensitivity, not confidence intervals. Gray = approximate 12-pixel masked strip.',fontsize=8)
        fig.text(.075,.010,'Physical units assume DX in metres/pixel and DT in seconds; the geometric surface offset is inferred.',fontsize=8)
        fig.subplots_adjust(bottom=.14,top=.90,wspace=.24);save(fig,dest/'horizontal_integral')
    # Shared scales across pairs allow a direct visual comparison of gradients.
    gradstat={}
    for component,flag,name,title in [(0,'gradient_accepted_xx','du_dx',r'$\partial u/\partial x$'),(1,'gradient_accepted_yy','dw_dz',r'$\partial w/\partial z$')]:
        vals=[]
        for pair in [80,100]:
            s=samples[pair];vals.extend((s['gradient'][:,component,component][s[flag]]/float(s['DT'])).tolist())
        limit=max(float(np.percentile(np.abs(vals),99)),.1) if vals else .1
        gradstat[name]=dict(color_limit=limit,color_scale_rule='99th percentile absolute accepted values across both pairs; clipping shown by colorbar extensions')
        fig,axes=plt.subplots(2,1,figsize=(16,6.7),sharex=True)
        for pair,ax in zip([80,100],axes):
            s=samples[pair];shape=(len(s['depth_axis_px']),len(s['x_axis_px']))
            a=np.where(s[flag],s['gradient'][:,component,component]/float(s['DT']),np.nan).reshape(shape)
            x=(s['x_axis_px']+.5)*float(s['DX'])*100;h=s['depth_axis_px']*float(s['DX'])*100
            # Cell edges define displayed sample support only; no gap interpolation.
            xe=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
            he=np.r_[0,(h[:-1]+h[1:])/2,1]
            cm=copy.copy(plt.get_cmap('RdBu_r'));cm.set_bad('#e4e7eb')
            pc=ax.pcolormesh(xe,he,np.ma.masked_invalid(a),cmap=cm,vmin=-limit,vmax=limit,shading='flat',rasterized=True)
            ax.set_xlim(0,2048*float(s['DX'])*100);ax.set_ylim(1,0);ax.set_ylabel('Local depth (cm)')
            ax.set_title('Pair %d · %s (s$^{-1}$)'%(pair,title),loc='left');fig.colorbar(pc,ax=ax,pad=.015,extend='both',label=r's$^{-1}$')
        axes[-1].set_xlabel('Horizontal position x (cm)')
        fig.text(.12,.03,'Values are Cartesian derivatives; the display follows local surface depth. Gray = unreported. Overlapping local windows limit spatial resolution. Physical units are assumed.',fontsize=9)
        fig.subplots_adjust(bottom=.15,top=.93,hspace=.45);save(fig,OUT/name)
    fig,axes=plt.subplots(2,1,figsize=(16,4.6))
    for pair,ax in zip([80,100],axes):quiver(ax,results[pair],raw[pair],surfaces[pair],pair)
    axes[0].set_xlabel('')
    fig.text(.12,.050,'Image-only estimates. Cyan = inferred local surface; orange = 1 cm below it. Both panels use the same arrow scale.',fontsize=8)
    fig.text(.12,.023,'Physical units assume DX in metres/pixel and DT in seconds. Height is relative to the median surface ordinate in each image.',fontsize=8)
    fig.subplots_adjust(bottom=.16,top=.87,hspace=.65);save(fig,OUT/'quiver_comparison')
    fig,axes=plt.subplots(2,1,figsize=(16,3.4))
    for pair,ax in zip([80,100],axes):quiver(ax,results[pair],raw[pair],surfaces[pair],pair,gain=20,min_depth_cm=.5)
    axes[0].set_xlabel('')
    fig.text(.12,.035,'Supplementary view of the same estimates: larger arrow gain reveals slower motion. Cyan = 0.5 cm local depth; orange = 1 cm. Physical units are assumed.',fontsize=9)
    fig.subplots_adjust(bottom=.18,top=.83,hspace=.65);save(fig,OUT/'quiver_deeper_half')
    fig,axes=plt.subplots(2,3,figsize=(13,10),sharey=True,sharex='col')
    for pair,axs in zip([80,100],axes):profile(fig,axs,profiles[pair],pair,'#167a91' if pair==80 else '#bd5527')
    fig.text(.075,.030,'Local surface depth. Gray = approximate 12-pixel masked strip. Accepted-segment and common-domain integrals cover different horizontal domains.',fontsize=8)
    fig.text(.075,.013,'Physical units assume DX in metres/pixel and DT in seconds; the geometric surface offset is inferred. Shading = model sensitivity, not confidence intervals.',fontsize=8)
    fig.subplots_adjust(bottom=.09,top=.95,wspace=.24,hspace=.36);save(fig,OUT/'horizontal_integral_comparison')
    (OUT/'figure_metadata.json').write_text(json.dumps(dict(quiver=stats,gradients=gradstat),indent=2)+'\n')
    print('Figure and data export complete',OUT,flush=True)

if __name__=='__main__':main()
