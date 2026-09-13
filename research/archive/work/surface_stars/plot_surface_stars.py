"""Add time-aligned observed-derived surface markers to frozen mean profiles."""
from pathlib import Path
import os,sys,json,hashlib,csv
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/surface_velocity_comparison'
BASE=ROOT/'outputs/piv_comparison_top_cm'
OF='#d54b28';PIV='#2363b1';STAR='#edbd36'
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white','svg.fonttype':'none'})

def read(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(fig,name):
    fig.savefig(str(OUT/name)+'.png',dpi=220,bbox_inches='tight')
    fig.savefig(str(OUT/name)+'.svg',bbox_inches='tight');plt.close(fig)

def select(p,pair):
    rows=np.flatnonzero(p['PIV__pairNum']==pair);assert len(rows)==1
    row=int(rows[0]);idx=int(p['PIV__IR_idx'][row])-1
    t=float(p['PIV__t'][row]);ir=float(p['USurf__t'][idx])
    assert idx==int(np.argmin(np.abs(p['USurf__t']-t)))
    sa=np.flatnonzero(p['Surfs__pairNum']==pair)
    assert len(sa)==2 and abs(float(p['Surfs__t'][sa[0]])-t)<1e-12
    dt=float(p['Surfs__dt_pair']);assert abs(float(p['Surfs__t'][sa[1]])-t-dt)<1e-12
    field='usurf0' if np.isfinite(p['USurf__usurf0'][idx]) else 'usurffilt'
    speed=float(p['USurf__'+field][idx]);assert np.isfinite(speed)
    finite=np.flatnonzero(np.isfinite(p['USurf__usurf0']))
    before=finite[finite<idx];after=finite[finite>idx]
    brackets=[]
    for side,jj in [('before',int(before[-1]) if len(before) else None),('after',int(after[0]) if len(after) else None)]:
        if jj is not None:brackets.append(dict(side=side,index_matlab=jj+1,time_s=float(p['USurf__t'][jj]),u_m_per_s=float(p['USurf__usurf0'][jj]),delta_from_piv_time_s=float(p['USurf__t'][jj]-t)))
    def number(key):
        v=float(p['USurf__'+key][idx]);return v if np.isfinite(v) else None
    return dict(pair=pair,experiment=str(p['exp_name']),piv_index_matlab=row+1,ir_index_matlab=idx+1,
        piv_frame_a_time_s=t,piv_frame_b_time_s=t+dt,piv_pair_midpoint_time_s=t+dt/2,
        ir_sample_time_s=ir,ir_minus_piv_a_time_s=ir-t,ir_mapping_verified_nearest=True,
        selected_field='USurf.'+field,selected_surface_velocity_m_per_s=speed,selected_surface_velocity_cm_per_s=speed*100,
        plotted_depth_mm=0.,raw_fresh_dot_velocity_m_per_s=number('usurf0'),
        filtered_velocity_m_per_s=number('usurffilt'),edge_method_velocity_m_per_s=number('usurf1'),
        composite_velocity_m_per_s=number('usurfComp'),Ndots_used=number('Ndots_used'),
        TMVTech=number('TMVTech'),corrmax0=number('corrmax0'),corrmax1=number('corrmax1'),
        nearest_finite_raw_brackets=brackets,IRfps=float(p['USurf__IRfps']),
        smoothing_window_samples=40,smoothing_window_40_over_fps_s=40/float(p['USurf__IRfps']),
        smoothing_endpoint_separation_39_over_fps_s=39/float(p['USurf__IRfps']),
        method_note='Stored usurffilt: usurf0 linearly interpolated across NaN gaps, then 40-sample moving mean. Used because raw usurf0 is missing at the matched IR index. No use of theory or uncertain usurf1/composite as marker.')

def axes_style(ax,p):
    ax.set_ylim(10,-.28);ax.set_yticks(np.arange(0,11,2))
    ax.grid(alpha=.2);ax.axvline(0,color='.6',lw=.7)
    ax.axhspan(0,12*float(p['DX'])*1000,color='.9',alpha=.8)
    ax.axhline(0,color='.45',lw=.65)
    ax.set_ylabel('Depth below local surface (mm)')

def plot_mean(ax,p,selection,own=False,label_value=True):
    h=p['depth_m']*1000
    a='of_original_mean_u_m_per_s' if own else 'of_matched_mean_u_m_per_s'
    b='piv_own_mean_u_m_per_s' if own else 'piv_matched_mean_u_m_per_s'
    ax.plot(p[a]*100,h,color=OF,lw=2,label='Image-only optical flow')
    ax.plot(p[b]*100,h,color=PIV,ls='--',lw=1.7,label='Supplied PIV')
    x=selection['selected_surface_velocity_cm_per_s']
    ax.scatter([x],[0],marker='*',s=200,facecolor=STAR,edgecolor='#222222',linewidth=1.,
        zorder=10,clip_on=False,label='IR surface estimate (smoothed)')
    if label_value:
        ax.annotate('%.2f cm/s'%x,xy=(x,0),xytext=(-12,0),textcoords='offset points',ha='right',va='center',fontsize=10,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.))
    axes_style(ax,p);ax.set_xlim(-.4,15.3);ax.set_xlabel('Mean horizontal velocity (cm/s)')
    ax.legend(loc='lower right',fontsize=8,framealpha=.95)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(HERE/'selected_run.npz')
    selected={pair:select(source,pair) for pair in [80,100]}
    paths={pair:BASE/('pair_%s'%pair)/'horizontal_integral_comparison.npz' for pair in [80,100]}
    before={str(p):sha(p) for p in paths.values()}
    profiles={pair:read(path) for pair,path in paths.items()}
    assert all(float(p['DX'])==float(source['Surfs__dx']) and float(p['DT'])==float(source['Surfs__dt_pair']) for p in profiles.values())
    fig,axes=plt.subplots(1,2,figsize=(11,6.7),sharex=True,sharey=True)
    for pair,ax in zip([80,100],axes):
        plot_mean(ax,profiles[pair],selected[pair]);ax.set_title('Pair %d · surface estimate %.2f cm/s'%(pair,selected[pair]['selected_surface_velocity_cm_per_s']),loc='left',pad=16)
    axes[1].set_ylabel('')
    fig.text(.085,.070,'Stars: USurf.usurffilt at the stored PIV-to-IR time match, placed at zero depth. Raw fresh-dot samples are missing at both matched times.',fontsize=9)
    fig.text(.085,.044,'The IR values are interpolated and smoothed over 40 samples (~0.93 s). They are surface references, not additional subsurface predictions.',fontsize=9)
    fig.text(.085,.019,'Red/blue means use identical horizontal segments at each depth; their curves and the previous local-depth convention are unchanged.',fontsize=9)
    fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.89,wspace=.17)
    save(fig,'mean_horizontal_velocity_with_surface')

    fig,axes=plt.subplots(2,3,figsize=(14.6,10.2),sharey=True,sharex='col')
    for pair,axs in zip([80,100],axes):
        p=profiles[pair];h=p['depth_m']*1000
        axs[0].plot(p['of_matched_integral_m2_per_s']*1e4,h,color=OF,lw=2,label='Image-only optical flow')
        axs[0].plot(p['piv_matched_integral_m2_per_s']*1e4,h,color=PIV,ls='--',lw=1.6,label='Supplied PIV')
        axes_style(axs[0],p);axs[0].set_xlabel(r'Horizontal integral $\int u\,dx$ (cm$^2$/s)')
        axs[0].set_title('Pair %d · same segments at each depth'%pair,loc='left',fontsize=11,pad=14);axs[0].legend(loc='lower right',fontsize=8)
        plot_mean(axs[1],p,selected[pair]);axs[1].set_title('Mean velocity + IR surface reference',loc='left',fontsize=11,pad=14)
        axs[2].plot(p['of_original_coverage_fraction']*100,h,color=OF,lw=1.3,ls=':',label='Optical-flow coverage')
        axs[2].plot(p['piv_own_coverage_fraction']*100,h,color=PIV,lw=1.3,ls='--',label='PIV coverage')
        axs[2].plot(p['matched_coverage_fraction']*100,h,color='#303a44',lw=1.7,label='Width used in red/blue comparison')
        axes_style(axs[2],p);axs[2].set_xlim(0,101);axs[2].set_xlabel('Horizontal coverage (%)')
        axs[2].set_title('Coverage used in the comparison',loc='left',fontsize=11,pad=14);axs[2].legend(loc='lower left',fontsize=8)
        for ax in axs:ax.tick_params(labelbottom=True)
    fig.text(.065,.061,'Stars at depth zero: smoothed IR surface estimates (USurf.usurffilt), 9.87 cm/s for pair 80 and 13.49 cm/s for pair 100.',fontsize=10)
    fig.text(.065,.040,'Raw fresh-dot values are absent at the matched times. The stored filtered series interpolates gaps and uses a 40-sample moving mean (~0.93 s).',fontsize=9)
    fig.text(.065,.020,'Original subsurface curves are unchanged. Each red/blue pair uses identical intervals at each depth; the available intervals vary with depth.',fontsize=9)
    fig.text(.065,.004,'Gray = previously masked strip; prior local-depth geometry is retained. Surface velocity is not converted into an integral without a supported surface width.',fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,bottom=.15,top=.94,hspace=.37,wspace=.25)
    save(fig,'horizontal_integral_vs_piv_with_surface')

    fig,axes=plt.subplots(1,2,figsize=(11,6.7),sharex=True,sharey=True)
    for pair,ax in zip([80,100],axes):
        plot_mean(ax,profiles[pair],selected[pair],own=True);ax.set_title('Pair %d · each method’s available width'%pair,loc='left',pad=16)
    axes[1].set_ylabel('')
    fig.text(.085,.065,'Supplement: means computed over each method’s own available segments. The red and blue curves therefore cover different horizontal locations.',fontsize=9)
    fig.text(.085,.040,'Stars use the same smoothed IR surface estimates at depth zero. Original curves and local-depth geometry are unchanged.',fontsize=9)
    fig.text(.085,.015,'Use mean_horizontal_velocity_with_surface for the comparison on identical horizontal segments.',fontsize=9)
    fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.89,wspace=.17)
    save(fig,'mean_own_coverage_with_surface')

    records=list(selected.values())
    after={str(p):sha(p) for p in paths.values()};assert before==after
    metadata={'experiment':'ExpLCL_1_03','selection':records,'original_profile_sha256':before,'original_profiles_unchanged':True,
        'surface_measurement_used_to_refit_or_extend_velocity':False,
        'units':'USurf downstream velocity is already m/s; multiplied by100 for cm/s without a sign change.',
        'background_source':'data/PIVFileContents.md',
        'manual_ptv_note':'ManualPTVOutput.md documents a separate particle-match format; results.mat is the campaign surface-results file.',
        'source_file_sha256':sha('data/results.mat')}
    (OUT/'surface_marker_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    columns=['pair','piv_frame_a_time_s','ir_sample_time_s','ir_minus_piv_a_time_s','ir_index_matlab','selected_field','selected_surface_velocity_m_per_s','selected_surface_velocity_cm_per_s','plotted_depth_mm']
    with (OUT/'surface_markers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows({k:r[k] for k in columns} for r in records)
    arr={k:np.array([r[k] for r in records]) for k in columns}
    np.savez_compressed(OUT/'surface_markers.npz',**arr)
    savemat(str(OUT/'surface_markers.mat'),arr,long_field_names=True,do_compression=True)
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':main()
