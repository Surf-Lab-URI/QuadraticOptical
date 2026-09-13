"""Render identical-domain, held-out PIV integral comparisons."""
from quiver_compare import *
from matplotlib.colors import ListedColormap

def profile_axes(ax,p):
    ax.set_ylim(10,0);ax.grid(alpha=.2);ax.axvline(0,color='.6',lw=.7)
    ax.axhspan(0,12*float(p['DX'])*1000,color='.9',alpha=.8)
    ax.set_ylabel('Depth below local surface (mm)')

def curves(ax,p,stem,mean=False):
    scale=100 if mean else 1e4
    suffix='_mean_u_m_per_s' if mean else '_integral_m2_per_s'
    h=p['depth_m']*1000
    ax.plot(p['of_'+stem+suffix]*scale,h,color=OF,lw=2,label='Image-only optical flow')
    ax.plot(p['piv_'+stem+suffix]*scale,h,color=PIV,ls='--',lw=1.6,label='Supplied PIV')
    profile_axes(ax,p)
    ax.set_xlabel('Mean horizontal velocity (cm/s)' if mean else r'Horizontal integral $\int u\,dx$ (cm$^2$/s)')

def main():
    ps={pair:read(WORK/('pair_%s_integral_comparison.npz'%pair)) for pair in [80,100]}
    fig,axes=plt.subplots(2,3,figsize=(14.6,10.2),sharey=True,sharex='col')
    for pair,axs in zip([80,100],axes):
        p=ps[pair];h=p['depth_m']*1000
        curves(axs[0],p,'matched');curves(axs[1],p,'matched',mean=True)
        axs[0].set_title('Pair %d · same segments at each depth'%pair,loc='left',fontsize=11)
        axs[1].set_title('Mean over those same segments',loc='left',fontsize=11)
        axs[0].legend(loc='lower right',fontsize=8)
        axs[1].legend(loc='lower right',fontsize=8)
        axs[2].plot(p['of_original_coverage_fraction']*100,h,color=OF,lw=1.3,ls=':',label='Optical-flow coverage')
        axs[2].plot(p['piv_own_coverage_fraction']*100,h,color=PIV,lw=1.3,ls='--',label='PIV coverage')
        axs[2].plot(p['matched_coverage_fraction']*100,h,color='#303a44',lw=1.7,label='Width used in left panel')
        axs[2].set_xlim(0,101);axs[2].set_xlabel('Horizontal coverage (%)')
        axs[2].set_title('Coverage used in the comparison',loc='left',fontsize=11)
        profile_axes(axs[2],p);axs[2].legend(loc='lower left',fontsize=8)
        for ax in axs:ax.tick_params(labelbottom=True)
    fig.text(.065,.046,'Each red/blue pair integrates exactly the same horizontal intervals. No missing interval is filled or counted as zero.',fontsize=10)
    fig.text(.065,.025,'Available segments change with depth. No nonempty set of segments is supported by BOTH methods at every depth from 1.13 to 10 mm.',fontsize=9)
    fig.text(.065,.007,'Gray strip: approximately 12 masked pixels. Units assume DX in metres/pixel and DT in seconds; local surface uses the existing inferred offset.',fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.95,hspace=.36,wspace=.25)
    save(fig,OUT/'horizontal_integral_vs_piv')

    fig,axes=plt.subplots(2,3,figsize=(14.6,10.2),sharey=True,sharex='col')
    for pair,axs in zip([80,100],axes):
        p=ps[pair]
        h=p['depth_m']*1000
        for ax,suffix,scale,xlabel in [(axs[0],'integral_m2_per_s',1e4,r'Horizontal integral $\int u\,dx$ (cm$^2$/s)'),(axs[1],'mean_u_m_per_s',100,'Mean horizontal velocity (cm/s)')]:
            ax.plot(p['of_original_'+suffix]*scale,h,color=OF,lw=2,label='Original optical-flow profile')
            ax.plot(p['piv_own_'+suffix]*scale,h,color=PIV,lw=1.6,ls='--',label='PIV on its available segments')
            profile_axes(ax,p);ax.set_xlabel(xlabel);ax.legend(loc='lower right',fontsize=8)
        axs[0].set_title('Pair %d · each method uses its own segments'%pair,loc='left',fontsize=11)
        axs[1].set_title('Mean over each method’s available width',loc='left',fontsize=11)
        axs[2].plot(p['of_original_coverage_fraction']*100,h,color=OF,lw=2,label='Optical-flow width')
        axs[2].plot(p['piv_own_coverage_fraction']*100,h,color=PIV,lw=1.6,ls='--',label='PIV width')
        axs[2].set_xlim(0,101);axs[2].set_xlabel('Horizontal coverage (%)')
        axs[2].set_title('These horizontal widths differ',loc='left',fontsize=11)
        profile_axes(axs[2],p);axs[2].legend(loc='lower left',fontsize=8)
        for ax in axs:ax.tick_params(labelbottom=True)
    fig.text(.065,.045,'Supplement: the previous optical-flow integral is preserved here, alongside PIV integrated over its own available segments.',fontsize=10)
    fig.text(.065,.024,'The curves cover different horizontal locations and widths. Their integral difference therefore combines velocity differences with coverage differences.',fontsize=9)
    fig.text(.065,.006,'Use horizontal_integral_vs_piv for the identical-segment comparison. Gray = masked strip; physical units and surface offset are assumed.',fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.95,hspace=.36,wspace=.25);save(fig,OUT/'original_profiles_and_piv_own_coverage')

    # Display the exact horizontal interval masks used by the numerical integration.
    fig,axes=plt.subplots(2,1,figsize=(16,6.3),sharex=True)
    for pair,ax in zip([80,100],axes):
        p=ps[pair];h=p['depth_m']*1000
        joint=p['matched_interval_mask'];of=p['of_original_interval_mask'];piv=p['piv_own_interval_mask']
        val=of.astype(int)+2*piv.astype(int)
        bounds=p['horizontal_interval_bounds_px'];xe=np.r_[bounds[:,0],bounds[-1,1]]
        xe=(xe+.5)*float(p['DX'])*100
        he=np.r_[0,(h[:-1]+h[1:])/2,10]
        ax.pcolormesh(xe,he,val,cmap=ListedColormap(['#edf0f3','#f1c3af','#a7c8ec','#087d80']),vmin=0,vmax=3,shading='flat',rasterized=True)
        ax.axhline(p['common_depth_band_m'][0]*1000,color='#34434a',ls=':',lw=.8)
        ax.set_ylim(10,0);ax.set_xlim(0,2048*float(p['DX'])*100)
        ax.set_ylabel('Local depth (mm)')
        ax.set_title('Pair %d · dark segments are shared at each depth; none remain shared at every depth from 1.13 to 10 mm'%pair,loc='left',fontsize=11)
    axes[-1].set_xlabel('Horizontal position x (cm)')
    legend=[Line2D([0],[0],color=c,lw=8,label=l) for c,l in [('#edf0f3','Neither method'),('#f1c3af','Optical flow only'),('#a7c8ec','PIV only'),('#087d80','Both: used for comparison')]]
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9)
    fig.subplots_adjust(left=.065,right=.985,bottom=.14,top=.93,hspace=.43)
    save(fig,OUT/'horizontal_segments_used')

    # Copy the exact comparison arrays into the deliverable folder for reuse.
    import shutil
    for pair,p in ps.items():
        dest=OUT/('pair_'+str(pair));dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'horizontal_integral_comparison.npz',**p)
        savemat(str(dest/'horizontal_integral_comparison.mat'),p,long_field_names=True,do_compression=True)
        for ext in ['csv','json']:
            source=WORK/('pair_%s_integral_comparison.%s'%(pair,ext))
            if source.exists():shutil.copy2(source,dest/('horizontal_integral_comparison.'+ext))

if __name__=='__main__':main()
