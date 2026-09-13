"""Side-by-side derivative maps with explicit operator and support labels."""
from quiver_compare import *
import copy, shutil

def main():
    data={pair:read(WORK/('pair_%s_gradient_comparison.npz'%pair)) for pair in [80,100]}
    info={}
    for component,mathname in [('du_dx',r'$\partial u/\partial x$'),('dw_dz',r'$\partial w/\partial z$')]:
        vals=np.concatenate([data[pair][kind+'_'+component+'_assumed_per_s'].ravel() for pair in data for kind in ['of','piv']])
        dv=np.concatenate([data[pair]['difference_'+component+'_assumed_per_s'].ravel() for pair in data])
        lim=max(.1,float(np.nanpercentile(np.abs(vals),99)))
        dlim=max(.1,float(np.nanpercentile(np.abs(dv),99)))
        info[component]={'method_color_limit_per_s':lim,'difference_color_limit_per_s':dlim,
            'color_rule':'99th percentile absolute finite values pooled across both pairs; symmetric scales; extensions flag clipping.',
            'method_full_range_per_s':[float(np.nanmin(vals)),float(np.nanmax(vals))],
            'difference_full_range_per_s':[float(np.nanmin(dv)),float(np.nanmax(dv))]}
        fig,axes=plt.subplots(2,3,figsize=(17,7.1),sharex=True,sharey=True)
        for pair,axs in zip([80,100],axes):
            p=data[pair];x=p['x_axis_assumed_cm'];h=p['depth_axis_assumed_cm']*10
            xe=np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
            he=np.r_[0,(h[:-1]+h[1:])/2,10]
            for ax,kind,title in zip(axs,['of','piv','difference'],['Optical flow · saved analytic derivative','PIV · 16-pixel central difference','Optical flow − PIV · shared points']):
                a=p[kind+'_'+component+'_assumed_per_s'];limit=dlim if kind=='difference' else lim
                cm=copy.copy(plt.get_cmap('RdBu_r'));cm.set_bad('#e4e7eb')
                pc=ax.pcolormesh(xe,he,np.ma.masked_invalid(a),cmap=cm,vmin=-limit,vmax=limit,shading='flat',rasterized=True)
                ax.set_ylim(10,0);ax.set_xlim(0,2048*float(p['DX'])*100)
                ax.set_title(('Pair %s · '%pair if kind=='of' else '')+title,loc='left',fontsize=10)
                ax.set_xlabel('Horizontal position x (cm)');ax.tick_params(labelbottom=True)
                fig.colorbar(pc,ax=ax,pad=.02,extend='both',label=r's$^{-1}$',fraction=.045)
            axs[0].set_ylabel('Local depth (mm)')
        fig.suptitle(mathname+' comparison over the top 1 cm',fontsize=15,y=.99)
        fig.text(.055,.059,'Cartesian derivatives displayed against depth below the local surface. Gray = unavailable. The two methods have different smoothing.',fontsize=9)
        fig.text(.055,.035,'OF and PIV share a color scale across both pairs; differences use their own scale. Limits = 99th percentile of absolute values; extremes remain in data.',fontsize=9)
        fig.text(.055,.012,'PIV differentiation precedes surface-depth sampling. The supplied PIV does not initialize or alter optical flow. Physical units and surface offset are assumed.',fontsize=8)
        fig.subplots_adjust(left=.055,right=.985,bottom=.17,top=.89,hspace=.48,wspace=.27)
        save(fig,OUT/(component+'_vs_piv'))
    for pair,p in data.items():
        dest=OUT/('pair_'+str(pair));dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'gradient_comparison.npz',**p)
        savemat(str(dest/'gradient_comparison.mat'),p,long_field_names=True,do_compression=True)
        shutil.copy2(WORK/('pair_%s_gradient_comparison.json'%pair),dest/'gradient_comparison_metrics.json')
    (OUT/'gradient_figure_metadata.json').write_text(json.dumps(info,indent=2)+'\n')

if __name__=='__main__':main()
