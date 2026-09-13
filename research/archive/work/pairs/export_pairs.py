"""Scientific plots and MATLAB products for the new full-image pairs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MPLCONFIGDIR']=os.path.abspath('work/mplcache')
from pathlib import Path
import argparse,json,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import savemat

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'work/pairs'
CYAN='#1ed5ed';ORANGE='#ffad38';INK='#193348';GRAY='#d7dde3'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run(pair,vlim=None,figures_only=False):
    w=BASE/str(pair);out=ROOT/'outputs'/('pair_'+str(pair));out.mkdir(exist_ok=True)
    inp=read(w/'inputs.npz');z=read(w/'results.npz')
    summary=json.loads((w/'summary.json').read_text())
    q=z['query_full'];d=z['disp'];g=z['gradient'];ok=z['accepted'].astype(bool)
    gx=z['gradient_accepted_xx'].astype(bool);gz=z['gradient_accepted_yy'].astype(bool)
    c=z['classical_displacement_native_at_grid'];cv=z['classical_available_at_grid'].astype(bool)
    err=np.linalg.norm(d-c,axis=1);both=ok&cv
    dx=float(inp['DX']);dt=float(inp['DT']);raw=inp['rawA'];va=inp['va'];sa=inp['surface_a']
    h,width=raw.shape;xx=np.arange(7,width,8);yy=np.arange(7,h,8)
    ix=np.rint((q[:,0]-7)/8).astype(int);iy=np.rint((q[:,1]-7)/8).astype(int)
    assert np.allclose(q,np.c_[xx[ix],yy[iy]])
    xe=np.r_[xx-4,xx[-1]+4];ye=np.r_[yy-4,yy[-1]+4]
    def raster(a,mask=None):
        a=np.asarray(a,float).copy()
        if mask is not None:a[~mask]=np.nan
        im=np.full((len(yy),len(xx)),np.nan);im[iy,ix]=a
        return np.ma.masked_invalid(im)
    cmap=copy.copy(plt.get_cmap('gray'));cmap.set_bad('#151b23')
    background=np.ma.array(raw,mask=~va)
    first=va.argmax(axis=0)
    def bg(ax,limits=(0,2047,2047,0),image=True):
        if image:
            ax.imshow(background,cmap=cmap,vmin=25,vmax=150,origin='upper',extent=(-.5,width-.5,h-.5,-.5),interpolation='nearest',alpha=.82)
            ax.set_facecolor('#151b23')
        else:ax.set_facecolor(GRAY)
        ax.fill_between(np.arange(width),0,first,color='#edf0f4',lw=0,zorder=3)
        ax.plot(np.arange(width),first,color='#8e608b',lw=1,zorder=4)
        ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='Image x (pixels)',ylabel='Image y (pixels, downward)')
        ax.set_aspect('equal')
    def quiv(ax,mask,vec,color,scale,width=.0015,alpha=1,zorder=5):
        return ax.quiver(q[mask,0],q[mask,1],vec[mask,0],vec[mask,1],angles='xy',scale_units='xy',scale=1/scale,color=color,width=width,
                         headwidth=3.6,headlength=4.5,headaxislength=4.1,minlength=.05,alpha=alpha,zorder=zorder)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    handles=[Line2D([],[],color=CYAN,lw=3,label='Conservative hybrid'),Line2D([],[],color=ORANGE,lw=1.6,label='Supplied native PIV'),Line2D([],[],color='#8e608b',lw=1.5,label='Boundary of retained image data')]
    sparse=((ix%4)==0)&((iy%4)==0)
    fig,ax=plt.subplots(figsize=(10.5,11.8));fig.subplots_adjust(top=.87,bottom=.17,left=.09,right=.96)
    fig.suptitle('Pair {} | full-image velocity comparison'.format(pair),x=.09,y=.985,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.09,.937,'A-to-B displacement vectors shown at 4× scale, sampled every 32 pixels',fontsize=11,color=INK)
    shown_ends=np.r_[q[sparse&ok]+4*d[sparse&ok],q[sparse&cv]+4*c[sparse&cv]]
    display_limits=(min(0,float(shown_ends[:,0].min())-5),max(width-1,float(shown_ends[:,0].max())+5),
                    max(h-1,float(shown_ends[:,1].max())+5),min(0,float(shown_ends[:,1].min())-5))
    bg(ax,display_limits);ax.scatter(q[ok,0],q[ok,1],s=.14,color=CYAN,alpha=.16,lw=0,zorder=4)
    a=quiv(ax,sparse&ok,d,CYAN,4,width=.0017)
    quiv(ax,sparse&cv,c,ORANGE,4,width=.0009,alpha=.87,zorder=6)
    ax.quiverkey(a,.81,1.045,5,'5 pixels / pair',coordinates='axes',labelpos='E',fontproperties={'size':10})
    ax.legend(handles=handles,loc='upper left',framealpha=.94,fontsize=10)
    fig.text(.09,.113,'{:,} / {:,} hybrid vectors pass. Faint cyan dots show the full accepted grid; arrows are thinned for readability.'.format(int(ok.sum()),len(ok)),fontsize=9.8,color=INK)
    fig.text(.09,.073,'Classical vectors use their own finite native samples and image-availability checks; they are not filtered by agreement.',fontsize=9.5,color=INK)
    fig.text(.09,.035,'The supplied PIV initializes the coarse stage. This is a method comparison, not independent ground-truth validation.',fontsize=9.5,color=INK)
    save(fig,'hybrid_piv_overlay')

    fig,axs=plt.subplots(2,2,figsize=(15,11.5));fig.subplots_adjust(top=.88,bottom=.16,wspace=.17,hspace=.30)
    fig.suptitle('Pair {} | near-surface comparison across the image'.format(pair),x=.075,y=.985,ha='left',fontsize=19,fontweight='bold',color=INK)
    fig.text(.075,.934,'Same fields; actual displacement scale (1×), sampled every 16 pixels. These are detailed views of the full-image analysis.',fontsize=10.5)
    for ax,(lo,hi) in zip(axs.ravel(),[(0,650),(450,1150),(950,1650),(1450,2048)]):
        bg(ax,(lo,min(hi,2047),780,340));sel=(q[:,0]>=lo+20)&(q[:,0]<hi-30)&(q[:,1]>=340)&(q[:,1]<765)&((ix%2)==0)&((iy%2)==0)
        quiv(ax,sel&ok,d,CYAN,1,width=.0033);quiv(ax,sel&cv,c,ORANGE,1,width=.0014,alpha=.88,zorder=6)
        ax.set_title('x = {}–{} pixels'.format(lo,hi-1),loc='left')
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.032))
    save(fig,'hybrid_piv_near_surface')

    vals=np.r_[np.abs(g[gx,0,0]/dt),np.abs(g[gz,1,1]/dt)]
    limit=float(vlim) if vlim else max(.2,float(np.percentile(vals,99.5))) if len(vals) else .2
    limit=float(np.ceil(limit*5)/5)
    fig,axs=plt.subplots(1,2,figsize=(15,9.7));fig.subplots_adjust(top=.86,bottom=.15,left=.065,right=.93,wspace=.18)
    fig.suptitle('Pair {} | conservative velocity derivatives'.format(pair),x=.065,y=.985,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.065,.934,'u and x point right; w and z point upward. Each derivative has its own conservative sensitivity screen.',fontsize=10.5)
    clipped={}
    for ax,(row,col,mask,title,key) in zip(axs,[(0,0,gx,'∂u/∂x','du_dx'),(1,1,gz,'∂w/∂z','dw_dz')]):
        im=ax.pcolormesh(xe,ye,raster(g[:,row,col]/dt,mask),cmap='RdBu_r',vmin=-limit,vmax=limit,shading='flat',rasterized=True)
        bg(ax,image=False);ax.set_title('{} | {:,} accepted locations'.format(title,int(mask.sum())),loc='left',fontsize=12)
        clipped[key]=int(np.sum(np.abs(g[mask,row,col]/dt)>limit))
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.025,extend='both');cb.set_label('s⁻¹, assuming DT is seconds')
    fig.text(.065,.086,'Both signs cancel in ∂w/∂z = ∂dᵧ/∂y ÷ DT. Gray indicates withheld estimates; light gray above the mask contains no usable data.',fontsize=9.5)
    fig.text(.065,.044,'DT = {} assumed seconds. Colors saturate at ±{:.2f} s⁻¹; full numerical values are retained. No incompressibility constraint is imposed.'.format(dt,limit),fontsize=9.5)
    save(fig,'du_dx_dw_dz')

    fig,axs=plt.subplots(2,1,figsize=(15,9));fig.subplots_adjust(top=.87,bottom=.13,left=.065,right=.90,hspace=.32)
    fig.suptitle('Pair {} | near-surface derivative detail'.format(pair),x=.065,y=.985,ha='left',fontsize=20,fontweight='bold',color=INK)
    fig.text(.065,.937,'Full horizontal width; the same accepted fields and color scale as the full-image maps.',fontsize=10.5)
    for ax,(row,col,mask,title) in zip(axs,[(0,0,gx,'∂u/∂x'),(1,1,gz,'∂w/∂z')]):
        im=ax.pcolormesh(xe,ye,raster(g[:,row,col]/dt,mask),cmap='RdBu_r',vmin=-limit,vmax=limit,shading='flat',rasterized=True)
        bg(ax,(0,width-1,780,340),image=False);ax.set_title(title,loc='left',fontsize=13)
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.025,pad=.025,extend='both');cb.set_label('s⁻¹, assuming DT is seconds')
    fig.text(.065,.077,'Gray: withheld estimates. Purple: boundary of retained image data. Both derivatives use separate acceptance masks.',fontsize=9.5)
    fig.text(.065,.038,'Colors saturate at ±{:.2f} s⁻¹. These are detailed views of the full-image calculation; all accepted values are exported.'.format(limit),fontsize=9.5)
    save(fig,'du_dx_dw_dz_near_surface')

    for row,col,mask,name,label in [(0,0,gx,'du_dx','∂u/∂x'),(1,1,gz,'dw_dz','∂w/∂z')]:
        fig,ax=plt.subplots(figsize=(9,10.1));fig.subplots_adjust(top=.87,bottom=.13,left=.09,right=.88)
        fig.suptitle('Pair {} | {}'.format(pair,label),x=.09,y=.98,ha='left',fontsize=21,fontweight='bold',color=INK)
        im=ax.pcolormesh(xe,ye,raster(g[:,row,col]/dt,mask),cmap='RdBu_r',vmin=-limit,vmax=limit,shading='flat',rasterized=True)
        bg(ax,image=False);cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.035,extend='both');cb.set_label('s⁻¹, assuming DT is seconds')
        fig.text(.09,.072,'{:,} accepted locations. Gray: withheld. Derivative of interval-averaged motion at the frame-A position.'.format(int(mask.sum())),fontsize=9.5)
        fig.text(.09,.036,'DT = {} assumed seconds; colors saturate at ±{:.2f} s⁻¹. Values in pixel-gradient units are also exported.'.format(dt,limit),fontsize=9.5)
        save(fig,name)

    fig,axs=plt.subplots(1,2,figsize=(15,9));fig.subplots_adjust(top=.86,bottom=.15,left=.07,right=.93,wspace=.30)
    fig.suptitle('Pair {} | agreement and accepted coverage'.format(pair),x=.07,y=.985,ha='left',fontsize=19,fontweight='bold',color=INK)
    emax=max(.25,float(np.percentile(err[both],99))) if both.any() else 1
    im=axs[0].pcolormesh(xe,ye,raster(err,both),cmap='magma',vmin=0,vmax=emax,shading='flat',rasterized=True);bg(axs[0],image=False)
    axs[0].set_title('Hybrid–classical endpoint difference',loc='left');cb=fig.colorbar(im,ax=axs[0],fraction=.046,pad=.025,extend='max');cb.set_label('pixels / pair')
    histmax=max(emax*1.2,.5);overflow=int(np.sum(err[both]>histmax))
    axs[1].hist(err[both],bins=np.linspace(0,histmax,65),color='#087c83',alpha=.85)
    axs[1].set(xlabel='Endpoint difference (pixels / pair)',ylabel='Accepted hybrid locations',title='Coincident, available native PIV samples')
    if both.any():
        mean=float(err[both].mean());median=float(np.median(err[both]));p95=float(np.percentile(err[both],95));axs[1].text(.96,.94,'All comparisons: N = {:,}\nMean = {:.3f} px\nMedian = {:.3f} px\n95th percentile = {:.3f} px\n{} values beyond plotted histogram range'.format(int(both.sum()),mean,median,p95,overflow),transform=axs[1].transAxes,ha='right',va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    fig.text(.07,.084,'Agreement is summarized only where the hybrid passes and a native classical vector is available. Larger differences are not used to reject vectors.',fontsize=9.5)
    fig.text(.07,.04,'Both methods see the same image pair, and supplied PIV starts the coarse fit. No manual or independent reference vectors were supplied.',fontsize=9.5)
    save(fig,'piv_comparison')

    if figures_only:
        print('Refreshed figures only for pair',pair,flush=True)
        return

    dc=d.copy();dc[~ok]=np.nan;gc=g.copy();du=g[:,0,0].copy();dw=g[:,1,1].copy();du[~gx]=np.nan;dw[~gz]=np.nan
    data=dict(grid_xy_zero_based_px=q,grid_xy_matlab_one_based_px=q+1,
              displacement_raw_px_per_pair=d,displacement_conservative_px_per_pair=dc,
              displacement_gradient_raw=gc,du_dx_pixel_gradient_conservative=du,dw_dz_pixel_gradient_conservative=dw,
              velocity_uw_conservative_assumed_mps=dc*np.array([1,-1])*dx/dt,
              du_dx_conservative_assumed_per_s=du/dt,dw_dz_conservative_assumed_per_s=dw/dt,
              accepted_vector=ok.astype(np.uint8),accepted_du_dx=gx.astype(np.uint8),accepted_dw_dz=gz.astype(np.uint8),
              classical_native_displacement_at_grid_px=c,classical_native_available_at_grid=cv.astype(np.uint8),
              comparison_endpoint_difference_px=err,comparison_available=both.astype(np.uint8),
              DX_from_supplied_MAT=dx,DT_from_supplied_MAT=dt,physical_units_confirmed=np.uint8(0),
              units_note='Pixels are primary; assumed SI requires DX metres/pixel and DT seconds. u/x right, w/z up. Both diagonal derivatives are Gii/DT.',
              derivative_note='Analytic derivative of blended A-to-B displacement divided by DT, attached to A source position; finite-time gradient proxy. Separate component screens; not calibrated error bars.',
              method_note='Original conservative hybrid masked quadratic r13; affine13,quadratic19,mask14 comparisons; C2 blend16; entire image depth; native suppliedPIV for comparison and dense suppliedPIV for coarse initialization.',
              validation_note='Classical PIV is a related method comparison and initializer, not independent ground truth. No manual references for this pair.',
              surface_y_zero_based_px=sa,surface_b_y_zero_based_px=inp['surface_b'],actual_first_valid_row=first)
    data['distance_below_retained_boundary_px']=q[:,1]-np.interp(q[:,0],np.arange(width),first)
    data['surface_geometry_inferred']=np.uint8(inp['surface_geometry_inferred'])
    data['surface_trace_offset_pixels']=inp['surface_trace_offset_pixels']
    data['surface_geometry_note']='surfacePIVImg minus 12 pixels transfers the prior exporter convention; geometric surface is not independently recovered from these masked pairs. Actual availability is independently verified.'
    for k,v in z.items():
        if k not in ['query_full','disp','gradient'] and isinstance(v,np.ndarray) and v.dtype.kind in 'biuf':data['diagnostic_'+k]=v.astype(np.uint8) if v.dtype.kind=='b' else v
    for k,v in inp.items():
        if k.startswith('classical') or k in ['native_x_zero','native_y_zero','physical_units_confirmed','surface_geometry_inferred','surface_trace_offset_pixels']:
            if isinstance(v,np.ndarray) and v.dtype.kind in 'biuf':data['source_'+k]=v
    savemat(out/'conservative_field.mat',data,do_compression=True,long_field_names=True)
    np.savez_compressed(out/'conservative_field.npz',**data)
    report_extra={'pair':pair,'grid_nodes':len(ok),'accepted_vector_count':int(ok.sum()),'accepted_du_dx_count':int(gx.sum()),'accepted_dw_dz_count':int(gz.sum()),
                  'classical_native_available_count':int(cv.sum()),'comparison_count':int(both.sum()),'comparison_mean_px':float(np.mean(err[both])) if both.any() else None,
                  'comparison_median_px':float(np.median(err[both])) if both.any() else None,'comparison_rmse_px':float(np.sqrt(np.mean(err[both]**2))) if both.any() else None,
                  'gradient_color_limit_assumed_per_s':limit,'gradient_color_saturation_counts':clipped,
                  'comparison_map_color_limit_px':emax,'histogram_upper_limit_px':histmax,'histogram_overflow_count':overflow,
                  'DX':dx,'DT':dt,'physical_units_confirmed':False}
    (out/'plot_and_export_summary.json').write_text(json.dumps(report_extra,indent=2));(out/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(report_extra,indent=2),flush=True)
    return report_extra

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',type=int,required=True);p.add_argument('--vlim',type=float);p.add_argument('--figures-only',action='store_true');a=p.parse_args();run(a.pair,a.vlim,a.figures_only)
