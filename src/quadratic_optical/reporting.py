"""Portable field exports, conservative coverage plots and local HTML reports."""
from pathlib import Path
import json,html,os
from urllib.parse import quote
import numpy as np
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter
from .core.finalize_fields import ConservativeEvaluator
from .matio import read_experiment_fields
from .core.fit_fields import atomic_npz

OF='#d54b28';PIV='#2363b1';MANUAL='#17806d'

# Velocity/gradient field panels. These are presentational settings, held here
# rather than in the JSON configuration on purpose: every configuration key
# except workers and piv_quality enters the preparation signature, so adding one
# would invalidate every existing output directory for a change that alters no
# number. Limits are fixed rather than per-pair percentiles so that panels from
# different pairs of the same experiment are directly comparable.
FIELD_SMOOTH_PX=40.             # box filter width in image pixels
FIELD_MIN_VALID=.5              # minimum valid fraction within the kernel
FIELD_U_RANGE=(-.005,.12)       # m/s
FIELD_W_ABS=.02                 # m/s, symmetric
FIELD_DUDX_ABS=8.               # s^-1, symmetric
FIELD_CONTOUR_CM_S=1.           # isotach interval on the smoothed u panel; 0 disables
FIELD_BELOW='#ff00ff';FIELD_ABOVE='#39ff14';FIELD_ABSENT='#b8b8b8'
FIELD_DATUM_FRAMES=20           # leading surface frames averaged for the still-water datum
FIELD_SURFACE_LINE='#00e5ff'    # free-surface profile drawn on the z panels
FIELD_FIG_WIDTH_IN=17.          # panels are drawn at a true 1:1 aspect, so they are wide and short
FIELD_FIG_EXTRA_IN=1.9          # vertical allowance for titles, labels and the colour bar

_datum_cache={} 

def jsonable(value):
    """Strict JSON values, with absent/nonfinite diagnostics represented by null."""
    if isinstance(value, dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):return jsonable(value.tolist())
    if isinstance(value, np.generic):return jsonable(value.item())
    if isinstance(value, Path):return str(value)
    if isinstance(value, float) and not np.isfinite(value):return None
    if value is None or isinstance(value, (str, bool, int, float)):return value
    raise TypeError('Unsupported report metadata value: '+type(value).__name__)

def write_json(path, values):
    path=Path(path);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(jsonable(values),indent=2,allow_nan=False)+'\n',encoding='utf8')
    temporary.replace(path)

def surface_reference(directory, record=None):
    """Persist the selected observation separately; a later comparison retains it."""
    path=Path(directory)/'surface_ir_reference.json'
    if record is None and path.exists():record=json.loads(path.read_text(encoding='utf8'))
    if record is None:return None
    if not isinstance(record,dict):raise ValueError('IR surface reference must be a selection record.')
    if record.get('available',False):
        if record.get('selected_field') not in ['USurf.usurf0','USurf.usurffilt']:
            raise ValueError('Only indexed raw or filtered observational IR surface values may be starred.')
        velocity=record.get('velocity_m_per_s')
        if velocity is None or not np.isfinite(float(velocity)):
            raise ValueError('An available IR surface observation must have finite velocity.')
        if not record.get('comparison_only',False) or float(record.get('plotting_depth_m',0.))!=0:
            raise ValueError('IR references must be comparison-only observations plotted at zero depth.')
    record=jsonable(record);write_json(path,record)
    return record

def quality_label(comparison):
    if comparison is None:return ''
    quality=comparison['summary']['quality']
    return 'PIV: finite correlation required' if quality=='correlation' else 'PIV: supplied finite vectors; correlation not required'

def read(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def save(fig,directory,name):
    fig.savefig(directory/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(directory/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def sample_plot_grid(directory,depth_m):
    ev=ConservativeEvaluator(directory,requested_depth_m=depth_m)
    width=len(ev.inputs['surface_a'])
    x=np.arange(7,width,8,dtype=float)
    h=np.r_[np.arange(0,ev.max_depth_px,4.),ev.max_depth_px]
    h=np.unique(h);xx,hh=np.meshgrid(x,h)
    yy=hh+np.interp(xx,np.arange(width),ev.inputs['surface_a'])
    s=ev.evaluate_conservative(np.c_[xx.ravel(),yy.ravel()])
    s.update(x_axis_px=x,depth_axis_px=h,DX=ev.DX,DT=ev.DT)
    atomic_npz(Path(directory)/'plot_samples.npz',**s)
    return s

def field_export(directory):
    r=read(directory/'results.npz');dx=float(r['DX']);dt=float(r['DT'])
    n=len(r['query'])
    if r['query'].shape!=(n,2) or r['disp'].shape!=(n,2) or r['gradient'].shape!=(n,2,2):
        raise ValueError('Frozen field coordinates, displacements and gradients have inconsistent dimensions.')
    if any(r[k].shape!=(n,) for k in ['depth','accepted','gradient_accepted_xx','gradient_accepted_yy']):
        raise ValueError('Frozen field acceptance/depth arrays have inconsistent lengths.')
    physical={'x_m':(r['query'][:,0]+.5)*dx,'depth_m':r['depth']*dx,
        'z_up_m':(np.median(r['surface_a'])-r['query'][:,1])*dx,
        'u_m_per_s':np.where(r['accepted'],r['disp'][:,0]*dx/dt,np.nan),
        'w_m_per_s':np.where(r['accepted'],-r['disp'][:,1]*dx/dt,np.nan),
        'du_dx_per_s':np.where(r['gradient_accepted_xx'],r['gradient'][:,0,0]/dt,np.nan),
        'dw_dz_per_s':np.where(r['gradient_accepted_yy'],r['gradient'][:,1,1]/dt,np.nan),
        'accepted':r['accepted'],'gradient_accepted_xx':r['gradient_accepted_xx'],'gradient_accepted_yy':r['gradient_accepted_yy'],
        'query_px':r['query'],'disp_px':r['disp'],'gradient_px_per_px':r['gradient'],
        'DX':dx,'DT':dt,'physical_units_confirmed':bool(r.get('physical_units_confirmed',False)),
        'gradient_definition':'G[i,j] = d image displacement component i / d image coordinate j; x right, y down',
        'screening_note':'Physical velocities and diagonal derivatives are masked; raw disp_px and gradient_px_per_px require the associated acceptance masks.'}
    atomic_npz(directory/'velocity_gradients.npz',**physical)
    savemat(str(directory/'velocity_gradients.mat'),physical,long_field_names=True,do_compression=True)
    names=['x_m','z_up_m','depth_m','u_m_per_s','w_m_per_s','du_dx_per_s','dw_dz_per_s','accepted','gradient_accepted_xx','gradient_accepted_yy']
    np.savetxt(directory/'velocity_gradients.csv',np.column_stack([physical[k] for k in names]),delimiter=',',header=','.join(names),comments='')
    return r

def quiver(directory,r,comparison):
    with np.load(directory/'inputs.npz',allow_pickle=False) as f:raw=f['rawA'].astype(float);surface=f['surface_a']
    q=r['query'];dx=float(r['DX'])*100;dt=float(r['DT']);sr=float(np.median(surface));dmax=float(r['requested_depth_m'])*100
    fig,ax=plt.subplots(figsize=(14,4.5))
    low=max(0,int(np.floor(surface.min()))-3);high=min(raw.shape[0],int(np.ceil(surface.max()+dmax/dx))+3)
    crop=raw[low:high].copy();yy=np.arange(low,high)[:,None]
    crop[(yy<surface[None,:])|(yy>surface[None,:]+dmax/dx)]=np.nan
    ax.imshow(crop,cmap='gray_r',vmin=0,vmax=180,alpha=.3,interpolation='nearest',aspect='equal',extent=[0,raw.shape[1]*dx,(sr-high+.5)*dx,(sr-low+.5)*dx])
    x=(np.arange(raw.shape[1])+.5)*dx
    ax.plot(x,(sr-surface)*dx,color='#087c87',lw=.8);ax.plot(x,(sr-surface)*dx-dmax,color='#b66d13',lw=.8)
    ux=np.unique(q[:,0]);uy=np.unique(q[:,1]);thin=np.isin(q[:,0],ux[::4])&np.isin(q[:,1],uy[::2])
    series=[]
    if comparison is not None:
        v=comparison['velocity']
        if not np.array_equal(v['query_px'],q):raise ValueError('PIV comparison arrow origins differ from the frozen reporting grid.')
        series.append((v['piv_disp_px'],v['piv_available'],PIV,'Supplied PIV'))
    series.append((r['disp'],r['accepted'],OF,'Image-only optical flow'))
    gain=4.;endpoints=[];handles=[];speeds=[];drawn=[]
    for disp,ok,color,label in series:
        use=ok&thin&np.isfinite(disp).all(axis=1)
        origin=np.c_[(q[use,0]+.5)*dx,(sr-q[use,1])*dx]
        velocity=np.c_[disp[use,0],-disp[use,1]]*dx/dt
        handles.append(ax.quiver(origin[:,0],origin[:,1],velocity[:,0],velocity[:,1],color=color,
            angles='xy',scale_units='xy',scale=1/(gain*dt),width=.001,label=label))
        endpoints.extend([origin,origin+gain*dt*velocity]);speeds.extend(np.linalg.norm(velocity,axis=1))
        drawn.append(dict(method=label,source_indices=np.flatnonzero(use),origin_cm=origin,
                          displayed_tip_cm=origin+gain*dt*velocity))
    extent=np.array([[0.,(sr-surface.max())*dx-dmax],[raw.shape[1]*dx,(sr-surface.min())*dx]])
    points=np.concatenate([extent]+endpoints)
    limits=np.c_[points.min(axis=0),points.max(axis=0)]
    pad=np.maximum(np.ptp(points,axis=0)*.025,3*dx)
    ax.set_xlim(limits[0,0]-pad[0],limits[0,1]+pad[0]);ax.set_ylim(limits[1,0]-pad[1],limits[1,1]+pad[1])
    positive=np.asarray(speeds);positive=positive[np.isfinite(positive)&(positive>0)]
    candidate=float(np.percentile(positive,70)) if positive.size else 1.
    power=10.**np.floor(np.log10(candidate));key_velocity=float(min([1,2,5,10],key=lambda v:abs(v-candidate/power))*power)
    ax.quiverkey(handles[-1],.84,1.04,key_velocity,'%g cm/s'%key_velocity,labelpos='E',coordinates='axes',color='#222')
    ax.set_xlabel('Horizontal position (cm)');ax.set_ylabel('Height relative to median surface (cm)')
    ax.set_title(directory.name+' · top %.2f cm · arrows 4× pair displacement'%dmax,loc='left');ax.legend(loc='upper left',bbox_to_anchor=(0,-.25),ncol=2,frameon=False)
    note='Shared sampled origins and physical scale; unavailable estimates omitted. '+quality_label(comparison)
    fig.text(.12,.01,note,fontsize=8)
    fig.subplots_adjust(bottom=.24,top=.82)
    write_json(directory/'quiver_display_metadata.json',dict(arrow_gain=gain,DT_s=dt,
        arrow_key_cm_per_s=key_velocity,xlim_cm=ax.get_xlim(),ylim_cm=ax.get_ylim(),series=drawn,
        velocity_scaling='displayed arrow length in cm = velocity in cm/s × DT × arrow_gain'))
    save(fig,directory,'quiver')

def mesh(ax,s,a,limit,title):
    x=(s['x_axis_px']+.5)*float(s['DX'])*100;h=s['depth_axis_px']*float(s['DX'])*1000
    cm=plt.get_cmap('RdBu_r').with_extremes(bad='#e4e7eb')
    pc=ax.pcolormesh(x,h,np.ma.masked_invalid(a),cmap=cm,vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
    ax.set_ylim(h[-1],0);ax.set_title(title,loc='left',fontsize=10);ax.set_xlabel('Horizontal position (cm)');ax.set_ylabel('Local depth (mm)')
    return pc

def gradients(directory,s,comparison):
    shape=(len(s['depth_axis_px']),len(s['x_axis_px']))
    if comparison is None:
        fig,axes=plt.subplots(2,1,figsize=(13,6))
        for j,(name,flag) in enumerate([('du/dx','gradient_accepted_xx'),('dw/dz','gradient_accepted_yy')]):
            a=np.where(s[flag],s['gradient'][:,j,j]/float(s['DT']),np.nan).reshape(shape)
            v=a[np.isfinite(a)];lim=max(.1,float(np.percentile(np.abs(v),99))) if v.size else .1
            pc=mesh(axes[j],s,a,lim,'Optical flow '+name);fig.colorbar(pc,ax=axes[j],label='s⁻¹',extend='both')
        name='gradients'
    else:
        g=comparison['gradients'];fig,axes=plt.subplots(2,3,figsize=(16,7))
        if not np.array_equal(g['query_px'],s['query']):raise ValueError('Comparison gradient coordinates differ from the frozen plot grid.')
        for j,comp in enumerate(['du_dx','dw_dz']):
            of=g['of_'+comp+'_per_s'];piv=g['piv_'+comp+'_per_s'];diff=np.where(np.isfinite(of)&np.isfinite(piv),of-piv,np.nan)
            if of.shape!=shape or piv.shape!=shape:raise ValueError('Gradient comparison arrays do not match the surface-relative plot axes.')
            values=np.r_[of.ravel(),piv.ravel()];values=values[np.isfinite(values)]
            limit=max(.1,float(np.percentile(np.abs(values),99))) if values.size else .1
            dv=diff[np.isfinite(diff)];dl=max(.1,float(np.percentile(np.abs(dv),99))) if dv.size else .1
            for ax,a,lim,label in zip(axes[j],[of,piv,diff],[limit,limit,dl],['Optical flow (analytic)','PIV (16-pixel difference)','Optical flow − PIV']):
                pc=mesh(ax,s,a,lim,['du/dx','dw/dz'][j]+' · '+label);fig.colorbar(pc,ax=ax,label='s⁻¹',extend='both',fraction=.046,pad=.02)
        name='gradient_comparison'
    fig.text(.07,.015,'Cartesian derivatives against local depth. Gray = unavailable; symmetric 99th-percentile limits; different smoothing retained. '+quality_label(comparison),fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.97]);save(fig,directory,name)

def profiles(directory,comparison,surface_record):
    p=read(directory/'integration_profile.npz');h=p['depth_m']*1000
    fig,axs=plt.subplots(2,3,figsize=(13,9),sharey=True)
    total_width=float(p['target_full_width_m'])
    if comparison is None:
        bands=p['common_depth_band_actual_m']*1000
        rows=[(p['covered_segment_integral_assumed_m2_per_s'],p['covered_width_mean_u_assumed_m_per_s'],
               p['coverage_fraction'],None,None,'Accepted intervals at each depth'),
              (p['common_domain_integral_assumed_m2_per_s'],p['common_domain_mean_u_assumed_m_per_s'],
               np.where(p['common_domain_available_at_depth'],float(p['common_domain_width_m'])/total_width,np.nan),
               None,None,'Fixed common horizontal domain')]
    else:
        c=comparison['integrals']
        if not np.array_equal(c['depth_m'],p['depth_m']):raise ValueError('Comparison profile depths differ from the frozen integral samples.')
        bands=c['common_depth_band_m']*1000
        rows=[(c['of_matched_integral_m2_per_s'],c['of_matched_mean_u_m_per_s'],c['matched_width_m']/total_width,
               c['piv_matched_integral_m2_per_s'],c['piv_matched_mean_u_m_per_s'],'Shared intervals at each depth'),
              (c['of_common_integral_m2_per_s'],c['of_common_mean_u_m_per_s'],
               np.where(c['common_available_at_depth'],float(c['common_width_m'])/total_width,np.nan),
               c['piv_common_integral_m2_per_s'],c['piv_common_mean_u_m_per_s'],'Fixed common horizontal domain')]
    for axes,(integral,mean,width,piv_integral,piv_mean,title) in zip(axs,rows):
        if piv_integral is not None:
            axes[0].plot(piv_integral*1e4,h,color=PIV,ls='--',label='PIV on identical intervals')
            axes[1].plot(piv_mean*100,h,color=PIV,ls='--',label='PIV on identical intervals')
        axes[0].plot(integral*1e4,h,color=OF,label='Image-only optical flow')
        axes[1].plot(mean*100,h,color=OF,label='Image-only optical flow')
        axes[2].plot(width*100,h,color='#34434a',label='Horizontal width used')
        if not np.isfinite(integral).any():
            for ax in axes:ax.text(.5,.5,'No supported domain',ha='center',va='center',transform=ax.transAxes,color='#68757e')
        if surface_record and surface_record.get('available',False):
            velocity=float(surface_record['velocity_m_per_s'])*100
            label='IR surface (smoothed)' if surface_record['selected_field']=='USurf.usurffilt' else 'IR surface (raw fresh-dot)'
            axes[1].scatter([velocity],[0],marker='*',s=180,color='#edbd36',edgecolor='#222',zorder=10,clip_on=False,label=label)
        for ax in axes:
            ax.set_ylim(h[-1],-.06*h[-1]);ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=7)
        for ax in axes[:2]:
            lo,hi=ax.dataLim.intervalx
            if np.isfinite([lo,hi]).all():
                # Avoid magnifying roundoff-scale differences in a nearly
                # constant profile, while retaining physical units on ticks.
                minimum_span=max(.001,.1*max(abs(lo),abs(hi)))
                if hi-lo<minimum_span:
                    middle=(lo+hi)/2;ax.set_xlim(middle-minimum_span/2,middle+minimum_span/2)
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
        axes[0].set_title(title,loc='left',fontsize=11)
        axes[0].set_xlabel('Horizontal integral (cm²/s)');axes[1].set_xlabel('Mean horizontal velocity (cm/s)')
        axes[2].set_xlabel('Horizontal coverage (%)');axes[2].set_xlim(0,101)
        axes[0].set_ylabel('Depth below local surface (mm)')
    fig.suptitle(directory.name+' · depth profiles',x=.08,ha='left')
    band_note='Fixed domain: intersection of supported horizontal intervals across %.3g–%.3g mm depth.'%tuple(bands)
    fig.text(.08,.055,band_note,fontsize=8)
    fig.text(.08,.035,'No missing segments are filled. Surface stars are comparisons only; they do not constrain the field. '+quality_label(comparison),fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.93,wspace=.25,hspace=.34);save(fig,directory,'profiles')
    savemat(str(directory/'horizontal_integral.mat'),p,long_field_names=True,do_compression=True)


def _field_figsize(x_mm,y_mm,width=FIELD_FIG_WIDTH_IN,extra=FIELD_FIG_EXTRA_IN):
    """Figure size that leaves the axes close to its true 1:1 shape.

    With an equal aspect the axes box is set by the data, so sizing the figure to
    match keeps the surrounding whitespace small; tight bounding boxes on save
    trim whatever is left.
    """
    span_x=float(np.nanmax(x_mm)-np.nanmin(x_mm));span_y=float(np.nanmax(y_mm)-np.nanmin(y_mm))
    if span_x<=0 or span_y<=0:return (width,4.)
    return (width,min(12.,max(2.4,width*span_y/span_x+extra)))


def _masked_box(field,valid,size,min_valid):
    """Box filter that averages over accepted samples only, so gaps do not spread."""
    f=np.where(valid,np.nan_to_num(field),0.)
    total=uniform_filter(f,size=size,mode='nearest')
    share=uniform_filter(valid.astype(float),size=size,mode='nearest')
    out=np.divide(total,share,out=np.full_like(total,np.nan),where=share>0)
    return np.where(share>=min_valid,out,np.nan)


def _field_image(directory,name,values,cmap_name,low,high,coords,title,bar,contours=None):
    x_mm,z_mm=coords
    below=above=0.
    finite=np.isfinite(values)
    if finite.any():
        below=100.*np.sum(values[finite]<low)/finite.sum()
        above=100.*np.sum(values[finite]>high)/finite.sum()
    cmap=plt.get_cmap(cmap_name).with_extremes(bad=FIELD_ABSENT,under=FIELD_BELOW,over=FIELD_ABOVE)
    fig,ax=plt.subplots(figsize=_field_figsize(x_mm,z_mm))
    extent=[x_mm[0],x_mm[-1],z_mm[-1],z_mm[0]]
    # Equal aspect in matching units: one millimetre is the same length on both
    # axes, so feature slopes are not distorted.
    image=ax.imshow(np.ma.masked_invalid(values),extent=extent,origin='upper',aspect='equal',
                    cmap=cmap,vmin=low,vmax=high,interpolation='nearest')
    if contours is not None and len(contours):
        grid_x,grid_z=np.meshgrid(x_mm,z_mm)
        lines=ax.contour(grid_x,grid_z,np.ma.masked_invalid(values),levels=contours,
                         colors='white',linewidths=.8)
        lines.set_path_effects([patheffects.withStroke(linewidth=1.9,foreground='black')])
        marked=contours[1::2] if len(contours)>6 else contours
        for text in ax.clabel(lines,levels=marked,fmt=lambda v:'%g'%round(v*100,3),
                              fontsize=7,inline=True,inline_spacing=6):
            text.set_path_effects([patheffects.withStroke(linewidth=2.,foreground='black')])
    ax.set_xlabel('horizontal position x (mm)');ax.set_ylabel('depth below local surface (mm)')
    ax.set_title(title,fontsize=11,pad=20)
    ax.text(.5,1.012,'scale %.4g to %.4g   |   clipped: %.2f%% below (magenta), %.2f%% above (green)'
            '   |   grey = no accepted estimate'%(low,high,below,above),
            transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='0.35')
    # An aspect-locked axes is shorter than its subplot slot, so tie the colour
    # bar to the drawn axes rather than letting it span the original height.
    bar_axes=make_axes_locatable(ax).append_axes('right',size='0.9%',pad=.12,axes_class=plt.Axes)
    fig.colorbar(image,cax=bar_axes,extend='both').set_label(bar)
    save(fig,directory,name)
    return {'clipped_below_percent':round(below,3),'clipped_above_percent':round(above,3)}


def _still_water_datum(record):
    """Image row of the still-water level from a campaign results file.

    Averages the first FIELD_DATUM_FRAMES frames of ``Surfs.surfsPIV``, which is
    the same reference the campaign's ``Surfs.eta`` uses, so z here and eta there
    share a zero. Returns ``(row, note)``; ``row`` is None when no campaign file
    was supplied or its surface record cannot be read, and ``note`` always says
    which happened. Cached per file because the campaign read is seconds long.
    """
    if not record:
        return None,'no IR/campaign results file was supplied with this run'
    source=record.get('source');experiment=record.get('experiment')
    if not source or not experiment:
        return None,'the surface record names no campaign file and experiment'
    if not Path(source).exists():
        return None,'campaign file '+str(source)+' is not present on this machine'
    key=(str(source),str(experiment))
    if key not in _datum_cache:
        try:
            values,_=read_experiment_fields(source,experiment,['Surfs.surfsPIV'])
            frames=np.asarray(values['Surfs.surfsPIV'],float)[:FIELD_DATUM_FRAMES]
            usable=np.isfinite(frames)&(frames>=0)
            if not usable.any():
                _datum_cache[key]=(None,'Surfs.surfsPIV holds no usable rows in the first %d frames'%FIELD_DATUM_FRAMES)
            else:
                _datum_cache[key]=(float(frames[usable].mean()),
                    'mean of the first %d surface frames in %s (Surfs.surfsPIV, %.1f%% of samples usable); '
                    'the same reference as that campaign\'s Surfs.eta'
                    %(min(FIELD_DATUM_FRAMES,len(frames)),Path(source).name,100.*usable.mean()))
        except Exception as problem:
            _datum_cache[key]=(None,'campaign surface record unreadable (%s: %s)'%(type(problem).__name__,problem))
    return _datum_cache[key]


def _field_image_z(directory,name,values,cmap_name,low,high,grid,surface_mm,title,bar,
                   datum_note,contours=None):
    """Same field drawn against absolute height z, with the free surface profile."""
    grid_x,grid_z=grid
    below=above=0.
    finite=np.isfinite(values)
    if finite.any():
        below=100.*np.sum(values[finite]<low)/finite.sum()
        above=100.*np.sum(values[finite]>high)/finite.sum()
    cmap=plt.get_cmap(cmap_name).with_extremes(bad=FIELD_ABSENT,under=FIELD_BELOW,over=FIELD_ABOVE)
    fig,ax=plt.subplots(figsize=_field_figsize(grid_x[0],grid_z))
    mesh_values=np.ma.masked_invalid(values)
    image=ax.pcolormesh(grid_x,grid_z,mesh_values,cmap=cmap,vmin=low,vmax=high,shading='nearest')
    if contours is not None and len(contours):
        lines=ax.contour(grid_x,grid_z,mesh_values,levels=contours,colors='white',linewidths=.8)
        lines.set_path_effects([patheffects.withStroke(linewidth=1.9,foreground='black')])
        marked=contours[1::2] if len(contours)>6 else contours
        for text in ax.clabel(lines,levels=marked,fmt=lambda v:'%g'%round(v*100,3),
                              fontsize=7,inline=True,inline_spacing=6):
            text.set_path_effects([patheffects.withStroke(linewidth=2.,foreground='black')])
    ax.plot(grid_x[0],surface_mm,color=FIELD_SURFACE_LINE,linewidth=1.4,
            path_effects=[patheffects.withStroke(linewidth=2.8,foreground='black')],
            label='free surface',zorder=5)
    ax.axhline(0.,color='0.25',linewidth=.8,linestyle='--',zorder=4)
    ax.set_xlabel('horizontal position x (mm)');ax.set_ylabel('height z (mm)')
    ax.set_xlim(grid_x[0][0],grid_x[0][-1])
    ax.set_ylim(np.nanmin(grid_z),max(np.nanmax(surface_mm),0.)+1.)
    ax.set_aspect('equal')
    ax.set_title(title,fontsize=11,pad=28)
    ax.text(.5,1.052,'scale %.4g to %.4g   |   clipped: %.2f%% below (magenta), %.2f%% above (green)'
            '   |   grey = no accepted estimate'%(low,high,below,above),
            transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='0.35')
    ax.text(.5,1.012,'z = 0 at '+datum_note,transform=ax.transAxes,ha='center',va='bottom',
            fontsize=8,color='0.35')
    ax.legend(loc='lower right',fontsize=8,framealpha=.85)
    # An aspect-locked axes is shorter than its subplot slot, so tie the colour
    # bar to the drawn axes rather than letting it span the original height.
    bar_axes=make_axes_locatable(ax).append_axes('right',size='0.9%',pad=.12,axes_class=plt.Axes)
    fig.colorbar(image,cax=bar_axes,extend='both').set_label(bar)
    save(fig,directory,name)
    return {'clipped_below_percent':round(below,3),'clipped_above_percent':round(above,3)}


def field_panels(directory,s,surface_record=None):
    """Depth-rectified velocity and gradient images, with their acceptance masks applied.

    Returns the settings record, or None when the display grid is too small.
    The du/dx panel is a finite difference of the smoothed horizontal velocity.
    It is deliberately not the screened analytic gradient, which is stricter and
    stays in velocity_gradients.csv; this one is masked only by acceptance of u.
    """
    directory=Path(directory)
    x=np.asarray(s['x_axis_px'],float);z=np.asarray(s['depth_axis_px'],float)
    rows,columns=len(z),len(x)
    if rows<2 or columns<2:return None
    dx=float(s['DX']);dt=float(s['DT'])
    accepted=np.asarray(s['accepted']).reshape(rows,columns)
    disp=np.asarray(s['disp']).reshape(rows,columns,2)
    u=np.where(accepted,disp[...,0]*dx/dt,np.nan)
    w=np.where(accepted,-disp[...,1]*dx/dt,np.nan)
    span_x=float(x[1]-x[0]);span_z=float(z[1]-z[0])
    cells=(max(1,int(np.rint(FIELD_SMOOTH_PX/span_z))),max(1,int(np.rint(FIELD_SMOOTH_PX/span_x))))
    smooth_u=_masked_box(u,accepted,cells,FIELD_MIN_VALID)
    smooth_w=_masked_box(w,accepted,cells,FIELD_MIN_VALID)
    dudx=np.gradient(smooth_u,span_x*dx,axis=1)
    step=FIELD_CONTOUR_CM_S/100.
    levels=np.arange(step,FIELD_U_RANGE[1]+step/2,step) if FIELD_CONTOUR_CM_S>0 else np.array([])
    coords=(x*dx*1000.,z*dx*1000.)
    tag='%gpx'%FIELD_SMOOTH_PX
    name=Path(directory).name
    contour_note='   (contours every %g cm/s)'%FIELD_CONTOUR_CM_S if len(levels) else ''
    jobs=[('field_u',u,'magma',FIELD_U_RANGE[0],FIELD_U_RANGE[1],
           name+'   horizontal velocity u  (unsmoothed)','u  (m s$^{-1}$)',None),
          ('field_w',w,'RdBu_r',-FIELD_W_ABS,FIELD_W_ABS,
           name+'   vertical velocity w, positive up  (unsmoothed)','w  (m s$^{-1}$)',None),
          ('field_u_smooth'+tag,smooth_u,'magma',FIELD_U_RANGE[0],FIELD_U_RANGE[1],
           '%s   horizontal velocity u,  %g px smoothed%s'%(name,FIELD_SMOOTH_PX,contour_note),
           'u  (m s$^{-1}$)',levels),
          ('field_w_smooth'+tag,smooth_w,'RdBu_r',-FIELD_W_ABS,FIELD_W_ABS,
           '%s   vertical velocity w (positive up),  %g px smoothed'%(name,FIELD_SMOOTH_PX),
           'w  (m s$^{-1}$)',None),
          ('field_dudx_from_smooth'+tag,dudx,'PuOr_r',-FIELD_DUDX_ABS,FIELD_DUDX_ABS,
           '%s   du/dx from %g px smoothed u  (finite difference)'%(name,FIELD_SMOOTH_PX),
           r'$\partial u/\partial x$  (s$^{-1}$)',levels if False else None)]
    clipping={}
    for base,values,cmap_name,low,high,title,bar,contours in jobs:
        clipping[base+'.png']=_field_image(directory,base,values,cmap_name,low,high,coords,
                                           title,bar,contours)
    # Absolute-height versions. Each sample keeps its own image row, so the
    # lattice is not rectangular in z and is drawn as a mesh rather than resampled.
    query_y=np.asarray(s['query'])[:,1].reshape(rows,columns)
    surface_y=query_y[0]-z[0]
    datum_row,datum_note=_still_water_datum(surface_record)
    if datum_row is None:
        fallback=float(np.mean(surface_y))
        datum_source='frame'
        datum_note=('the mean surface of this frame pair (row %.2f) because %s'%(fallback,datum_note))
        datum_row=fallback
    else:
        datum_source='campaign'
        datum_note='the still-water level (row %.2f): %s'%(datum_row,datum_note)
    grid=(np.tile(coords[0],(rows,1)),(datum_row-query_y)*dx*1000.)
    surface_mm=(datum_row-surface_y)*dx*1000.
    for base,values,cmap_name,low,high,title,bar,contours in jobs:
        clipping[base+'_z.png']=_field_image_z(directory,base+'_z',values,cmap_name,low,high,
                                               grid,surface_mm,title+'   [height z]',bar,
                                               datum_note,contours)
    record={'smoothing_px':FIELD_SMOOTH_PX,'kernel_cells_depth_x':list(cells),
            'kernel_px_depth_x':[cells[0]*span_z,cells[1]*span_x],
            'grid_spacing_px':{'x':span_x,'depth':span_z},
            'minimum_valid_fraction':FIELD_MIN_VALID,
            'colour_limits':{'u_m_per_s':list(FIELD_U_RANGE),
                             'w_m_per_s':[-FIELD_W_ABS,FIELD_W_ABS],
                             'dudx_per_s':[-FIELD_DUDX_ABS,FIELD_DUDX_ABS]},
            'contour_interval_cm_s':FIELD_CONTOUR_CM_S,
            'contour_levels_cm_s':[round(float(v)*100,3) for v in levels],
            'flag_colours':{'below':FIELD_BELOW,'above':FIELD_ABOVE,'no_estimate':FIELD_ABSENT},
            'source':'plot_samples.npz display grid, depth-rectified, acceptance mask applied',
            'dudx_note':'central finite difference of the smoothed u along x; not the screened analytic gradient',
            'z_datum':{'source':datum_source,'image_row':datum_row,'description':datum_note,
                       'frames_averaged':FIELD_DATUM_FRAMES if datum_source=='campaign' else None,
                       'surface_mean_z_mm':float(np.mean(surface_mm)),
                       'convention':'z increases upward, metres converted to mm; depth panels are unaffected'},
            'clipping_percent':clipping}
    write_json(directory/'field_panels.json',record)
    return record


def manual_arrows(directory,record):
    """Predicted and hand-matched arrows from the same origins, at true aspect."""
    directory=Path(directory)
    with np.load(directory/'inputs.npz',allow_pickle=False) as f:
        raw=f['rawA'].astype(float);surface=f['surface_a'].astype(float)
    dx=float(record['DX'])*1000.                       # mm per pixel
    origin=np.asarray(record['origin_zero_based'],float).reshape(2)
    source=np.asarray(record['source_px'],float)-origin
    manual=np.asarray(record['manual_disp_px'],float)
    predicted=np.asarray(record['predicted_disp_px'],float)
    accepted=np.asarray(record['accepted_mask'],bool)
    reference=float(np.median(surface))
    gain=4.
    tips=np.concatenate([source+gain*manual,source[accepted]+gain*predicted[accepted]]) if accepted.any() else source+gain*manual
    span=np.concatenate([source,tips])
    pad=18.
    left=max(0,int(np.floor(span[:,0].min()-pad)));right=min(raw.shape[1],int(np.ceil(span[:,0].max()+pad)))
    top=max(0,int(np.floor(span[:,1].min()-pad)));bottom=min(raw.shape[0],int(np.ceil(span[:,1].max()+pad)))
    crop=raw[top:bottom,left:right]
    fig,ax=plt.subplots(figsize=(14,5.2))
    ax.imshow(crop,cmap='gray_r',vmin=0,vmax=180,alpha=.35,interpolation='nearest',aspect='equal',
              extent=[left*dx,right*dx,(reference-bottom)*dx,(reference-top)*dx])
    columns=np.arange(left,right)
    ax.plot((columns+.5)*dx,(reference-surface[left:right])*dx,color='#087c87',lw=1.,label='surface (frame A)')
    def draw(values,mask,colour,label):
        if not mask.any():return
        ax.quiver(source[mask,0]*dx,(reference-source[mask,1])*dx,
                  values[mask,0]*dx,-values[mask,1]*dx,color=colour,angles='xy',
                  scale_units='xy',scale=1./gain,width=.0022,label=label)
    draw(manual,np.ones(len(source),bool),MANUAL,'Hand-matched (%d)'%len(source))
    draw(predicted,accepted,OF,'Image-only optical flow (%d accepted)'%int(accepted.sum()))
    if (~accepted).any():
        ax.plot(source[~accepted,0]*dx,(reference-source[~accepted,1])*dx,'x',color='#b03a3a',
                ms=4.5,mew=.9,label='withheld by the screen (%d)'%int((~accepted).sum()))
    ax.set_xlabel('horizontal position x (mm)');ax.set_ylabel('height above median surface (mm)')
    ax.set_title('%s   hand-matched particles vs image-only prediction   (arrows %gx)'
                 %(directory.name,gain),fontsize=11,pad=16)
    best=[b for b in record['statistics'] if b['subset']=='passing vector screen']
    if best and best[0]['count']:
        b=best[0]
        ax.text(.5,1.006,'endpoint disagreement, screened: mean %.3f px, median %.3f px, rms %.3f px  '
                '(n=%d of %d)'%(b['mean_px'],b['median_px'],b['rms_px'],b['count'],len(source)),
                transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='0.35')
    ax.legend(loc='lower right',fontsize=8,framealpha=.9)
    save(fig,directory,'manual_comparison')


def manual_block(record):
    """Report section: the static figure plus an inline adjustable-magnification view."""
    if record is None:return ''
    source=np.asarray(record['source_px'],float);manual=np.asarray(record['manual_disp_px'],float)
    predicted=np.asarray(record['predicted_disp_px'],float);accepted=np.asarray(record['accepted_mask'],bool)
    finite=np.isfinite(predicted).all(axis=1)
    points=[{'x':round(float(source[i,0]),2),'y':round(float(source[i,1]),2),
             'mx':round(float(manual[i,0]),3),'my':round(float(manual[i,1]),3),
             'px':round(float(predicted[i,0]),3) if finite[i] else None,
             'py':round(float(predicted[i,1]),3) if finite[i] else None,
             'a':bool(accepted[i])} for i in range(len(source))]
    rows=''.join('<tr><td>%s</td><td>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>'%(
        html.escape(b['subset']),b['count'],
        '&mdash;' if b['mean_px'] is None else '%.4f'%b['mean_px'],
        '&mdash;' if b['median_px'] is None else '%.4f'%b['median_px'],
        '&mdash;' if b['rms_px'] is None else '%.4f'%b['rms_px']) for b in record['statistics'])
    source_name=html.escape(Path(record['manual']['path']).name)
    return ('<h2>Hand-matched particle comparison</h2>'
        '<p>Sparse particles matched by eye between the two frames, read only after the prediction '
        'was frozen. They are a held-out reference: not detection seeds, not constraints, not '
        'training labels, and no correction is applied to the field. Arrows share an origin, so the '
        'gap between arrowheads is the disagreement and its direction. Source: <code>'+source_name+
        '</code>.</p>'
        '<figure><a href="manual_comparison.svg"><img src="manual_comparison.png" alt="hand-matched comparison"></a></figure>'
        '<table><thead><tr><th>Subset</th><th>Count</th><th>Mean (px)</th><th>Median (px)</th>'
        '<th>RMS (px)</th></tr></thead><tbody>'+rows+'</tbody></table>'
        '<p class="mnote">Endpoint disagreement is the distance between the predicted and '
        'hand-matched arrow endpoints. Manual picks carry their own picking error and sit in one '
        'horizontal band, so this is a check, not a ground-truth error bound.</p>'
        '<div class="mviewer"><div class="mbar">'
        '<label>arrow magnification <input id="mgain" type="range" min="1" max="40" step="1" value="4"></label>'
        '<span id="mgainv">4&times;</span>'
        '<label><input id="mshowrej" type="checkbox"> show withheld points</label></div>'
        '<canvas id="mcanvas" width="1160" height="430"></canvas>'
        '<p class="mnote"><span style="color:'+MANUAL+'">&#9632;</span> hand-matched &nbsp; '
        '<span style="color:'+OF+'">&#9632;</span> optical flow &nbsp; '
        '<span style="color:#b03a3a">&#9632;</span> withheld by the screen</p></div>'
        '<script>(function(){var P='+json.dumps(points)+';'
        'var c=document.getElementById("mcanvas"),g=c.getContext("2d");'
        'var sl=document.getElementById("mgain"),lab=document.getElementById("mgainv"),'
        'rej=document.getElementById("mshowrej");'
        'var xs=P.map(function(p){return p.x}),ys=P.map(function(p){return p.y});'
        'var x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs),'
        'y0=Math.min.apply(null,ys),y1=Math.max.apply(null,ys);'
        'function draw(){var k=+sl.value;lab.textContent=k+"\u00d7";'
        'var pad=30,w=c.width-2*pad,h=c.height-2*pad;'
        'var s=Math.min(w/Math.max(x1-x0,1),h/Math.max(y1-y0,1));'
        'var ox=pad+(w-(x1-x0)*s)/2,oy=pad+(h-(y1-y0)*s)/2;'
        'g.clearRect(0,0,c.width,c.height);g.lineWidth=1.1;'
        'function arrow(px,py,dx,dy,col){var X=ox+(px-x0)*s,Y=oy+(py-y0)*s,'
        'X2=X+dx*k*s,Y2=Y+dy*k*s;g.strokeStyle=col;g.fillStyle=col;'
        'g.beginPath();g.moveTo(X,Y);g.lineTo(X2,Y2);g.stroke();'
        'var a=Math.atan2(Y2-Y,X2-X),L=4.5;g.beginPath();g.moveTo(X2,Y2);'
        'g.lineTo(X2-L*Math.cos(a-0.4),Y2-L*Math.sin(a-0.4));'
        'g.lineTo(X2-L*Math.cos(a+0.4),Y2-L*Math.sin(a+0.4));g.closePath();g.fill();}'
        'P.forEach(function(p){arrow(p.x,p.y,p.mx,p.my,"'+MANUAL+'");});'
        'P.forEach(function(p){if(p.px===null)return;'
        'if(p.a)arrow(p.x,p.y,p.px,p.py,"'+OF+'");'
        'else if(rej.checked)arrow(p.x,p.y,p.px,p.py,"#b03a3a");});}'
        'sl.addEventListener("input",draw);rej.addEventListener("change",draw);draw();})();</script>')


_FIELD_ORDER = ['field_u.png', 'field_w.png']


def _extra_field_panels(directory):
    """Optional velocity/gradient panels written alongside by an external script.

    Returns an empty string when no field_*.png files are present, so the report
    is unchanged for runs that do not have them. Purely presentational: this
    function reads no numerical data and participates in no signature.
    """
    names = sorted(p.name for p in Path(directory).glob('field_*.png'))
    if not names:
        return ''
    names.sort(key=lambda n: (_FIELD_ORDER.index(n) if n in _FIELD_ORDER else len(_FIELD_ORDER), n))
    head = ('<h2>Velocity and gradient fields</h2><p>Depth-rectified fields on the display grid, '
            'masked by the acceptance flags. Colour limits are shared across all pairs so panels are '
            'directly comparable between pairs. Clipped samples are flagged <strong>magenta</strong> '
            '(below range) and <strong>green</strong> (above); grey marks locations with no accepted '
            'estimate. The <code>du/dx</code> panel is a finite difference of the smoothed horizontal '
            'velocity and is <em>not</em> the package screened analytic gradient, which is stricter and '
            'remains in <code>velocity_gradients.csv</code>. See '
            '<a href="field_panels.json">field_panels.json</a> for limits, kernel size, the z '
            'datum and clipping fractions.</p><p>Panels ending <code>_z</code> use absolute height '
            'z with the free surface drawn on top; the others use depth below the local surface. '
            'Each z panel states its own datum: the still-water level from the campaign results '
            'file when one was supplied, otherwise that frame pair\'s own mean surface.</p>')
    figs = ''
    for n in names:
        img = '<img src="' + n + '" alt="' + n[:-4].replace('_', ' ') + '">'
        if (Path(directory) / (n[:-4] + '.svg')).exists():
            img = '<a href="' + n[:-4] + '.svg">' + img + '</a>'
        figs += '<figure>' + img + '<figcaption><code>' + html.escape(n) + '</code></figcaption></figure>'
    return head + figs


def render_pair(directory,comparison=None,surface_record=None,manual_record=None):
    directory=Path(directory);r=field_export(directory);s=read(directory/'plot_samples.npz')
    surface_record=surface_reference(directory,surface_record)
    quiver(directory,r,comparison);gradients(directory,s,comparison);profiles(directory,comparison,surface_record)
    field_panels(directory,s,surface_record)
    if manual_record is not None:
        manual_arrows(directory,manual_record)
        write_json(directory/'manual_comparison.json',
                   {k:v for k,v in manual_record.items() if not isinstance(v,np.ndarray)})
    notes={'piv_comparison':comparison['summary'] if comparison else None,'ir_surface':surface_record}
    write_json(directory/'comparison_notes.json',notes)
    comparison_links=''
    if comparison is not None:
        savemat(str(directory/'piv_comparison.mat'),{k:comparison[k] for k in ['velocity','gradients','integrals']},long_field_names=True,do_compression=True)
        quality=comparison['summary']['quality'];base='comparison/'+quote(quality,safe='')+'/'
        comparison_links='<p><strong>'+html.escape(quality_label(comparison))+'.</strong> No agreement with optical flow is used as a quality gate. '
        comparison_links+='<a href="piv_comparison.mat">PIV comparison MATLAB data</a> · <a href="'+base+'velocity.csv">Compared velocities</a> · <a href="'+base+'gradients.csv">Compared gradients</a> · <a href="'+base+'integrals.csv">Compared integral profiles</a></p>'
    if surface_record is not None:comparison_links+='<p><a href="surface_ir_reference.json">Exact IR field selection, mapping and timing record</a></p>'
    files=['quiver','gradient_comparison' if comparison else 'gradients','profiles']
    title=html.escape(directory.name)
    body=''.join('<figure><a href="'+f+'.svg"><img src="'+f+'.png" alt="'+f.replace('_',' ')+'"></a></figure>' for f in files)
    extra=_extra_field_panels(directory)
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+title+'</title><style>body{font:17px/1.6 system-ui;margin:2rem auto;max-width:1200px;padding:0 1rem;color:#14283b}img{width:100%;height:auto}figure{margin:2rem 0}a{color:#185ea0}code{background:#eef3f6;padding:.15em}footer{border-top:1px solid #ccc;margin-top:2rem}.mviewer{border:1px solid #dde3ea;border-radius:8px;padding:10px 13px;margin:1.5rem 0;background:#fafcfe}.mbar{display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-size:13px;margin-bottom:8px}.mviewer canvas{width:100%;height:auto;background:#fff;border:1px solid #e3e8ee;border-radius:6px}.mnote{font-size:13px;color:#4a5866}table{border-collapse:collapse;font-size:14px}th,td{padding:.35rem .7rem;border-bottom:1px solid #dde3ea;text-align:right}th:first-child,td:first-child{text-align:left}</style><h1>'+title+'</h1><p>Image-only optical flow. Missing estimates remain missing. PIV and IR data, when available, are comparisons applied after prediction.</p><p><a href="velocity_gradients.csv">Velocity/gradient CSV</a> · <a href="velocity_gradients.mat">MATLAB field</a> · <a href="horizontal_integral.mat">Image-only MATLAB integral profiles</a> · <a href="comparison_notes.json">Comparison settings and IR record</a></p>'+comparison_links+body+extra+manual_block(manual_record)+'<footer>Gradient colors use the 99th percentile; raw values remain in the numerical data. Click a plot to open its vector-format version.</footer></html>'
    (directory/'index.html').write_text(doc)

def render_batch(output,rows):
    output=Path(output)
    items=[]
    for row in rows:
        name=html.escape(row['pair']);status=html.escape(row['status']);error=html.escape(row.get('error',''))
        link='<a href="'+quote(row['pair'],safe='')+'/index.html">'+name+'</a>' if row['status']=='complete' or row.get('prediction_report_available',False) else name
        items.append('<tr><td>'+link+'</td><td>'+status+'</td><td>'+error+'</td></tr>')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>QuadraticOptical batch results</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:3rem auto;padding:1rem;color:#14283b}td,th{text-align:left;padding:.8rem;border-bottom:1px solid #ccd}a{color:#185ea0}</style><h1>Batch results</h1><p>Each image pair is estimated independently. Open a completed pair for velocities, gradients, coverage, and optional PIV/IR comparisons.</p><table><thead><tr><th>Pair</th><th>Status</th><th>Detail</th></tr></thead><tbody>'+''.join(items)+'</tbody></table></html>')
