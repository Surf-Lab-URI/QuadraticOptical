"""Compare frozen image-only estimates with a supplied native PIV field.

``compare_pair(directory, piv_path, quality='correlation', surface_records=None)``
reads the completed ``inputs.npz``, ``results.npz``, ``plot_samples.npz`` and
``integration_profile.npz``. It never initializes, fits, screens, or modifies the
image-only estimator. The returned mapping contains ``velocity``, ``gradients``,
``integrals``, ``summary`` and the output ``directory``. Numeric arrays are also
written to ``directory/comparison/<quality>/{velocity,gradients,integrals}.npz``;
CSV tables and a JSON summary accompany them.

Velocity arrays use ``query_px`` (source-image local, x right/y down), native
``of_disp_px``/``piv_disp_px`` and independent/common masks. Physical ``u`` is
rightward and ``w`` upward, in m/s. Gradient maps expose separate 2-D
``of_du_dx_per_s``, ``piv_du_dx_per_s``, ``common_of_du_dx_per_s``,
``common_piv_du_dx_per_s`` and ``difference_du_dx_per_s`` arrays, and analogous
``dw_dz`` fields. Their axes are ``x_axis_px`` and ``depth_axis_px``. Gradients
are Cartesian derivatives even when the display follows local surface depth.
Integral arrays contain the original and joint interval masks, width in metres,
integral in m^2/s, and width-averaged velocity in m/s at every saved ``depth_m``.

Default PIV quality requires finite correlation, finite vector components, an
optional supplied valid-water mask, and actual source-image availability.
``quality='finite'`` explicitly accepts saved finite velocities even if their
correlations are missing. Neither mode imposes a score cutoff or fills gaps.
Both compare the same native saved PIV, which may already be postprocessed by
its producer. Correlation is a quality-availability rule, not a calibrated
uncertainty. The supplied PIV is a comparison, not ground truth.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _read_piv(path):
    # Importing the estimator is deliberately unnecessary in this module.
    from .matio import read_native_piv
    return read_native_piv(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _value(mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise ValueError('Missing required saved array: ' + ' or '.join(names))


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def strict_bilinear(x_axis, y_axis, values, node_valid, query):
    """Interpolate only when every positive-weight corner is valid and finite.

    Exact grid nodes do not require their zero-weight neighbors. There is no
    extrapolation; invalid outputs are NaN. Values may have trailing components.
    """
    x = np.asarray(x_axis, float)
    y = np.asarray(y_axis, float)
    values = np.asarray(values, float)
    valid = np.asarray(node_valid, bool)
    q = np.asarray(query, float).reshape(-1, 2)
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2:
        raise ValueError('Native axes must be one-dimensional with at least two nodes.')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError('Native axes must be finite and strictly increasing.')
    if values.shape[:2] != (len(y), len(x)) or valid.shape != values.shape[:2]:
        raise ValueError('Native values and validity must have [y,x] leading dimensions.')
    trailing = values.shape[2:]
    flat = values.reshape(len(y), len(x), -1)
    inside = (np.isfinite(q).all(axis=1) & (q[:, 0] >= x[0]) &
              (q[:, 0] <= x[-1]) & (q[:, 1] >= y[0]) & (q[:, 1] <= y[-1]))
    sx = np.where(inside, q[:, 0], x[0])
    sy = np.where(inside, q[:, 1], y[0])
    ix = np.clip(np.searchsorted(x, sx, side='right') - 1, 0, len(x)-2)
    iy = np.clip(np.searchsorted(y, sy, side='right') - 1, 0, len(y)-2)
    tx = (sx-x[ix])/(x[ix+1]-x[ix])
    ty = (sy-y[iy])/(y[iy+1]-y[iy])
    weights = np.stack([(1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty], axis=1)
    cx = np.stack([ix, ix+1, ix, ix+1], axis=1)
    cy = np.stack([iy, iy, iy+1, iy+1], axis=1)
    corner = flat[cy, cx]
    positive = weights > 0
    supported = inside & np.all(~positive | (valid[cy, cx] & np.isfinite(corner).all(axis=2)), axis=1)
    output = np.sum(weights[:, :, None] * np.where(positive[:, :, None], corner, 0.), axis=1)
    output[~supported] = np.nan
    return output.reshape((len(q),) + trailing), supported


def _visible(mask, query):
    q = np.asarray(query, float).reshape(-1, 2)
    mask = np.asarray(mask, float)
    inside = (np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) &
              (q[:, 0] <= mask.shape[1]-1) & (q[:, 1] >= 0) &
              (q[:, 1] <= mask.shape[0]-1))
    result = np.zeros(len(q), bool)
    result[inside] = map_coordinates(mask, [q[inside, 1], q[inside, 0]],
                                    order=1, mode='constant', cval=0.) > .99
    return result


class NativeComparison:
    """A comparison-only native PIV field and its explicit quality policy.

    ``native`` follows ``matio.read_native_piv``. Its optional ``mask`` must be
    Boolean True for valid/retained water. ``geometry`` is frozen image-only
    input data. ``origin0`` maps local source-image coordinates to native image
    coordinates; surfaces and availability always remain local to the source.
    """
    def __init__(self, native, geometry, quality='correlation'):
        if quality not in ('correlation', 'finite'):
            raise ValueError("quality must be 'correlation' or 'finite'.")
        self.quality = quality
        self.x = np.asarray(native['x_px'], float)
        self.y = np.asarray(native['y_px'], float)
        self.disp = np.asarray(native['disp_px'], float)
        self.DX = float(geometry['DX'])
        self.DT = float(geometry['DT'])
        if self.DX <= 0 or self.DT <= 0 or not np.isfinite([self.DX, self.DT]).all():
            raise ValueError('DX and DT must be finite positive metres/pixel and seconds.')
        for key, value in [('DX', self.DX), ('DT', self.DT)]:
            if native.get(key) is not None and not np.isclose(float(native[key]), value, rtol=1e-10, atol=0.):
                raise ValueError('Native PIV '+key+' differs from the frozen image-only calibration.')
        self.origin = np.asarray(geometry.get('origin0', [0., 0.]), float).reshape(2)
        self.availability = np.asarray(_value(geometry, 'availability_a', 'va'))
        self.surface = np.asarray(geometry['surface_a'], float).ravel()
        if self.availability.ndim != 2 or len(self.surface) != self.availability.shape[1]:
            raise ValueError('Frozen surface_a and source availability shapes differ.')
        if self.disp.shape != (len(self.y), len(self.x), 2):
            raise ValueError('Native displacement must have shape [y,x,2].')
        if self.x.ndim != 1 or self.y.ndim != 1 or min(len(self.x),len(self.y)) < 2 or np.any(np.diff(self.x)<=0) or np.any(np.diff(self.y)<=0):
            raise ValueError('Native x/y axes must be strictly increasing.')
        self.dcor = native.get('dcor')
        if quality == 'correlation' and self.dcor is None:
            raise ValueError("Native PIV has no dcor correlation field. To compare its saved finite velocities explicitly, use quality='finite' (CLI: --piv-quality finite).")
        self.finite = np.isfinite(self.disp).all(axis=2)
        self.correlation_valid = np.ones(self.finite.shape, bool)
        if self.dcor is not None:
            self.dcor = np.asarray(self.dcor, float)
            if self.dcor.shape != self.finite.shape:
                raise ValueError('Native dcor shape differs from displacement.')
            self.correlation_valid = np.isfinite(self.dcor)
        mask = native.get('mask')
        self.supplied_mask = np.ones(self.finite.shape, bool)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != self.finite.shape or mask.dtype.kind != 'b':
                raise ValueError('matio must normalize the optional PIV mask to Boolean True=valid water.')
            self.supplied_mask = mask.copy()
        xx, yy = np.meshgrid(self.x, self.y)
        self.source_valid = self.source_available(np.c_[xx.ravel(), yy.ravel()]-self.origin).reshape(xx.shape)
        self.finite_available = self.finite & self.supplied_mask & self.source_valid
        self.valid = self.finite_available.copy()
        if quality == 'correlation':
            self.valid &= self.correlation_valid
        self.metadata = native.get('metadata', {})

    def source_available(self, query_local):
        q = np.asarray(query_local, float).reshape(-1, 2)
        under = q[:, 1] >= np.interp(q[:, 0], np.arange(len(self.surface)), self.surface)
        return _visible(self.availability, q) & under

    def sample(self, query_local):
        q = np.asarray(query_local, float).reshape(-1, 2)
        output, valid = strict_bilinear(self.x, self.y, self.disp, self.valid, q+self.origin)
        valid &= self.source_available(q)
        output[~valid] = np.nan
        return output, valid

    def sample_gradient(self, values, masks, query_local):
        q = np.asarray(query_local, float).reshape(-1, 2)
        output = np.full((len(q), 2), np.nan)
        valid = np.zeros((len(q), 2), bool)
        source = self.source_available(q)
        for component in range(2):
            output[:, component], available = strict_bilinear(
                self.x, self.y, values[..., component], masks[..., component], q+self.origin)
            valid[:, component] = available & source
            output[~valid[:, component], component] = np.nan
        return output, valid


def cartesian_gradient(displacement, valid, x_axis, y_axis, half_span_px=8.):
    """Native diagonal central differences with an exact +/-8px baseline.

    All native nodes between both endpoints and their interpolation brackets
    must be valid. On a 4px grid these are center, +/-4 and +/-8. Other strictly
    increasing spacings use linear endpoint interpolation without crossing a
    missing node. Derivatives are computed before any surface rectification.
    """
    d = np.asarray(displacement, float)
    if d.shape != (len(y_axis),len(x_axis),2) or np.asarray(valid).shape != d.shape[:2]:
        raise ValueError('Native displacement, validity and derivative axes differ.')
    valid = np.asarray(valid, bool) & np.isfinite(d).all(axis=2)
    span = float(half_span_px)
    if not np.isfinite(span) or span <= 0:
        raise ValueError('Derivative half-span must be finite and positive.')
    gradient = np.full(d.shape, np.nan)
    supported = np.zeros(d.shape, bool)
    for component, axis in [(0, np.asarray(x_axis,float)), (1, np.asarray(y_axis,float))]:
        if len(axis)<2 or np.any(np.diff(axis)<=0):
            raise ValueError('Derivative axes must be strictly increasing.')
        field = d[..., component] if component==0 else d[..., component].T
        good = valid if component==0 else valid.T
        gout = gradient[..., component] if component==0 else gradient[..., component].T
        vout = supported[..., component] if component==0 else supported[..., component].T
        for index, center in enumerate(axis):
            lo, hi = center-span, center+span
            if lo < axis[0] or hi > axis[-1]:
                continue
            left = int(np.searchsorted(axis,lo,side='right')-1)
            right = int(np.searchsorted(axis,hi,side='left'))
            all_nodes = good[:,left:right+1].all(axis=1)
            endpoint = []
            for position in [lo,hi]:
                base = int(np.clip(np.searchsorted(axis,position,side='right')-1,0,len(axis)-2))
                fraction = (position-axis[base])/(axis[base+1]-axis[base])
                value = (np.where(1-fraction>0,field[:,base],0.)*(1-fraction) +
                         np.where(fraction>0,field[:,base+1],0.)*fraction)
                endpoint.append(value)
            values = (endpoint[1]-endpoint[0])/(2*span)
            all_nodes &= np.isfinite(values)
            gout[:,index] = np.where(all_nodes,values,np.nan)
            vout[:,index] = all_nodes
    return gradient, supported


def _query(saved, origin):
    if 'query' in saved:
        return np.asarray(saved['query'],float)
    if 'query_px' in saved:
        return np.asarray(saved['query_px'],float)
    return np.asarray(_value(saved,'query_full'),float)-origin


def _diagonals(saved):
    gradient = np.asarray(_value(saved,'G','gradient'),float)
    values = np.stack([gradient[:,0,0],gradient[:,1,1]],axis=1)
    masks = np.stack([_value(saved,'accepted_du_dx','gradient_accepted_xx'),
                      _value(saved,'accepted_dw_dz','gradient_accepted_yy')],axis=1).astype(bool)
    return values, masks


def saved_cartesian_difference(query, displacement, accepted, half_span_px=8.):
    """Matched operator using only saved center and exact +/-8px OF neighbors."""
    q = np.asarray(query,float)
    d = np.asarray(displacement,float)
    accepted = np.asarray(accepted,bool)
    key = lambda p: tuple(np.round(p,10))
    lookup = {key(p):index for index,p in enumerate(q)}
    if len(lookup)!=len(q):
        raise ValueError('Saved report coordinates must be unique.')
    values = np.full((len(q),2),np.nan)
    valid = np.zeros((len(q),2),bool)
    neighbors = np.full((len(q),2,2),-1,int)
    for component in range(2):
        offset = np.zeros(2);offset[component]=half_span_px
        for index,point in enumerate(q):
            minus = lookup.get(key(point-offset),-1)
            plus = lookup.get(key(point+offset),-1)
            neighbors[index,component]=[minus,plus]
            if minus<0 or plus<0:
                continue
            indices=[minus,index,plus]
            if accepted[indices].all() and np.isfinite(d[indices]).all():
                values[index,component]=(d[plus,component]-d[minus,component])/(2*half_span_px)
                valid[index,component]=True
    return values,valid,neighbors


def _metrics(first,second,valid):
    a=np.asarray(first);b=np.asarray(second);valid=np.array(valid,bool,copy=True)
    if a.ndim>valid.ndim:
        valid &= np.isfinite(a).all(axis=-1)&np.isfinite(b).all(axis=-1)
    else:
        valid &= np.isfinite(a)&np.isfinite(b)
    a,b=a[valid],b[valid]
    if not len(a):
        return {'count':0}
    delta=a-b
    magnitude=np.linalg.norm(delta,axis=-1) if delta.ndim>1 else np.abs(delta)
    result=dict(count=int(len(a)),mean_difference=np.mean(delta,axis=0),
        median_absolute_difference=float(np.median(magnitude)),
        mean_absolute_difference=float(np.mean(magnitude)),
        rms_difference=float(np.sqrt(np.mean(magnitude*magnitude))),
        absolute_difference_p90=float(np.percentile(magnitude,90)))
    if a.ndim==1:
        result['pearson_correlation']=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None
    return _jsonable(result)


def _velocity_comparison(native,results):
    q=_query(results,native.origin)
    of=np.asarray(results['disp'],float)
    of_ok=np.asarray(results['accepted'],bool)&np.isfinite(of).all(axis=1)
    piv,piv_ok=native.sample(q)
    common=of_ok&piv_ok
    out=dict(query_px=q,native_query_px=q+native.origin,of_disp_px=of,piv_disp_px=piv,
             of_available=of_ok,piv_available=piv_ok,common_available=common,
             displacement_difference_px=np.where(common[:,None],of-piv,np.nan),
             DX=np.array(native.DX),DT=np.array(native.DT))
    for label,disp,mask in [('of',of,of_ok),('piv',piv,piv_ok)]:
        out[label+'_u_m_per_s']=np.where(mask,disp[:,0]*native.DX/native.DT,np.nan)
        out[label+'_w_m_per_s']=np.where(mask,-disp[:,1]*native.DX/native.DT,np.nan)
    return out,dict(displacement_px=_metrics(of,piv,common),
        velocity_m_per_s=_metrics(of*native.DX/native.DT,piv*native.DX/native.DT,common),
        of_available=int(of_ok.sum()),piv_available=int(piv_ok.sum()),common_available=int(common.sum()))


def _gradient_comparison(native,results,samples):
    q=_query(samples,native.origin)
    native_g,native_good=cartesian_gradient(native.disp,native.valid,native.x,native.y)
    piv,piv_good=native.sample_gradient(native_g,native_good,q)
    of,of_good=_diagonals(samples)
    common=of_good&piv_good&np.isfinite(of)
    x=np.asarray(samples.get('x_axis_px',[]),float)
    h=np.asarray(samples.get('depth_axis_px',[]),float)
    shape=(len(h),len(x)) if len(h)*len(x)==len(q) and len(q)>0 else (len(q),)
    out=dict(query_px=q,native_query_px=q+native.origin,x_axis_px=x,depth_axis_px=h,
        x_axis_m=(x+native.origin[0]+.5)*native.DX,depth_axis_m=h*native.DX,
        plot_shape=np.array(shape),of_gradient_per_pair=of,piv_gradient_per_pair=piv,
        of_valid=of_good,piv_valid=piv_good,common_valid=common,
        native_x_px=native.x,native_y_px=native.y,
        native_gradient_per_pair=native_g,native_gradient_valid=native_good,
        component_names=np.array(['du_dx','dw_dz']),DX=np.array(native.DX),DT=np.array(native.DT),
        derivative_half_span_px=np.array(8.),derivative_full_span_px=np.array(16.))
    labels={'of':np.where(of_good,of/native.DT,np.nan),
            'piv':np.where(piv_good,piv/native.DT,np.nan),
            'common_of':np.where(common,of/native.DT,np.nan),
            'common_piv':np.where(common,piv/native.DT,np.nan),
            'difference':np.where(common,(of-piv)/native.DT,np.nan)}
    for index,name in enumerate(['du_dx','dw_dz']):
        for label,array in labels.items():
            out[label+'_'+name+'_per_s']=array[:,index].reshape(shape)
    rq=_query(results,native.origin)
    fd,fd_good,neighbors=saved_cartesian_difference(rq,results['disp'],results['accepted'])
    pg,pg_good=native.sample_gradient(native_g,native_good,rq)
    ag,ag_good=_diagonals(results)
    matched=fd_good&pg_good
    stricter=matched&ag_good&np.isfinite(ag)
    out.update(matched_query_px=rq,matched_of_gradient_per_pair=fd,
        matched_piv_gradient_per_pair=pg,matched_of_valid=fd_good,matched_piv_valid=pg_good,
        matched_valid=matched,matched_neighbor_indices=neighbors,matched_and_analytic_valid=stricter,
        matched_of_gradient_per_s=np.where(matched,fd/native.DT,np.nan),
        matched_piv_gradient_per_s=np.where(matched,pg/native.DT,np.nan),
        reporting_analytic_of_gradient_per_pair=ag)
    metrics={}
    for index,name in enumerate(['du_dx','dw_dz']):
        metrics[name]=dict(analytic_of_vs_piv_per_s=_metrics(of[:,index]/native.DT,piv[:,index]/native.DT,common[:,index]),
            matched_operator_per_s=_metrics(fd[:,index]/native.DT,pg[:,index]/native.DT,matched[:,index]),
            analytic_on_matched_and_analytic_support_per_s=_metrics(ag[:,index]/native.DT,pg[:,index]/native.DT,stricter[:,index]),
            matched_on_matched_and_analytic_support_per_s=_metrics(fd[:,index]/native.DT,pg[:,index]/native.DT,stricter[:,index]))
    return out,metrics


def interval_mask(nodes):
    nodes=np.asarray(nodes,bool)
    if nodes.ndim!=2 or nodes.shape[1]<3 or nodes.shape[1]%2!=1:
        raise ValueError('Simpson samples must be [depth,2*intervals+1].')
    return nodes[:,:-2:2]&nodes[:,1::2]&nodes[:,2::2]


def _simpson(values,width):
    return width[None,:]*(values[:,:-2:2]+4*values[:,1::2]+values[:,2::2])/6


def _supported_sum(values,mask):
    total=np.sum(np.where(mask,values,0.),axis=1)
    total[~mask.any(axis=1)]=np.nan
    return total


def domain_integral(values,mask,interval_width_m):
    """Integrate only accepted Simpson intervals; empty domains have NaN Q/u."""
    values=np.asarray(values,float);mask=np.asarray(mask,bool);width=np.asarray(interval_width_m,float)
    if mask.shape!=(len(values),(values.shape[1]-1)//2) or len(width)!=mask.shape[1] or np.any(width<=0):
        raise ValueError('Interval masks/widths do not match the Simpson samples.')
    mask=mask&interval_mask(np.isfinite(values))
    total=_supported_sum(_simpson(values,width),mask)
    observed=np.sum(width[None,:]*mask,axis=1)
    mean=np.divide(total,observed,out=np.full(len(values),np.nan),where=observed>0)
    full=np.where(mask.all(axis=1),total,np.nan)
    return dict(integral_m2_per_s=total,width_m=observed,mean_u_m_per_s=mean,
                coverage_fraction=observed/width.sum(),full_width_integral_m2_per_s=full)


def _integral_comparison(native,profile):
    h=np.asarray(profile['depth_m'],float)
    if h.ndim!=1 or not len(h) or not np.isfinite(h).all() or np.any(h<0):
        raise ValueError('Saved depths must be a nonempty finite nonnegative vector.')
    x=np.asarray(_value(profile,'horizontal_sample_x_px','x_sample_px'),float)
    if x.ndim!=1 or len(x)<3 or len(x)%2!=1 or np.any(np.diff(x)<=0):
        raise ValueError('Saved horizontal Simpson samples must have odd length and increase.')
    if not np.allclose(x[1::2],(x[:-2:2]+x[2::2])/2,rtol=0,atol=1e-9):
        raise ValueError('Saved integration samples are not endpoint/midpoint Simpson intervals.')
    y=np.asarray(profile.get('sample_query_y_px',h[:,None]/native.DX+
        np.interp(x,np.arange(len(native.surface)),native.surface)[None,:]),float)
    if y.shape!=(len(h),len(x)):
        raise ValueError('Saved depth/query-y arrays have incompatible shapes.')
    if 'sample_displacement_px' in profile:
        of=np.asarray(profile['sample_displacement_px'],float)[...,0]*native.DX/native.DT
    else:
        of=np.asarray(_value(profile,'sample_primary_u_m_per_s','sample_primary_u_assumed_m_per_s'),float)
    of_nodes=np.asarray(profile['sample_accepted'],bool)&np.isfinite(of)
    if of.shape!=y.shape or of_nodes.shape!=y.shape:
        raise ValueError('Frozen integration velocities and query geometry differ.')
    width=(x[2::2]-x[:-2:2])*native.DX
    if 'horizontal_interval_width_m' in profile and not np.allclose(profile['horizontal_interval_width_m'],width,rtol=1e-12,atol=1e-15):
        raise ValueError('Frozen physical interval widths differ from DX and pixel coordinates.')
    query=np.stack([np.broadcast_to(x,y.shape),y],axis=-1).reshape(-1,2)
    disp,piv_nodes=native.sample(query)
    piv=disp[:,0].reshape(y.shape)*native.DX/native.DT
    piv_nodes=piv_nodes.reshape(y.shape)&np.isfinite(piv)
    of_mask=interval_mask(of_nodes);piv_mask=interval_mask(piv_nodes);matched=of_mask&piv_mask
    if 'interval_accepted' in profile and not np.array_equal(profile['interval_accepted'],of_mask):
        raise ValueError('Frozen OF interval mask does not match its accepted samples.')
    bounds=np.asarray(profile.get('common_depth_band_requested_m',[20*native.DX,h.max()]),float)
    band=(h>=bounds[0]-1e-15)&(h<=bounds[1]+1e-15)
    common=matched[band].all(axis=0) if band.any() else np.zeros(matched.shape[1],bool)
    common_width=float(width[common].sum())
    common_available=(common_width>0)&matched[:,common].all(axis=1)
    fixed=np.broadcast_to(common,matched.shape)&common_available[:,None]
    out=dict(depth_m=h,x_sample_px=x,sample_query_y_px=y,interval_bounds_px=np.c_[x[:-2:2],x[2::2]],
        interval_width_m=width,target_full_width_m=np.array(width.sum()),DX=np.array(native.DX),DT=np.array(native.DT),
        of_sample_u_m_per_s=of,piv_sample_u_m_per_s=piv,
        of_sample_available=of_nodes,piv_sample_available=piv_nodes,
        of_original_interval_mask=of_mask,piv_own_interval_mask=piv_mask,matched_interval_mask=matched,
        common_interval_mask=common,common_available_at_depth=common_available,
        common_depth_band_m=np.array([h[band].min(),h[band].max()]) if band.any() else np.array([np.nan,np.nan]),
        common_width_m=np.array(common_width),common_coverage_fraction=np.array(common_width/width.sum()),
        all_requested_depth_common_width_m=np.array(width[matched.all(axis=0)].sum()))
    for domain,mask in [('matched',matched),('common',fixed),('original',of_mask),('own',piv_mask)]:
        sides=[('of',of),('piv',piv)] if domain in ('matched','common') else [('of',of)] if domain=='original' else [('piv',piv)]
        for side,values in sides:
            stats=domain_integral(values,mask,width)
            for key,value in stats.items():
                out[side+'_'+domain+'_'+key]=value
        if domain in ('matched','common'):
            out[domain+'_integral_difference_m2_per_s']=out['of_'+domain+'_integral_m2_per_s']-out['piv_'+domain+'_integral_m2_per_s']
            out[domain+'_mean_difference_m_per_s']=out['of_'+domain+'_mean_u_m_per_s']-out['piv_'+domain+'_mean_u_m_per_s']
    out['matched_width_m']=out['of_matched_width_m']
    out['matched_coverage_fraction']=out['of_matched_coverage_fraction']
    frozen_q=profile.get('covered_segment_integral_m2_per_s',profile.get('covered_segment_integral_assumed_m2_per_s'))
    if frozen_q is not None and not np.allclose(out['of_original_integral_m2_per_s'],frozen_q,rtol=1e-12,atol=1e-15,equal_nan=True):
        raise ValueError('Frozen OF integral differs under identical accepted Simpson intervals.')
    return out,dict(depth_count=len(h),interval_count=len(width),target_full_width_m=float(width.sum()),
        common_width_m=common_width,common_coverage_fraction=common_width/width.sum(),
        common_depth_band_m=out['common_depth_band_m'],
        all_requested_depth_common_width_m=float(out['all_requested_depth_common_width_m']))


def _write_csv(path,columns):
    columns={key:np.asarray(value).reshape(-1) for key,value in columns.items()}
    sizes={len(value) for value in columns.values()}
    if len(sizes)!=1:
        raise ValueError('Tabular columns must have the same number of rows.')
    with Path(path).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def compare_pair(directory,piv_path,quality='correlation',surface_records=None):
    """Export held-out PIV comparisons without changing the frozen estimator.

    ``surface_records`` are optional surface-velocity annotations, retained as
    metadata for plotting only; they never replace frozen geometry or gates.
    ``quality='correlation'`` rejects NaN/inf dcor but applies no score cutoff.
    ``quality='finite'`` is an explicit saved-finite-velocity comparison.
    Returns in-memory arrays as documented in this module's high-level API.
    """
    directory=Path(directory);piv_path=Path(piv_path)
    required={key:directory/(key+'.npz') for key in ['inputs','results','plot_samples','integration_profile']}
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError('Complete the image-only prediction and export first; missing '+str(path))
    immutable=list(required.values())+[directory/(name+'.npz') for name in
        ['main','affine','large_window','margin14','reverse','ptv_tracks'] if (directory/(name+'.npz')).is_file()]
    before={str(path):_sha(path) for path in immutable}
    arrays={key:_read_npz(path) for key,path in required.items()}
    inputs=arrays['inputs']
    if 'supplied_velocity_used' in inputs and bool(inputs['supplied_velocity_used']):
        raise ValueError('This API requires frozen image-only predictions; supplied_velocity_used is true.')
    if 'image_only' in inputs and not bool(inputs['image_only']):
        raise ValueError('This API requires frozen image-only inputs.')
    for name in ['results','plot_samples','integration_profile']:
        for key in ['DX','DT']:
            if key in arrays[name] and not np.isclose(float(arrays[name][key]),float(inputs[key]),rtol=1e-12,atol=0):
                raise ValueError('Frozen '+name+' calibration differs from inputs.')
    native_data=_read_piv(piv_path)
    native=NativeComparison(native_data,inputs,quality=quality)
    velocity,velocity_stats=_velocity_comparison(native,arrays['results'])
    gradients,gradient_stats=_gradient_comparison(native,arrays['results'],arrays['plot_samples'])
    integrals,integral_stats=_integral_comparison(native,arrays['integration_profile'])
    after={str(path):_sha(path) for path in immutable}
    if after!=before:
        raise RuntimeError('A frozen image-only artifact changed during comparison.')
    output=directory/'comparison'/quality
    summary=_jsonable(dict(quality=quality,comparison_only=True,image_only_artifacts_unchanged=True,
        PIV_used_in_prediction=False,refitting_performed=False,quality_cutoff=None,
        quality_rule='Finite dcor, finite vector, optional supplied valid-water mask and actual source availability.' if quality=='correlation' else 'Saved finite vector, optional supplied valid-water mask and actual source availability; dcor not required.',
        native_counts=dict(finite_vectors=int(native.finite.sum()),source_and_mask_finite=int(native.finite_available.sum()),
            selected_available=int(native.valid.sum()),finite_vector_without_finite_dcor=int((native.finite_available&~native.correlation_valid).sum())),
        native_metadata=native_data.get('metadata',{}),native_path=str(piv_path),native_sha256=_sha(piv_path),
        frozen_sha256=before,comparison_code_sha256=_sha(__file__),DX=native.DX,DT=native.DT,
        units='DX in metres/pixel; DT in seconds; velocity m/s; diagonal gradients 1/s; horizontal integral m^2/s.',
        coordinate_rule='Native MAT coordinates are zero-based x-right/y-down. Local frozen source-image queries add origin0 when sampling native PIV. u=dx*DX/DT; w=-dy*DX/DT; du/dx=G00/DT; dw/dz=G11/DT.',
        interpolation='Exact native nodes when coincident, otherwise bilinear interpolation requiring every positive-weight valid native contributor and valid query source. No extrapolation or filling of missing values.',
        gradient_operator='Main OF derivatives are saved analytic Cartesian derivatives with original separate component masks. PIV central differences use +/-8px endpoints, all intervening native nodes and endpoint brackets valid; native derivatives are then strictly sampled at OF plot positions. A separate matched-operator audit uses only saved accepted OF center and exact +/-8px neighbors.',
        gradient_scope='Native stencils may use valid points outside the displayed depth band to estimate an in-band derivative. The matched saved-OF operator requires neighbors within its saved reporting domain.',
        integration_rule='Frozen endpoint/midpoint Simpson intervals are preserved, nominally 2px with a possible shorter final interval. Joint intervals require both methods valid at all three sample positions. Integrate u dx horizontally, without an arc-length factor. Missing intervals are excluded, never filled; an empty-domain integral and mean are NaN, width is zero.',
        common_domain_rule='Fixed joint support is the intersection across the frozen requested common-depth band; if the band is absent it begins 20 pixels below the local surface. No fixed-domain value is reported where any interval of that domain is unavailable.',
        comparison_interpretation='Native saved PIV may include its producer postprocessing. It is held out of image-only fitting but is not independent ground truth. No OF-PIV difference gate is applied.',
        surface_records=surface_records,velocity=velocity_stats,gradients=gradient_stats,integrals=integral_stats))
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('velocity',velocity),('gradients',gradients),('integrals',integrals)]:
        np.savez_compressed(output/(name+'.npz'),**data)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    _write_csv(output/'velocity.csv',dict(x_px=velocity['query_px'][:,0],y_px=velocity['query_px'][:,1],
        of_u_m_per_s=velocity['of_u_m_per_s'],of_w_m_per_s=velocity['of_w_m_per_s'],
        piv_u_m_per_s=velocity['piv_u_m_per_s'],piv_w_m_per_s=velocity['piv_w_m_per_s'],
        of_available=velocity['of_available'],piv_available=velocity['piv_available'],common_available=velocity['common_available']))
    columns=dict(x_px=gradients['query_px'][:,0],y_px=gradients['query_px'][:,1])
    for index,name in enumerate(['du_dx','dw_dz']):
        for label in ['of','piv','common_of','common_piv','difference']:
            columns[label+'_'+name+'_per_s']=gradients[label+'_'+name+'_per_s']
        columns['common_'+name+'_valid']=gradients['common_valid'][:,index]
    _write_csv(output/'gradients.csv',columns)
    profile_keys=['depth_m','matched_width_m','matched_coverage_fraction',
        'matched_integral_difference_m2_per_s','matched_mean_difference_m_per_s',
        'common_available_at_depth','common_integral_difference_m2_per_s','common_mean_difference_m_per_s']
    profile_keys += [side+'_'+domain+'_'+stat for side,domain in
        [('of','matched'),('piv','matched'),('of','common'),('piv','common'),('of','original'),('piv','own')]
        for stat in ['integral_m2_per_s','width_m','coverage_fraction','mean_u_m_per_s','full_width_integral_m2_per_s']]
    _write_csv(output/'integrals.csv',{key:integrals[key] for key in profile_keys})
    return dict(directory=output,velocity=velocity,gradients=gradients,integrals=integrals,summary=summary)
