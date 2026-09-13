"""Held-out native-PIV gradient comparison; never fits or changes optical flow.

The main comparison pairs the saved analytic image-only gradients with native
PIV Cartesian central differences over a 16-pixel baseline, strictly sampled
at the saved surface-following plotting positions. A separate audit applies
that same finite-difference operator to both saved displacement fields.
"""
from pathlib import Path
import argparse
import json
import numpy as np

from native_piv import NativePIV, strict_bilinear, visible, depth, sha

HERE = Path(__file__).resolve().parent
IMAGE_ONLY = HERE.parent / 'image_only_cm'
COMPONENTS = ['du_dx', 'dw_dz']


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def native_diagonal_difference(displacement, valid, native_spacing=4., half_span=8.):
    """Return (G00,G11), with all intervening native nodes required valid.

    G11 is d(dy_image)/d(y_image); flipping both vertical coordinates and
    velocities leaves this diagonal derivative unchanged.
    """
    d = np.asarray(displacement, float)
    valid = np.asarray(valid, bool)
    m = int(round(half_span / native_spacing))
    if m < 1 or not np.isclose(m * native_spacing, half_span):
        raise ValueError('Half-span must be an integer native-grid multiple.')
    if d.shape[:2] != valid.shape or d.shape[2:] != (2,):
        raise ValueError('Expected [y,x,2] displacement and [y,x] validity.')
    g = np.full(d.shape, np.nan)
    good = np.zeros(d.shape, bool)
    # Both components must be finite for a supplied vector to be available.
    valid = valid & np.isfinite(d).all(axis=2)
    for component, axis in [(0, 1), (1, 0)]:
        center = [slice(None), slice(None)]
        center[axis] = slice(m, d.shape[axis] - m)
        center = tuple(center)
        stencil_ok = np.ones(valid[center].shape, bool)
        for offset in range(-m, m + 1):
            loc = [slice(None), slice(None)]
            loc[axis] = slice(m + offset, d.shape[axis] - m + offset)
            stencil_ok &= valid[tuple(loc)]
        plus = [slice(None), slice(None)]
        minus = [slice(None), slice(None)]
        plus[axis] = slice(2*m, None)
        minus[axis] = slice(None, -2*m)
        values = (d[tuple(plus)][..., component] -
                  d[tuple(minus)][..., component]) / (2 * half_span)
        g[..., component][center] = np.where(stencil_ok, values, np.nan)
        good[..., component][center] = stencil_ok
    return g, good


def sample_gradient(native, gradients, validity, query):
    """Sample each component with its own stencil mask and no gap bridging."""
    q = np.asarray(query, float).reshape(-1, 2)
    output = np.full((len(q), 2), np.nan)
    good = np.zeros((len(q), 2), bool)
    query_source = (visible(native.geometry['availability_a'], q) &
                    (depth(native.geometry['surface_a'], q) >= 0))
    for component in range(2):
        output[:, component], corners = strict_bilinear(
            native.x, native.y, gradients[..., component],
            validity[..., component], q)
        good[:, component] = corners & query_source
        output[~good[:, component], component] = np.nan
    return output, good


def reporting_diagonal_difference(results, half_span=8.):
    """Same +/-8 px operator on existing accepted Cartesian report-grid values.

    The sparse saved report grid is 8 pixels apart. Missing neighbors, including
    those outside the saved top-centimetre domain, remain unsupported.
    """
    q = np.asarray(results['query'], float)
    d = np.asarray(results['disp'], float)
    accepted = np.asarray(results['accepted'], bool)
    if not np.all(q == np.round(q)):
        raise ValueError('Reporting-grid coordinates must be exact integers.')
    lookup = {tuple(p): i for i, p in enumerate(q.astype(int))}
    if len(lookup) != len(q):
        raise ValueError('Duplicate reporting-grid coordinates.')
    g = np.full((len(q), 2), np.nan)
    good = np.zeros((len(q), 2), bool)
    neighbors = np.full((len(q), 2, 2), -1, int)
    for component in range(2):
        offset = np.zeros(2, int)
        offset[component] = int(half_span)
        if offset[component] != half_span:
            raise ValueError('Expected integer reporting-grid half-span.')
        for i, point in enumerate(q.astype(int)):
            minus = lookup.get(tuple(point-offset), -1)
            plus = lookup.get(tuple(point+offset), -1)
            neighbors[i, component] = [minus, plus]
            if minus < 0 or plus < 0:
                continue
            indices = [minus, i, plus]
            if accepted[indices].all() and np.isfinite(d[indices]).all():
                g[i, component] = (d[plus, component]-d[minus, component])/(2*half_span)
                good[i, component] = True
    return g, good, neighbors


def scalar_metrics(a, b, valid, dt):
    """Scalar differences OF minus PIV; comparison is not ground-truth error."""
    use = np.asarray(valid, bool) & np.isfinite(a) & np.isfinite(b)
    aa, bb = np.asarray(a)[use], np.asarray(b)[use]
    if not len(aa):
        return {'count': 0}
    delta = aa-bb
    corr = None
    if len(aa)>1 and np.std(aa)>0 and np.std(bb)>0:
        corr = float(np.corrcoef(aa,bb)[0,1])
    base = dict(count=int(len(aa)), pearson_correlation=corr)
    for scale, label in [(1., 'per_pair'), (1./dt, 'assumed_per_s')]:
        dif = delta*scale
        base[label] = dict(mean_of=float(np.mean(aa)*scale),
            mean_piv=float(np.mean(bb)*scale), signed_mean_difference=float(np.mean(dif)),
            median_difference=float(np.median(dif)), mean_absolute_difference=float(np.mean(np.abs(dif))),
            median_absolute_difference=float(np.median(np.abs(dif))),
            root_mean_square_difference=float(np.sqrt(np.mean(dif*dif))),
            absolute_difference_p90=float(np.percentile(np.abs(dif),90)))
    return base


def run_pair(pair):
    folder = IMAGE_ONLY / str(pair)
    source_paths = [folder/'results.npz', folder/'plot_samples.npz', folder/'inputs.npz']
    before = {str(p): sha(p) for p in source_paths}
    r = read_npz(source_paths[0]); s = read_npz(source_paths[1])
    native = NativePIV(pair)
    if not np.all(np.diff(native.x)==4) or not np.all(np.diff(native.y)==4):
        raise ValueError('Expected native PIV grid spacing of 4 px.')
    if float(s['DX']) != native.DX or float(s['DT']) != native.DT:
        raise ValueError('Comparison calibration mismatch.')
    g, gv = native_diagonal_difference(native.disp, native.valid_source)
    piv, piv_ok = sample_gradient(native, g, gv, s['query'])
    of = np.stack([s['gradient'][:,0,0],s['gradient'][:,1,1]],axis=1)
    of_ok = np.stack([s['gradient_accepted_xx'],s['gradient_accepted_yy']],axis=1)
    common = of_ok & piv_ok & np.isfinite(of)
    of_fd, of_fd_ok, neighbors = reporting_diagonal_difference(r)
    piv_fd, piv_fd_ok = sample_gradient(native, g, gv, r['query'])
    matched = of_fd_ok & piv_fd_ok
    analytic_report = np.stack([r['gradient'][:,0,0],r['gradient'][:,1,1]],axis=1)
    analytic_report_ok = np.stack([r['gradient_accepted_xx'],r['gradient_accepted_yy']],axis=1)
    matched_and_analytic = matched & analytic_report_ok
    nx = len(s['x_axis_px']); nh = len(s['depth_axis_px'])
    data = dict(query=s['query'], x_axis_px=s['x_axis_px'], depth_axis_px=s['depth_axis_px'],
        x_axis_assumed_cm=(s['x_axis_px']+.5)*native.DX*100,
        depth_axis_assumed_cm=s['depth_axis_px']*native.DX*100,
        plot_shape=np.array([nh,nx]), component_names=np.array(COMPONENTS),
        DX=np.array(native.DX), DT=np.array(native.DT), pair=np.array(pair),
        of_gradient_per_pair=of, of_gradient_valid=of_ok,
        piv_gradient_per_pair=piv, piv_gradient_valid=piv_ok,
        common_gradient_valid=common,
        of_gradient_assumed_per_s=np.where(of_ok,of/native.DT,np.nan),
        piv_gradient_assumed_per_s=np.where(piv_ok,piv/native.DT,np.nan),
        common_of_gradient_assumed_per_s=np.where(common,of/native.DT,np.nan),
        common_piv_gradient_assumed_per_s=np.where(common,piv/native.DT,np.nan),
        gradient_difference_assumed_per_s=np.where(common,(of-piv)/native.DT,np.nan),
        native_x_px=native.x, native_y_px=native.y,
        native_gradient_per_pair=g, native_gradient_valid=gv,
        matched_query=r['query'], matched_of_gradient_per_pair=of_fd,
        matched_piv_gradient_per_pair=piv_fd, matched_of_valid=of_fd_ok,
        matched_piv_valid=piv_fd_ok, matched_valid=matched,
        matched_neighbor_indices=neighbors,
        matched_of_gradient_assumed_per_s=np.where(matched,of_fd/native.DT,np.nan),
        matched_piv_gradient_assumed_per_s=np.where(matched,piv_fd/native.DT,np.nan),
        matched_difference_assumed_per_s=np.where(matched,(of_fd-piv_fd)/native.DT,np.nan),
        reporting_analytic_of_gradient_per_pair=analytic_report,
        matched_and_analytic_valid=matched_and_analytic,
        native_spacing_px=np.array(4.), derivative_half_span_px=np.array(8.),
        derivative_full_span_px=np.array(16.), supplied_piv_used_for_prediction=np.array(False),
        comparison_only=np.array(True), units_are_assumed=np.array(True))
    # Explicit 2-D plot-ready fields avoid component/reshape ambiguity for figures.
    for component, name in enumerate(COMPONENTS):
        for label, array in [('of',data['of_gradient_assumed_per_s']),
                             ('piv',data['piv_gradient_assumed_per_s']),
                             ('common_of',data['common_of_gradient_assumed_per_s']),
                             ('common_piv',data['common_piv_gradient_assumed_per_s']),
                             ('difference',data['gradient_difference_assumed_per_s'])]:
            data[label+'_'+name+'_assumed_per_s'] = array[:,component].reshape(nh,nx)
    metadata = dict(pair=pair, comparison_only=True, supplied_piv_used_for_prediction=False,
        model_recomputed=False, calibration=dict(DX=native.DX,DT=native.DT,
            physical_unit_assumption='DX is metres/pixel; DT is seconds. MAT numeric fields have no verified unit labels.'),
        native_metadata=native.metadata, component_order=COMPONENTS,
        native_operator='Cartesian centered [d(x+8)-d(x-8)]/16 for G00 and [dy(y+8)-dy(y-8)]/16 for G11. Native spacing 4 px; every center and intermediate +/-4,+/-8 node must be source-valid and have both finite supplied displacement components.',
        source_validity='Finite native supplied vector, actual retained source-image availability >0.99 under bilinear sampling, and source below inferred geometric surface. No dcor or target-visibility filter in the primary comparison.',
        rectified_sampling='Differentiate on Cartesian native grid first, then strict positive-weight bilinear interpolation at saved OF plotting positions. Every contributing native derivative node must pass its own component stencil; query source must be visible and underwater. No extrapolation or interpolation across missing native nodes.',
        primary_of_operator='Saved analytic Cartesian derivatives of the existing blended quadratic field, using the unchanged separate G00 and G11 conservative masks. Display coordinates follow local surface depth, but derivatives are not taken along constant local depth.',
        vertical_sign='Physical w=-dy*DX/DT and z=-y*DX imply dw/dz=G11/DT; signs cancel.',
        derivative_support='The PIV stencil can include valid nodes beyond the displayed top-centimetre boundary to estimate a derivative at a displayed center. No source predictions are extended. Matched OF finite differences require saved center and both +/-8px Cartesian neighbors, all accepted; unavailable saved-domain neighbors are omitted.',
        matched_operator='The same +/-8px central difference and 16px denominator are applied to both displacements at existing exact Cartesian report-grid nodes. OF center and neighbors must be accepted; PIV requires all five native nodes on that axis. Separate componentwise common masks.',
        interpretation='PIV is a held-out comparison, not a velocity or gradient ground truth. Main operator regularization differs; matched-operator results isolate part of that distinction. No comparison discrepancy threshold is used.',
        requested_domain='Full image width, 0 to 1 cm below local inferred source surface under the stated calibration. Original image-only acceptance masks are unchanged.',
        plot_shape=[nh,nx], counts={}, metrics={},
        source_sha256=dict(before,**{str(native.mat_path):sha(native.mat_path)}),
        code_sha256={str(Path(__file__)):sha(__file__),str(HERE/'native_piv.py'):sha(HERE/'native_piv.py')})
    for component, name in enumerate(COMPONENTS):
        metadata['counts'][name]=dict(optical_flow_rectified=int(of_ok[:,component].sum()),
            piv_rectified=int(piv_ok[:,component].sum()), common_rectified=int(common[:,component].sum()),
            matched_operator=int(matched[:,component].sum()),
            matched_and_analytic=int(matched_and_analytic[:,component].sum()))
        metadata['metrics'][name]=dict(
            analytic_of_vs_piv=scalar_metrics(of[:,component],piv[:,component],common[:,component],native.DT),
            matched_operator=scalar_metrics(of_fd[:,component],piv_fd[:,component],matched[:,component],native.DT),
            analytic_of_vs_piv_on_matched_and_analytic_support=scalar_metrics(analytic_report[:,component],piv_fd[:,component],matched_and_analytic[:,component],native.DT),
            matched_operator_on_matched_and_analytic_support=scalar_metrics(of_fd[:,component],piv_fd[:,component],matched_and_analytic[:,component],native.DT))
    after = {str(p):sha(p) for p in source_paths}
    if before != after:
        raise RuntimeError('A scientific source changed during comparison.')
    metadata['scientific_sources_unchanged']=True
    np.savez_compressed(HERE/('pair_%s_gradient_comparison.npz'%pair),**data)
    (HERE/('pair_%s_gradient_comparison.json'%pair)).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(pair=pair,counts=metadata['counts'],scientific_sources_unchanged=True)),flush=True)
    return metadata


def self_test():
    x=np.arange(3.,84.,4.);y=np.arange(3.,84.,4.);xx,yy=np.meshgrid(x,y)
    d=np.stack([.2+.03*xx+.005*xx**2+.2*yy,
                -.4+.1*xx-.02*yy+.003*yy**2],axis=2)
    valid=np.ones(xx.shape,bool)
    g,good=native_diagonal_difference(d,valid)
    expected=np.stack([.03+.01*xx,-.02+.006*yy],axis=2)
    error=float(np.max(np.abs(g[good]-expected[good])))
    assert error<1e-13
    valid[9,8]=False
    _,holes=native_diagonal_difference(d,valid)
    assert not holes[9,6:11,0].any()
    assert not holes[7:12,8,1].any()
    # Exact native-node queries need no zero-weight neighbors; a fractional
    # query crossing a missing derivative is rejected, never filled.
    scalar=g[...,0];mask=good[...,0].copy();mask[9,8]=False
    sampled,ok=strict_bilinear(x,y,scalar,mask,np.array([[x[7],y[9]],[x[7]+1,y[9]],[x[0]-1,y[5]]]))
    assert ok.tolist()==[True,False,False]
    q=np.c_[xx.ravel(),yy.ravel()]
    # Use an actual 8px Cartesian sublattice for the matched operator.
    select=((q[:,0]-3)%8==0)&((q[:,1]-3)%8==0)
    rr=dict(query=q[select],disp=d.reshape(-1,2)[select],accepted=np.ones(select.sum(),bool))
    gg,ggood,_=reporting_diagonal_difference(rr)
    ee=expected.reshape(-1,2)[select]
    matched_error=float(np.max(np.abs(gg[ggood]-ee[ggood])))
    assert matched_error<1e-13
    out=dict(status='pass',quadratic_analytic_gradient_max_error=error,
        matched_operator_quadratic_max_error=matched_error,
        missing_intermediate_nodes_not_bridged=True,positive_weight_corner_test=True,
        vertical_sign='Both w and z flip relative to image coordinates; G11 sign unchanged.')
    (HERE/'gradient_comparison_self_test.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pairs',nargs='+',type=int,default=[80,100]);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        for pair in args.pairs:run_pair(pair)
