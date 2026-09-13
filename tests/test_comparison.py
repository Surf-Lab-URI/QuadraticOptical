"""Synthetic scientific-contract tests; no full experimental images needed."""
import hashlib
import json

import numpy as np
import pytest

from quadratic_optical import comparison as c


def native_fixture(origin=(0,0)):
    origin=np.array(origin,float)
    x=np.arange(0,80,4.)+origin[0]
    y=np.arange(0,64,4.)+origin[1]
    xx,yy=np.meshgrid(x,y)
    disp=np.zeros(xx.shape+(2,));disp[...,0]=.4;disp[...,1]=.2
    native=dict(x_px=x,y_px=y,disp_px=disp,dcor=np.ones(xx.shape),mask=None,DX=.01,DT=.1,metadata={'test':True})
    geometry=dict(DX=np.array(.01),DT=np.array(.1),origin0=origin,
        availability_a=np.ones((64,80),bool),surface_a=np.zeros(80),
        image_only=np.array(True),supplied_velocity_used=np.array(False))
    return native,geometry


def test_strict_interpolation_rejects_gaps_but_ignores_zero_weight_nan():
    x=np.array([0.,4.,8.,12.]);y=np.array([0.,4.,8.])
    values=np.arange(12,dtype=float).reshape(3,4)
    values[0,1]=np.nan;valid=np.isfinite(values)
    q=np.array([[0,0],[1,0],[8,4],[13,4],[np.nan,0]])
    sampled,ok=c.strict_bilinear(x,y,values,valid,q)
    assert ok.tolist()==[True,False,True,False,False]
    assert sampled[0]==0 and sampled[2]==6
    assert np.isnan(sampled[~ok]).all()


def test_quality_requires_correlation_and_applies_mask_and_source():
    native,geometry=native_fixture()
    native['dcor'][4,6]=np.nan
    native['dcor'][5,6]=-10.  # finite scores are not thresholded
    native['mask']=np.ones(native['dcor'].shape,bool);native['mask'][4,7]=False
    geometry['availability_a'][16,32]=False
    cor=c.NativeComparison(native,geometry,'correlation')
    finite=c.NativeComparison(native,geometry,'finite')
    q=np.array([[24,16],[24,20],[28,16],[32,16]])
    assert cor.sample(q)[1].tolist()==[False,True,False,False]
    assert finite.sample(q)[1].tolist()==[True,True,False,False]
    native['dcor']=None
    with pytest.raises(ValueError,match="quality='finite'"):
        c.NativeComparison(native,geometry)
    assert c.NativeComparison(native,geometry,'finite').sample(q)[1][0]


def test_native_coordinate_origin_and_upward_velocity_sign():
    native,geometry=native_fixture(origin=(10,20))
    n=c.NativeComparison(native,geometry)
    results=dict(query=np.array([[24.,16.]]),disp=np.array([[.4,.2]]),accepted=np.array([True]))
    out,stats=c._velocity_comparison(n,results)
    np.testing.assert_array_equal(out['native_query_px'],[[34,36]])
    np.testing.assert_allclose(out['of_u_m_per_s'],[.04])
    np.testing.assert_allclose(out['of_w_m_per_s'],[-.02])
    np.testing.assert_allclose(out['piv_w_m_per_s'],[-.02])
    assert stats['displacement_px']['rms_difference']==0


def test_cartesian_gradient_quadratic_and_missing_intermediate_node():
    x=np.arange(0,84,4.);y=np.arange(0,68,4.);xx,yy=np.meshgrid(x,y)
    d=np.stack([.03*xx+.005*xx**2+.2*yy,-.02*yy+.003*yy**2+.1*xx],axis=2)
    good=np.ones(xx.shape,bool)
    g,ok=c.cartesian_gradient(d,good,x,y)
    expected=np.stack([.03+.01*xx,-.02+.006*yy],axis=2)
    np.testing.assert_allclose(g[ok],expected[ok],atol=1e-14)
    good[7,7]=False
    _,ok=c.cartesian_gradient(d,good,x,y)
    assert not ok[7,5:10,0].any()
    assert not ok[5:10,7,1].any()
    assert ok[7,10,0] and ok[10,7,1]
    # Flipping both physical w and z retains G11's sign, then divide by DT.
    assert expected[8,8,1]>.0


def test_cartesian_gradient_supports_nonfour_pixel_native_spacing():
    x=np.array([0.,3.,7.,10.,15.,21.,26.,33.,40.])
    y=np.array([0.,5.,11.,18.,23.,30.,37.])
    xx,yy=np.meshgrid(x,y)
    d=np.stack([.3*xx+.1*yy,-.4*yy+.2*xx],axis=2)
    g,ok=c.cartesian_gradient(d,np.ones(xx.shape,bool),x,y)
    np.testing.assert_allclose(g[...,0][ok[...,0]],.3,atol=1e-14)
    np.testing.assert_allclose(g[...,1][ok[...,1]],-.4,atol=1e-14)
    assert ok[...,0].any() and ok[...,1].any()


def test_saved_matched_operator_uses_exact_neighbors_and_acceptance():
    x=np.arange(3.,52.,8.);y=np.arange(7.,40.,8.);xx,yy=np.meshgrid(x,y)
    q=np.c_[xx.ravel(),yy.ravel()]
    d=np.c_[.01*q[:,0]**2,-.03*q[:,1]**2]
    accepted=np.ones(len(q),bool)
    g,ok,_=c.saved_cartesian_difference(q,d,accepted)
    expected=np.c_[.02*q[:,0],-.06*q[:,1]]
    np.testing.assert_allclose(g[ok],expected[ok],atol=1e-14)
    center=len(x)+3;accepted[center]=False
    _,ok,_=c.saved_cartesian_difference(q,d,accepted)
    assert not ok[center].any()
    assert not ok[center-1,0] and not ok[center+1,0]
    assert not ok[center-len(x),1] and not ok[center+len(x),1]


def test_simpson_integrals_preserve_gaps_empty_nan_and_physical_width():
    x=np.arange(9.)
    u=np.broadcast_to(x**3,(3,9)).copy()
    nodes=np.ones((3,9),bool);nodes[1,4]=False;nodes[2]=False
    mask=c.interval_mask(nodes)
    stats=c.domain_integral(u,mask,np.ones(4)*2)
    np.testing.assert_allclose(stats['integral_m2_per_s'][:2],[1024.,(2**4/4)+(8**4-6**4)/4])
    np.testing.assert_allclose(stats['width_m'],[8,4,0])
    assert np.isnan(stats['integral_m2_per_s'][2]) and np.isnan(stats['mean_u_m_per_s'][2])
    assert np.isnan(stats['full_width_integral_m2_per_s'][1:]).all()


def frozen_fixture(directory):
    native,geometry=native_fixture()
    native['dcor'][4,6]=np.nan
    x=np.arange(12.,45.,8.);h=np.array([16.,24.,32.]);xx,yy=np.meshgrid(x,h)
    q=np.c_[xx.ravel(),yy.ravel()];n=len(q)
    result=dict(query=q,disp=np.tile([.5,.1],(n,1)),accepted=np.ones(n,bool),
        G=np.zeros((n,2,2)),accepted_du_dx=np.ones(n,bool),accepted_dw_dz=np.ones(n,bool),DX=np.array(.01),DT=np.array(.1))
    samples=dict(result,x_axis_px=x,depth_axis_px=h)
    sx=np.arange(8.,49.);sy=np.broadcast_to(h[:,None],(len(h),len(sx))).copy()
    profile=dict(depth_m=h*.01,horizontal_sample_x_px=sx,sample_query_y_px=sy,
        sample_displacement_px=np.tile([.5,.1],(len(h),len(sx),1)),sample_accepted=np.ones(sy.shape,bool),
        horizontal_interval_width_m=np.ones((len(sx)-1)//2)*.02,
        interval_accepted=np.ones((len(h),(len(sx)-1)//2),bool),common_depth_band_requested_m=np.array([.16,.32]),
        DX=np.array(.01),DT=np.array(.1))
    profile['covered_segment_integral_m2_per_s']=np.ones(len(h))*.05*.4
    for name,data in [('inputs',geometry),('results',result),('plot_samples',samples),('integration_profile',profile),('main',dict(coefficients=np.arange(12.)))]:
        np.savez_compressed(directory/(name+'.npz'),**data)
    path=directory/'supplied.mat';path.write_bytes(b'synthetic loader fixture')
    return native,path


def test_complete_comparison_is_read_only_and_keeps_quality_runs_separate(tmp_path,monkeypatch):
    native,piv_path=frozen_fixture(tmp_path)
    monkeypatch.setattr(c,'_read_piv',lambda path:native)
    paths=list(tmp_path.glob('*.npz'))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:digest(p) for p in paths}
    record=[dict(x_px=24,velocity_m_per_s=99.,comparison_only=True)]
    cor=c.compare_pair(tmp_path,piv_path,surface_records=record)
    finite=c.compare_pair(tmp_path,piv_path,quality='finite')
    assert {p.name:digest(p) for p in paths}==before
    assert cor['directory']==tmp_path/'comparison'/'correlation'
    assert finite['directory']==tmp_path/'comparison'/'finite'
    assert cor['summary']['surface_records']==record
    assert cor['summary']['PIV_used_in_prediction'] is False
    assert cor['summary']['image_only_artifacts_unchanged'] is True
    assert cor['integrals']['matched_width_m'][0]<finite['integrals']['matched_width_m'][0]
    np.testing.assert_allclose(finite['integrals']['of_matched_integral_m2_per_s'],.02)
    np.testing.assert_allclose(finite['integrals']['piv_matched_integral_m2_per_s'],.016)
    np.testing.assert_allclose(finite['integrals']['matched_mean_difference_m_per_s'],.01)
    np.testing.assert_allclose(finite['integrals']['of_matched_width_m'],finite['integrals']['piv_matched_width_m'])
    assert cor['gradients']['of_du_dx_per_s'].shape==(3,5)
    for directory in [cor['directory'],finite['directory']]:
        assert set(p.name for p in directory.iterdir())=={'velocity.npz','gradients.npz','integrals.npz','summary.json','velocity.csv','gradients.csv','integrals.csv'}
        json.loads((directory/'summary.json').read_text())
        with np.load(directory/'gradients.npz',allow_pickle=False) as archive:
            assert archive['component_names'].tolist()==['du_dx','dw_dz']


def test_missing_frozen_artifacts_and_calibration_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError,match='Complete the image-only prediction'):
        c.compare_pair(tmp_path,tmp_path/'absent.mat')
    native,geometry=native_fixture();native['DT']=.2
    with pytest.raises(ValueError,match='calibration'):
        c.NativeComparison(native,geometry)
