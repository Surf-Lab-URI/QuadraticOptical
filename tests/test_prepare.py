from copy import deepcopy
import hashlib
import json

import h5py
import numpy as np
from PIL import Image
import pytest
from scipy.io import savemat

from quadratic_optical.config import DEFAULTS, for_pair, load_config, merge, validate
from quadratic_optical.discovery import ImagePair
from quadratic_optical import prepare as preparation
from quadratic_optical import matio


def config(**changes):
    base=merge(DEFAULTS,dict(dx_m_per_px=.0001,dt_s=.01,depth_m=.005,
                           surface=dict(mode='constant',index_base=0,row_a=12)))
    return validate(merge(base,changes))


def pair(tmp_path,shape=(96,144),dtype=np.uint8):
    y,x=np.indices(shape);raw=((x*7+y*11)%200+20).astype(dtype)
    if dtype==np.uint16:raw=raw*256
    paths=[]
    for fr in 'AB':
        p=tmp_path/f'Experiment_80_img{fr}.tif';Image.fromarray(raw).save(p);paths.append(p)
    return ImagePair('Experiment_80',*paths,experiment='Experiment',pair_number=80),raw


def read(directory):
    with np.load(directory/'inputs.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}


def native_metadata(path,width,format='v5',velocity=123.,dx=.0001):
    data={'compVel':dict(DX=dx,DT=.01,delta_x=np.full((3,7),velocity),delta_z=np.full((3,7),-velocity),dcor=np.ones((3,7))),
          'imSurfa':dict(surfacePIVImg=np.full((1,width),13.)),
          'imSurfb':dict(surfacePIVImg=np.full((1,width),14.))}
    if format=='v5':savemat(path,data,do_compression=True)
    else:
        with h5py.File(path,'w') as f:
            for section,fields in data.items():
                g=f.create_group(section)
                for name,value in fields.items():g.create_dataset(name,data=np.atleast_2d(value).T)


def test_rectangular_uint8_grid_and_core_metadata(tmp_path):
    p,raw=pair(tmp_path);out=tmp_path/'out';manifest=preparation.prepare(p,out,config())
    z=read(out)
    assert z['A'].shape==(96,144) and np.isfinite(z['A']).all()
    np.testing.assert_array_equal(z['rawA'],raw)
    np.testing.assert_array_equal(z['surface_a'],np.full(144,12.))
    assert z['va'].dtype==bool and np.all(z['va'][22:]) and not z['va'][:22].any()
    assert np.all(z['points'][:,0]%8==7) and np.all(z['points'][:,1]%8==7)
    assert z['points'][:,0].max()==143 and z['points'][:,1].max()<=95
    np.testing.assert_array_equal(z['source_depth'],z['points'][:,1]-12)
    assert bool(z['image_only']) and not bool(z['supplied_velocity_used'])
    assert 'surface_trace_offset_px' in z and 'surface_trace_offset_pixels' not in z
    assert manifest['mat_dataset_reads']==[]


@pytest.mark.parametrize('format',['v5','v73'])
def test_metadata_only_loading_never_uses_supplied_velocity(tmp_path,monkeypatch,format):
    p,raw=pair(tmp_path);path=tmp_path/'meta.mat';native_metadata(path,raw.shape[1],format)
    p=ImagePair(p.name,p.image_a,p.image_b,path,experiment=p.experiment,pair_number=80)
    requests=[];original=preparation.read_mat_fields
    def selective(path,keys):
        requests.append(keys.copy());return original(path,keys)
    monkeypatch.setattr(preparation,'read_mat_fields',selective)
    cfg=config(dx_m_per_px=None,dt_s=None,surface=dict(mode='piv_mat',index_base=1))
    out=tmp_path/'out';before=preparation.prepare(p,out,cfg);z=read(out)
    assert requests==[['compVel.DX','compVel.DT','imSurfa.surfacePIVImg','imSurfb.surfacePIVImg']]
    assert all(not any(word in key.lower() for word in ['delta','dcor','velocity']) for key in z if key!='supplied_velocity_used')
    assert z['surface_a'][0]==12 and z['surface_b'][0]==13
    # Changing only held-out velocity arrays must neither seed nor invalidate
    # image-only preparation: only selected metadata enter the signature.
    native_metadata(path,raw.shape[1],format,velocity=-987.)
    after=preparation.prepare(p,out,cfg)
    assert before['preparation_signature']==after['preparation_signature']
    assert before['inputs_sha256']==after['inputs_sha256']


def test_sidecar_surfaces_masks_and_offset(tmp_path):
    p,raw=pair(tmp_path);surface=tmp_path/'surface.npz';yy=np.indices(raw.shape)[0]
    mask=yy>=18;mask[40:50,30:40]=False
    np.savez(surface,surface_a=np.full(144,18.),surface_b=np.full((1,144),19.),availability_a=mask,availability_b=mask.astype(np.uint8))
    p=ImagePair(p.name,p.image_a,p.image_b,surface_file=surface)
    out=tmp_path/'out';preparation.prepare(p,out,config(surface=dict(mode='auto',index_base=1,offset_px=-2)))
    z=read(out)
    np.testing.assert_array_equal(z['surface_a'],np.full(144,16.))
    np.testing.assert_array_equal(z['availability_a'],mask)
    np.testing.assert_array_equal(z['va'],mask&(yy>=26))
    assert float(z['surface_trace_offset_px'])==-2 and bool(z['surface_geometry_inferred'])


@pytest.mark.parametrize('bad_value',[np.nan,2,-1])
def test_invalid_sidecar_mask_rejected(tmp_path,bad_value):
    p,raw=pair(tmp_path);surface=tmp_path/'surface.npz';mask=np.ones(raw.shape);mask[0,0]=bad_value
    np.savez(surface,surface_a=np.full(144,12.),surface_b=np.full(144,12.),availability_a=mask)
    p=ImagePair(p.name,p.image_a,p.image_b,surface_file=surface)
    with pytest.raises(ValueError,match='finite binary'):preparation.prepare(p,tmp_path/'out',config(surface=dict(mode='sidecar')))


def test_intensity_scaling_and_original_zero_boundary(tmp_path):
    p,raw=pair(tmp_path,dtype=np.uint16);raw[:12]=0
    for path in [p.image_a,p.image_b]:Image.fromarray(raw).save(path)
    with pytest.raises(ValueError,match='8-bit intensity'):preparation.prepare(p,tmp_path/'unscaled',config())
    cfg=config(intensity_scale=1/256,intensity_offset=5,
               surface=dict(mode='nonzero_boundary',offset_px=-2),availability='nonzero_boundary')
    out=tmp_path/'scaled';preparation.prepare(p,out,cfg);z=read(out)
    np.testing.assert_array_equal(z['rawA'],raw/256+5)
    assert np.all(z['surface_a']==10)
    assert not z['availability_a'][:12].any() and z['availability_a'][12:].all()
    with pytest.raises(ValueError,match=r'\[0,255\]'):preparation.prepare(p,tmp_path/'badscale',config(intensity_scale=1/128))


def test_missing_calibration_surface_and_too_shallow_fail_early(tmp_path):
    p,raw=pair(tmp_path)
    with pytest.raises(ValueError,match='No surface'):preparation.prepare(p,tmp_path/'none',config(surface=dict(mode='auto')))
    with pytest.raises(ValueError,match='Calibration/surface'):preparation.prepare(p,tmp_path/'nocal',config(dx_m_per_px=None))
    with pytest.raises(ValueError,match='at least 20 pixels'):config(depth_m=.001)
    mat=tmp_path/'meta.mat';native_metadata(mat,144,dx=.001)
    # A metadata-based failure must occur before any image is opened.
    missing=ImagePair(p.name,tmp_path/'missingA.tif',tmp_path/'missingB.tif',mat)
    with pytest.raises(ValueError,match='at least 20 pixels'):
        preparation.prepare(missing,tmp_path/'shallow',config(dx_m_per_px=None))


@pytest.mark.parametrize('changed',['image','calibration','sidecar','saved_inputs','incomplete'])
def test_resume_rejects_changed_sources_and_preserves_original(tmp_path,changed):
    p,raw=pair(tmp_path);mat=tmp_path/'meta.mat';native_metadata(mat,144)
    sidecar=tmp_path/'surface.npz';np.savez(sidecar,surface_a=np.full(144,12.),surface_b=np.full(144,12.))
    p=ImagePair(p.name,p.image_a,p.image_b,mat,sidecar)
    cfg=config(dx_m_per_px=None,surface=dict(mode='auto'));out=tmp_path/'out'
    before=preparation.prepare(p,out,cfg)
    assert preparation.prepare(p,out,cfg)==before
    input_bytes=(out/'inputs.npz').read_bytes()
    if changed=='image':
        raw[30,30]+=1;Image.fromarray(raw).save(p.image_a)
    elif changed=='calibration':native_metadata(mat,144,dx=.00011)
    elif changed=='sidecar':np.savez(sidecar,surface_a=np.full(144,13.),surface_b=np.full(144,12.))
    elif changed=='saved_inputs':(out/'inputs.npz').write_bytes(input_bytes+b'changed')
    else:(out/'input_manifest.json').unlink()
    expected_bytes=(out/'inputs.npz').read_bytes()
    with pytest.raises(ValueError,match='Existing results|Incomplete input'):preparation.prepare(p,out,cfg)
    assert (out/'inputs.npz').read_bytes()==expected_bytes


def test_mask_shape_image_shape_and_empty_column_errors(tmp_path):
    p,raw=pair(tmp_path);Image.fromarray(raw[:90]).save(p.image_b)
    with pytest.raises(ValueError,match='dimensions differ'):preparation.prepare(p,tmp_path/'shape',config())
    Image.fromarray(raw).save(p.image_b);raw[:,0]=0;Image.fromarray(raw).save(p.image_a)
    with pytest.raises(ValueError,match='every column'):preparation.prepare(p,tmp_path/'column',config(availability='nonzero_boundary'))
    maskfile=tmp_path/'mask.npz';np.savez(maskfile,surface_a=np.full(144,12.),surface_b=np.full(144,12.),availability_a=np.ones((96,96),bool))
    p=ImagePair(p.name,p.image_a,p.image_b,surface_file=maskfile)
    with pytest.raises(ValueError,match='mask must match'):preparation.prepare(p,tmp_path/'maskshape',config(surface=dict(mode='sidecar')))


@pytest.mark.parametrize('change',[{'intensity_offset':float('nan')},{'workers':True},{'grid_phase_px':float('inf')},
    {'surface':{'offset_pixels':-12}},{'surface':None},{'pairs':[]},{'pairs':{'name':[]}}, {'depth_m':True}])
def test_config_invalid_values_are_explicit(change):
    with pytest.raises(ValueError):config(**change)


def test_pair_overrides_are_isolated_and_validated():
    parent=config(pairs={'first':{'surface':{'row_a':18},'dt_s':.02}})
    first=for_pair(parent,'first');second=for_pair(parent,'second')
    assert first['surface']['row_a']==18 and first['dt_s']==.02
    assert second['surface']['row_a']==12 and parent['surface']['row_a']==12


def test_resume_allows_runtime_and_comparison_policy_changes(tmp_path):
    p,_=pair(tmp_path);out=tmp_path/'out';cfg=config(workers=1,piv_quality='correlation')
    original=preparation.prepare(p,out,cfg)
    original_bytes=(out/'inputs.npz').read_bytes()
    resumed=preparation.prepare(p,out,config(workers=2,piv_quality='finite'))
    assert resumed['preparation_signature']==original['preparation_signature']
    assert (out/'inputs.npz').read_bytes()==original_bytes
    # Initial preparation configuration remains documented in full.
    assert resumed['config']['workers']==1 and resumed['config']['piv_quality']=='correlation'
    with pytest.raises(ValueError,match='different inputs/settings'):
        preparation.prepare(p,out,config(depth_step_m=.0002))


def test_minimum_integration_band_matches_core_strict_boundary():
    assert config(depth_m=20*.0001)['depth_m']==.002
    with pytest.raises(ValueError,match='at least 20 pixels'):
        config(depth_m=np.nextafter(.002,0).item())
