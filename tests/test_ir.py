import h5py
import numpy as np
import pytest
from scipy.io import savemat
from quadratic_optical.ir import select_surface_values
from quadratic_optical.matio import read_experiment_fields


def run(name='test'):
    return dict(exp_name=name,
        PIV=dict(pairNum=[0,80,100],IR_idx=[1,2,3],t=[0.,1.01,2.01]),
        Surfs=dict(pairNum=[0,0,80,80,100,100],t=[0.,.01,1.01,1.02,2.01,2.02],dt_pair=.01,spp=1.),
        USurf=dict(t=[0.,1.,2.],usurf0=[.11,np.nan,np.nan],usurffilt=[.12,.22,np.nan],IRfps=1.,Ndots_used=[1,0,0]),
        enormous_unused=np.zeros((1000,1000)))


@pytest.mark.parametrize('compressed',[False,True])
def test_campaign_selected_run_and_raw_filtered_missing(tmp_path, compressed):
    p=tmp_path/'campaign.mat';exps=np.empty((1,3),object);exps[0]=[run('other'),run('test'),np.array([])]
    savemat(p,dict(exps=exps),do_compression=compressed)
    records=select_surface_values(p,'test',[0,80,100,9])
    assert [r['status'] for r in records]==['selected','selected','no_finite_surface_value','pair_not_found']
    # The filtered estimate is preferred even though a raw value exists here;
    # the raw one stays recorded so the rejected observation is still visible.
    assert records[0]['selected_field']=='USurf.usurffilt' and records[0]['velocity_m_per_s']==.12
    assert records[0]['indexed_raw_usurf0_m_per_s']==.11
    assert records[0]['raw_observation_available_at_index'] is True
    assert records[0]['representation']=='smoothed IR surface estimate'
    assert records[1]['selected_field']=='USurf.usurffilt' and records[1]['velocity_m_per_s']==.22
    assert records[1]['IR_index_matlab']==2 and records[1]['IR_index_python']==1
    assert records[1]['PIV_row_matlab']==2 and records[1]['PIV_B_time_s']==1.02
    assert records[1]['IR_index_is_nearest_time'] is True
    assert records[1]['experiment_index_matlab']==2
    assert not records[1]['theory_fallback_used'] and not records[1]['whole_campaign_loaded']


def test_run_top_level_no_name_identity_unverified(tmp_path):
    p=tmp_path/'run.mat';data=run();del data['exp_name'];savemat(p,data,do_compression=True)
    r=select_surface_values(p,'configured-name',[80])[0]
    assert r['available'] and not r['experiment_identity_verified']
    assert 'exp_name' in r['missing_fields']


def test_missing_mapping_invalid_index_and_duplicate_pair(tmp_path):
    p=tmp_path/'run.mat';data=run();del data['PIV']['IR_idx'];savemat(p,data)
    assert select_surface_values(p,'test',[80])[0]['status']=='missing_pair_mapping'
    data=run();data['PIV']['IR_idx']=[0,1.5,100];savemat(p,data)
    rows=select_surface_values(p,'test',[0,80,100])
    assert [r['status'] for r in rows]==['invalid_or_missing_IR_index','invalid_or_missing_IR_index','IR_index_out_of_range']
    data['PIV']['pairNum']=[80,80,100];savemat(p,data)
    with pytest.raises(ValueError,match='Duplicate'):select_surface_values(p,'test',[80])


def test_mismatched_or_duplicate_experiment(tmp_path):
    p=tmp_path/'run.mat';savemat(p,run('different'))
    with pytest.raises(ValueError,match='does not match'):select_surface_values(p,'test',[80])
    exps=np.empty((1,2),object);exps[0]=[run(),run()];savemat(p,dict(exps=exps),do_compression=True)
    with pytest.raises(ValueError,match='found 2'):select_surface_values(p,'test',[80])


def h5_run(file, name, exp):
    group=file.create_group(name)
    for key,value in exp.items():
        if isinstance(value,dict):
            sub=group.create_group(key)
            for k,v in value.items():
                a=np.atleast_2d(v);sub.create_dataset(k,data=a.T)
        elif isinstance(value,str):
            ds=group.create_dataset(key,data=np.array([[ord(c)] for c in value],dtype='u2'));ds.attrs['MATLAB_class']=b'char'
        else:group.create_dataset(key,data=np.atleast_2d(value).T)
    return group


def test_hdf5_campaign_cells_and_run_top_level(tmp_path):
    p=tmp_path/'campaign.mat'
    with h5py.File(p,'w') as f:
        other=h5_run(f,'r1',run('other'));target=h5_run(f,'r2',run())
        exps=f.create_dataset('exps',(2,1),dtype=h5py.ref_dtype);exps[0,0]=other.ref;exps[1,0]=target.ref
    rows=select_surface_values(p,'test',[0,80])
    assert rows[0]['velocity_m_per_s']==.12 and rows[1]['velocity_m_per_s']==.22
    assert rows[1]['source_format']=='v7.3' and rows[1]['experiment_index_matlab']==2
    fields,meta=read_experiment_fields(p,'test',['USurf.usurf0'])
    assert set(fields)=={'exp_name','USurf.usurf0'}
    assert fields['USurf.usurf0'].shape==(1,3)


def test_hdf5_top_level_scalar_struct_references(tmp_path):
    p=tmp_path/'run.mat'
    with h5py.File(p,'w') as f:
        group=h5_run(f,'run',run())
        for key,obj in group.items():
            ds=f.create_dataset(key,(1,1),dtype=h5py.ref_dtype)
            ds[0,0]=obj.ref
    row=select_surface_values(p,'test',[80])[0]
    assert row['velocity_m_per_s']==.22 and row['experiment_identity_verified']
    assert row['experiment_index_matlab'] is None


def test_filtered_is_preferred_and_raw_is_only_a_fallback(tmp_path):
    """Pin the preference itself, in both directions.

    The raw fresh-dot value rests on one to three dots and scatters by about
    0.037 m/s against the filtered series, so it must not win where both exist.
    It must still be used where the filtered value is absent, so that a file
    without a filtered series is not silently left blank.
    """
    both=run()
    both['USurf']['usurf0']=[.11,np.nan,np.nan]
    both['USurf']['usurffilt']=[.12,.22,np.nan]
    p=tmp_path/'both.mat';savemat(p,both,do_compression=True)
    r=select_surface_values(p,'test',[0])[0]
    assert r['selected_field']=='USurf.usurffilt' and r['velocity_m_per_s']==.12

    only_raw=run()
    only_raw['USurf']['usurf0']=[.11,np.nan,np.nan]
    only_raw['USurf']['usurffilt']=[np.nan,.22,np.nan]
    q=tmp_path/'raw_only.mat';savemat(q,only_raw,do_compression=True)
    r2=select_surface_values(q,'test',[0])[0]
    assert r2['selected_field']=='USurf.usurf0' and r2['velocity_m_per_s']==.11
    assert r2['representation']=='raw fresh-dot IR surface observation'
