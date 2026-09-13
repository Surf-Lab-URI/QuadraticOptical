from pathlib import Path
import h5py
import numpy as np
import pytest
from scipy.io import savemat
from quadratic_optical import matio


def h5_numeric(group, name, value, cls=b'double'):
    a = np.asarray(value)
    d = group.create_dataset(name, data=a.transpose(tuple(reversed(range(a.ndim)))))
    d.attrs['MATLAB_class'] = cls
    return d


@pytest.mark.parametrize('compressed', [False, True])
def test_v5_selective_shapes_and_types(tmp_path, monkeypatch, compressed):
    p = tmp_path/'input.mat'
    grid = np.arange(12.).reshape(3,4); grid[1,2] = np.nan
    values = dict(geometry=dict(DX=.001, surface=np.arange(4.)[None,:]),
                  payload=dict(grid=grid, logical=np.array([[True,False]]),
                               complex=grid+1j*grid, unused=np.ones((300,400))))
    savemat(p, values, do_compression=compressed)
    calls = []
    decode = matio._decode
    def spy(reader, header, mat_header):
        calls.append(header['dims'])
        return decode(reader, header, mat_header)
    monkeypatch.setattr(matio, '_decode', spy)
    got = matio.read_mat_fields(p, ['geometry.DX','geometry.surface','payload.grid','payload.logical','payload.complex'])
    assert len(calls) == 5 and (300,400) not in calls
    assert got['geometry.DX'].shape == (1,1)
    assert got['geometry.surface'].shape == (1,4)
    np.testing.assert_equal(got['payload.grid'], grid)
    np.testing.assert_equal(got['payload.complex'], grid+1j*grid)
    assert got['payload.logical'].dtype == bool
    with pytest.raises(KeyError):matio.read_mat_fields(p, ['geometry.missing'])
    assert matio.read_mat_fields(p, ['geometry.missing'], missing='ignore') == {}


def test_hdf5_orientation_references_and_logical(tmp_path):
    p = tmp_path/'input.mat';grid = np.arange(24.).reshape(2,3,4)
    with h5py.File(p,'w',userblock_size=512) as f:
        refs=f.create_group('#refs#');struct=refs.create_group('struct')
        h5_numeric(struct,'grid',grid)
        h5_numeric(struct,'logical',np.array([[1,0],[0,1]]),b'logical')
        d=f.create_dataset('nested',(1,1),dtype=h5py.ref_dtype);d[0,0]=struct.ref
    got=matio.read_mat_fields(p,['nested.grid','nested.logical'])
    np.testing.assert_equal(got['nested.grid'],grid)
    assert got['nested.logical'].shape==(2,2) and got['nested.logical'].dtype==bool


@pytest.mark.parametrize('format', ['v5','v73'])
def test_native_piv_nonsquare_coordinates_sign_and_mask(tmp_path, format):
    p=tmp_path/'piv.mat';dx=np.arange(6.).reshape(2,3)
    data=dict(xPIV=np.array([[4,8,12]]),zPIV=np.array([[4,8]]),delta_x=dx,
              delta_z=dx+2,DX=.0001,DT=.01,dcor=np.ones((2,3)),mask=np.array([[1,0,1],[1,np.nan,1]]))
    if format=='v5':savemat(p,dict(compVel=data),do_compression=True)
    else:
        with h5py.File(p,'w') as f:
            group=f.create_group('compVel')
            for key,value in data.items():h5_numeric(group,key,np.atleast_2d(value))
    got=matio.read_native_piv(p,require_dcor=True)
    np.testing.assert_equal(got['x_px'],[3,7,11]);np.testing.assert_equal(got['y_px'],[3,7])
    np.testing.assert_equal(got['dx_px'],dx);np.testing.assert_equal(got['dy_px'],-dx-2)
    np.testing.assert_equal(got['mask'],[[True,False,True],[True,False,True]])
    assert not got['metadata']['dense_fields_read']


def test_native_missing_dcor_explicit_mask_and_nonbinary_mask(tmp_path):
    p=tmp_path/'piv.mat'
    data=dict(xPIV=[[4,8]],zPIV=[[4,8]],delta_x=np.zeros((2,2)),delta_z=np.zeros((2,2)),DX=1,DT=1)
    savemat(p,dict(compVel=data))
    assert matio.read_native_piv(p)['dcor'] is None
    with pytest.raises(KeyError,match='dcor'):matio.read_native_piv(p,require_dcor=True)
    with pytest.raises(KeyError,match='mask'):matio.read_native_piv(p,mask_field='compVel.mask')
    data['mask']=np.ones((2,2))*255;savemat(p,dict(compVel=data))
    with pytest.raises(ValueError,match='1=valid'):matio.read_native_piv(p)


def test_big_endian_v5_and_empty_optional_struct(tmp_path):
    import struct
    p=tmp_path/'big.mat'
    def element(kind,data):return struct.pack('>II',kind,len(data))+data+b'\0'*((-len(data))%8)
    prefix=b'MATLAB 5.0 MAT-file, synthetic big-endian fixture'.ljust(116,b' ')+b'\0'*8+struct.pack('>H',256)+b'MI'
    matrix=element(6,struct.pack('>II',6,0))+element(5,struct.pack('>ii',2,3))+element(1,b'grid')+element(9,np.arange(6.,dtype='>f8').tobytes())
    p.write_bytes(prefix+element(14,matrix))
    got=matio.read_mat_fields(p,['grid'])
    np.testing.assert_equal(got['grid'],np.arange(6.).reshape((2,3),order='F'))
    p=tmp_path/'empty.mat';savemat(p,dict(USurf=np.array([])),do_compression=True)
    assert matio.read_mat_fields(p,['USurf.t'],missing='ignore')=={}
