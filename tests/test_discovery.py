from pathlib import Path
import pytest
from quadratic_optical.discovery import discover,identify


def touch(folder,*names):
    folder.mkdir(parents=True,exist_ok=True)
    for name in names:(folder/name).touch()


def test_pairing_by_suffix_metadata_and_number(tmp_path):
    touch(tmp_path,'ExpLCL_1_03_80_imgB.TIFF','ExpLCL_1_03_80_imgA.tif',
          'ExpLCL_1_03_80_PIV.mat','ExpLCL_1_03_80_surface.npz',
          'second_A.png','second_b.png','ExpLCL_1_03_80_imgA_mask.tif','unrelated.png')
    pairs,incomplete=discover(tmp_path)
    assert not incomplete and len(pairs)==2
    p=next(p for p in pairs if p.name=='ExpLCL_1_03_80')
    assert p.image_a.name.endswith('imgA.tif') and p.image_b.name.endswith('imgB.TIFF')
    assert p.experiment=='ExpLCL_1_03' and p.pair_number==80
    assert p.piv_mat.is_file() and p.surface_file.is_file()
    assert identify('not_numbered')==(None,None)
    assert identify('Exp-0080')==('Exp',80)


def test_incomplete_duplicate_and_empty_stems(tmp_path):
    touch(tmp_path,'good_a.tif','good_b.tif','missing_a.tif','_imgA.tif','_imgB.tif')
    with pytest.raises(ValueError,match='Incomplete'):discover(tmp_path)
    pairs,incomplete=discover(tmp_path,skip_incomplete=True)
    assert [p.name for p in pairs]==['good'] and incomplete==['missing']
    touch(tmp_path,'good_imgA.png')
    with pytest.raises(ValueError,match='Ambiguous duplicate'):discover(tmp_path,skip_incomplete=True)


def test_recursive_pairing_and_output_name_collision(tmp_path):
    touch(tmp_path/'a','exp_1_imgA.tif','exp_1_imgB.tif')
    touch(tmp_path/'b','exp_1_imgA.tif','exp_1_imgB.tif')
    with pytest.raises(ValueError,match='No image pairs'):discover(tmp_path)
    pairs,_=discover(tmp_path,recursive=True)
    assert [p.name for p in pairs]==['a__exp_1','b__exp_1']
    touch(tmp_path,'a__exp_1_imgA.tif','a__exp_1_imgB.tif')
    with pytest.raises(ValueError,match='Duplicate output name'):discover(tmp_path,recursive=True)


def test_missing_directory_no_pairs_and_directory_sidecar(tmp_path):
    with pytest.raises(ValueError,match='does not exist'):discover(tmp_path/'missing')
    with pytest.raises(ValueError,match='No image pairs'):discover(tmp_path)
    touch(tmp_path,'good_a.tif','good_b.tif');(tmp_path/'good_PIV.mat').mkdir()
    with pytest.raises(ValueError,match='sidecar file'):discover(tmp_path)


def test_unsafe_parent_output_name_is_rejected(tmp_path):
    touch(tmp_path,'.._a.tif','.._b.tif')
    with pytest.raises(ValueError,match='Unsafe output name'):discover(tmp_path)
