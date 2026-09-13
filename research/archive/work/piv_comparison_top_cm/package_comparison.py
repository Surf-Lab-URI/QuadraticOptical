"""Validate comparison exports and package figures, data and audit records."""
from quiver_compare import *
from scipy.io import loadmat
import zipfile, shutil

def main():
    checks={}
    for pair in [80,100]:
        dest=OUT/('pair_'+str(pair))
        for stem in ['velocity_comparison','gradient_comparison','horizontal_integral_comparison']:
            n=read(dest/(stem+'.npz'));m=loadmat(str(dest/(stem+'.mat')),squeeze_me=True)
            count=0
            for key,value in n.items():
                if np.issubdtype(value.dtype,np.number) or value.dtype==bool:
                    assert key in m,key
                    a=np.asarray(m[key]);b=np.asarray(value).squeeze()
                    assert a.shape==b.shape,(key,a.shape,b.shape)
                    assert np.allclose(a,b,rtol=0,atol=0,equal_nan=True),key
                    count+=1
            checks[str(pair)+'_'+stem]={'numeric_arrays_round_trip_exact':count}
        for stem in ['gradient','integral']:
            source=read(WORK/('pair_%s_%s_comparison.npz'%(pair,stem)))
            target=read(dest/(('gradient' if stem=='gradient' else 'horizontal_integral')+'_comparison.npz'))
            assert set(source)==set(target)
            for key in source:
                if source[key].dtype.kind in 'US':assert np.array_equal(source[key],target[key])
                else:assert np.allclose(source[key],target[key],rtol=0,atol=0,equal_nan=True),key
        shutil.copy2(WORK/('pair_%s_independent_comparison_audit.json'%pair),dest/'independent_comparison_audit.json')
    audits=OUT/'audits';audits.mkdir(exist_ok=True)
    for name in ['native_piv_audit.json','quiver_export_audit.json','gradient_comparison_self_test.json','integral_comparison_self_test.json','native_piv_review.md','comparison_final_review.md']:
        shutil.copy2(WORK/name,audits/name)
    (audits/'export_round_trip_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Code is included for scientific reproducibility; original prediction archive is separate.
    code=OUT/'comparison_code';code.mkdir(exist_ok=True)
    for source in WORK.glob('*.py'):shutil.copy2(source,code/source.name)
    (code/'README.md').write_text('These scripts were run from work/piv_comparison_top_cm under the task root. They use frozen inputs/models/samples in work/image_only_cm, retained in the earlier image_only_top_cm_results.zip archive, plus the original supplied native PIV MAT files. The scripts perform comparison and display only. They do not refit optical flow. Restore this relative directory structure to rerun. Source paths and SHA-256 hashes are recorded in the comparison metadata. Scientific dependencies: NumPy, SciPy, h5py, Matplotlib.\n')
    manifest={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_manifest.json'}
    (OUT/'file_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=OUT.parent/'piv_comparison_top_cm.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        entries=len(z.infolist())
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,entries=entries,sha256=sha(archive),export_checks='all pass'),indent=2))

if __name__=='__main__':main()
