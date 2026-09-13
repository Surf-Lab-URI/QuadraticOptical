"""Package fresh results, source, and exact inputs for reproducible inspection."""
from pathlib import Path
import json,hashlib,zipfile,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'outputs/image_only_top_cm'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    source_files=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))
    source_files += [HERE/'dependency_initializer_review.md',HERE/'integration_plan.md',HERE/'integration_api.md',
        ROOT/'work/full_width/fit_local_cached.py',ROOT/'work/pairs/process_executor.py',
        ROOT/'work/ptv_model.py',ROOT/'work/field_model.py']
    exact=[]
    for pair in [80,100]:
        folder=HERE/str(pair)
        assert (folder/'independent_image_only_audit.json').exists()
        audit=json.loads((folder/'independent_image_only_audit.json').read_text())
        assert audit.get('passed',False),audit
        fields=json.loads((folder/'independent_field_reconstruction.json').read_text())
        assert fields.get('passed',False),fields
        names=['inputs.npz','coarse_search.npz','coarse_affine.npz','coarse_bootstrap48.npz',
            'particle_candidates.npz','ptv_tracks.npz','seeds.npz','main.npz','affine.npz',
            'large_window.npz','margin14.npz','reverse.npz','results.npz','integration_profile.npz','plot_samples.npz',
            'independent_field_reconstruction_samples.npz']
        exact += [folder/name for name in names]
        exact += list(folder.glob('*.json'))
        exact += list(folder.glob('bootstrap*comparison.npz'))
        exact += list(folder.glob('tracking_prior_bootstrap_audit.npz'))
        branch=folder/'bootstrap48'
        if branch.is_dir():
            exact += [p for p in branch.glob('*.npz') if p.name!='inputs.npz']
            exact += list(branch.glob('*.json'))
    payload=[p for p in source_files+exact if p.is_file()]
    manifest=dict(image_only=True,supplied_velocity_used=False,
        source_note='Only fresh image-only data and reusable algorithm source. No previous velocity results or supplied velocity arrays are included.',
        exact_inputs_note='Normalized images, unchanged raw pixel arrays, geometry, calibration and masks are included to preserve artifact hashes. Original TIFFs are identified by hash in input manifests.',
        alternate_input_note='Any bootstrap48 branch uses exactly its parent pair inputs.npz, identified by the same SHA256. This duplicate payload is omitted from the branch archive; copy or link the parent inputs.npz when rerunning the branch.',
        artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in payload})
    (OUT/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'outputs/image_only_top_cm_results.zip'
    with zipfile.ZipFile(str(archive),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(str(p),str(p.relative_to(OUT.parent)))
        for p in payload:z.write(str(p),'reproducibility/'+str(p.relative_to(ROOT)))
    # Verify stored CRCs, file inventory, and the dependency exclusion.
    with zipfile.ZipFile(str(archive)) as z:
        assert z.testzip() is None
        assert not any('work/pairs/80/' in n or 'work/pairs/100/' in n for n in z.namelist())
        entries=len(z.namelist())
    report=dict(archive_path=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        entries=entries,crc_pass=True,image_only=True,supplied_velocity_used=False)
    (OUT/'package_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
