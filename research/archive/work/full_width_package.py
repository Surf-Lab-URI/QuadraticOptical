"""Package only the completed conservative full-width deliverables and provenance."""
from pathlib import Path
import hashlib,json,zipfile
dst=Path('outputs/full_width_conservative_bundle.zip');items={}
for p in Path('outputs').glob('full_width_conservative_*'):
    if p.suffix!='.zip':items[str(p)]=p
for p in Path('work/full_width').iterdir():
    if p.is_file() and (p.suffix in ['.npz','.json','.md','.py'] or p.name=='possible_target_conflict.png'):items[str(p)]=p
for name in ['inputs.npz','main.npz','affine.npz','large_window.npz','margin14.npz','reverse.npz','results.npz','summary.json','coarse_affine.npz','ptv_tracks.npz']:
    p=Path('work/large_frame')/name;items[str(p)]=p
for name in ['full_width_prepare.py','full_width_finalize.py','full_width_export.py','full_width_verify_exports.py','full_width_report.py',
             'field_model.py','ptv_model.py','extended_flow.py','metadata.npz','final_results.npz','final_summary.json']:
    p=Path('work')/name;items[str(p)]=p
items['README.md']=Path('work/full_width_bundle_readme.md')
manifest={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in items.items()}
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,p in items.items():z.write(str(p),name)
    z.writestr('manifest_sha256.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dst) as z:
    assert z.testzip() is None
    for name,rec in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==rec['sha256']
info=dict(path=str(dst.resolve()),files=len(items)+1,bytes=dst.stat().st_size,zip_integrity=True,all_sha256_verified=True)
Path('work/full_width/bundle_integrity.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
