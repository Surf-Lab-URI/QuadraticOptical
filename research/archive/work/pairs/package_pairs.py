"""Package completed user-facing results, audits and portable analysis source."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs';WORK=ROOT/'work/pairs'
readme='''Conservative hybrid predictions: observations 80 and 100

Open pair_80/analysis_report.md or pair_100/analysis_report.md for scientific interpretation.
Each folder includes a full-image classical-PIV overlay, detailed near-surface overlays,
du/dx and dw/dz maps, comparison diagnostics, and MATLAB/NumPy numerical arrays.

Use conservative arrays and their separate acceptance masks. Image displacement has
x right and y down. Physical u/x point right, w/z point up; both diagonal derivatives
are Gii/DT. Physical units remain conditional because DX and DT lack source unit labels.
The inferred geometric surface convention and verified actual image availability are
recorded separately. Classical PIV is a related-method comparison and coarse initializer,
not independent experimental ground truth.

The analysis_code folder contains runnable source, including the unchanged local solver
and field evaluator. Prepared input arrays and intermediate fit checkpoints are omitted
from this results download; they can be regenerated from the six supplied input files.
Original input images/MAT files are not duplicated in this archive.

Reproduction from analysis_code (Python with NumPy, SciPy, Pillow, h5py, Matplotlib):
  python work/pairs/prepare_pairs.py --input-dir /path/to/your/files 80 100
  python work/pairs/process_executor.py --pair 80 --workers 4
  python work/pairs/process_executor.py --pair 100 --workers 4
  python work/pairs/process_fit_pairs.py --pairs 80 100 --workers 4 --checkpoint-every 2048
  python work/pairs/finalize_pairs.py --pairs 80 100
  python work/pairs/build_deliverables.py
  python work/pairs/deep_quiver_pairs.py

The same conservative thresholds are used for both fresh image pairs. The depth limit
of the earlier observation123 analysis is removed. Each diagonal derivative is screened
separately; no zero-divergence condition is imposed. Model stages are resumable.
The fork-process launchers change only execution scheduling. Their
stratified benchmarks reproduced every tested numerical output bitwise. The
original thread-based driver remains available as a portable fallback.

SHA256SUMS.json lists every packaged member except itself.
'''
code=[WORK/k for k in ['prepare_pairs.py','track_pairs.py','process_executor.py','process_fit_pairs.py','fit_pairs.py','finalize_pairs.py','export_pairs.py','report_pairs.py','build_deliverables.py','deep_quiver_pairs.py','audit_inputs.py','audit_tracks.py','audit_final_fields.py']]
code += [ROOT/'work/extended_flow.py',ROOT/'work/ptv_model.py',ROOT/'work/field_model.py',ROOT/'work/full_width/fit_local_cached.py']
members={}
for pair in [80,100]:
    folder=OUT/('pair_'+str(pair))
    audit=json.loads((WORK/str(pair)/'final_fields_audit.json').read_text())
    if not audit['passed']:raise RuntimeError('Final field audit failed for '+str(pair))
    track_audit=json.loads((WORK/str(pair)/'track_independent_audit.json').read_text())
    if not track_audit['all_checks_pass']:raise RuntimeError('Track audit failed for '+str(pair))
    for p in sorted(folder.iterdir()):
        if p.is_file():members[str(p.relative_to(OUT))]=p.read_bytes()
    for k in ['input_manifest.json','input_audit.json','track_independent_audit.json','final_fields_audit.json','coarse_ptv_summary.json','fit_summary.json']:
        p=WORK/str(pair)/k
        if p.exists():members['audit/pair_'+str(pair)+'/'+k]=p.read_bytes()
for p in code:members['analysis_code/'+str(p.relative_to(ROOT))]=p.read_bytes()
for k in ['track_driver_audit.json','process_executor_benchmark.json','fit_executor_benchmark.json','input_audit.json','runtime.json']:
    p=WORK/k
    if p.exists():members['audit/'+k]=p.read_bytes()
members['README.txt']=readme.encode()
members['pairs_80_100_summary.json']=(OUT/'pairs_80_100_summary.json').read_bytes()
manifest={name:{'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()} for name,blob in members.items()}
members['SHA256SUMS.json']=json.dumps(manifest,indent=2).encode()
target=OUT/'pairs_80_100_conservative_results.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,blob in members.items():z.writestr(name,blob)
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for name,entry in manifest.items():
        blob=z.read(name);assert len(blob)==entry['bytes'];assert hashlib.sha256(blob).hexdigest()==entry['sha256']
result=dict(file=str(target),members=len(members),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),all_members_verified=True)
(WORK/'bundle_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
