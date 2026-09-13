from pathlib import Path
import shutil,json,zipfile,hashlib
root=Path.cwd();bundle=root/'work'/'bundle';(bundle/'work').mkdir(parents=True,exist_ok=True);(bundle/'inputs').mkdir(exist_ok=True);(bundle/'outputs').mkdir(exist_ok=True)
scripts=['extended_flow.py','variant_run.py','field_model.py','ptv_refine.py','ptv_model.py','final_reverse_check.py','prepare_final.py','export_results.py','synthetic_validate.py','numeric_qa.py']
for name in scripts:
 s=(root/'work'/name).read_text()
 s=s.replace('historical-user-files/Downloads','inputs').replace('work','work').replace('outputs','outputs')
 (bundle/'work'/name).write_text(s)
checkpoints=['metadata.npz','propagate0.npz','seed_dis.npz','baseline_reference_matlab.npz','baseline_ncc.npz','final_spline13.npz','final_ptv13.npz','final_ptv19.npz','final_affine13.npz','final_ptv13_margin14.npz','final_reverse_field.npz','ptv_tracks.npz','seed_ptv.npz']
for name in checkpoints:shutil.copy2(root/'work'/name,bundle/'work'/name)
notes=['reproduction_check.json','methods_research.md','data_audit.md','ptv_report.md','spline_report.md','synthetic_validation.md','synthetic_validation.json','numeric_qa.json','baseline_report.md','final_review.md','final_summary.json','baseline_metrics_matlab.json','spline_converged_metrics.json','ptv_metrics.json','final_ptv13_blended_metrics.json','final_affine13_blended_metrics.json','final_ptv19_blended_metrics.json','final_ptv13_margin14_blended_metrics.json']
(bundle/'notes').mkdir(exist_ok=True)
for name in notes:shutil.copy2(root/'work'/name,bundle/'notes'/name)
for p in Path('historical-user-files/Downloads').glob('ExpLCL_1_03-123*'):
 if p.suffix in ['.mat','.tif']:shutil.copy2(p,bundle/'inputs'/p.name)
(bundle/'requirements.txt').write_text('numpy\nscipy\nmatplotlib\nPillow\nh5py\n')
(bundle/'README.md').write_text('''# Capillary-lump analysis: source and image-derived checkpoints

The input TIFFs and original MAT are copies; the originals were not modified.

Run from this extracted directory in a Python environment with the packages in requirements.txt:

    python run_analysis.py

This reruns automatic detection/tracking, the four final affine/quadratic image refinements, correct backward registration, field blending, screening and exports. Outputs are written under outputs/. Manual endpoints enter comparison only, never the fitted image field.

The coarse image-only seed `work/propagate0.npz` is frozen explicitly. It came from the exploratory NCC/DIS/affine stage and is included because that stage evolved during the investigation, including correction of MATLAB coordinate bookkeeping. This package reproduces the final analysis from this supplied image-derived coarse checkpoint, rather than claiming an untouched one-command implementation of a published algorithm from raw images. The baseline comparison checkpoints and alternative spline seed are likewise frozen. No manual correspondence endpoints were incorporated in any seed.

For fast field queries without rerunning tracking, use:

    python query_field.py 195 67

Coordinates are zero-based source A pixels. Returned displacement is pixels per pair, x right/y down. `gradient[component,axis]` differentiates that same source-indexed displacement. Raw queries are estimates, not automatically screened measurements; consult output masks for support.

The primary MATLAB output offers named coordinate, displacement, gradient and support arrays. In the NumPy output, query rows 0:200 are manual source positions, rows 200:927 are the727 image grid nodes, and remaining rows are the flattened85×166 dense display grid. `dense_x`/`dense_y` identify the latter. Apply `accepted` to displacement, `gradient_accepted` to derivatives. Raw estimates are preserved so flagged predictions remain reviewable.

Unknown time and spatial calibration are NaN. With length/pixel L and seconds/pair dt, physical horizontal velocity is L*dx/dt; upward vertical velocity is -L*dy/dt. The horizontal velocity gradient is gradient[:,0,0]/dt for isotropic calibration. Source versus midpoint coordinates are distinct finite-time representations.

Numerics were run using Python3.8.3, NumPy1.18.5, SciPy1.5.0, Matplotlib3.2.2, Pillow7.2.0 and h5py2.10.0 already installed on the machine. Newer libraries may cause small numerical/optimizer differences; the frozen final-model checkpoints allow exact field inspection. No dependency downloads were needed. notes/ contains the method audits and comparison records. input_and_checkpoint_sha256.json records hashes.
''')
(bundle/'query_field.py').write_text('''import sys,numpy as np\nsys.path.insert(0,"work")\nfrom field_model import LocalField\nq=np.array([[float(sys.argv[1]),float(sys.argv[2])]])\nd,g=LocalField("work/final_ptv13.npz").evaluate(q)\nprint("A coordinates (px):",q[0])\nprint("Displacement x/y_down (px/pair):",d[0])\nprint("Source-coordinate displacement gradient [component,axis]:\\n",g[0])\nprint("Raw estimate; inspect support/sensitivity masks before interpretation.")\n''')
(bundle/'run_analysis.py').write_text('''from pathlib import Path
import subprocess,sys,numpy as np,os
os.chdir(Path(__file__).resolve().parent)
Path('outputs').mkdir(exist_ok=True)
def run(*args):
    subprocess.run([sys.executable,*args],check=True)
run('work/ptv_refine.py')
f=np.load('work/ptv_quad10.npz');n=int(f['grid_count'])
np.savez('work/seed_ptv.npz',points=f['points'][:n],params=np.concatenate([f['disp'][:n,:,None],f['gradient'][:n]*13],axis=2),radius=13)
for name,radius,order,margin in [('final_ptv13',13,2,10),('final_ptv19',19,2,10),('final_affine13',13,1,10),('final_ptv13_margin14',13,2,14)]:
    run('work/variant_run.py','--name',name,'--radius',str(radius),'--order',str(order),'--seedfile','work/seed_ptv.npz','--maskednorm','--margin',str(margin))
run('work/field_model.py')
run('work/final_reverse_check.py')
f=np.load('work/final_ptv13.npz');r=np.load('work/final_ptv13_reverse.npz');data={k:r[k] for k in r.files};data['points']=f['points']+f['params'][:,:,0];data['radius']=13
np.savez('work/final_reverse_field.npz',**data)
run('work/prepare_final.py')
run('work/export_results.py')
print('Final numerical fields and figures are in outputs/.')
''')
included=[bundle/'work'/name for name in scripts+checkpoints]+list((bundle/'inputs').glob('*'))+[bundle/'notes'/name for name in notes]+[bundle/name for name in ['README.md','requirements.txt','run_analysis.py','query_field.py']]
manifest={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in included}
(bundle/'input_and_checkpoint_sha256.json').write_text(json.dumps(manifest,indent=2))
zip_path=root/'outputs'/'analysis_code_and_checkpoints.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in included+[bundle/'input_and_checkpoint_sha256.json']:
  z.write(p,arcname=str(p.relative_to(bundle)))
print(zip_path,'bytes',zip_path.stat().st_size)
