from pathlib import Path
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
