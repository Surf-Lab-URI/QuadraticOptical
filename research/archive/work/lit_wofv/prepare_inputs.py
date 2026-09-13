"""Freeze cropped inputs and separate manual references for future MATLAB run."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import savemat
base=Path('historical-user-files/Downloads');out=Path(__file__).resolve().parent
meta=np.load(out.parent/'metadata.npz')
a_path=base/'ExpLCL_1_03-123_imgA.tif';b_path=base/'ExpLCL_1_03-123_imgB.tif'
A=np.asarray(Image.open(a_path));B=np.asarray(Image.open(b_path));sa=meta['surfa'].ravel()-1;sb=meta['surfb'].ravel()-1
x0,y0,width,height=32,0,448,256
A=A[y0:y0+height,x0:x0+width];B=B[y0:y0+height,x0:x0+width]
yy,xx=np.indices(A.shape);globalx=xx+x0;globaly=yy+y0
validA=globaly>=sa[globalx]+14;validB=globaly>=sb[globalx]+14
savemat(str(out/'fit_inputs.mat'),dict(A=A,B=B,valid_A=validA,valid_B=validB,common_valid=validA&validB,crop_origin0=[x0,y0],surface_A0=sa,surface_B0=sb))
p=meta['p'];q=p[:,0].T-1;truth=p[:,1].T-p[:,0].T;depth=q[:,1]-np.interp(q[:,0],np.arange(501),sa)
savemat(str(out/'manual_reference.mat'),dict(source0=q,displacement=truth,depth=depth))
manifest=dict(source_commit='71c434fd63daf2f7054cf0cb90f0dee72daa56c0',source_repository='https://github.com/Shrediquette/PIVlab',images=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [a_path,b_path]],crop_origin0=[x0,y0],crop_shape=[height,width],manual_count=len(q),near40_count=int(np.sum(depth<40)),common_mask='intersection of both per-frame liquid masks at fixed image coordinates; each excludes depth <14 px; this is not a warped dual mask',status='prepared but MATLAB execution blocked before initialization')
(out/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
