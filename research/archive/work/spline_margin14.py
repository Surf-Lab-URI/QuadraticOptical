"""Image-only cubic B-spline flow with bending regularization.
No manual target data are read before fixed registrations finish.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,json,time
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.ndimage import gaussian_filter,map_coordinates,uniform_filter
from scipy.optimize import minimize
from PIL import Image
from pathlib import Path
W=Path('work');D=Path('historical-user-files/Downloads');meta=np.load(W/'metadata.npz')
Araw=np.asarray(Image.open(D/'ExpLCL_1_03-123_imgA.tif'),dtype=float);Braw=np.asarray(Image.open(D/'ExpLCL_1_03-123_imgB.tif'),dtype=float)
sa=meta['surfa'].ravel()-1;sb=meta['surfb'].ravel()-1
spacing=16.;xmin,xmax,ymin,ymax=50.,400.,10.,190.;ox=xmin-2*spacing;oy=ymin-2*spacing;nx=int(np.ceil((xmax-xmin)/spacing))+5;ny=int(np.ceil((ymax-ymin)/spacing))+5;nc=nx*ny

def basis(points,derivative=None):
 t=(np.asarray(points)-[ox,oy])/spacing;base=np.floor(t).astype(int);f=t-base
 def b(f,deriv):
  if deriv:return np.array([-.5*(1-f)**2,1.5*f*f-2*f,-1.5*f*f+f+.5,.5*f*f]).T/spacing
  return np.array([(1-f)**3,3*f**3-6*f*f+4,-3*f**3+3*f*f+3*f+1,f**3]).T/6
 wx=b(f[:,0],derivative=='x');wy=b(f[:,1],derivative=='y');ii=base[:,0,None]+np.arange(-1,3);jj=base[:,1,None]+np.arange(-1,3)
 cols=(jj[:,:,None]*nx+ii[:,None,:]).reshape(len(t),16);vals=(wy[:,:,None]*wx[:,None,:]).reshape(len(t),16)
 return sparse.csr_matrix((vals.ravel(),(np.repeat(np.arange(len(t)),16),cols.ravel())),shape=(len(t),nc))

def bending():
 rows=[];cols=[];vals=[];nrow=0
 for yy in range(ny):
  for xx in range(1,nx-1):
   for dx,v in [(-1,1),(0,-2),(1,1)]:rows.append(nrow);cols.append(yy*nx+xx+dx);vals.append(v)
   nrow+=1
 for yy in range(1,ny-1):
  for xx in range(nx):
   for dy,v in [(-1,1),(0,-2),(1,1)]:rows.append(nrow);cols.append((yy+dy)*nx+xx);vals.append(v)
   nrow+=1
 for yy in range(ny-1):
  for xx in range(nx-1):
   for dx,dy,v in [(0,0,1),(1,0,-1),(0,1,-1),(1,1,1)]:rows.append(nrow);cols.append((yy+dy)*nx+xx+dx);vals.append(v*np.sqrt(2))
   nrow+=1
 return sparse.csr_matrix((vals,(rows,cols)),shape=(nrow,nc))
R=bending();H=R.T@R
# Seed from images only: reliable affine registration nodes, robust smooth fit.
seed=np.load(W/'propagate0.npz');sp=seed['points'];sd=seed['params'][:,:,0];sn=seed['ncc'];use=(sn>.65)&np.all(np.isfinite(sd),axis=1);sp=sp[use];sd=sd[use];basew=((sn[use]-.65)/.35+.1)**2;S=basis(sp);weights=basew.copy();C=np.zeros((nc,2))
for _ in range(5):
 sw=np.sqrt(weights);system=sparse.vstack([S.multiply(sw[:,None]),R*.15]);
 for axis in range(2):C[:,axis]=lsmr(system,np.r_[sd[:,axis]*sw,np.zeros(R.shape[0])],atol=1e-7,btol=1e-7,maxiter=500)[0]
 err=np.linalg.norm(S@C-sd,axis=1);weights=basew/np.sqrt(1+(err/3)**2)
init=C.copy();print('control grid',nx,ny,'seed nodes',len(sp),'seed median fit residual',np.median(np.linalg.norm(S@C-sd,axis=1)),flush=True)

def normalize(I,sigma):
 sm=gaussian_filter(I,sigma);hp=sm-gaussian_filter(I,7);rms=np.sqrt(gaussian_filter(hp*hp,9)+25);return np.clip(hp/rms,-3,4)
yg,xg=np.mgrid[10:191,50:401];points=np.column_stack([xg.ravel(),yg.ravel()]).astype(float);source_depth=points[:,1]-np.interp(points[:,0],np.arange(501),sa);use=source_depth>=14;points=points[use];Q=basis(points)
# Data include only fluid pixels, target mask uses smooth barrier on depth<10.
def solve(lam):
 coeff=init.copy();traces=[];t=time.time()
 for sigma,maxiter in [(1.6,1000),(.8,1600)]:
  A=normalize(Araw,sigma);B=normalize(Braw,sigma);ap=A[points[:,1].astype(int),points[:,0].astype(int)];sbprime=np.gradient(sb);last={};count=[0]
  def objective(flat):
   c=flat.reshape(nc,2);uv=Q@c;dest=points+uv;coords=[dest[:,1],dest[:,0]];ix=np.clip(np.floor(dest[:,0]).astype(int),0,499);iy=np.clip(np.floor(dest[:,1]).astype(int),0,499);fx=np.clip(dest[:,0]-ix,0,1);fy=np.clip(dest[:,1]-iy,0,1);b00=B[iy,ix];b10=B[iy,ix+1];b01=B[iy+1,ix];b11=B[iy+1,ix+1];bp=(1-fx)*(1-fy)*b00+fx*(1-fy)*b10+(1-fx)*fy*b01+fx*fy*b11;gx=(1-fy)*(b10-b00)+fy*(b11-b01);gy=(1-fx)*(b01-b00)+fx*(b11-b10)
   residual=bp-ap;delta=.65;rho=delta*delta*(np.sqrt(1+(residual/delta)**2)-1);psi=residual/np.sqrt(1+(residual/delta)**2)
   # Invalid mapped samples get a weak fixed residual weight and mask barrier;
   # this avoids rewarding flow for losing liquid-particle support.
   dep=dest[:,1]-np.interp(dest[:,0],np.arange(501),sb);invalid=np.minimum(dep-10,0);barrier=.02*np.mean(invalid*invalid);bargrad=.04*invalid/len(points)
   gradpix=np.column_stack([psi*gx,psi*gy])/len(points);gradpix[:,0]-=bargrad*(sb[ix+1]-sb[ix]);gradpix[:,1]+=bargrad
   hc=H@c;reg=lam*np.sum(c*hc)/R.shape[0];grad=Q.T@gradpix+2*lam*hc/R.shape[0]
   loss=float(np.mean(rho)+reg+barrier);count[0]+=1;last.update(data=float(np.mean(rho)),bend=float(reg),barrier=float(barrier),loss=loss,eval=count[0])
   return loss,grad.ravel()
  # Finite-difference audit at initial stage, a random directional derivative.
  if sigma==1.6:
   rng=np.random.default_rng(271);direction=rng.normal(size=coeff.size);direction/=np.linalg.norm(direction);base=coeff.ravel();val,g=objective(base);eps=1e-4;fd=(objective(base+eps*direction)[0]-objective(base-eps*direction)[0])/(2*eps);an=np.dot(g,direction);print('gradient audit',lam,fd,an,'difference',abs(fd-an),flush=True)
  result=minimize(objective,coeff.ravel(),jac=True,method='L-BFGS-B',bounds=[(-50,90),(-40,40)]*nc,options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-7,'maxls':30,'maxcor':12})
  coeff=result.x.reshape(nc,2);final=objective(result.x)[0];traces.append({'sigma':sigma,'success':bool(result.success),'message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'gradient_inf':float(np.max(np.abs(result.jac))),**last});print('lambda',lam,'sigma',sigma,'elapsed',round(time.time()-t,1),traces[-1],flush=True)
 # Export at regular field nodes plus dense integer ROI for independent plotting.
 gp=np.array([(x,y) for y in range(24,177,8) for x in range(60,385,8) if y>=sa[x]+12],float);G=basis(gp);disp=G@coeff;dx=basis(gp,'x')@coeff;dy=basis(gp,'y')@coeff
 dense=np.full((501,501,2),np.nan);allp=np.column_stack([xg.ravel(),yg.ravel()]);dense[yg,xg]=(basis(allp)@coeff).reshape(xg.shape+(2,))
 # Local NCC diagnostic on15x15 windows after dense warping.
 A=normalize(Araw,.8);B=normalize(Braw,.8);dest=points+Q@coeff;aw=A[points[:,1].astype(int),points[:,0].astype(int)];bw=map_coordinates(B,[dest[:,1],dest[:,0]],order=1,mode='nearest');mask=np.zeros((501,501));aa=mask.copy();bb=mask.copy();iy=points[:,1].astype(int);ix=points[:,0].astype(int);mask[iy,ix]=1;aa[iy,ix]=aw;bb[iy,ix]=bw
 blur=lambda v:uniform_filter(v,15,mode='constant')*225;n=blur(mask);ma=blur(aa)/np.maximum(n,1);mb=blur(bb)/np.maximum(n,1);cov=blur(aa*bb)/np.maximum(n,1)-ma*mb;va=blur(aa*aa)/np.maximum(n,1)-ma*ma;vb=blur(bb*bb)/np.maximum(n,1)-mb*mb;localncc=cov/np.sqrt(np.maximum(va*vb,1e-9));localncc[n<100]=np.nan;gncc=localncc[gp[:,1].astype(int),gp[:,0].astype(int)]
 tag=str(lam).replace('.','p');name='spline_margin14_l'+tag
 np.savez_compressed(W/(name+'.npz'),points=gp,disp=disp,params=np.stack([disp,dx,dy],axis=2),ncc=gncc,coefficients=coeff,initial_coefficients=init,flow=dense,local_ncc=localncc,radius=1.,order=1,warp_degree=3,spacing=spacing,lam=lam,origin=[ox,oy],grid_shape=[ny,nx],source_margin=14,derivative_units='pixel displacement per pixel')
 (W/(name+'_optimization.json')).write_text(json.dumps(traces,indent=2));return name,coeff,traces
if __name__=='__main__':
 results=[]
 for lam in [.001]:results.append(solve(lam))
 # Validation happens only after all fixed image-only runs finish.
 p=meta['p'];xy=p[:,0,:].T-1;truth=(p[:,1,:]-p[:,0,:]).T;Qm=basis(xy);depth=xy[:,1]-np.interp(xy[:,0],np.arange(501),sa);report=[]
 for name,c,traces in [('spline_initial',init,[])]+results:
  pred=Qm@c;e=np.linalg.norm(pred-truth,axis=1);metrics={}
  for label,sel in [('all',np.ones(len(xy),bool)),('depth<40',depth<40),('depth40-80',(depth>=40)&(depth<80)),('depth80+',depth>=80)]:
   a=e[sel];metrics[label]={'n':len(a),'median':float(np.median(a)),'mean':float(np.mean(a)),'rmse':float(np.sqrt(np.mean(a*a))),'p90':float(np.percentile(a,90)),'under1':float(np.mean(a<1)),'under2':float(np.mean(a<2))}
  report.append({'name':name,'metrics':metrics});np.savez_compressed(W/(name+'_validation.npz'),xy=xy,pred=pred,truth=truth,error=e,depth=depth);print('VALIDATION',name,json.dumps(metrics),flush=True)
 (W/'spline_margin14_metrics.json').write_text(json.dumps(report,indent=2))
