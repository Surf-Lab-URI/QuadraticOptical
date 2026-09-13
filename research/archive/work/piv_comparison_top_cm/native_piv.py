"""Comparison-only native PIV loader; never modifies image-only model inputs.

Primary availability retains finite supplied vectors with source-image support.
Target-endpoint and correlation support are separate sensitivity diagnostics.
No dense delta_x1/delta_z1 values are read and no missing values are filled.
"""
from pathlib import Path
import hashlib
import h5py
import numpy as np
from scipy.ndimage import map_coordinates

HERE=Path(__file__).resolve().parent
IMAGE_ONLY=HERE.parent/'image_only_cm'
SOURCE=Path('historical-user-files/Downloads')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def visible(mask,query):
    q=np.asarray(query,float).reshape(-1,2)
    inside=np.isfinite(q).all(axis=1)&(q[:,0]>=0)&(q[:,0]<=mask.shape[1]-1)&(q[:,1]>=0)&(q[:,1]<=mask.shape[0]-1)
    out=np.zeros(len(q),bool)
    out[inside]=map_coordinates(np.asarray(mask,float),[q[inside,1],q[inside,0]],order=1,mode='constant',cval=0.)>.99
    return out


def depth(surface,query):
    q=np.asarray(query,float).reshape(-1,2)
    return q[:,1]-np.interp(q[:,0],np.arange(len(surface)),surface)


def strict_bilinear(x_axis,y_axis,values,node_valid,query):
    """No extrapolation; every mathematically positive-weight corner required.

    values has shape [y,x,...]; returns (interpolated_values, supported).
    Invalid corners with exactly zero weight do not invalidate an exact node.
    """
    x=np.asarray(x_axis);y=np.asarray(y_axis);v=np.asarray(values);q=np.asarray(query,float).reshape(-1,2)
    if v.shape[:2]!=(len(y),len(x)) or np.asarray(node_valid).shape!=v.shape[:2]:raise ValueError('Native array shape mismatch.')
    extra=v.shape[2:];flat=v.reshape(len(y),len(x),-1)
    inside=np.isfinite(q).all(axis=1)&(q[:,0]>=x[0])&(q[:,0]<=x[-1])&(q[:,1]>=y[0])&(q[:,1]<=y[-1])
    ii=np.clip(np.searchsorted(x,np.where(inside,q[:,0],x[0]),side='right')-1,0,len(x)-2)
    jj=np.clip(np.searchsorted(y,np.where(inside,q[:,1],y[0]),side='right')-1,0,len(y)-2)
    tx=(np.where(inside,q[:,0],x[0])-x[ii])/(x[ii+1]-x[ii])
    ty=(np.where(inside,q[:,1],y[0])-y[jj])/(y[jj+1]-y[jj])
    weights=np.stack([(1-tx)*(1-ty),tx*(1-ty),(1-tx)*ty,tx*ty],axis=1)
    ci=np.stack([ii,ii+1,ii,ii+1],axis=1);cj=np.stack([jj,jj,jj+1,jj+1],axis=1)
    corner=flat[cj,ci];positive=weights>0
    valid=inside&np.all(~positive|(np.asarray(node_valid)[cj,ci]&np.isfinite(corner).all(axis=2)),axis=1)
    out=np.sum(weights[:,:,None]*np.where(positive[:,:,None],corner,0.),axis=1)
    out[~valid]=np.nan
    return out.reshape((len(q),)+extra),valid


class NativePIV:
    def __init__(self,pair,input_path=None,mat_path=None):
        self.pair=str(pair)
        self.input_path=Path(input_path) if input_path else IMAGE_ONLY/self.pair/'inputs.npz'
        self.mat_path=Path(mat_path) if mat_path else SOURCE/('ExpLCL_1_03_'+self.pair+'_PIV.mat')
        with np.load(self.input_path,allow_pickle=False) as z:
            if not bool(z['image_only']) or bool(z['supplied_velocity_used']):raise ValueError('Expected unchanged fresh image-only inputs.')
            self.geometry={k:z[k] for k in ['availability_a','availability_b','surface_a','surface_b','DX','DT']}
        self.DX=float(self.geometry['DX']);self.DT=float(self.geometry['DT'])
        with h5py.File(self.mat_path,'r') as f:
            c=f['compVel'];self.x=c['xPIV'][:].ravel()-1;self.y=c['zPIV'][:].ravel()-1
            self.dx=c['delta_x'][:].T.copy();self.dy=-c['delta_z'][:].T.copy();self.dcor=c['dcor'][:].T.copy()
            self.metadata={k:float(c[k][0,0]) for k in ['DX','DT','IW','GS']}
        if self.metadata['DX']!=self.DX or self.metadata['DT']!=self.DT:raise ValueError('Calibration differs from image-only inputs.')
        if np.any(np.diff(self.x)<=0) or np.any(np.diff(self.y)<=0):raise ValueError('Native axes must increase.')
        self.disp=np.stack([self.dx,self.dy],axis=2)
        self.finite=np.isfinite(self.disp).all(axis=2)
        xx,yy=np.meshgrid(self.x,self.y);q=np.c_[xx.ravel(),yy.ravel()]
        shape=xx.shape
        self.source_depth=depth(self.geometry['surface_a'],q).reshape(shape)
        self.source_visible=visible(self.geometry['availability_a'],q).reshape(shape)
        self.source_underwater=self.source_depth>=0
        self.valid_source=self.finite&self.source_visible&self.source_underwater
        target=q+self.disp.reshape(-1,2)
        self.target_depth=depth(self.geometry['surface_b'],target).reshape(shape)
        self.target_visible=visible(self.geometry['availability_b'],target).reshape(shape)
        with np.errstate(invalid='ignore'):
            self.valid_source_target=self.valid_source&self.target_visible&(self.target_depth>=0)
        self.valid_with_dcor=self.valid_source&np.isfinite(self.dcor)

    def sample(self,query):
        q=np.asarray(query,float).reshape(-1,2)
        d,corners=strict_bilinear(self.x,self.y,self.disp,self.valid_source,q)
        source_visible=visible(self.geometry['availability_a'],q)
        source_depth=depth(self.geometry['surface_a'],q)
        available=corners&source_visible&(source_depth>=0)
        d[~available]=np.nan
        correlation,cor_corners=strict_bilinear(self.x,self.y,self.dcor,self.valid_with_dcor,q)
        dcor_available=available&cor_corners;correlation[~dcor_available]=np.nan
        _,target_corners=strict_bilinear(self.x,self.y,self.disp,self.valid_source_target,q)
        target=q+d
        target_visible=visible(self.geometry['availability_b'],target)
        target_depth=depth(self.geometry['surface_b'],target)
        with np.errstate(invalid='ignore'):
            available_with_target=available&target_corners&target_visible&(target_depth>=0)
        return dict(query=q,disp=d,available=available,source_visible=source_visible,source_depth=source_depth,
            dcor=correlation,dcor_available=dcor_available,target_visible=target_visible,target_depth=target_depth,
            available_with_target=available_with_target,
            available_with_dcor=dcor_available)


def load_native(pair,**kwargs):
    return NativePIV(pair,**kwargs)
