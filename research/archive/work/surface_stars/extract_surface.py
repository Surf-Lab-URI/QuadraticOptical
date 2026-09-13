"""Stream a MATLAB v5 campaign file, retaining only selected-run scalar/vector data."""
from pathlib import Path
import struct,zlib,json,hashlib
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=Path('data/results.mat')
TARGET='ExpLCL_1_03'

class Stream:
    def __init__(self,file,size):
        self.file=file;self.remaining=size;self.z=zlib.decompressobj();self.pending=b'';self.buf=bytearray();self.pos=0
    def pump(self):
        if not self.pending and self.remaining:
            n=min(65536,self.remaining);self.pending=self.file.read(n);self.remaining-=len(self.pending)
        out=self.z.decompress(self.pending,65536);self.pending=self.z.unconsumed_tail
        self.buf.extend(out)
        if not out and not self.pending and not self.remaining:raise EOFError('Compressed stream ended')
    def read(self,n):
        while len(self.buf)<n:self.pump()
        b=bytes(self.buf[:n]);del self.buf[:n];self.pos+=n;return b
    def skip(self,n):
        while n:
            k=min(n,65536);self.read(k);n-=k

def tag(r):
    a,b=struct.unpack('<II',r.read(8))
    if a>>16:return a&65535,a>>16,struct.pack('<I',b)[:a>>16]
    return a,b,None

def value(r,t):
    typ,n,small=t
    if small is not None:return small
    a=r.read(n);r.skip((-n)%8);return a

def skip(r,t):
    typ,n,small=t
    if small is None:r.skip(n+(-n)%8)

def header(r,t):
    typ,n,small=t
    assert typ==14 and small is None
    end=r.pos+n
    flags=np.frombuffer(value(r,tag(r)),dtype='<u4');cls=int(flags[0])&255
    dims=np.frombuffer(value(r,tag(r)),dtype='<i4')
    name=value(r,tag(r)).decode('utf8')
    return cls,dims,name,end

def finish(r,end):
    assert r.pos<=end,(r.pos,end)
    r.skip(end-r.pos)

def read_leaf(r,t):
    cls,dims,name,end=header(r,t)
    if not np.prod(dims):finish(r,end);return np.array([])
    elem=tag(r);typ=elem[0];data=value(r,elem)
    dtypes={1:'i1',2:'u1',3:'<i2',4:'<u2',5:'<i4',6:'<u4',7:'<f4',9:'<f8',12:'<i8',13:'<u8',16:'u1',17:'<u2',18:'<u4'}
    if cls==4:
        if typ in [1,2,16]:out=data.decode('utf8')
        elif typ in [3,4,17]:out=data.decode('utf-16-le')
        elif typ==18:out=data.decode('utf-32-le')
        else:raise ValueError('Unknown char type '+str(typ))
    else:
        out=np.frombuffer(data,dtype=dtypes[typ]).copy().reshape(tuple(dims),order='F').squeeze()
    finish(r,end);return out

def fields(r):
    length=int(np.frombuffer(value(r,tag(r)),dtype='<i4')[0])
    names=value(r,tag(r));return [names[i:i+length].split(b'\0')[0].decode('utf8') for i in range(0,len(names),length)]

WANTED={
 'USurf':{'t','IRfps','IRDX','usurf0','usurf1','usurfComp','usurffilt','TMVTech','Ndots_used','corrmax0','corrmax1','corr_threshold','surfVelFit','A','t_0','t_lf','maxSurfVel'},
 'PIV':{'t','pairNum','IR_idx'},
 'Surfs':{'t','pairNum','IR_idx','dx','dt_pair','pps','spp'},
}

def read_substruct(r,t,which):
    cls,dims,name,end=header(r,t);assert cls==2 and np.prod(dims)==1
    names=fields(r);out={}
    for field in names:
        t=tag(r)
        if field in WANTED[which]:out[field]=read_leaf(r,t)
        else:skip(r,t)
    finish(r,end);return out,names

def read_run(r,t,index):
    cls,dims,name,end=header(r,t)
    if cls!=2 or np.prod(dims)==0:finish(r,end);return None,{'index_matlab':index,'empty':True}
    assert np.prod(dims)==1
    names=fields(r);out={};exp=None;subfields={}
    assert 'exp_name' in names,names
    for field in names:
        t=tag(r)
        if field=='exp_name':exp=read_leaf(r,t);out[field]=exp
        elif field in WANTED:out[field],subfields[field]=read_substruct(r,t,field)
        elif field=='number_of_pair':out[field]=read_leaf(r,t)
        else:skip(r,t)
    finish(r,end)
    print('Read run',index,exp,flush=True)
    return (out if exp==TARGET else None),{'index_matlab':index,'exp_name':exp,'fields':names,'selected_substruct_fields':subfields}

def main():
    records=[];selected=None
    with SOURCE.open('rb') as f:
        hdr=f.read(128);assert hdr[126:128]==b'IM'
        while True:
            start=f.tell();b=f.read(8)
            if not b:break
            typ,n=struct.unpack('<II',b)
            assert typ==15,'Expected compressed top-level elements'
            r=Stream(f,n);t=tag(r);cls,dims,name,end=header(r,t)
            if name=='exps':
                assert cls==1
                for i in range(int(np.prod(dims))):
                    result,record=read_run(r,tag(r),i+1);records.append(record)
                    if result is not None:
                        assert selected is None,'Duplicate target experiment';selected=result
                finish(r,end)
            f.seek(start+8+n)
    assert selected is not None,'Experiment not found'
    arrays={section+'__'+k:np.asarray(v) for section,sub in selected.items() if isinstance(sub,dict) for k,v in sub.items()}
    arrays.update(exp_name=np.array(selected['exp_name']),number_of_pair=np.asarray(selected['number_of_pair']))
    np.savez_compressed(HERE/'selected_run.npz',**arrays)
    (HERE/'extraction_metadata.json').write_text(json.dumps({'source':str(SOURCE),'source_bytes':SOURCE.stat().st_size,'target':TARGET,'method':'Streaming MATLAB v5 parser; large image/profile arrays skipped, selected small fields retained.','runs':records,'selected_fields':list(arrays)},indent=2)+'\n')
    print('Selected fields',[(k,v.shape) for k,v in arrays.items()],flush=True)

if __name__=='__main__':main()
