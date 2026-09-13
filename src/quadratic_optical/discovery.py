"""Pair discovery is based on explicit A/B suffixes, never directory order."""
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass(frozen=True)
class ImagePair:
    name: str
    image_a: Path
    image_b: Path
    piv_mat: object = None
    surface_file: object = None
    experiment: object = None
    pair_number: object = None

def identify(name):
    m=re.match(r'^(.+?)[_-](\d+)$',name)
    return (m.group(1),int(m.group(2))) if m else (None,None)

def discover(directory,recursive=False,skip_incomplete=False):
    root=Path(directory).expanduser().resolve()
    if not root.is_dir():raise ValueError('Image directory does not exist: '+str(root))
    found={}
    for p in sorted(root.rglob('*') if recursive else root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ['.tif','.tiff','.png']:continue
        m=re.match(r'^(.+?)(?:_img([AB])|_([AB]))$',p.stem,re.I)
        if not m:continue
        key=(p.parent,m.group(1));frame=(m.group(2) or m.group(3)).upper()
        if frame in found.setdefault(key,{}):raise ValueError('Ambiguous duplicate '+frame+' image for '+str(key))
        found[key][frame]=p
    pairs=[];incomplete=[];names=set()
    for (parent,stem),frames in sorted(found.items(),key=lambda item:str(item[0])):
        relative=parent.relative_to(root)
        name='__'.join(relative.parts+(stem,))
        if name in ['.','..']:raise ValueError('Unsafe output name from image filenames: '+name)
        if set(frames)!={'A','B'}:
            incomplete.append(name)
            continue
        if name in names:raise ValueError('Duplicate output name after recursive discovery: '+name)
        names.add(name);exp,num=identify(stem)
        piv=parent/(stem+'_PIV.mat');surface=parent/(stem+'_surface.npz')
        for sidecar in (piv,surface):
            if sidecar.exists() and not sidecar.is_file():raise ValueError('Expected a sidecar file, found a directory: '+str(sidecar))
        pairs.append(ImagePair(name,frames['A'],frames['B'],piv if piv.exists() else None,
            surface if surface.exists() else None,exp,num))
    if incomplete and not skip_incomplete:raise ValueError('Incomplete A/B pairs: '+', '.join(incomplete)+'. Supply the missing image or use --skip-incomplete.')
    if not pairs:raise ValueError('No image pairs found. Expected NAME_imgA.tif + NAME_imgB.tif or NAME_a.tif + NAME_b.tif (also TIFF/PNG).')
    return pairs,incomplete
