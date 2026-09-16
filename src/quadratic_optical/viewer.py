"""Self-contained browser viewer: vector fields drawn over the particle images.

One HTML file per image pair, with no external dependencies and nothing fetched
at open time, so it can be copied or emailed and still work. The frames are
embedded losslessly: the point of the tool is to flip between A and B and judge
by eye whether the arrows match the particles that moved, and lossy compression
smears exactly the specks being judged.

Layers are whatever that pair actually has. Every processed pair has the
image-only field; supplied PIV needs a companion MAT; hand-matched picks exist
for only a few pairs. Controls show what is present and nothing else.
"""
from pathlib import Path
import base64, html, io, json
import numpy as np
from PIL import Image

from .matio import read_native_piv

SURFACE_MARGIN_PX = 24.     # headroom above the highest surface point, for wave crests
PIV_STRIDE = 4              # native PIV is 4 px; decimate before embedding

# Report figures, in reading order, appended below the interactive canvas.
FIGURE_ORDER = ['quiver', 'gradient_comparison', 'gradients', 'profiles', 'manual_comparison',
                'field_u', 'field_u_z', 'field_w', 'field_w_z',
                'field_u_smooth40px', 'field_u_smooth40px_z',
                'field_w_smooth40px', 'field_w_smooth40px_z',
                'field_dudx_from_smooth40px', 'field_dudx_from_smooth40px_z']


def _png_data_uri(values):
    """Lossless 8-bit PNG as a data URI."""
    array = np.clip(np.rint(np.asarray(values, float)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode='L').save(buffer, format='PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii')


def crop_rows(surface_a, surface_b, depth_px, height, margin=SURFACE_MARGIN_PX):
    """Rows spanning a margin above the highest surface to depth_px below the lowest."""
    top = int(max(0, np.floor(min(surface_a.min(), surface_b.min())-margin)))
    bottom = int(min(height, np.ceil(max(surface_a.max(), surface_b.max())+depth_px)+1))
    if bottom-top < 8:
        raise ValueError('Crop is degenerate; check the surface traces and requested depth.')
    return top, bottom


def _vectors(x, y, dx, dy, keep):
    """Compact rows for the browser: origin and displacement, rounded."""
    keep = np.asarray(keep, bool)
    return [[round(float(a), 1), round(float(b), 1), round(float(c), 3), round(float(d), 3)]
            for a, b, c, d in zip(np.asarray(x)[keep], np.asarray(y)[keep],
                                  np.asarray(dx)[keep], np.asarray(dy)[keep])]


def collect(directory, piv_path=None, manual_directory=None, piv_stride=PIV_STRIDE,
            depth_m=None, margin=SURFACE_MARGIN_PX):
    """Everything one viewer page needs, already cropped and decimated."""
    directory = Path(directory)
    with np.load(directory/'inputs.npz', allow_pickle=False) as f:
        raw_a = f['rawA'].astype(float); raw_b = f['rawB'].astype(float)
        surface_a = f['surface_a'].astype(float); surface_b = f['surface_b'].astype(float)
    frozen = np.load(directory/'results.npz', allow_pickle=True)
    dx_m = float(np.asarray(frozen['DX']).ravel()[0]); dt_s = float(np.asarray(frozen['DT']).ravel()[0])
    if depth_m is None:
        depth_m = float(np.asarray(frozen['requested_depth_m']).ravel()[0])
    origin = np.asarray(frozen['origin0'], float).reshape(2) if 'origin0' in frozen.files else np.zeros(2)
    height, width = raw_a.shape
    top, bottom = crop_rows(surface_a, surface_b, depth_m/dx_m, height, margin)

    layers = {}
    query = np.asarray(frozen['query'], float)
    disp = np.asarray(frozen['disp'], float)
    accepted = np.asarray(frozen['accepted'], bool)
    inside = (query[:, 1] >= top) & (query[:, 1] < bottom) & np.isfinite(disp).all(axis=1)
    layers['of'] = {'label': 'Image-only optical flow (fitting grid)', 'colour': '#d54b28',
                    'default_on': True,
                    'vectors': _vectors(query[:, 0], query[:, 1]-top, disp[:, 0], disp[:, 1],
                                        inside & accepted),
                    'withheld': _vectors(query[:, 0], query[:, 1]-top, disp[:, 0], disp[:, 1],
                                         inside & ~accepted),
                    'spacing_px': float(np.diff(np.unique(query[:, 0]))[0]) if len(np.unique(query[:, 0])) > 1 else 8.}

    if piv_path is not None and Path(piv_path).is_file():
        native = read_native_piv(piv_path)
        px = np.asarray(native['x_px'], float); py = np.asarray(native['y_px'], float)
        pd = np.asarray(native['disp_px'], float)
        mask = native.get('mask')
        step = max(1, int(piv_stride))
        rows = np.arange(0, len(py), step); cols = np.arange(0, len(px), step)
        grid_x, grid_y = np.meshgrid(px[cols]-origin[0], py[rows]-origin[1])
        sub = pd[np.ix_(rows, cols)]
        good = np.isfinite(sub).all(axis=2)
        if mask is not None:
            good &= np.asarray(mask)[np.ix_(rows, cols)].astype(bool)
        good &= (grid_y >= top) & (grid_y < bottom)
        layers['piv'] = {'label': 'Supplied PIV (every %d node)' % step, 'colour': '#2363b1',
                         'default_on': True,
                         'vectors': _vectors(grid_x.ravel(), grid_y.ravel()-top,
                                             sub[..., 0].ravel(), sub[..., 1].ravel(), good.ravel()),
                         'withheld': [], 'spacing_px': float(np.diff(px)[0]*step) if len(px) > 1 else 4.*step}

    if manual_directory:
        from .manual import find_manual
        manifest = json.loads((directory/'input_manifest.json').read_text())
        record = find_manual(manual_directory, manifest.get('experiment'), int(manifest['pair_number'])) \
            if manifest.get('pair_number') is not None else None
        if record is not None:
            source = record['source_px']-origin; move = record['displacement_px']
            keep = (source[:, 1] >= top) & (source[:, 1] < bottom)
            layers['manual'] = {'label': 'Hand-matched (%d)' % int(keep.sum()), 'colour': '#17806d',
                                'default_on': True,
                                'vectors': _vectors(source[:, 0], source[:, 1]-top,
                                                    move[:, 0], move[:, 1], keep),
                                'withheld': [], 'spacing_px': 0.}
            # The fitting grid never lands on a hand-picked particle, so a grid
            # arrow near a manual arrow is not the same measurement. Evaluate the
            # frozen field at the pick positions themselves for a true
            # one-to-one comparison: shared origin, so the gap between arrowheads
            # is the disagreement and its direction.
            from .manual import compare_manual
            paired = compare_manual(directory, record)
            predicted = np.asarray(paired['predicted_disp_px'], float)
            passed = np.asarray(paired['accepted_mask'], bool)
            usable = keep & np.isfinite(predicted).all(axis=1)
            layers['of_at_manual'] = {
                'label': 'Optical flow at those exact points (%d)' % int((usable & passed).sum()),
                'colour': '#d54b28', 'default_on': True, 'paired_with': 'manual',
                'vectors': _vectors(source[:, 0], source[:, 1]-top,
                                    predicted[:, 0], predicted[:, 1], usable & passed),
                'withheld': _vectors(source[:, 0], source[:, 1]-top,
                                     predicted[:, 0], predicted[:, 1], usable & ~passed),
                'spacing_px': 0.}
            # With a true one-to-one layer present, the dense grid starts hidden so
            # the first view is the comparison rather than a wash of arrows.
            layers['of']['default_on'] = False

    columns = np.arange(width)
    return {'pair': directory.name, 'width': int(width), 'height': int(bottom-top),
            'crop_top': int(top), 'crop_bottom': int(bottom),
            'DX': dx_m, 'DT': dt_s, 'depth_m': depth_m,
            'frames': {'A': _png_data_uri(raw_a[top:bottom]), 'B': _png_data_uri(raw_b[top:bottom])},
            'surface': {'A': [round(float(v), 2) for v in surface_a-top],
                        'B': [round(float(v), 2) for v in surface_b-top],
                        'x': [int(v) for v in columns[::16]]},
            'layers': layers}


def report_figures(directory, embed=False):
    """The pair's own report figures, in reading order, as HTML.

    Linked relatively by default: they already sit beside the page, so this costs
    nothing and keeps the viewer small. ``embed`` inlines them instead, for a page
    that must survive being moved on its own, at several megabytes a pair.
    """
    directory = Path(directory)
    found = sorted(p.name for p in directory.glob('*.png') if p.stem != 'viewer')
    def rank(name):
        stem = name[:-4]
        if stem in FIGURE_ORDER:
            return (FIGURE_ORDER.index(stem), name)
        return (len(FIGURE_ORDER), name)
    found.sort(key=rank)
    if not found:
        return ''
    pieces = []
    for name in found:
        stem = name[:-4]
        if embed:
            source = _png_data_uri(np.asarray(Image.open(directory/name).convert('L')))
            inner = '<img src="'+source+'" alt="'+html.escape(stem.replace('_', ' '))+'">'
        else:
            inner = '<img loading="lazy" src="'+name+'" alt="'+html.escape(stem.replace('_', ' '))+'">'
            if (directory/(stem+'.svg')).is_file():
                inner = '<a href="'+stem+'.svg">'+inner+'</a>'
        pieces.append('<figure><figcaption><code>'+html.escape(name)+'</code></figcaption>'+inner+'</figure>')
    note = ('' if embed else '<p class="note">Figures are linked from this pair\'s own directory '
            'rather than embedded, so the page stays small. Move the page on its own and these '
            'become blank; the interactive view above is self-contained either way.</p>')
    return ('<h2 class="figs">Report figures</h2>'+note+'<div class="figures">'+''.join(pieces)+'</div>')


_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>__TITLE__</title><style>
body{font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1e2530;margin:1.2rem auto;
 max-width:1400px;padding:0 1rem;background:#fff}
h1{font-size:19px;margin:0 0 .2rem}
.sub{color:#5a6775;font-size:13px;margin:0 0 .9rem}
.bar{display:flex;gap:15px;align-items:center;flex-wrap:wrap;background:#f7f9fc;border:1px solid #dde3ea;
 border-radius:8px;padding:9px 13px;margin:8px 0}
button{font:12.5px inherit;padding:5px 11px;border:1px solid #cfd6e0;background:#fff;border-radius:5px;cursor:pointer}
button.on{background:#1e2530;color:#fff;border-color:#1e2530}
label{font-size:12.5px;display:inline-flex;align-items:center;gap:6px}
input[type=range]{width:132px;vertical-align:middle}
canvas{width:100%;border:1px solid #dde3ea;border-radius:7px;display:block;background:#111;cursor:grab}
canvas.drag{cursor:grabbing}
.key{font-size:12.5px;color:#41505f;margin:8px 2px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:4px}
.note{font-size:12.5px;color:#5a6775;border-top:1px solid #e3e8ee;margin-top:1rem;padding-top:.7rem}
code{background:#eef3f6;padding:.1em .3em;border-radius:3px}
h2.figs{font-size:17px;margin:1.8rem 0 .3rem;border-top:1px solid #e3e8ee;padding-top:1.2rem}
.figures figure{margin:1.4rem 0}
.figures img{width:100%;height:auto;border:1px solid #e3e8ee;border-radius:6px;display:block}
.figures figcaption{font-size:12px;color:#5a6775;margin-bottom:.3rem}
</style>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<div class="bar">
  <strong style="font-size:12.5px">frame</strong>
  <span><button id="fA" class="on">A</button><button id="fB">B</button></span>
  <button id="flip">flip A/B</button>
  <label><input type="checkbox" id="surf" checked> surface</label>
  <span style="flex:1"></span>
  <button id="zin">+</button><button id="zout">&minus;</button><button id="reset">reset view</button>
</div>
<div class="bar" id="layerbar"><strong style="font-size:12.5px">layers</strong></div>
<div class="bar">
  <label>arrow length <input type="range" id="gain" min="1" max="60" step="1" value="8"></label>
  <span id="gainv" style="font-size:12.5px;width:34px">8&times;</span>
  <label>density <input type="range" id="dens" min="1" max="8" step="1" value="1"></label>
  <span id="densv" style="font-size:12.5px;width:70px">every node</span>
  <label><input type="checkbox" id="rej"> show withheld</label>
</div>
<canvas id="c" width="1800" height="420"></canvas>
<div class="key" id="key"></div>
<p class="note">Drag to pan, scroll to zoom. Arrows are drawn at the chosen multiple of the true
pixel displacement, from each vector's own origin. Withheld points are locations the conservative
screen declined to report; they are hidden by default. Frames are embedded losslessly so individual
particles stay sharp when flipping. __FOOT__</p>
<script>
var D=__DATA__;
var img={},ready=0,names=['A','B'];
names.forEach(function(n){var i=new Image();i.onload=function(){ready++;draw()};i.src=D.frames[n];img[n]=i});
var view={s:1,x:0,y:0},frame='A',gain=8,dens=1,showRej=false,showSurf=true;
var on={};Object.keys(D.layers).forEach(function(k){on[k]=D.layers[k].default_on!==false});
var c=document.getElementById('c'),g=c.getContext('2d');
function fit(){var s=c.width/D.width;view.s=s;view.x=0;view.y=(c.height-D.height*s)/2;}
function bar(){
  var b=document.getElementById('layerbar'),k=document.getElementById('key'),h='',kh='';
  Object.keys(D.layers).forEach(function(id){var L=D.layers[id];
    h+='<button data-l="'+id+'" class="'+(on[id]?'on':'')+'">'+L.label+'</button>';
    kh+='<span class="sw" style="background:'+L.colour+'"></span>'+L.label+' &nbsp; ';});
  b.innerHTML='<strong style="font-size:12.5px">layers</strong>'+h;k.innerHTML=kh;
  b.querySelectorAll('button').forEach(function(btn){btn.onclick=function(){
    var id=btn.dataset.l;on[id]=!on[id];btn.className=on[id]?'on':'';draw();};});
}
function arrow(X,Y,DX,DY){
  g.beginPath();g.moveTo(X,Y);g.lineTo(X+DX,Y+DY);g.stroke();
  var L=Math.hypot(DX,DY);if(L<2.5)return;var a=Math.atan2(DY,DX),h=Math.min(5,L*.45);
  g.beginPath();g.moveTo(X+DX,Y+DY);
  g.lineTo(X+DX-h*Math.cos(a-.42),Y+DY-h*Math.sin(a-.42));
  g.lineTo(X+DX-h*Math.cos(a+.42),Y+DY-h*Math.sin(a+.42));g.closePath();g.fill();
}
function draw(){
  g.setTransform(1,0,0,1,0,0);g.fillStyle='#111';g.fillRect(0,0,c.width,c.height);
  if(ready<2)return;
  g.imageSmoothingEnabled=view.s<2;
  g.setTransform(view.s,0,0,view.s,view.x,view.y);
  g.drawImage(img[frame],0,0);
  if(showSurf){g.strokeStyle='#37d0e0';g.lineWidth=1.2/view.s;g.beginPath();
    var S=D.surface[frame],X=D.surface.x;
    X.forEach(function(x,i){var y=S[x];if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();}
  Object.keys(D.layers).forEach(function(id){
    if(!on[id])return;var L=D.layers[id];
    g.strokeStyle=L.colour;g.fillStyle=L.colour;g.lineWidth=1.1/view.s;
    var step=(L.spacing_px>0)?dens:1;
    var pitch=L.spacing_px>0?L.spacing_px:1;
    L.vectors.forEach(function(v){
      if(step>1&&(Math.round(v[0]/pitch)%step||Math.round(v[1]/pitch)%step))return;
      arrow(v[0],v[1],v[2]*gain,v[3]*gain);});
    if(showRej&&L.withheld.length){g.strokeStyle='#ff5a5a';g.fillStyle='#ff5a5a';
      L.withheld.forEach(function(v){
        if(step>1&&(Math.round(v[0]/pitch)%step||Math.round(v[1]/pitch)%step))return;
        arrow(v[0],v[1],v[2]*gain,v[3]*gain);});}
  });
}
function setFrame(n){frame=n;document.getElementById('fA').className=n==='A'?'on':'';
  document.getElementById('fB').className=n==='B'?'on':'';draw();}
document.getElementById('fA').onclick=function(){setFrame('A')};
document.getElementById('fB').onclick=function(){setFrame('B')};
document.getElementById('flip').onclick=function(){setFrame(frame==='A'?'B':'A')};
document.getElementById('surf').onchange=function(e){showSurf=e.target.checked;draw()};
document.getElementById('rej').onchange=function(e){showRej=e.target.checked;draw()};
document.getElementById('gain').oninput=function(e){gain=+e.target.value;
  document.getElementById('gainv').textContent=gain+'\\u00d7';draw()};
document.getElementById('dens').oninput=function(e){dens=+e.target.value;
  document.getElementById('densv').textContent=dens===1?'every node':'every '+dens+parseSuffix(dens);draw()};
function parseSuffix(n){return n===2?'nd':n===3?'rd':'th'}
document.getElementById('zin').onclick=function(){zoom(1.4,c.width/2,c.height/2)};
document.getElementById('zout').onclick=function(){zoom(1/1.4,c.width/2,c.height/2)};
document.getElementById('reset').onclick=function(){fit();draw()};
function zoom(k,cx,cy){var s=Math.max(c.width/D.width*.5,Math.min(60,view.s*k));
  var r=s/view.s;view.x=cx-(cx-view.x)*r;view.y=cy-(cy-view.y)*r;view.s=s;draw();}
c.addEventListener('wheel',function(e){e.preventDefault();
  var r=c.getBoundingClientRect(),k=c.width/r.width;
  zoom(e.deltaY<0?1.12:1/1.12,(e.clientX-r.left)*k,(e.clientY-r.top)*k);},{passive:false});
var drag=null;
c.addEventListener('pointerdown',function(e){var r=c.getBoundingClientRect(),k=c.width/r.width;
  drag={x:(e.clientX-r.left)*k-view.x,y:(e.clientY-r.top)*k-view.y};c.className='drag';
  c.setPointerCapture(e.pointerId);});
c.addEventListener('pointermove',function(e){if(!drag)return;var r=c.getBoundingClientRect(),k=c.width/r.width;
  view.x=(e.clientX-r.left)*k-drag.x;view.y=(e.clientY-r.top)*k-drag.y;draw();});
c.addEventListener('pointerup',function(){drag=null;c.className='';});
window.addEventListener('keydown',function(e){
  if(e.key==='a'||e.key==='A')setFrame('A');
  if(e.key==='b'||e.key==='B')setFrame('B');
  if(e.key===' '){e.preventDefault();setFrame(frame==='A'?'B':'A');}});
bar();fit();draw();
</script>
__FIGURES__
</html>"""


def build(directory, output=None, piv_path=None, manual_directory=None,
          piv_stride=PIV_STRIDE, depth_m=None, margin=SURFACE_MARGIN_PX,
          embed_figures=False):
    """Write one self-contained viewer page for a completed pair."""
    directory = Path(directory)
    data = collect(directory, piv_path, manual_directory, piv_stride, depth_m, margin)
    output = Path(output) if output else directory/'viewer.html'
    counts = ', '.join('%s %d' % (key, len(value['vectors'])) for key, value in data['layers'].items())
    subtitle = ('%d &times; %d px crop, rows %d&ndash;%d of the frame &middot; %.1f mm below the '
                'deepest surface point &middot; layers: %s'
                % (data['width'], data['height'], data['crop_top'], data['crop_bottom'],
                   data['depth_m']*1000., counts or 'none'))
    foot = ('Keys: <code>A</code>/<code>B</code> select a frame, <code>space</code> flips. '
            'Supplied PIV and hand-matched picks are comparisons read after the prediction was '
            'frozen; neither constrains the field.')
    figures = report_figures(directory, embed_figures)
    page = (_PAGE.replace('__TITLE__', html.escape(data['pair']+' — field viewer'))
                 .replace('__FIGURES__', figures)
                 .replace('__SUBTITLE__', subtitle).replace('__FOOT__', foot)
                 .replace('__DATA__', json.dumps(data, separators=(',', ':'))))
    output.write_text(page, encoding='utf8')
    return {'path': str(output), 'bytes': output.stat().st_size,
            'crop_rows': [data['crop_top'], data['crop_bottom']],
            'figures': figures.count('<figure>'),
            'layers': {k: len(v['vectors']) for k, v in data['layers'].items()}}


_INDEX = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>__TITLE__</title><style>
body{font:14px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;color:#1e2530;margin:2rem auto;
 max-width:1000px;padding:0 1rem}
h1{font-size:20px;margin:0 0 .3rem}
p.sub{color:#5a6775;font-size:13px;margin:0 0 1.2rem}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:.45rem .7rem;border-bottom:1px solid #e3e8ee;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:#2b3542;font-weight:600;border-bottom:1px solid #cfd6e0}
a{color:#185ea0;text-decoration:none}a:hover{text-decoration:underline}
tr:hover{background:#f7f9fc}
.note{font-size:12.5px;color:#5a6775;border-top:1px solid #e3e8ee;margin-top:1.4rem;padding-top:.7rem}
.y{color:#17806d}.n{color:#b9c2cc}
</style>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<table><thead><tr><th>Pair</th><th>Optical flow</th><th>PIV</th><th>Hand-matched</th>
<th>One-to-one</th><th>Size</th></tr></thead><tbody>__ROWS__</tbody></table>
<p class="note">Each page is self-contained: the particle frames with their vector layers, an A/B
flip, zoom and pan, and sliders for arrow length and density. Where hand-matched picks exist the
field is also evaluated at those exact positions, so prediction and pick share an origin and the gap
between arrowheads is the disagreement.</p></html>"""


def build_all(root, piv_directory=None, manual_directory=None, piv_stride=PIV_STRIDE,
              depth_m=None, margin=SURFACE_MARGIN_PX, progress=print, embed_figures=False):
    """A viewer for every completed pair under a batch root, plus an index page."""
    root = Path(root)
    pairs = sorted(d for d in root.iterdir()
                   if d.is_dir() and (d/'results.npz').is_file() and (d/'inputs.npz').is_file())
    if not pairs:
        raise ValueError('No completed pair directories under '+str(root))
    rows = []
    for directory in pairs:
        piv = None
        if piv_directory:
            candidate = Path(piv_directory)/(directory.name+'_PIV.mat')
            piv = candidate if candidate.is_file() else None
        record = build(directory, None, piv, manual_directory, piv_stride, depth_m, margin,
                       embed_figures)
        layers = record['layers']
        rows.append((directory.name, layers, record['bytes']))
        progress('  %-22s %.1f MB  %s' % (directory.name, record['bytes']/1e6,
                 ', '.join('%s %d' % kv for kv in layers.items())))
    def cell(layers, key):
        count = layers.get(key)
        return ('<td class="y">%d</td>' % count) if count else '<td class="n">&mdash;</td>'
    body = ''.join('<tr><td><a href="%s/viewer.html">%s</a></td>%s%s%s%s<td>%.1f MB</td></tr>'
                   % (html.escape(name), html.escape(name), cell(layers, 'of'), cell(layers, 'piv'),
                      cell(layers, 'manual'), cell(layers, 'of_at_manual'), size/1e6)
                   for name, layers, size in rows)
    total = sum(size for _, _, size in rows)
    with_manual = sum(1 for _, layers, _ in rows if layers.get('manual'))
    page = (_INDEX.replace('__TITLE__', html.escape(root.name+' — field viewers'))
                  .replace('__SUBTITLE__', '%d pairs, %.0f MB total, %d with hand-matched picks'
                           % (len(rows), total/1e6, with_manual))
                  .replace('__ROWS__', body))
    index = root/'viewers.html'
    index.write_text(page, encoding='utf8')
    return {'index': str(index), 'pairs': len(rows), 'bytes': total}
