import json, numpy as np
from pathlib import Path
Q=Path('/media/surflab/LC_Working24/LC/FabMarcNovDec2014/data/Longitudinal/PIVdt10ms_IRlas1_8hz/ExpLCL_1_03/Results_Surflab/quadratic_optical')
PAIRS=[(80,'calm'),(100,'calm'),(115,'onset'),(123,'onset'),(125,'waves'),(140,'waves'),(144,'waves')]
ROOTS=[('baseline','analysis_L2'),('mild','accept_mild'),('aggressive','accept_aggressive')]
ONSET=117
H=['<title>ExpLCL_1_03 acceptance profiles</title>','<style>',
'body{font:14px/1.55 system-ui,-apple-system,sans-serif;margin:0;padding:24px;max-width:1080px;color:#1a1a1a}',
'table{border-collapse:collapse;width:100%;margin:16px 0}',
'th,td{border:1px solid #d8d8d8;padding:6px 9px;text-align:left}',
'th{background:#f4f4f4;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}',
'tr.sep td{border-top:2px solid #999}a{color:#0a58ca;text-decoration:none}a:hover{text-decoration:underline}',
'code{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-size:13px}',
'.note{color:#555;font-size:13px}.man{color:#0a7a3d}.no{color:#999}',
'@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}th{background:#242424}',
'th,td{border-color:#3a3a3a}code{background:#242424}.note{color:#aaa}a{color:#6ea8fe}}',
'</style>','<h1>ExpLCL_1_03 &mdash; acceptance profile comparison</h1>',
'<p class="note">Identical fits in all three columns; only the final screening rule differs, '
'so every difference is the rule and not a refit. Wave onset is pair %d. '
'Pairs with hand-matched picks carry a manual layer in the viewer, plus optical flow '
'evaluated at the pick positions themselves.</p>'%ONSET,
'<table><tr><th>pair</th><th>profile</th><th>accepted</th><th>shallowest</th>'
'<th>5&ndash;9&nbsp;px</th><th>vs manual (median px)</th><th>viewer</th></tr>']
for n,tag in PAIRS:
    first=True
    for lbl,root in ROOTS:
        d=Q/root/('ExpLCL_1_03_%d'%n)
        if not (d/'results.npz').exists(): continue
        s=json.loads((d/'summary.json').read_text())
        r=np.load(d/'results.npz',allow_pickle=True)
        a=r['accepted']; dep=r['depth']
        top=float(dep[a].min()) if a.any() else float('nan')
        band=(dep>=5)&(dep<9); sh=100*a[band].mean() if band.any() else 0.
        man='&mdash;'
        st=d/'status.json'
        if st.exists():
            rec=json.loads(st.read_text()).get('manual_comparison')
            if rec:
                S={x['subset']:x for x in rec['statistics']}
                p=S.get('passing vector screen',{})
                if isinstance(p.get('median_px'),(int,float)):
                    man='<span class="man">%.3f</span> <span class="note">(n=%d)</span>'%(
                        p['median_px'],p.get('count',0))
        v=d/'viewer.html'
        cls=' class="sep"' if (first and n!=PAIRS[0][0]) else ''
        lead='<td rowspan="3"><b>%d</b><br><span class="note">%s</span></td>'%(n,tag) if first else ''
        H.append('<tr%s>%s<td>%s</td><td class="n">%d</td><td class="n">%.2f px</td>'
                 '<td class="n">%.1f%%</td><td class="n">%s</td><td>%s</td></tr>'%(
                 cls,lead,lbl,s['accepted_grid'],top,sh,man,
                 '<a href="%s">open</a>'%v.relative_to(Q) if v.exists() else '&mdash;'))
        first=False
H.append('</table>')
H.append('<p class="note"><b>vs manual</b> is the median distance between the predicted and '
         'hand-matched arrow endpoints, over picks where the rule returned a vector. Its n grows '
         'as the rule relaxes, so compare the medians alongside the counts. '
         'Pair 144 sits late in the wave evolution where out-of-plane motion prevents good '
         'near-surface manual estimates; it is included for completeness, not as evidence.</p>')
H.append('<p class="note">Rebuild with <code>quadratic-optical run &lt;input&gt; --output &lt;new dir&gt; '
         '--acceptance mild</code>. The values applied are recorded in each pair\'s '
         '<code>summary.json</code> under <code>acceptance_profile</code>.</p>')
(Q/'acceptance_profiles.html').write_text('\n'.join(H)+'\n')
print('wrote',Q/'acceptance_profiles.html')
