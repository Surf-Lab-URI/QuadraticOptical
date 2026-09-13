"""Independent exported quiver values, signs, masks and displayed tip audit."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from native_piv import NativePIV,sha

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/piv_comparison_top_cm'


def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and np.allclose(a,b,rtol=0,atol=0,equal_nan=True)


def run():
    meta=json.loads((OUT/'quiver_display_metadata.json').read_text())
    for plot in ['quiver_overlay','quiver_overlay_deeper_half']:
        for key in ['xlim_cm','ylim_cm','arrow_gain','min_depth_cm']:
            assert meta['80_'+plot][key]==meta['100_'+plot][key]
    before=json.loads((HERE/'native_piv_audit.json').read_text())['pairs'];report={}
    for pair in [80,100]:
        n=NativePIV(pair);r=read(n.input_path.parent/'results.npz');dest=OUT/('pair_'+str(pair))
        a=read(dest/'velocity_comparison.npz');mat=loadmat(dest/'velocity_comparison.mat',squeeze_me=True)
        metrics=json.loads((dest/'velocity_comparison_metrics.json').read_text())
        q=a['query_xy_zero_based_px'];sample=n.sample(q);factor=n.DX/n.DT*100
        assert same(q,r['query']) and same(a['piv_available'],sample['available'])
        assert same(a['image_only_accepted'],r['accepted'])
        assert same(a['joint_available'],r['accepted']&sample['available'])
        for component,label,sign in [(0,'u',1),(1,'w',-1)]:
            expected=sample['disp'][:,component]*factor*sign
            assert same(a['piv_'+label+'_cm_per_s'],expected)
            expected=np.where(r['accepted'],r['disp'][:,component]*factor*sign,np.nan)
            assert same(a['image_only_'+label+'_cm_per_s'],expected)
        for k in a:
            if a[k].dtype.kind in 'biufc':assert same(np.squeeze(a[k]),np.squeeze(mat[k])),k
        assert sha(n.input_path)==before[str(pair)]['input_sha256']
        assert sha(n.input_path.parent/'results.npz')==before[str(pair)]['results_sha256']==metrics['result_sha256']
        assert sha(n.mat_path)==before[str(pair)]['mat_sha256']==metrics['native_mat_sha256']
        dx=n.DX*100;surface=n.geometry['surface_a'];sr=np.median(surface);counts={}
        for plot in ['quiver_overlay','quiver_overlay_deeper_half']:
            m=meta[str(pair)+'_'+plot];gain=m['arrow_gain'];h0=m['min_depth_cm']
            thin=((q[:,0].astype(int)-7)%32==0)&((q[:,1].astype(int)-7)%16==0)&(r['depth']*dx>=h0)
            limits=np.array([m['xlim_cm'],m['ylim_cm']]);counts[plot]={}
            for name,d,ok in [('image_only',r['disp'],r['accepted']),('piv',sample['disp'],sample['available'])]:
                use=thin&ok
                ends=np.c_[(q[use,0]+.5+gain*d[use,0])*dx,(sr-q[use,1]-gain*d[use,1])*dx]
                clipped=((ends<limits[:,0])|(ends>limits[:,1])).any(axis=1)
                assert int(use.sum())==m[name+'_shown']
                assert not clipped.any(),(pair,plot,name,int(clipped.sum()))
                counts[plot][name]=dict(drawn=int(use.sum()),clipped_tips=int(clipped.sum()))
        report[str(pair)]=dict(status='passed',exact_native_node_values_and_signs=True,
            source_only_PIV_masks_match_shared_loader=True,original_OF_values_and_masks_unchanged=True,
            assumed_cm_per_s_conversion_correct=True,mat_and_npz_numeric_fields_identical=True,
            pooled_panel_limits_and_arrow_gain_identical_between_pairs=True,
            original_image_only_inputs_results_and_native_MAT_unchanged=True,plots=counts,
            quiver_key_rule='Axes data are centimetres per image pair; key U=speed_cm_per_s*DT, then same displayed gain as both methods.')
    (HERE/'quiver_export_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':run()
