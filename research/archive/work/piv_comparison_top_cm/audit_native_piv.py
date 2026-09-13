"""Read-only audit of supplied native PIV and shared comparison interpolation."""
from pathlib import Path
import json
import numpy as np
from native_piv import NativePIV,strict_bilinear,sha

HERE=Path(__file__).resolve().parent


def self_test():
    x=np.array([3.,7.,11.]);y=np.array([3.,7.,11.]);xx,yy=np.meshgrid(x,y)
    v=2*xx-3*yy+5;valid=np.ones(v.shape,bool)
    q=np.array([[3,3],[11,11],[4.2,9.1],[7,3]],float)
    out,ok=strict_bilinear(x,y,v,valid,q)
    assert ok.all() and np.allclose(out,2*q[:,0]-3*q[:,1]+5,atol=1e-14,rtol=0)
    broken=v.copy();broken[1,1]=np.nan;vok=np.isfinite(broken)
    query=np.array([[3,3],[11,11],[3,7],[3,5],[5,5],[11.001,7],[2.999,7],[np.nan,5]])
    out,ok=strict_bilinear(x,y,broken,vok,query)
    expected=np.array([1,1,1,1,0,0,0,0],bool)
    assert np.array_equal(ok,expected) and np.isnan(out[~ok]).all()
    vv=np.stack([v,2*v],axis=2)
    got,ok=strict_bilinear(x,y,vv,valid,q)
    assert ok.all() and np.allclose(got,np.c_[2*q[:,0]-3*q[:,1]+5,2*(2*q[:,0]-3*q[:,1]+5)],atol=1e-14,rtol=0)
    return dict(linear_scalar_and_vector_interpolation_exact=True,zero_weight_invalid_corners_ignored=True,
        positive_weight_missing_corners_rejected=True,no_extrapolation=True,nonfinite_queries_rejected=True)


def run():
    report={'interpolator_tests':self_test(),'pairs':{}}
    for pair in [80,100]:
        n=NativePIV(pair);input_hash=sha(n.input_path)
        result_path=n.input_path.parent/'results.npz';result_hash=sha(result_path)
        top=(n.source_depth>=0)&(n.source_depth<=.01/n.DX+1e-9);valid=n.valid_source&top
        with np.load(result_path,allow_pickle=False) as z:q=z['query'].copy()
        sample=n.sample(q)
        ix=np.rint((q[:,0]-n.x[0])/4).astype(int);iy=np.rint((q[:,1]-n.y[0])/4).astype(int)
        inside=(ix>=0)&(ix<len(n.x))&(iy>=0)&(iy<len(n.y))
        exact=np.zeros(len(q),bool);exact[inside]=(n.x[ix[inside]]==q[inside,0])&(n.y[iy[inside]]==q[inside,1])
        ids=np.flatnonzero(exact&sample['available'])
        assert np.array_equal(sample['disp'][ids],n.disp[iy[ids],ix[ids]])
        assert sha(n.input_path)==input_hash and sha(result_path)==result_hash
        report['pairs'][str(pair)]=dict(native_shape=list(n.dx.shape),metadata=n.metadata,
            zero_based_x_range=[float(n.x[0]),float(n.x[-1])],zero_based_y_range=[float(n.y[0]),float(n.y[-1])],
            native_spacing_px=4.,native_finite_vectors=int(n.finite.sum()),
            full_frame_finite_dcor=int(np.isfinite(n.dcor).sum()),
            full_frame_finite_vectors_without_dcor=int((n.finite&~np.isfinite(n.dcor)).sum()),
            component_missing_masks_identical=bool(np.array_equal(np.isfinite(n.dx),np.isfinite(n.dy))),
            native_top_cm_finite_vectors=int((n.finite&top).sum()),source_supported_top_cm_vectors=int(valid.sum()),
            source_supported_top_cm_without_dcor=int((valid&~np.isfinite(n.dcor)).sum()),
            source_supported_top_cm_without_dcor_percent=float(100*(valid&~np.isfinite(n.dcor)).sum()/valid.sum()),
            source_supported_top_cm_with_dcor=int((valid&np.isfinite(n.dcor)).sum()),
            source_and_target_supported_top_cm_vectors=int((n.valid_source_target&top).sum()),
            original_report_query_count=len(q),report_queries_exact_native_node=int(exact.sum()),
            report_queries_with_native_source_support=int(sample['available'].sum()),
            report_queries_with_native_source_support_and_dcor=int(sample['available_with_dcor'].sum()),
            report_queries_with_native_source_target_support=int(sample['available_with_target'].sum()),
            exact_node_samples_bitwise_equal_to_native_values=True,
            image_only_inputs_unchanged=True,image_only_results_unchanged=True,
            input_sha256=input_hash,results_sha256=result_hash,mat_sha256=sha(n.mat_path))
    (HERE/'native_piv_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':run()
