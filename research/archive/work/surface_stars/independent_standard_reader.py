"""Audit one bounded MATLAB cell with SciPy's standard v5 reader.

Only the first campaign cell is reconstructed as a small standalone MAT stream;
the remaining experiment cells and function workspace are not loaded.
"""
from pathlib import Path
from io import BytesIO
import struct,json
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab.streams import ZlibInputStream

HERE=Path(__file__).resolve().parent
SOURCE=Path('data/results.mat')


def element(stream):
    a,b=struct.unpack('<II',stream.read(8))
    if a>>16:return a&65535,struct.pack('<I',b)[:a>>16]
    data=stream.read(b)
    if b%8:stream.read((-b)%8)
    return a,data


def finite_float(value):
    v=float(value)
    return v if np.isfinite(v) else None


def run():
    with SOURCE.open('rb') as f:
        header=f.read(128);assert header[126:128]==b'IM'
        f.seek(1034);tag=f.read(8);typ,size=struct.unpack('<II',tag)
        assert typ==15 and size==894922055
        z=ZlibInputStream(f,size)
        matrix_type,matrix_bytes=struct.unpack('<II',z.read(8));assert matrix_type==14
        _,flags=element(z);_,dims=element(z);_,name=element(z)
        assert name==b'exps'
        assert int(np.frombuffer(flags,dtype='<u4')[0])&255==1
        assert np.prod(np.frombuffer(dims,dtype='<i4'))==13
        first_tag=z.read(8);first_type,first_bytes=struct.unpack('<II',first_tag)
        assert first_type==14
        payload=z.read(first_bytes)
        assert len(payload)==first_bytes
    loaded=loadmat(BytesIO(header+first_tag+payload),squeeze_me=True,struct_as_record=False)
    keys=[key for key in loaded if key not in ['__header__','__version__','__globals__']]
    assert len(keys)==1,keys
    run_obj=loaded[keys[0]]
    if isinstance(run_obj,np.ndarray) and run_obj.size==1:run_obj=run_obj.item()
    assert str(run_obj.exp_name)=='ExpLCL_1_03'
    expected_path=HERE/'selected_run.npz'
    checks={}
    with np.load(expected_path,allow_pickle=False) as extracted:
        for key in extracted.files:
            if '__' in key:
                section,field=key.split('__',1);expected=np.asarray(getattr(getattr(run_obj,section),field)).squeeze()
            else:expected=np.asarray(getattr(run_obj,key)).squeeze()
            actual=extracted[key].squeeze()
            same=actual.shape==expected.shape
            if same:
                same=np.all((actual==expected)|(np.isnan(actual)&np.isnan(expected))) if actual.dtype.kind in 'fc' else np.array_equal(actual,expected)
            checks[key]=bool(same)
            assert same,key
    P=run_obj.PIV;S=run_obj.Surfs;U=run_obj.USurf
    pnum=np.asarray(P.pairNum).ravel();pt=np.asarray(P.t).ravel();ir_idx=np.asarray(P.IR_idx).ravel()
    times=np.asarray(U.t).ravel();raw=np.asarray(U.usurf0).ravel();filt=np.asarray(U.usurffilt).ravel()
    assert np.all(np.diff(times)>0)
    entries={}
    for pair in [80,100]:
        found=np.flatnonzero(pnum==pair);assert len(found)==1;row=int(found[0]);idx=int(ir_idx[row])-1
        assert ir_idx[row]==idx+1 and 0<=idx<len(times)
        nearest=int(np.argmin(abs(times-pt[row])));assert idx==nearest
        surface_rows=np.flatnonzero(np.asarray(S.pairNum).ravel()==pair)
        st=np.asarray(S.t).ravel()[surface_rows]
        assert len(st)==2 and st[0]==pt[row]
        valid=np.flatnonzero(np.isfinite(raw))
        before=valid[times[valid]<=times[idx]];after=valid[times[valid]>=times[idx]]
        def entry(k):
            if k is None:return None
            return dict(IR_index_matlab=int(k+1),time_s=float(times[k]),
                raw_usurf0_m_per_s=float(raw[k]),raw_usurf0_cm_per_s=float(raw[k]*100),
                time_minus_PIV_A_s=float(times[k]-pt[row]),
                Ndots_used=float(np.asarray(U.Ndots_used).ravel()[k]),
                corrmax0=finite_float(np.asarray(U.corrmax0).ravel()[k]))
        left=entry(int(before[-1])) if len(before) else None;right=entry(int(after[0])) if len(after) else None
        chosen='usurf0' if np.isfinite(raw[idx]) else 'usurffilt'
        value=raw[idx] if chosen=='usurf0' else filt[idx]
        assert np.isfinite(value)
        entries[str(pair)]=dict(PIV_pair_number_zero_based=pair,PIV_row_matlab=row+1,
            PIV_A_time_s=float(pt[row]),PIV_B_time_s=float(st[1]),PIV_midpoint_time_s=float(st.mean()),
            IR_index_matlab=idx+1,IR_index_python=idx,IR_sample_time_s=float(times[idx]),
            IR_time_minus_PIV_A_s=float(times[idx]-pt[row]),IR_time_minus_PIV_midpoint_s=float(times[idx]-st.mean()),
            selected_field=chosen,selected_surface_velocity_m_per_s=float(value),selected_surface_velocity_cm_per_s=float(100*value),
            indexed_raw_usurf0_m_per_s=finite_float(raw[idx]),indexed_usurffilt_m_per_s=float(filt[idx]),
            indexed_TMVTech=float(np.asarray(U.TMVTech).ravel()[idx]),
            indexed_Ndots_used=float(np.asarray(U.Ndots_used).ravel()[idx]),
            indexed_corrmax0=finite_float(np.asarray(U.corrmax0).ravel()[idx]),
            nearest_finite_raw_before=left,nearest_finite_raw_after=right,
            bracketing_raw_time_gap_s=(right['time_s']-left['time_s']) if left and right else None,
            plotting_depth_m=0.,raw_observation_available_at_index=bool(np.isfinite(raw[idx])))
    fps=float(U.IRfps);dt=float(np.median(np.diff(times)))
    report=dict(status='passed',experiment=str(run_obj.exp_name),experiment_cell_matlab=1,
        standard_reader='scipy.io.loadmat on original first cell matrix, not the custom extraction parser',
        first_cell_uncompressed_payload_bytes=int(first_bytes),whole_campaign_not_loaded=True,
        extracted_field_count=len(checks),all_extracted_fields_exact_including_NaNs=all(checks.values()),field_checks=checks,
        IRfps=fps,IR_sample_median_spacing_s=dt,nominal_40_sample_moving_mean_duration_s=40/fps,
        span_between_first_and_last_of_40_samples_s=39/fps,
        PIV_pair_interval_s=float(S.spp),within_pair_delay_s=float(S.dt_pair),
        corr_threshold=float(U.corr_threshold),pairs=entries,
        recommendation='Use indexed usurffilt for both stars, explicitly labelled interpolated and 40-sample smoothed IR surface estimates. Do not relabel missing raw observations, shift to another raw time silently, or use theory/usurf1/usurfComp.',
        timing_note='PIV.t is the A-frame time. The stored PIV.IR_idx is the nearest USurf.t, with MATLAB one-based indexing. No extra IR exposure-midpoint convention is inferred.',
        direction_and_units='All USurf velocities are downstream-positive m/s. Multiply by 100 for the existing cm/s mean-velocity axes; no image-y sign flip and no DX/DT conversion is applied again.',
        baseline_note='New PIV documentation identifies DX as metres/pixel and dcor NaN as rejected. Existing as-delivered PIV/OF comparison curves remain unchanged for this surface-star addition.')
    (HERE/'independent_selection_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False));return report


if __name__=='__main__':run()
