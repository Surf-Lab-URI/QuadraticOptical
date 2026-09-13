from plot_surface_stars import *
import shutil,zipfile

def main():
    metadata=json.loads((OUT/'surface_marker_metadata.json').read_text())
    selected=metadata['selection']
    prof={pair:read(BASE/('pair_%s'%pair)/'horizontal_integral_comparison.npz') for pair in [80,100]}
    assert np.array_equal(prof[80]['depth_m'],prof[100]['depth_m'])
    arrays={'pair':np.array([80,100]),'depth_mm':prof[80]['depth_m']*1000,
        'surface_velocity_cm_per_s':np.array([s['selected_surface_velocity_cm_per_s'] for s in selected]),
        'surface_depth_mm':np.zeros(2)}
    for new,old,scale in [
        ('optical_flow_mean_cm_per_s','of_matched_mean_u_m_per_s',100),
        ('piv_mean_cm_per_s','piv_matched_mean_u_m_per_s',100),
        ('matched_coverage_percent','matched_coverage_fraction',100),
        ('of_own_coverage_mean_cm_per_s','of_original_mean_u_m_per_s',100),
        ('piv_own_coverage_mean_cm_per_s','piv_own_mean_u_m_per_s',100)]:
        arrays[new]=np.column_stack([prof[p][old]*scale for p in [80,100]])
    np.savez_compressed(OUT/'mean_velocity_plot_data.npz',**arrays)
    savemat(str(OUT/'mean_velocity_plot_data.mat'),arrays,long_field_names=True,do_compression=True)
    header=['depth_mm'];cols=[arrays['depth_mm']]
    for j,pair in enumerate([80,100]):
        for k in ['optical_flow_mean_cm_per_s','piv_mean_cm_per_s','matched_coverage_percent']:
            header.append('pair_%s_%s'%(pair,k));cols.append(arrays[k][:,j])
    np.savetxt(OUT/'mean_velocity_profiles.csv',np.column_stack(cols),delimiter=',',header=','.join(header),comments='')
    doc='''# Surface stars on the mean-horizontal-velocity comparison

Stars have been added at **zero depth**, using the surface-velocity record for experiment **ExpLCL_1_03** in the supplied campaign `results.mat`.

| Pair | Surface velocity | Selected record | PIV A time | Matched IR time |
|---|---:|---|---:|---:|
| 80 | **9.8672 cm/s** | `USurf.usurffilt(480)` | 11.111111 s | 11.112000 s |
| 100 | **13.4860 cm/s** | `USurf.usurffilt(600)` | 13.888889 s | 13.890000 s |

Indices in this table use MATLAB's one-based convention. The experiment is `exps{1}`, verified by its stored `exp_name`. Pair numbering is zero-based; pair 80 and pair 100 are PIV rows 81 and 101. The stored `PIV.IR_idx` values were verified to select the nearest IR times. Their offsets from the PIV A exposures are only 0.889 and 1.111 milliseconds. No extra timing shift or sign reversal is applied.

## Why this surface-velocity representation

The background notes recommend fresh-dot `usurf0` where available and `usurffilt` for a dense smoothed trace. At both matched IR indices, **`usurf0` is NaN and `Ndots_used` is zero**. The selected `usurffilt` values interpolate gaps in the fresh-dot record and apply a 40-sample moving mean (nominally 0.926 seconds at the stored IR frame rate). These are therefore labeled **smoothed IR surface estimates**, rather than instantaneous raw observations.

The other image-correlation method, `usurf1`, and its composite fallback have uncertain accuracy according to the supplied documentation. Their values at these times are approximately 7.648 and 11.986 cm/s; they are recorded in the metadata but are not the plotted stars. The theoretical `U_msv98` profiles and linear-fit velocities are not used as observations.

The nearest finite fresh-dot values surrounding each missing interval are:

| Pair | Earlier raw time / velocity | Later raw time / velocity | Raw-sample gap |
|---|---|---|---:|
| 80 | 10.74160 s / 9.35280 cm/s | 11.15830 s / 10.16355 cm/s | 0.41670 s |
| 100 | 13.51960 s / 12.99768 cm/s | 13.93630 s / 14.00629 cm/s | 0.41670 s |

These neighboring measurements are documented for context; they are not substituted as though measured at the PIV time. No uncertainty interval is inferred from the smoothing window or the bracket values.

## Figures and data

- `mean_horizontal_velocity_with_surface.png` / `.svg`: the requested mean profiles with gold stars. Both subsurface methods use exactly the same horizontal segments at each depth.
- `horizontal_integral_vs_piv_with_surface.png` / `.svg`: the full comparison figure, with stars added only to its mean-velocity panels. A scalar surface velocity is not an integral and is not plotted on the integral axis.
- `mean_own_coverage_with_surface.png` / `.svg`: a supplementary view retaining each method's own horizontal averaging segments, as in the previous separate-coverage figure.
- `surface_markers.csv`, `.mat`, `.npz`: the two star values, time matches and selected field.
- `mean_velocity_plot_data.mat`, `.npz`, and `mean_velocity_profiles.csv`: reusable mean profiles and coverage. In the matrix files, columns correspond to pairs 80 and 100, in that order.
- `surface_marker_metadata.json`: precise values, timing, alternative fields, raw brackets, source hashes and selection rationale.

The existing optical-flow and PIV curves have **not been recomputed, extended or adjusted to meet the stars**. Their source hashes are unchanged. The stars are a surface reference from the IR record; the supplied records do not establish that its spatial averaging footprint equals the depth-dependent PIV/optical-flow comparison segments. The plot preserves the previous local-depth geometry, including its inferred surface offset, and does not resolve that convention from this scalar IR record.

The new `PIVFileContents.md` confirms that `DX` is in metres per pixel and velocities in this results file are downstream-positive m/s. It also states that NaN `dcor` identifies rejected PIV estimates. The existing comparison retained finite saved PIV velocities without applying that correlation mask; this star-only update preserves that baseline. A comparison limited to correlation-accepted PIV would be a different filtering choice; it has not been silently applied here.

`ManualPTVOutput.md` documents the separate manually matched particle-file format. The supplied `results.mat` is the campaign surface-results file, so its IR surface-velocity record is used for these stars.

The selected campaign fields were extracted without loading the full campaign into memory and independently checked against SciPy's standard MAT reader for the selected experiment. All 29 selected fields matched exactly, including missing values. The independent selection audit is included.
'''
    (OUT/'README.md').write_text(doc)
    shutil.copy2(HERE/'independent_selection_review.json',OUT/'independent_selection_review.json')
    (OUT/'file_manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='file_manifest.json'},indent=2)+'\n')
    archive=OUT.parent/'surface_velocity_comparison.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.name))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        print('Archive verified:',archive,'entries',len(z.infolist()),'bytes',archive.stat().st_size)

if __name__=='__main__':main()
