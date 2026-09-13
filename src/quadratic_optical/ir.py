"""Comparison-only IR surface references; no theory or velocity-model fallback."""
from pathlib import Path
import numpy as np
from .matio import read_experiment_fields

_FIELDS = [
    "PIV.pairNum", "PIV.IR_idx", "PIV.t",
    "Surfs.pairNum", "Surfs.IR_idx", "Surfs.t", "Surfs.dt_pair", "Surfs.spp",
    "USurf.t", "USurf.usurf0", "USurf.usurffilt", "USurf.IRfps", "USurf.IRDX",
    "USurf.Ndots_used", "USurf.corrmax0", "USurf.TMVTech",
]


def _finite(value):
    return float(value) if np.isfinite(value) else None


def select_surface_values(path, experiment, pair_numbers):
    """Return one explicit selection record per requested zero-based PIV pair.

    Pair numbers are matched to stored ``PIV.pairNum``; ``PIV.IR_idx`` is a
    MATLAB one-based index into ``USurf``. A finite indexed ``usurf0`` is used
    first, otherwise finite ``usurffilt``. Nothing substitutes a neighboring raw
    time, theoretical profile, composite method, or a guessed IR index. Missing
    data give ``available=False`` and a reason, never a fabricated observation.

    Surface velocities in these files are downstream-positive m/s; no DX/DT
    conversion or image-y sign change applies. The raw IR path is not required
    and is never fetched. Timing/identity validation and absent fields are
    recorded per row. Filtered values are labelled smoothed estimates, without
    assuming a universal smoothing window for arbitrary run files.
    """
    requested = list(pair_numbers)
    if any(not np.isfinite(p) or int(p) != p or p < 0 for p in requested):
        raise ValueError("pair_numbers must contain nonnegative integer labels")
    values, metadata = read_experiment_fields(path, experiment, _FIELDS)
    arrays = {key: np.asarray(value).ravel(order="F") for key, value in values.items()
              if key != "exp_name"}
    def at(key, index):
        array = arrays.get(key)
        return None if array is None or index < 0 or index >= len(array) else _finite(array[index])
    def scalar(key):
        array = arrays.get(key)
        return _finite(array[0]) if array is not None and len(array) == 1 else None
    rows = []
    for requested_pair in requested:
        pair = int(requested_pair)
        row = dict(pair_number_zero_based=pair, pair=pair, experiment=experiment,
            source=str(Path(path)), comparison_only=True, available=False,
            selected_field=None, velocity_m_per_s=None,
            selected_surface_velocity_m_per_s=None, plotting_depth_m=0.,
            representation=None, raw_observation_available_at_index=False,
            missing_fields=metadata["missing_fields"],
            experiment_identity_verified=metadata["experiment_identity_verified"],
            experiment_index_matlab=metadata["experiment_index_matlab"],
            source_format=metadata["format"], whole_campaign_loaded=False,
            PIV_row_matlab=None, IR_index_matlab=None, IR_index_python=None,
            PIV_A_time_s=None, PIV_B_time_s=None, PIV_midpoint_time_s=None,
            IR_sample_time_s=None, IR_time_minus_PIV_A_s=None,
            IR_time_minus_PIV_midpoint_s=None, IR_index_is_nearest_time=None,
            IRfps=scalar("USurf.IRfps"), IRDX=scalar("USurf.IRDX"),
            within_pair_delay_s=scalar("Surfs.dt_pair"),
            between_pair_interval_s=scalar("Surfs.spp"),
            timing_note="Stored PIV.IR_idx selects the IR sample; PIV.t denotes A-frame time. No additional timing shift.",
            units="downstream-positive m/s", theory_fallback_used=False)
        rows.append(row)
        if "PIV.pairNum" not in arrays or "PIV.IR_idx" not in arrays:
            row["status"] = "missing_pair_mapping"
            continue
        matches = np.flatnonzero(arrays["PIV.pairNum"] == pair)
        if len(matches) > 1:
            raise ValueError("Duplicate PIV pair label: " + str(pair))
        if not len(matches):
            row["status"] = "pair_not_found"
            continue
        j = int(matches[0]); row["PIV_row_matlab"] = j + 1
        index = at("PIV.IR_idx", j)
        row["PIV_A_time_s"] = at("PIV.t", j)
        if index is None or index < 1 or int(index) != index:
            row["status"] = "invalid_or_missing_IR_index"
            continue
        k = int(index) - 1
        row.update(IR_index_matlab=k + 1, IR_index_python=k,
                   IR_sample_time_s=at("USurf.t", k))
        raw, filtered = at("USurf.usurf0", k), at("USurf.usurffilt", k)
        row.update(indexed_raw_usurf0_m_per_s=raw, indexed_usurffilt_m_per_s=filtered,
            indexed_Ndots_used=at("USurf.Ndots_used", k),
            indexed_corrmax0=at("USurf.corrmax0", k), indexed_TMVTech=at("USurf.TMVTech", k))
        lengths = [len(arrays[key]) for key in ("USurf.t", "USurf.usurf0", "USurf.usurffilt") if key in arrays]
        if lengths and k >= max(lengths):
            row["status"] = "IR_index_out_of_range"
            continue
        # A supplied time vector defines the valid record range when available.
        if "USurf.t" in arrays and k >= len(arrays["USurf.t"]):
            row["status"] = "IR_index_out_of_range"
            continue
        chosen = "USurf.usurf0" if raw is not None else ("USurf.usurffilt" if filtered is not None else None)
        value = raw if raw is not None else filtered
        row.update(selected_field=chosen, velocity_m_per_s=value,
            selected_surface_velocity_m_per_s=value, available=value is not None,
            status="selected" if value is not None else "no_finite_surface_value",
            raw_observation_available_at_index=raw is not None,
            representation=("raw fresh-dot IR surface observation" if raw is not None else
                "smoothed IR surface estimate" if filtered is not None else None))
        a_time, ir_time = row["PIV_A_time_s"], row["IR_sample_time_s"]
        if a_time is not None and ir_time is not None:
            row["IR_time_minus_PIV_A_s"] = ir_time - a_time
            times = arrays["USurf.t"]
            if np.isfinite(times).all() and np.all(np.diff(times) > 0):
                row["IR_index_is_nearest_time"] = int(np.argmin(abs(times - a_time))) == k
            else:
                row["timing_warning"] = "IR time axis is nonfinite or not strictly increasing; nearest-time check unavailable"
        if "Surfs.pairNum" in arrays and "Surfs.t" in arrays:
            sj = np.flatnonzero(arrays["Surfs.pairNum"] == pair)
            st = [at("Surfs.t", int(i)) for i in sj]
            if len(st) == 2 and all(t is not None for t in st) and st[1] >= st[0]:
                row["Surfs_A_time_s"] = st[0]
                row["PIV_B_time_s"] = st[1]
                row["PIV_midpoint_time_s"] = .5 * (st[0] + st[1])
                row["Surfs_A_matches_PIV_A"] = bool(np.isclose(st[0], a_time, atol=1e-9, rtol=0)) if a_time is not None else None
                if ir_time is not None:
                    row["IR_time_minus_PIV_midpoint_s"] = ir_time - row["PIV_midpoint_time_s"]
        if "USurf.t" in arrays and "USurf.usurf0" in arrays and ir_time is not None:
            times, raw_values = arrays["USurf.t"], arrays["USurf.usurf0"]
            if len(times) == len(raw_values):
                valid = np.flatnonzero(np.isfinite(times) & np.isfinite(raw_values))
                before = valid[times[valid] <= ir_time]; after = valid[times[valid] >= ir_time]
                def bracket(indices, earlier):
                    if not len(indices):
                        return None
                    b = int(indices[np.argmax(times[indices]) if earlier else np.argmin(times[indices])])
                    return dict(IR_index_matlab=b + 1, time_s=float(times[b]),
                                velocity_m_per_s=float(raw_values[b]))
                row["nearest_finite_raw_before"] = bracket(before, True)
                row["nearest_finite_raw_after"] = bracket(after, False)
    return rows
