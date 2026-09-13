"""Selective MATLAB input, with MATLAB logical array orientation preserved.

Numeric arrays retain their MATLAB dimensions: an image/grid is ``[y, x]``.
MATLAB v7.3 stores dimensions in reverse order, so HDF5 numeric axes are reversed
exactly once. Row/column vectors are not squeezed by ``read_mat_fields``.

The v5 reader streams compressed elements and gives *selected matrices only* to
SciPy. Unrequested nested fields are skipped without constructing their arrays.
This matters when a geometry-only read shares a struct with velocity/image data.
"""
from io import BytesIO
from pathlib import Path
import struct
import zlib

import h5py
import numpy as np
from scipy.io import loadmat

__all__ = ["read_mat_fields", "read_native_piv", "read_experiment_fields"]


class _Reader:
    def __init__(self, file):
        self.file, self.pos = file, 0

    def read(self, count):
        data = self.file.read(count)
        if len(data) != count:
            raise EOFError("Truncated MATLAB element")
        self.pos += count
        return data

    def skip(self, count):
        while count:
            size = min(count, 65536)
            self.read(size)
            count -= size


class _Inflated(_Reader):
    """Bounded compressed input/output buffers, including while skipping arrays."""
    def __init__(self, file, size):
        super().__init__(file)
        self.remaining = size
        self.decompressor = zlib.decompressobj()
        self.pending = b""
        self.buffer = bytearray()

    def read(self, count):
        while len(self.buffer) < count:
            if not self.pending and self.remaining:
                size = min(self.remaining, 65536)
                self.pending = self.file.read(size)
                if len(self.pending) != size:
                    raise EOFError("Truncated compressed MATLAB element")
                self.remaining -= size
            out = self.decompressor.decompress(self.pending, 65536)
            self.pending = self.decompressor.unconsumed_tail
            self.buffer.extend(out)
            if not out and not self.pending and not self.remaining:
                raise EOFError("Compressed MATLAB element ended before its matrix")
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        self.pos += count
        return data


def _tag(reader, endian):
    raw = reader.read(8)
    first, second = struct.unpack(endian + "II", raw)
    # The two uint16 entries have the same semantic order on either endian.
    kind, size = struct.unpack(endian + "HH", raw[:4])
    if first > 18:
        if size > 4:
            raise ValueError("Invalid small MATLAB element")
        return kind, size, raw[4:4 + size], raw
    return first, second, None, raw


def _element(reader, tag):
    _, size, small, raw = tag
    if small is not None:
        return small, raw
    data = reader.read(size)
    padding = reader.read((-size) % 8)
    return data, raw + data + padding


def _skip(reader, tag):
    if tag[2] is None:
        reader.skip(tag[1] + (-tag[1]) % 8)


def _header(reader, tag, endian):
    if tag[0] != 14 or tag[2] is not None:
        raise ValueError("Expected a MATLAB matrix element")
    end = reader.pos + tag[1]
    flags, flags_raw = _element(reader, _tag(reader, endian))
    dims, dims_raw = _element(reader, _tag(reader, endian))
    name, name_raw = _element(reader, _tag(reader, endian))
    flag = int(np.frombuffer(flags, dtype=endian + "u4")[0])
    return dict(kind=flag & 255, logical=bool(flag & 0x200),
                dims=tuple(np.frombuffer(dims, dtype=endian + "i4")),
                name=name.decode("utf8"), end=end,
                prefix=flags_raw + dims_raw + name_raw,
                metadata_prefix=flags_raw + dims_raw, tag=tag)


def _finish(reader, header):
    if reader.pos > header["end"]:
        raise ValueError("MATLAB element exceeded its declared length")
    reader.skip(header["end"] - reader.pos)
    reader.skip((-header["tag"][1]) % 8)


def _fields(reader, endian):
    data, _ = _element(reader, _tag(reader, endian))
    length = int(np.frombuffer(data, dtype=endian + "i4")[0])
    data, _ = _element(reader, _tag(reader, endian))
    if length < 1 or len(data) % length:
        raise ValueError("Invalid MATLAB struct field table")
    return [data[i:i + length].split(b"\0", 1)[0].decode("utf8")
            for i in range(0, len(data), length)]


def _decode(reader, header, mat_header):
    """Use SciPy's standard decoder on one selected complete matrix."""
    endian = "<" if mat_header[126:128] == b"IM" else ">"
    # Nested struct leaves have empty names, which SciPy otherwise interprets
    # as __function_workspace__. Give only this isolated matrix a stable name.
    name = b"selected"
    name_element = struct.pack(endian + "II", 1, len(name)) + name
    data = header["metadata_prefix"] + name_element + reader.read(header["end"] - reader.pos)
    reader.skip((-header["tag"][1]) % 8)
    stream = BytesIO(mat_header + struct.pack(endian + "II", 14, len(data)) + data)
    obj = loadmat(stream, squeeze_me=False, struct_as_record=False,
                  chars_as_strings=False, mat_dtype=False)
    keys = [key for key in obj if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError("Selected MATLAB matrix could not be decoded")
    value = np.asarray(obj[keys[0]])
    return value.astype(bool) if header["logical"] else value


def _walk(reader, header, path, wanted, output, mat_header, endian):
    if path in wanted:
        output[path] = _decode(reader, header, mat_header)
        return
    descendants = [key for key in wanted if key.startswith(path + ".")]
    if not descendants:
        _finish(reader, header)
        return
    if np.prod(header["dims"]) == 0:
        _finish(reader, header)
        return
    if header["kind"] != 2 or np.prod(header["dims"]) != 1:
        raise ValueError("Dotted paths require scalar structs: " + path)
    for name in _fields(reader, endian):
        tag = _tag(reader, endian)
        child = path + "." + name
        if any(key == child or key.startswith(child + ".") for key in descendants):
            _walk(reader, _header(reader, tag, endian), child, wanted,
                  output, mat_header, endian)
        else:
            _skip(reader, tag)
    _finish(reader, header)


def _v5_roots(path, visitor):
    with Path(path).open("rb") as file:
        mat_header = file.read(128)
        if len(mat_header) != 128 or mat_header[126:128] not in (b"IM", b"MI"):
            raise ValueError("Expected a MATLAB v5 or v7.3 file")
        endian = "<" if mat_header[126:128] == b"IM" else ">"
        while True:
            start = file.tell()
            raw = file.read(8)
            if not raw:
                break
            if len(raw) != 8:
                raise EOFError("Truncated top-level MATLAB tag")
            kind, size = struct.unpack(endian + "II", raw)
            if kind == 15:
                reader = _Inflated(file, size)
                tag = _tag(reader, endian)
                header = _header(reader, tag, endian)
                visitor(reader, header, mat_header, endian)
                # Compressed top-level elements are not padded in v5 files.
                file.seek(start + 8 + size)
            elif kind == 14:
                reader = _Reader(file)
                tag = (kind, size, None, raw)
                header = _header(reader, tag, endian)
                visitor(reader, header, mat_header, endian)
                file.seek(start + 8 + size + (-size) % 8)
            else:
                file.seek(start + 8 + size + (-size) % 8)


def _h5_unwrap(file, obj):
    while isinstance(obj, h5py.Dataset) and h5py.check_dtype(ref=obj.dtype):
        refs = np.asarray(obj[()])
        if refs.size != 1:
            break
        ref = refs.ravel()[0]
        if not ref:
            raise KeyError("Empty MATLAB object reference")
        obj = file[ref]
    return obj


def _h5_array(file, obj):
    obj = _h5_unwrap(file, obj)
    if not isinstance(obj, h5py.Dataset) or h5py.check_dtype(ref=obj.dtype):
        raise ValueError("Requested MATLAB field is not a numeric/character array")
    value = np.asarray(obj[()])
    matlab_class = obj.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii")
    if obj.attrs.get("MATLAB_empty", False):
        return np.empty(tuple(int(x) for x in value.ravel()))
    if value.ndim > 1:
        value = value.transpose(tuple(reversed(range(value.ndim))))
    if value.dtype.names and {"real", "imag"}.issubset(value.dtype.names):
        value = value["real"] + 1j * value["imag"]
    if matlab_class == "logical":
        value = value.astype(bool)
    elif matlab_class == "char":
        value = np.array([chr(int(c)) for c in value.ravel()], dtype="U1").reshape(value.shape)
    return value


def _h5_fields(file, root, keys, missing):
    output = {}
    for key in keys:
        obj = root
        try:
            for part in key.split("."):
                obj = _h5_unwrap(file, obj)
                if not isinstance(obj, h5py.Group) or part not in obj:
                    raise KeyError(key)
                obj = obj[part]
            output[key] = _h5_array(file, obj)
        except KeyError:
            if missing == "raise":
                raise KeyError("Missing MATLAB field: " + key) from None
    return output


def read_mat_fields(path, dotted_keys, *, missing="raise"):
    """Read only explicit fields, preserving MATLAB shapes and missing NaNs.

    ``dotted_keys`` uses scalar-struct paths, e.g. ``compVel.DX``. Missing keys
    raise ``KeyError`` by default; ``missing='ignore'`` omits them. No computed
    velocity or fallback dataset is read unless explicitly named here.
    """
    if missing not in ("raise", "ignore"):
        raise ValueError("missing must be 'raise' or 'ignore'")
    keys = list(dict.fromkeys([dotted_keys] if isinstance(dotted_keys, str) else dotted_keys))
    if any(not isinstance(key, str) or not key or any(not p for p in key.split(".")) for key in keys):
        raise ValueError("Expected nonempty dotted MATLAB field names")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            return _h5_fields(file, file, keys, missing)
    output = {}
    def visitor(reader, header, mat_header, endian):
        _walk(reader, header, header["name"], keys, output, mat_header, endian)
    _v5_roots(path, visitor)
    absent = [key for key in keys if key not in output]
    if absent and missing == "raise":
        raise KeyError("Missing MATLAB fields: " + ", ".join(absent))
    return output


def read_native_piv(path, *, require_dcor=False, mask_field=None):
    """Read supplied *native-grid* PIV for comparison, without quality filtering.

    ``x_px,y_px`` are zero-based image pixel centers; ``dx_px,dy_px`` have
    shape ``[y,x]``. Stored ``delta_z`` is positive upward, so image-down
    ``dy_px=-delta_z``. Velocities are not converted to physical units here.
    ``mask`` defaults to optional ``compVel.mask``; an explicitly named absent
    mask raises. Masks must use 1=valid water, 0=invalid air; returned masks are
    boolean, with nonfinite entries invalid. ``dcor`` is None when absent unless required. Dense interpolated
    ``delta_x1/delta_z1`` are never read. No square-array transpose is guessed.
    """
    required = ["compVel." + name for name in ("xPIV", "zPIV", "delta_x", "delta_z", "DX", "DT")]
    optional = ["compVel." + name for name in ("dcor", "IW", "GS")]
    mask_key = mask_field or "compVel.mask"
    values = read_mat_fields(path, required + optional + [mask_key], missing="ignore")
    absent = [key for key in required if key not in values]
    if absent:
        raise KeyError("Missing native PIV fields: " + ", ".join(absent))
    if mask_field and mask_key not in values:
        raise KeyError("Missing explicitly requested PIV mask: " + mask_key)
    if require_dcor and "compVel.dcor" not in values:
        raise KeyError("Native PIV dcor is absent; use an explicit finite-vector quality mode to proceed")
    x = np.asarray(values["compVel.xPIV"], float).ravel() - 1
    y = np.asarray(values["compVel.zPIV"], float).ravel() - 1
    if len(x) < 2 or len(y) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("Native PIV axes must be finite and strictly increasing")
    dx = np.asarray(values["compVel.delta_x"], float)
    dy = -np.asarray(values["compVel.delta_z"], float)
    expected = (len(y), len(x))
    if dx.shape != expected or dy.shape != expected:
        raise ValueError("Native PIV displacement arrays must have MATLAB [y,x] shape " + str(expected))
    dcor = np.asarray(values["compVel.dcor"], float) if "compVel.dcor" in values else None
    mask = np.asarray(values[mask_key]) if mask_key in values else None
    for name, value in [("dcor", dcor), ("mask", mask)]:
        if value is not None and value.shape != expected:
            raise ValueError("Native PIV " + name + " shape does not match its axes")
    def scalar(key):
        a = np.asarray(values["compVel." + key])
        if a.size != 1:
            raise ValueError("PIV " + key + " must be a scalar")
        return float(a.item())
    if mask is not None:
        finite_mask = np.isfinite(mask)
        if not np.isin(mask[finite_mask], [0, 1]).all():
            raise ValueError("PIV mask must use 1=valid water and 0=invalid air")
        mask = finite_mask & (mask == 1)
    DX, DT = scalar("DX"), scalar("DT")
    if not np.isfinite(DX) or not np.isfinite(DT) or DX <= 0 or DT <= 0:
        raise ValueError("PIV DX and DT must be finite and positive")
    return dict(x_px=x, y_px=y, dx_px=dx, dy_px=dy,
                disp_px=np.stack([dx, dy], axis=-1), dcor=dcor, mask=mask,
                DX=DX, DT=DT, IW=scalar("IW") if "compVel.IW" in values else None,
                GS=scalar("GS") if "compVel.GS" in values else None,
                metadata=dict(source=str(Path(path)), fields_read=list(values),
                    array_order="[y,x]", coordinate_base=0,
                    vertical_displacement="dy_px=-delta_z (image down)",
                    dense_fields_read=False, quality_filter_applied=False,
                    mask_field=mask_key if mask is not None else None,
                    mask_semantics="True=valid/retained; source 1=water, 0=air; NaN invalid"))


def _text(value):
    array = np.asarray(value)
    if array.dtype.kind not in "US":
        raise ValueError("Experiment name is not a MATLAB character array")
    return "".join(array.astype(str).ravel()).rstrip("\0 ")


def read_experiment_fields(path, experiment, dotted_keys):
    """Select one campaign ``exps`` run, or a run file with top-level sections.

    Streams MATLAB v5 compressed campaigns with bounded buffers, retaining only
    requested small leaves of one run at a time. v7.3 dereferences only each
    run's name and the selected run's requested fields. Missing optional fields
    are omitted. Returns ``(fields, metadata)``. Duplicate experiment names raise.
    Top-level run files may omit ``exp_name``; metadata records that unverified
    identity explicitly, and never claims an experiment match from the filename.
    """
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("An explicit experiment name is required")
    requested = [dotted_keys] if isinstance(dotted_keys, str) else list(dotted_keys)
    keys = list(dict.fromkeys(["exp_name"] + requested))
    chosen = []
    records = []
    top = {}
    campaign = False
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:
            if "exps" not in file:
                top = _h5_fields(file, file, keys, "ignore")
            else:
                campaign = True
                exps = file["exps"]
                if isinstance(exps, h5py.Dataset) and h5py.check_dtype(ref=exps.dtype):
                    # Reverse storage dimensions, then MATLAB column-major order.
                    refs = exps[()]
                    refs = refs.transpose(tuple(reversed(range(refs.ndim)))).ravel(order="F")
                    runs = [file[ref] if ref else None for ref in refs]
                elif isinstance(exps, h5py.Group):
                    # A scalar struct stored directly under exps is also valid.
                    runs = [exps]
                else:
                    raise ValueError("Unsupported HDF5 campaign exps representation")
                for index, run in enumerate(runs, 1):
                    if run is None:
                        continue
                    names = _h5_fields(file, _h5_unwrap(file, run), ["exp_name"], "ignore")
                    name = _text(names["exp_name"]) if "exp_name" in names else None
                    records.append(dict(index_matlab=index, experiment=name))
                    if name == experiment:
                        chosen.append((_h5_fields(file, _h5_unwrap(file, run), keys, "ignore"), index))
    else:
        def visitor(reader, header, mat_header, endian):
            nonlocal campaign
            if header["name"] != "exps":
                _walk(reader, header, header["name"], keys, top, mat_header, endian)
                return
            campaign = True
            if header["kind"] != 1:
                raise ValueError("MATLAB v5 campaign exps must be a cell array")
            for index in range(1, int(np.prod(header["dims"])) + 1):
                run_header = _header(reader, _tag(reader, endian), endian)
                if np.prod(run_header["dims"]) == 0:
                    _finish(reader, run_header)
                    continue
                if run_header["kind"] != 2 or np.prod(run_header["dims"]) != 1:
                    raise ValueError("Campaign run must be a scalar struct")
                output = {}
                for name in _fields(reader, endian):
                    tag = _tag(reader, endian)
                    if any(key == name or key.startswith(name + ".") for key in keys):
                        _walk(reader, _header(reader, tag, endian), name, keys,
                              output, mat_header, endian)
                    else:
                        _skip(reader, tag)
                _finish(reader, run_header)
                name = _text(output["exp_name"]) if "exp_name" in output else None
                records.append(dict(index_matlab=index, experiment=name))
                if name == experiment:
                    chosen.append((output, index))
            _finish(reader, header)
        _v5_roots(path, visitor)
    if campaign:
        if len(chosen) != 1:
            raise ValueError("Expected one experiment %r; found %d" % (experiment, len(chosen)))
        selected, index = chosen[0]
        verified = True
    else:
        selected, index = top, None
        verified = "exp_name" in selected
        if verified and _text(selected["exp_name"]) != experiment:
            raise ValueError("Run-file exp_name does not match requested experiment")
    return selected, dict(source=str(Path(path)), experiment=experiment,
        campaign=campaign, experiment_index_matlab=index,
        experiment_identity_verified=verified, fields_read=list(selected),
        missing_fields=[key for key in keys if key not in selected],
        runs=records, whole_campaign_loaded=False,
        format="v7.3" if h5py.is_hdf5(path) else "v5")
