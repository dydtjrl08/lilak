"""Self-contained ROOT TTree stages, including full downstream observation arrays.

Use mktree explicitly: Uproot's assignment can create RNTuple on newer releases,
which is unsuitable for the user's ROOT6.24 server. No sidecar NPZ is required.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

SCHEMA = 'tpcdrum-reconstruction-root-v1'


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class StageWriter:
    """Stream rows to one ROOT file; publish atomically without overwriting.

    Numeric scalar values become scalar TTree branches. Numeric arrays become
    jagged primitive branches plus a shape branch. Diagnostics are a separate
    JSON string per event. No truth data is accepted in observation stages.
    """
    def __init__(self, path, stage, metadata=None):
        import uproot
        self.path = Path(path).expanduser().resolve()
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = dict(metadata or {})
        self.meta.update(schema=SCHEMA, stage=str(stage), complete=False,
                         root_storage='TTree; numeric arrays are embedded, no NPZ sidecar')
        self.meta.setdefault('role', 'observations')
        self.meta.setdefault('physics_accuracy_evaluated', False)
        fd, temporary = tempfile.mkstemp(prefix='.'+self.path.stem+'_', suffix='.root', dir=self.path.parent)
        os.close(fd)
        self.temporary = Path(temporary)
        self.file = uproot.recreate(self.temporary, compression=uproot.ZLIB(4))
        self.tree = None
        self.fields = None
        self.count = 0
        self.identities = set()
        self.closed = False

    @staticmethod
    def _numeric(name, value):
        if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', name) or name.endswith('__shape'):
            raise ValueError('Invalid/reserved ROOT field name: '+name)
        array = np.asarray(value)
        if array.dtype.kind not in 'biuf' or array.ndim > 8:
            raise ValueError('ROOT event fields must be numeric scalars/arrays: '+name)
        if array.dtype.kind == 'f' and np.isinf(array).any():
            raise ValueError('Infinite numeric field: '+name)
        if array.dtype.kind == 'b':
            dtype = np.dtype('bool') if array.ndim == 0 else np.dtype('uint8')
        elif array.dtype.kind in 'iu':
            dtype = np.dtype('int64')
        else:
            dtype = np.dtype('float32') if array.dtype.itemsize <= 4 else np.dtype('float64')
        return np.asarray(array, dtype=dtype), dtype

    def write(self, arrays, diagnostics=None):
        import awkward as ak
        if self.closed:
            raise RuntimeError('Stage writer is closed')
        names = sorted(arrays)
        if self.meta['role'] != 'truth' and any(n.startswith(('truth_', 'mc_target_')) for n in names):
            raise ValueError('Truth must be stored separately, not in inference-stage branches')
        numeric = {name: self._numeric(name, arrays[name]) for name in names}
        if 'entry' not in numeric or 'file_index' not in numeric:
            raise ValueError('Every ROOT row needs scalar entry and file_index')
        identity = tuple(int(numeric[n][0]) for n in ('file_index', 'entry'))
        if min(identity) < 0 or identity in self.identities:
            raise ValueError('Negative/duplicate ROOT event identity: '+str(identity))
        if self.fields is None:
            self.fields = {name: {'dtype': str(dtype), 'scalar': array.ndim == 0}
                           for name, (array, dtype) in numeric.items()}
            branch_types = {'record_json': 'string'}
            for name, field in self.fields.items():
                branch_types[name] = field['dtype'] if field['scalar'] else 'var * '+field['dtype']
                if not field['scalar']:
                    branch_types[name+'__shape'] = 'var * int64'
            self.tree = self.file.mktree('events', branch_types,
                                        title='One event per row; observation-driven reconstruction')
        if names != sorted(self.fields):
            raise ValueError('Field set changed between ROOT event rows')
        payload = {'record_json': [json.dumps(jsonable(diagnostics or {}), ensure_ascii=False, allow_nan=False)]}
        for name, (array, dtype) in numeric.items():
            spec = self.fields[name]
            if spec['scalar'] != (array.ndim == 0) or str(dtype) != spec['dtype']:
                raise ValueError('Scalar/dtype contract changed: '+name)
            if spec['scalar']:
                payload[name] = np.asarray([array.item()], dtype=dtype)
            else:
                flat = np.ascontiguousarray(array).reshape(-1)
                payload[name] = ak.unflatten(flat, [flat.size])
                payload[name+'__shape'] = ak.unflatten(np.asarray(array.shape, dtype=np.int64), [array.ndim])
        self.tree.extend(payload)
        self.identities.add(identity)
        self.count += 1

    def metrics(self, values):
        """Plottable summary metrics; names must distinguish diagnostic vs truth accuracy."""
        if 'metrics' in self.file.keys(recursive=False, cycle=False):
            raise ValueError('Metrics already written')
        tree = self.file.mktree('metrics', {'name': 'string', 'value': 'float64'})
        if values:
            tree.extend({'name': list(values), 'value': np.array(list(values.values()), dtype=float)})

    def histogram(self, name, samples, bins=60, range=None):
        values = np.asarray(samples, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        self.file['performance/'+name] = np.histogram(values, bins=bins, range=range)

    def close(self):
        if self.closed:
            return
        if self.count == 0:
            self.abort()
            raise ValueError('Refusing to publish an empty reconstruction stage')
        self.meta.update(complete=True, events=self.count, numeric_fields=self.fields)
        self.file['manifest'] = json.dumps(jsonable(self.meta), ensure_ascii=False, allow_nan=False)
        self.file.close()
        try:
            # Same-directory hard link publishes exclusively; a concurrent writer
            # cannot make an exists-check + replace overwrite an earlier result.
            os.link(self.temporary, self.path)
        finally:
            self.temporary.unlink(missing_ok=True)
            self.closed = True

    def abort(self):
        if not self.closed:
            self.file.close()
            self.temporary.unlink(missing_ok=True)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.abort()


def manifest(path):
    import uproot
    with uproot.open(path) as source:
        meta = json.loads(str(source['manifest']))
        if meta.get('schema') != SCHEMA or not meta.get('complete'):
            raise ValueError('Incomplete or incompatible reconstruction ROOT stage')
        if source['events'].classname != 'TTree':
            raise ValueError('Expected ROOT6.24-compatible TTree, not RNTuple')
        if int(meta['events']) != source['events'].num_entries:
            raise ValueError('Stage manifest/tree event counts differ')
        return meta


def read_stage(path, allow_truth=False, entry_indices=None):
    """Yield (numeric_arrays, diagnostics) using this ROOT file alone."""
    import uproot
    meta = manifest(path)
    if meta.get('role') in ('truth','evaluation') and not allow_truth:
        raise ValueError('Truth/evaluation ROOT cannot be used as an observation-stage input')
    fields = meta['numeric_fields']
    keys = ['record_json']
    for name, spec in fields.items():
        keys.append(name)
        if not spec['scalar']:
            keys.append(name+'__shape')
    with uproot.open(path) as source:
        tree = source['events']
        indices=range(tree.num_entries) if entry_indices is None else entry_indices
        for index in indices:
            index=int(index)
            if index<0 or index>=tree.num_entries:
                raise IndexError('Requested ROOT stage row is outside its TTree')
            row = tree.arrays(keys, entry_start=index, entry_stop=index+1, library='np')
            arrays = {}
            for name, spec in fields.items():
                value = np.asarray(row[name][0], dtype=spec['dtype'])
                if not spec['scalar']:
                    shape = tuple(int(x) for x in row[name+'__shape'][0])
                    if any(n < 0 for n in shape) or int(np.prod(shape)) != value.size:
                        raise ValueError('Invalid embedded array shape: '+name)
                    value = value.reshape(shape)
                arrays[name] = value.copy()
            yield arrays, json.loads(row['record_json'][0])
