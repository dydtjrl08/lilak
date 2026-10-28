"""Bounded on-disk index of explicit Geant4 material-import CSV inputs.

Each immutable input set is hashed/parsed/validated once per process. SQLite
stores the global cohort on disk and event queries retrieve only its path IDs.
The index is a temporary current-stage accelerator, not an interstage input;
successful imported ROOT records embed the material data they subsequently use.
"""
from __future__ import annotations

import atexit
from collections import OrderedDict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile


_CACHE = OrderedDict()
_MAX_INDEXES = 2


def _fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def _identity(row):
    path_id, segment_id = int(row[0]), int(row[1])
    if not 0 <= path_id <= 2**63-1 or segment_id < 0:
        raise ValueError('Invalid material path/segment identity')
    return path_id, segment_id


class MaterialIndex:
    def __init__(self, request_path, material_path, receipt, calibration_sha256, material_keys):
        self.pid = os.getpid()
        self.temporary = Path(tempfile.mkdtemp(prefix='tpcdrum_material_index_'))
        self.path = self.temporary/'materials.sqlite'
        self.connection = None
        try:
            self._build(request_path, material_path, receipt, calibration_sha256, set(material_keys))
            self.connection = sqlite3.connect('file:'+str(self.path)+'?mode=ro', uri=True)
            self.connection.execute('PRAGMA query_only=ON')
        except BaseException:
            self.close()
            raise

    def _build(self, request_path, material_path, receipt, calibration_sha256, material_keys):
        before = (_fingerprint(request_path), _fingerprint(material_path))
        request_hash, path_hash = _sha256(request_path), _sha256(material_path)
        checks = dict(executed=True, generated_events=0,
                      reconstructed_path_sha256=request_hash,
                      material_path_csv_sha256=path_hash,
                      material_manifest_sha256=calibration_sha256)
        if any(receipt.get(key) != value for key, value in checks.items()):
            raise ValueError('Native receipt does not bind executed geometry/material/request/output hashes')
        requests, pieces = 0, 0
        database = sqlite3.connect(str(self.path))
        try:
            database.execute('PRAGMA journal_mode=OFF')
            database.execute('PRAGMA synchronous=OFF')
            database.execute('PRAGMA foreign_keys=ON')
            database.executescript('''
              CREATE TABLE requests(path_id INTEGER NOT NULL, segment_id INTEGER NOT NULL,
                u0 REAL,v0 REAL,w0 REAL,u1 REAL,v1 REAL,w1 REAL,length_mm REAL NOT NULL,
                PRIMARY KEY(path_id,segment_id)) WITHOUT ROWID;
              CREATE TABLE pieces(path_id INTEGER NOT NULL,segment_id INTEGER NOT NULL,
                part INTEGER NOT NULL,material TEXT NOT NULL,length_mm REAL NOT NULL,volume TEXT NOT NULL,
                PRIMARY KEY(path_id,segment_id,part),
                FOREIGN KEY(path_id,segment_id) REFERENCES requests(path_id,segment_id)) WITHOUT ROWID;
            ''')
            with Path(request_path).open(newline='') as stream:
                reader = csv.reader(stream)
                if next(reader, None) != ['path_id', 'segment_id', 'u0', 'v0', 'w0', 'u1', 'v1', 'w1']:
                    raise ValueError('Unexpected reconstructed request CSV header')
                for row in reader:
                    if len(row) != 8:
                        raise ValueError('Malformed reconstructed request CSV row')
                    key = _identity(row)
                    coordinates = tuple(float(x) for x in row[2:])
                    if not all(math.isfinite(x) for x in coordinates):
                        raise ValueError('Nonfinite reconstructed request coordinates')
                    length = math.sqrt(sum((coordinates[j+3]-coordinates[j])**2 for j in range(3)))
                    database.execute('INSERT INTO requests VALUES(?,?,?,?,?,?,?,?,?)', (*key, *coordinates, length))
                    requests += 1
            with Path(material_path).open(newline='') as stream:
                reader = csv.reader(stream)
                if next(reader, None) != ['path_id', 'segment_id', 'part', 'material', 'length_mm', 'volume']:
                    raise ValueError('Unexpected material-path CSV header')
                for row in reader:
                    if len(row) != 6:
                        raise ValueError('Malformed material-path CSV row')
                    key = _identity(row)
                    part, material, length = int(row[2]), row[3], float(row[4])
                    if part < 0 or material not in material_keys or not math.isfinite(length) or length <= 0:
                        raise ValueError('Invalid navigated part/material/positive length')
                    database.execute('INSERT INTO pieces VALUES(?,?,?,?,?,?)', (*key, part, material, length, row[5]))
                    pieces += 1
            if not requests or not pieces:
                raise ValueError('No nonempty reconstructed/material path cohort')
            invalid = database.execute('''
              SELECT r.path_id,r.segment_id FROM requests r LEFT JOIN pieces p
              ON p.path_id=r.path_id AND p.segment_id=r.segment_id
              GROUP BY r.path_id,r.segment_id HAVING
                abs(coalesce(sum(p.length_mm),0)-r.length_mm)>1e-6 OR
                (r.length_mm>0 AND count(p.part)=0) OR
                (count(p.part)>0 AND (min(p.part)<>0 OR max(p.part)+1<>count(p.part)))
              LIMIT 1
            ''').fetchone()
            if invalid is not None:
                raise ValueError('Navigated material lengths/parts do not close submitted segment '+str(invalid))
            # Segment IDs represent the ordered submitted polyline, including
            # permitted zero-length segments. Check the entire cohort once.
            invalid = database.execute('''SELECT path_id FROM requests GROUP BY path_id
                HAVING min(segment_id)<>0 OR max(segment_id)+1<>count(*) LIMIT 1''').fetchone()
            if invalid is not None:
                raise ValueError('Missing/noncontiguous global reconstructed request segments')
            database.commit()
        except sqlite3.Error as error:
            raise ValueError('Invalid global material index (duplicate or orphan piece/request): '+str(error)) from error
        finally:
            database.close()
        if before != (_fingerprint(request_path), _fingerprint(material_path)):
            raise ValueError('Material CSV input changed during indexing')
        self.provenance = dict(native_export_receipt=receipt, request_sha256=request_hash,
            material_path_csv_sha256=path_hash, calibration_sha256=calibration_sha256,
            zero_length_request_segments_are_allowed=True, material_order='vertex to Si entrance',
            observation_only_path_required=True, native_execution_verified_by_receipt=True,
            detector_physics_accuracy_validated=False,
            import_index='temporary immutable SQLite; not an interstage dependency',
            global_request_segments=requests, global_material_pieces=pieces)

    def event_paths(self, own_request_rows):
        own = {}
        for row in own_request_rows:
            key = _identity(row)
            if key in own:
                raise ValueError('Duplicate current-event material request')
            own[key] = tuple(float(x) for x in row[2:])
        path_ids = sorted({key[0] for key in own})
        result = {}
        for path_id in path_ids:
            submitted = self.connection.execute('SELECT segment_id,u0,v0,w0,u1,v1,w1 FROM requests WHERE path_id=? ORDER BY segment_id',
                                               (path_id,)).fetchall()
            if {segment for segment, *_ in submitted} != {segment for pid, segment in own if pid == path_id}:
                raise ValueError('Native request CSV does not contain this exact ROOT reconstructed path')
            for segment, *coordinates in submitted:
                if any(abs(a-b) > 1e-10 for a, b in zip(own[path_id, segment], coordinates)):
                    raise ValueError('Native request CSV does not contain this exact ROOT reconstructed path')
            pieces = self.connection.execute('SELECT segment_id,part,material,length_mm,volume FROM pieces WHERE path_id=? ORDER BY segment_id,part',
                                            (path_id,)).fetchall()
            if not pieces:
                raise ValueError('This event/track material path is empty')
            result[path_id] = [dict(segment_id=segment, part=part, material=material, length_mm=length, volume=volume)
                               for segment, part, material, length, volume in pieces]
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close(); self.connection = None
        if self.pid == os.getpid():
            shutil.rmtree(self.temporary, ignore_errors=True)


def get_material_index(request_path, material_path, receipt, calibration_sha256, material_keys):
    """Return (bounded immutable index, cache_hit); no CSV work on a cache hit."""
    request_path, material_path = Path(request_path).expanduser(), Path(material_path).expanduser()
    receipt_key = json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False)
    key = (os.getpid(), _fingerprint(request_path), _fingerprint(material_path), receipt_key,
           str(calibration_sha256), tuple(sorted(material_keys)))
    if key in _CACHE:
        index = _CACHE.pop(key); _CACHE[key] = index
        return index, True
    index = MaterialIndex(request_path, material_path, receipt, calibration_sha256, material_keys)
    _CACHE[key] = index
    while len(_CACHE) > _MAX_INDEXES:
        _, expired = _CACHE.popitem(last=False)
        expired.close()
    return index, False


@atexit.register
def close_material_indexes():
    while _CACHE:
        _, index = _CACHE.popitem(last=False)
        index.close()
