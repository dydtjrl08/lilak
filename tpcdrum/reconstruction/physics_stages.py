"""Observation-only Si/material/energy ROOT-record handlers.

Every handler returns (numeric_updates, merged_diagnostics). Arrays and UTF-8
metadata are embedded in ROOT branches; upstream config files are not needed
once a successful material import is saved. Native Geant4 path tracing is a
separate, explicitly audited server operation, never replaced by a gas default.
"""
from __future__ import annotations

import csv
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from .algorithms.physics import (StoppingTable, intersect_detector, inverse_energy,
                                 silicon_entrance_energy)
from .calibration.material_index import get_material_index


def _bytes(value):
    return np.frombuffer(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode('utf-8'), dtype=np.uint8).copy()


def _decode(value):
    array = np.asarray(value)
    if array.ndim != 1 or np.any(array < 0) or np.any(array > 255):
        raise ValueError('Invalid embedded UTF-8 byte array')
    return json.loads(np.asarray(array, dtype=np.uint8).tobytes().decode('utf-8'))


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).expanduser().read_text())


def _k(record):
    for key in ('fit_ok', 'track_directions_uvw', 'components'):
        if key in record:
            return len(record[key])
    return 0


def _done(diagnostic, section, values, result):
    return values, dict(diagnostic, **{section: result})


def _geometry(record, config):
    if config.get('si_geometry') is not None:
        obj = _json(config['si_geometry'])
        return obj.get('detectors', [])
    if np.asarray(record.get('si_geometry_json_utf8', [])).size:
        return _decode(record['si_geometry_json_utf8'])['detectors']
    if np.asarray(record.get('stopping_metadata_json_utf8', [])).size:
        return _decode(record['stopping_metadata_json_utf8']).get('detectors', [])
    return []


def _energy_config(record, config):
    result = {}
    for key in ('si_geometry_json_utf8', 'stopping_metadata_json_utf8'):
        if np.asarray(record.get(key, [])).size:
            result.update(_decode(record[key]).get('energy_stage_config', {}))
    result.update(config.get('energy', {}))
    return result


def _fitted_polyline(record, diagnostic):
    if not diagnostic.get('fitting', {}).get('physical_3d', False):
        raise ValueError('Missing calibrated physical-3D fitting provenance')
    k = _k(record)
    points = np.asarray(record['fit_points_uvw_mm'], dtype=float)
    states = np.asarray(record['fit_states'], dtype=float)
    valid = np.asarray(record['fit_valid_layer'], dtype=bool)
    ok = np.asarray(record['fit_ok'], dtype=bool)
    if points.shape != (k, 12, 3) or states.shape != (k, 12, 5) or valid.shape != (k, 12):
        raise ValueError('Invalid fitted point/state/layer array shapes')
    if ok.shape != (k,) or not np.all(ok):
        raise ValueError('Incomplete physical fitted track set')
    result = []
    for track in range(k):
        ids = np.flatnonzero(valid[track] & np.isfinite(points[track]).all(axis=1) &
                             np.isfinite(states[track, :, :4]).all(axis=1))
        if len(ids) < 2:
            raise ValueError('Fewer than two valid fitted physical points')
        ids = ids[np.argsort(points[track, ids, 1], kind='stable')]
        p = points[track, ids]
        if np.any(np.diff(p[:, 1]) <= 0):
            raise ValueError('Fitted depth coordinates must increase strictly')
        last = states[track, ids[-1]]
        tangent = np.array([last[2], 1., last[3]], dtype=float)
        tangent /= np.linalg.norm(tangent)
        result.append((p, tangent, ids))
    return result


def _front_crossing(origin, direction, detector):
    """Front active face seen by this local tangent; normal sign is arbitrary."""
    thickness = float(detector['thickness_mm'])
    normal = np.asarray(detector['normal_uvw'], dtype=float)
    dot = float(direction@normal)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError('Missing positive active Si thickness')
    if abs(dot) < 1e-12:
        return {'valid': False, 'status': 'parallel_to_Si'}
    front = dict(detector, center_uvw_mm=(np.asarray(detector['center_uvw_mm'], dtype=float)
                                       -np.sign(dot)*thickness*.5*normal).tolist())
    result = intersect_detector(origin, direction, front)
    if 'center_plane_point_uvw_mm' in result:
        result['entrance_uvw_mm'] = result.pop('center_plane_point_uvw_mm')
    return result


def _association_empty(k, n):
    return dict(si_association_geometry_unique=np.int64(0), si_association_unique=np.int64(0),
                track_si_detector_id=np.full(k, -1, dtype=np.int64),
                track_si_observed_index=np.full(k, -1, dtype=np.int64),
                track_si_shared=np.zeros(k, dtype=np.uint8),
                track_si_entrance_uvw_mm=np.full((k, 3), np.nan),
                track_si_incidence_cosine=np.full(k, np.nan),
                track_last_fit_point_uvw_mm=np.full((k, 3), np.nan),
                track_last_fit_direction_uvw=np.full((k, 3), np.nan),
                track_first_fit_point_uvw_mm=np.full((k, 3), np.nan),
                si_observed_energy_aligned_mev=np.full(n, np.nan),
                si_geometry_observed_index=np.full(n, -1, dtype=np.int64),
                si_geometry_json_utf8=np.empty(0, dtype=np.uint8))


def associate_event(record, diagnostic, config):
    """Match observed module totals by NAME using the last local fitted tangent.

    A unique geometry assignment can still leave shared-module track energies
    unidentified. Both flags are exported separately; no MC-ID energy splitting.
    """
    k = _k(record)
    geometry_error = None
    try:
        detectors = _geometry(record, config)
    except (ValueError, TypeError, KeyError, OSError) as error:
        detectors, geometry_error = [], str(error)
    values = _association_empty(k, len(detectors))
    result = dict(status='missing_calibration', missing_inputs=[],
                  intersection_surface='front active Si face',
                  direction_definition='last valid local fitted downstream tangent',
                  mc_track_deposits_used=False, hypotheses=[])
    missing = result['missing_inputs']
    if not bool(record.get('has_observed_si_edep', False)):
        missing.append('observed per-module Si energy')
    if not detectors:
        missing.append('exported actual Si geometry')
    names = diagnostic.get('observed_si_names', diagnostic.get('si_detector_names'))
    if names is None and np.asarray(record.get('si_geometry_json_utf8', [])).size:
        names = _decode(record['si_geometry_json_utf8']).get('observed_names')
    if not names:
        missing.append('observed Si detector names')
    if missing:
        if geometry_error:result['message'] = geometry_error
        return _done(diagnostic, 'association', values, result)
    try:
        energies = np.asarray(record['si_edep_mev'], dtype=float)
        geometry_names = [d['name'] for d in detectors]
        if (len(set(names)) != len(names) or len(set(geometry_names)) != len(geometry_names) or
                set(names) != set(geometry_names) or energies.shape != (len(names),) or
                not np.isfinite(energies).all() or np.any(energies < 0)):
            raise ValueError('Observed Si name/energy and geometry names are not a one-to-one match')
        indices = np.array([names.index(name) for name in geometry_names], dtype=np.int64)
        aligned = energies[indices]
        cfg = _energy_config(record, config)
        embedded_cfg = {key: value for key, value in cfg.items()
                        if key in ('association_position_sigma_mm', 'association_gate_sigma',
                                   'si_energy_threshold_mev', 'max_si_hypotheses')}
        values.update(si_observed_energy_aligned_mev=aligned,
                      si_geometry_observed_index=indices,
                      si_geometry_json_utf8=_bytes(dict(detectors=detectors, observed_names=names,
                                                       energy_stage_config=embedded_cfg)))
        polyline = _fitted_polyline(record, diagnostic)
        sigma = cfg.get('association_position_sigma_mm')
        if sigma is None:
            raise ValueError('Missing explicit Si association position uncertainty [mm]')
        sigma = float(sigma)
        gate = float(cfg.get('association_gate_sigma', 3.))
        threshold = float(cfg.get('si_energy_threshold_mev', 0.))
        if not np.isfinite([sigma, gate, threshold]).all() or sigma <= 0 or gate < 0 or threshold < 0:
            raise ValueError('Invalid Si association uncertainty/gate/energy threshold')
        hits = np.flatnonzero(aligned > threshold).tolist()
        if not hits:
            result['status'] = 'no_Si_energy_observation'
            return _done(diagnostic, 'association', values, result)
        crossings, candidates = [], []
        for track, (p, direction, _) in enumerate(polyline):
            values['track_last_fit_point_uvw_mm'][track] = p[-1]
            values['track_first_fit_point_uvw_mm'][track] = p[0]
            values['track_last_fit_direction_uvw'][track] = direction
            row = [_front_crossing(p[-1], direction, d) for d in detectors]
            crossings.append(row)
            candidates.append([j for j in hits if 'outside_mm' in row[j] and
                               np.linalg.norm(row[j]['outside_mm']) <= sigma*gate])
        hypotheses = []
        limit = int(cfg.get('max_si_hypotheses', 256))
        if limit < 1:
            raise ValueError('Invalid Si hypothesis limit')
        truncated = False
        for assignment in product(*candidates):
            if set(assignment) != set(hits):
                continue
            if len(hypotheses) == limit:
                truncated = True
                break
            multiplicity = np.bincount(assignment, minlength=len(detectors))
            hypotheses.append(dict(track_detector_ids=list(assignment),
                shared=[bool(multiplicity[j] > 1) for j in assignment],
                geometry_cost=sum(float(np.sum(crossings[i][j]['outside_mm']**2))/sigma**2
                                  for i, j in enumerate(assignment))))
        hypotheses.sort(key=lambda h: (h['geometry_cost'], h['track_detector_ids']))
        result.update(hypotheses=hypotheses, hypothesis_search_truncated=truncated,
                      crossings=crossings)
        if len(hypotheses) != 1 or truncated:
            result['status'] = 'ambiguous_Si_association' if hypotheses else 'unexplained_positive_Si_module'
            return _done(diagnostic, 'association', values, result)
        chosen = hypotheses[0]
        ids = np.array(chosen['track_detector_ids'], dtype=np.int64)
        shared = np.array(chosen['shared'], dtype=np.uint8)
        values.update(si_association_geometry_unique=np.int64(1),
                      si_association_unique=np.int64(not np.any(shared)),
                      track_si_detector_id=ids, track_si_observed_index=indices[ids],
                      track_si_shared=shared,
                      track_si_entrance_uvw_mm=np.stack([crossings[i][j]['entrance_uvw_mm'] for i, j in enumerate(ids)]),
                      track_si_incidence_cosine=np.array([crossings[i][j]['incidence_cosine'] for i, j in enumerate(ids)]))
        result['status'] = 'unresolved_shared_detector' if np.any(shared) else 'ok'
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status='missing_or_invalid_calibration', message=str(error))
    return _done(diagnostic, 'association', values, result)


def stable_path_id(file_index, entry, track):
    """Collision-free signed-int64 packing within explicit dataset bounds."""
    file_index, entry, track = int(file_index), int(entry), int(track)
    if not 0 <= file_index < 2**16 or not 0 <= entry < 2**40 or not 0 <= track < 2**7:
        raise ValueError('Event/track identity exceeds stable material-path ID bounds')
    return np.int64((file_index << 47) | (entry << 7) | track)


def _request_empty(k):
    return dict(material_request_valid=np.int64(0),
                material_request_path_id=np.full(k, -1, dtype=np.int64),
                material_request_first_segment_id=np.full(k, -1, dtype=np.int64),
                material_request_track=np.empty(0, dtype=np.int64),
                material_request_segment_id=np.empty(0, dtype=np.int64),
                material_request_start_uvw_mm=np.empty((0, 3)),
                material_request_end_uvw_mm=np.empty((0, 3)))


def material_request_event(record, diagnostic, config):
    """Prepare each event's vertex->observed polyline->Si FRONT-face path."""
    association, diag = associate_event(record, diagnostic, config)
    row = dict(record, **association)
    k, values = _k(row), _request_empty(_k(row))
    values.update(association)
    result = dict(status='missing_physical_path', path_origin='observation-only fitted polyline',
                  native_Geant4_tracing_performed=False)
    if not bool(row.get('vertex_valid', False)) or not bool(row['si_association_geometry_unique']):
        result['missing_inputs'] = ['valid physical vertex and unique geometric Si assignment']
        return _done(diag, 'material_request', values, result)
    try:
        vertex = np.asarray(row['vertex_uvw_mm'], dtype=float)
        if vertex.shape != (3,) or not np.isfinite(vertex).all():
            raise ValueError('Invalid physical vertex')
        polylines = _fitted_polyline(row, diag)
        pathids, trackids, segmentids, starts, ends = [], [], [], [], []
        for track, (points, _, _) in enumerate(polylines):
            entrance = row['track_si_entrance_uvw_mm'][track]
            if vertex[1] > points[0, 1]+1e-8 or entrance[1] < points[-1, 1]-1e-8:
                raise ValueError('Path order must be vertex -> observed points -> Si entrance')
            pathids.append(stable_path_id(row['file_index'], row['entry'], track))
            polyline = np.vstack([vertex, points, entrance])
            for segment, (a, b) in enumerate(zip(polyline[:-1], polyline[1:])):
                trackids.append(track); segmentids.append(segment)
                starts.append(a); ends.append(b)
        values.update(material_request_valid=np.int64(1),
                      material_request_path_id=np.asarray(pathids, dtype=np.int64),
                      material_request_first_segment_id=np.ones(k, dtype=np.int64),
                      material_request_track=np.asarray(trackids, dtype=np.int64),
                      material_request_segment_id=np.asarray(segmentids, dtype=np.int64),
                      material_request_start_uvw_mm=np.asarray(starts, dtype=float).reshape(-1, 3),
                      material_request_end_uvw_mm=np.asarray(ends, dtype=float).reshape(-1, 3))
        result.update(status='ready_for_native_material_trace', segments=len(segmentids),
                      includes_Si_active_thickness=False, native_Geant4_tracing_performed=False)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status='invalid_physical_path', message=str(error))
    return _done(diag, 'material_request', values, result)


def _request_rows(record):
    if not bool(record.get('material_request_valid', False)):
        raise ValueError('No valid embedded reconstructed material request')
    ids = np.asarray(record['material_request_path_id'], dtype=np.int64)
    tracks = np.asarray(record['material_request_track'], dtype=np.int64)
    segments = np.asarray(record['material_request_segment_id'], dtype=np.int64)
    starts = np.asarray(record['material_request_start_uvw_mm'], dtype=float)
    ends = np.asarray(record['material_request_end_uvw_mm'], dtype=float)
    if tracks.ndim != 1 or segments.shape != tracks.shape or starts.shape != (len(tracks), 3) or ends.shape != starts.shape:
        raise ValueError('Invalid embedded material request arrays')
    if not np.isfinite(starts).all() or not np.isfinite(ends).all() or np.any(tracks < 0) or np.any(tracks >= len(ids)):
        raise ValueError('Invalid embedded material request values')
    rows = []
    for track, segment, a, b in zip(tracks, segments, starts, ends):
        expected = stable_path_id(record['file_index'], record['entry'], track)
        if ids[track] != expected:
            raise ValueError('Material path ID does not bind this ROOT event/track')
        rows.append((int(ids[track]), int(segment), *a.tolist(), *b.tolist()))
    if len({row[:2] for row in rows}) != len(rows):
        raise ValueError('Duplicate event material request segment')
    for path_id in ids:
        segments = sorted(row[1] for row in rows if row[0] == path_id)
        if segments != list(range(len(segments))) or not segments:
            raise ValueError('Missing/noncontiguous reconstructed request segments')
    boundary = np.asarray(record['material_request_first_segment_id'])
    if boundary.shape != ids.shape or np.any(boundary != 1):
        raise ValueError('Invalid first-observed-point segment boundary')
    return rows


def write_material_requests(input_root, csv_path, config=None, skip_invalid=False):
    """Aggregate embedded per-event requests into the native exporter's CSV.

    No original digi, model, hidden cache, or MC trajectory is opened. If the
    input ROOT is the vertex stage, config supplies the explicit Si calibration.
    A material-request stage ROOT already embeds it. Output must be a new file.
    Explicit skip_invalid=True retains skipped identities/reasons in the return
    audit; it is never used to calculate detector performance on a selected set.
    """
    from .stage_io import read_stage
    output = Path(csv_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count, ids, created, skipped, accepted_events = 0, set(), False, [], 0
    try:
        with output.open('x', newline='') as stream:
            created = True
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['path_id', 'segment_id', 'u0', 'v0', 'w0', 'u1', 'v1', 'w1'])
            for record, diag in read_stage(input_root):
                try:
                    if not bool(record.get('material_request_valid', False)):
                        values, _ = material_request_event(record, diag, config or {})
                        record = dict(record, **values)
                    rows = _request_rows(record)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    if not skip_invalid:raise
                    skipped.append(dict(file_index=int(record['file_index']), entry=int(record['entry']),
                                        reason=str(error)))
                    continue
                accepted_events += 1
                for row in rows:
                    if row[:2] in ids:
                        raise ValueError('Duplicate path/segment across ROOT rows')
                    ids.add(row[:2]); writer.writerow(row); count += 1
        if not count:
            raise ValueError('No valid material request segments')
    except BaseException:
        if created:output.unlink(missing_ok=True)
        raise
    return dict(csv=str(output), sha256=_hash(output), segments=count,
                accepted_events=accepted_events, skipped_events=skipped,
                skipped_event_count=len(skipped), skip_invalid_explicitly_requested=bool(skip_invalid),
                selection_is_not_a_detector_performance_denominator=True,
                inference_input='embedded observed reconstructed paths only')


def _table_empty():
    return dict(stopping_payload_valid=np.int64(0),
                stopping_table_offsets=np.zeros(1, dtype=np.int64),
                stopping_energy_mev=np.empty(0), stopping_range_mm=np.empty(0),
                stopping_dedx_mev_per_mm=np.empty(0),
                stopping_metadata_json_utf8=np.empty(0, dtype=np.uint8))


def _tables(record, config):
    """Return actual calibration tables and numeric embedding, never nominal defaults."""
    configured = config.get('energy', {}).get('stopping_tables')
    if configured is not None:
        obj = _json(configured)
        if obj.get('schema') != 'tpcdrum-geant4-material-export-v1':
            raise ValueError('Unsupported stopping export schema')
        digest = _hash(Path(configured).expanduser()) if not isinstance(configured, dict) else hashlib.sha256(_bytes(obj)).hexdigest()
        raw_tables = [r for r in obj.get('tables', []) if int(r.get('particle_pdg', 0)) == 1000020040]
        metadata = {key: value for key, value in obj.items() if key != 'tables'}
        metadata['calibration_sha256'] = digest
        metadata['table_metadata'] = [{key: value for key, value in raw.items()
                                      if key not in ('energy_mev', 'range_mm', 'dedx_mev_per_mm')}
                                     for raw in raw_tables]
        metadata['table_origin'] = 'explicit actual Geant4 export; no nominal gas default'
        tables = {}
        for raw in raw_tables:
            raw = dict(raw)
            provenance = dict(raw.get('provenance', {}), particle_pdg=1000020040,
                              calibration_sha256=digest)
            if not provenance.get('source'):
                raise ValueError('Stopping table lacks explicit source provenance')
            raw['provenance'] = provenance
            table = StoppingTable.from_mapping(raw)
            if table.material in tables:
                raise ValueError('Duplicate stopping material key')
            tables[table.material] = table
        if not tables:
            raise ValueError('No explicit alpha stopping tables')
        offsets = np.r_[0, np.cumsum([len(t.energy_mev) for t in tables.values()])]
        values = dict(stopping_payload_valid=np.int64(1), stopping_table_offsets=offsets.astype(np.int64),
                      stopping_energy_mev=np.concatenate([t.energy_mev for t in tables.values()]),
                      stopping_range_mm=np.concatenate([t.range_mm for t in tables.values()]),
                      stopping_dedx_mev_per_mm=np.concatenate([t.dedx_mev_per_mm for t in tables.values()]),
                      stopping_metadata_json_utf8=_bytes(metadata))
        return tables, metadata, values
    if not bool(record.get('stopping_payload_valid', False)):
        raise ValueError('Missing explicit actual stopping/range export')
    metadata = _decode(record['stopping_metadata_json_utf8'])
    raw_tables = metadata['table_metadata']
    offsets = np.asarray(record['stopping_table_offsets'], dtype=np.int64)
    energies = np.asarray(record['stopping_energy_mev'], dtype=float)
    ranges = np.asarray(record['stopping_range_mm'], dtype=float)
    dedx = np.asarray(record['stopping_dedx_mev_per_mm'], dtype=float)
    if (offsets.shape != (len(raw_tables)+1,) or offsets[0] != 0 or
            offsets[-1] != len(energies) or ranges.shape != energies.shape or dedx.shape != energies.shape or
            np.any(np.diff(offsets) < 2)):
        raise ValueError('Corrupt embedded stopping arrays/offsets')
    tables = {}
    for i, raw in enumerate(raw_tables):
        if int(raw['particle_pdg']) != 1000020040 or not raw.get('provenance', {}).get('source'):
            raise ValueError('Embedded stopping table lacks alpha/source provenance')
        a, b = offsets[i:i+2]
        table = StoppingTable(energies[a:b], ranges[a:b], dedx[a:b], raw['material'],
                              dict(raw['provenance'], particle_pdg=1000020040), raw['units'])
        if table.material in tables:
            raise ValueError('Duplicate embedded stopping material')
        tables[table.material] = table
    return tables, metadata, {key: record[key] for key in _table_empty()}


def _material_empty():
    values = _table_empty()
    values.update(material_paths_embedded_valid=np.int64(0),
                  material_path_track=np.empty(0, dtype=np.int64),
                  material_path_segment_id=np.empty(0, dtype=np.int64),
                  material_path_part=np.empty(0, dtype=np.int64),
                  material_path_table_index=np.empty(0, dtype=np.int64),
                  material_path_length_mm=np.empty(0),
                  material_path_provenance_json_utf8=np.empty(0, dtype=np.uint8))
    return values


def _same_geometry(detectors, exported):
    by_name = {d['name']: d for d in exported}
    for detector in detectors:
        if detector['name'] not in by_name:
            raise ValueError('Material export geometry omits an associated Si module')
        actual = by_name[detector['name']]
        for key in ('center_uvw_mm', 'axis_u_uvw', 'axis_v_uvw', 'normal_uvw', 'width_mm', 'height_mm', 'thickness_mm'):
            if not np.allclose(detector[key], actual[key], rtol=0., atol=1e-9):
                raise ValueError('Si association and material-export geometry differ: '+key)


def import_material_paths_event(record, diagnostic, config):
    """Bind native G4Navigator CSV/receipt to this event and embed all physics data."""
    values = _material_empty()
    result = dict(status='missing_calibration', native_geometry_verified_here=False,
                  missing_inputs=[])
    cfg = config.get('energy', {})
    try:
        if not cfg.get('material_paths_validated', False):
            raise ValueError('Actual material paths are not explicitly validated')
        for key in ('material_path_csv', 'material_path_request_csv', 'material_path_receipt'):
            if not cfg.get(key):
                result['missing_inputs'].append(key)
        if result['missing_inputs']:
            return _done(diagnostic, 'material_import', values, result)
        tables, metadata, payload = _tables(record, config)
        metadata['energy_stage_config'] = {key: value for key, value in _energy_config(record, config).items()
            if key in ('association_position_sigma_mm', 'association_gate_sigma', 'si_energy_threshold_mev',
                       'max_si_hypotheses', 'si_energy_sigma_mev', 'si_energy_covariance_mev2',
                       'alpha_stops_in_Si_validated', 'validated_kinetic_upper_bound_mev')}
        payload['stopping_metadata_json_utf8'] = _bytes(metadata)
        values.update(payload)
        _same_geometry(_geometry(record, config), metadata.get('detectors', []))
        receipt = _json(cfg['material_path_receipt'])
        request_path, material_path = (Path(cfg[k]).expanduser() for k in ('material_path_request_csv', 'material_path_csv'))
        own_requests = _request_rows(record)
        index, cache_hit = get_material_index(request_path, material_path, receipt,
                                              metadata['calibration_sha256'], list(tables))
        paths = index.event_paths(own_requests)
        table_ids = {name: i for i, name in enumerate(tables)}
        tracks, segments, parts, material_ids, distances = [], [], [], [], []
        for track, path_id in enumerate(np.asarray(record['material_request_path_id'])):
            if int(path_id) not in paths:
                raise ValueError('This event/track material path is missing')
            for piece in paths[int(path_id)]:
                if piece['material'] not in table_ids:
                    raise ValueError('Missing stopping table for a navigated material')
                tracks.append(track); segments.append(piece['segment_id']); parts.append(piece['part'])
                material_ids.append(table_ids[piece['material']]); distances.append(piece['length_mm'])
        provenance = dict(index.provenance)
        values.update(material_paths_embedded_valid=np.int64(1),
                      material_path_track=np.asarray(tracks, dtype=np.int64),
                      material_path_segment_id=np.asarray(segments, dtype=np.int64),
                      material_path_part=np.asarray(parts, dtype=np.int64),
                      material_path_table_index=np.asarray(material_ids, dtype=np.int64),
                      material_path_length_mm=np.asarray(distances, dtype=float),
                      material_path_provenance_json_utf8=_bytes(provenance))
        result.update(status='ok', provenance=provenance, materials_embedded=len(tables),
                      pieces_embedded=len(distances), uses_default_gas_path=False,
                      material_index_cache_hit=cache_hit, global_CSV_reparsed_for_this_event=not cache_hit)
    except (ValueError, KeyError, TypeError, IndexError, OSError, StopIteration) as error:
        result.update(status='missing_or_invalid_material_export', message=str(error))
    return _done(diagnostic, 'material_import', values, result)


def _embedded_paths(record, tables, metadata):
    if not bool(record.get('material_paths_embedded_valid', False)):
        raise ValueError('No validated material paths embedded in ROOT')
    provenance = _decode(record['material_path_provenance_json_utf8'])
    if provenance.get('calibration_sha256') != metadata['calibration_sha256']:
        raise ValueError('Embedded material paths and stopping/geometry calibration differ')
    arrays = [np.asarray(record[key]) for key in ('material_path_track', 'material_path_segment_id',
               'material_path_part', 'material_path_table_index', 'material_path_length_mm')]
    n = len(arrays[0])
    if any(a.shape != (n,) for a in arrays) or not np.isfinite(arrays[-1]).all() or np.any(arrays[-1] <= 0):
        raise ValueError('Corrupt embedded material path arrays')
    keys, names, paths = set(), list(tables), [[] for _ in range(_k(record))]
    for track, segment, part, material, length in zip(*arrays):
        track, segment, part, material = map(int, (track, segment, part, material))
        key = (track, segment, part)
        if not 0 <= track < len(paths) or not 0 <= material < len(names) or key in keys or min(segment, part) < 0:
            raise ValueError('Invalid/duplicate embedded material path identity')
        keys.add(key)
        paths[track].append(dict(material=names[material], length_mm=float(length),
                                 segment_id=segment, part=part))
    expected = {(row[0], row[1]): np.linalg.norm(np.asarray(row[5:])-np.asarray(row[2:5]))
                for row in _request_rows(record)}
    for track, pieces in enumerate(paths):
        pieces.sort(key=lambda p: (p['segment_id'], p['part']))
        path_id = int(record['material_request_path_id'][track])
        actual_segments = {p['segment_id'] for p in pieces}
        expected_nonzero = {segment for (pid, segment), length in expected.items() if pid == path_id and length > 0}
        if actual_segments != expected_nonzero:
            raise ValueError('Embedded material segments do not match the reconstructed request')
        for pid_segment, length in expected.items():
            if pid_segment[0] != path_id:
                continue
            selected = [p for p in pieces if p['segment_id'] == pid_segment[1]]
            if [p['part'] for p in selected] != list(range(len(selected))) or abs(sum(p['length_mm'] for p in selected)-length) > 1e-6:
                raise ValueError('Embedded path does not close its reconstructed request')
    return paths


def _energy_empty(k):
    return dict(track_vertex_energy_mev=np.full(k, np.nan),
                track_vertex_energy_sigma_mev=np.full(k, np.nan),
                track_vertex_energy_covariance_mev2=np.full((k, k), np.nan),
                track_first_point_energy_mev=np.full(k, np.nan),
                track_first_point_energy_sigma_mev=np.full(k, np.nan),
                track_first_point_energy_covariance_mev2=np.full((k, k), np.nan),
                track_si_entrance_energy_mev=np.full(k, np.nan),
                energy_first_point_prior_mev=np.full(k, np.nan),
                energy_ok=np.zeros(k, dtype=np.uint8))


def energy_event(record, diagnostic, config):
    """Invert actual per-event paths; keep first-observed and vertex T separate."""
    k, values = _k(record), _energy_empty(_k(record))
    # Predictable branch schema even if an event is missing its calibration.
    try:n_detectors = len(_geometry(record, config))
    except (ValueError, TypeError, KeyError, OSError):n_detectors = 0
    values.update({key: record.get(key, value) for key, value in _association_empty(k, n_detectors).items()})
    values.update({key: record.get(key, value) for key, value in _table_empty().items()})
    n_observed = len(record.get('si_edep_mev', []))
    values['si_energy_covariance_mev2'] = np.full((n_observed, n_observed), np.nan)
    result = dict(status='missing_calibration', tracks=[],
                  uncertainty_scope='supplied observed Si covariance only; no invented path/material/straggling variance',
                  eventwise_straggling_not_inverted=True, equal_split_imposed=False)
    row = dict(record)
    diag = diagnostic
    try:
        # Recompute association/request only as a freshness check, never silently
        # replace a traced path when a refit or vertex update changes geometry.
        current, _ = material_request_event(row, diag, config)
        if not bool(current.get('material_request_valid', False)):
            raise ValueError('Current fitted physical path/Si association is unavailable')
        for key in ('material_request_path_id', 'material_request_first_segment_id', 'material_request_track', 'material_request_segment_id',
                    'material_request_start_uvw_mm', 'material_request_end_uvw_mm'):
            if key not in row or np.asarray(row[key]).shape != np.asarray(current[key]).shape or not np.allclose(row[key], current[key], rtol=0., atol=1e-9):
                raise ValueError('Material paths are missing or stale after a fit/vertex update; request and trace again')
        values.update({key: current[key] for key in _association_empty(k, len(current['si_observed_energy_aligned_mev']))})
        tables, metadata, payload = _tables(row, config)
        values.update(payload)
        paths = _embedded_paths(row, tables, metadata)
        _same_geometry(_geometry(row, config), metadata.get('detectors', []))
        detectors = _geometry(row, config)
        cfg = _energy_config(row, config)
        covariance = cfg.get('si_energy_covariance_mev2', row.get('si_energy_covariance_mev2'))
        sigma = cfg.get('si_energy_sigma_mev', row.get('si_energy_sigma_mev'))
        n = len(row['si_edep_mev'])
        if covariance is None:
            if sigma is None:
                raise ValueError('Missing explicit observed Si energy covariance/uncertainty')
            sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (n,))
            if not np.isfinite(sigma).all() or np.any(sigma < 0):
                raise ValueError('Invalid observed Si energy uncertainty')
            covariance = np.diag(sigma**2)
        covariance = np.asarray(covariance, dtype=float)
        if (covariance.shape != (n, n) or not np.isfinite(covariance).all() or
                not np.allclose(covariance, covariance.T, atol=1e-12) or
                np.linalg.eigvalsh(covariance)[0] < -1e-10):
            raise ValueError('Invalid observed Si energy covariance')
        # Embed the explicit observation covariance for later ROOT-only repeats.
        values['si_energy_covariance_mev2'] = covariance
        vertex_jac, first_jac = np.full(k, np.nan), np.full(k, np.nan)
        for track in range(k):
            if current['track_si_shared'][track]:
                result['tracks'].append(dict(status='unresolved_shared_detector', track=track))
                continue
            det_id = int(current['track_si_detector_id'][track])
            detector = detectors[det_id]
            observed = int(current['track_si_observed_index'][track])
            material = detector['material']
            if material not in tables:
                raise ValueError('Missing stopping table for actual active Si material')
            entrance = silicon_entrance_energy(current['si_observed_energy_aligned_mev'][det_id],
                detector['thickness_mm'], current['track_si_incidence_cosine'][track], tables[material],
                stopping_verified=bool(cfg.get('alpha_stops_in_Si_validated', False)),
                kinetic_upper_bound_mev=cfg.get('validated_kinetic_upper_bound_mev'))
            if entrance['status'] != 'ok':
                result['tracks'].append(dict(entrance, track=track)); continue
            si_energy = entrance['energy_at_si_mev']
            si_sigma = float(np.sqrt(max(0., covariance[observed, observed])))
            vertex = inverse_energy(si_energy, paths[track], tables, si_sigma, path_validated=True)
            boundary = int(row['material_request_first_segment_id'][track])
            first_path = [p for p in paths[track] if p['segment_id'] >= boundary]
            first = inverse_energy(si_energy, first_path, tables, si_sigma, path_validated=True)
            values['track_si_entrance_energy_mev'][track] = si_energy
            values['track_vertex_energy_mev'][track] = vertex['vertex_energy_mev']
            values['track_vertex_energy_sigma_mev'][track] = vertex['boundary_only_sigma_mev']
            values['track_first_point_energy_mev'][track] = first['vertex_energy_mev']
            values['energy_first_point_prior_mev'][track] = first['vertex_energy_mev']
            values['track_first_point_energy_sigma_mev'][track] = first['boundary_only_sigma_mev']
            vertex_jac[track] = vertex['boundary_energy_jacobian']
            first_jac[track] = first['boundary_energy_jacobian']
            values['energy_ok'][track] = 1
            result['tracks'].append(dict(status='ok', track=track, Si_entrance=entrance,
                                          vertex=vertex, first_observed_point=first))
        chosen = current['track_si_observed_index']
        for a in np.flatnonzero(values['energy_ok']):
            for b in np.flatnonzero(values['energy_ok']):
                c = covariance[chosen[a], chosen[b]]
                values['track_vertex_energy_covariance_mev2'][a, b] = vertex_jac[a]*vertex_jac[b]*c
                values['track_first_point_energy_covariance_mev2'][a, b] = first_jac[a]*first_jac[b]*c
        result['status'] = 'ok' if k and np.all(values['energy_ok']) else 'unresolved_energy'
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        result.update(status='missing_or_invalid_calibration', message=str(error))
    return _done(diag, 'energy', values, result)
