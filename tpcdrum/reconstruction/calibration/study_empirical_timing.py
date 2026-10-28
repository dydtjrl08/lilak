"""Historical-MC timing study, separate from observation-only reconstruction.

Export measured layer statistics and ordered MC-plane crossings into SEPARATE
ROOT files. Fit a historical affine map only on the frozen training partition.
This is not a native transport or real-detector calibration and never enables
the production drift flag. Neither Si energies nor new DDE data are fabricated.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import time

import numpy as np

from ..preprocessing import LAYER_V_MM, preprocess
from ..raw_native.reader import ObservedReader
from ..raw_source import inspect
from ..stage_io import StageWriter, read_stage, manifest, sha256, jsonable


SPLITS = {'train': 0, 'validation': 1, 'test': 2}


def _selection(path):
    path = Path(path).expanduser().resolve(strict=True)
    config = json.loads(path.read_text())
    if config.get('configuration_frozen_before_waveforms_and_paths') is not True:
        raise ValueError('A pre-data frozen split manifest is required')
    selected = Path(config['selected_events_csv']).resolve(strict=True)
    if sha256(selected) != config['selected_events_sha256']:
        raise ValueError('Frozen event selection changed')
    with selected.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    seen = set(); groups = {}
    for row in rows:
        for key in ('request_index', 'input_file_index', 'entry', 'source_entry'):
            row[key] = int(row[key])
        identity = row['input_file_index'], row['entry']
        if identity in seen or row['split'] not in SPLITS:
            raise ValueError('Duplicate event or unknown split')
        seen.add(identity)
        previous = groups.setdefault(row['source_entry'], row['split'])
        if previous != row['split']:
            raise ValueError('Sibling source group leaked across partitions')
        expected = Path(config['input_roots'][row['input_file_index']]).resolve(strict=True)
        if Path(row['path']).resolve(strict=True) != expected:
            raise ValueError('CSV input and manifest path disagree')
    if len(rows) != config['counts']['events'] or len(groups) != config['counts']['source_groups']:
        raise ValueError('Frozen selection cardinality changed')
    return config, rows, sha256(path)


def export(manifest_path, output, root_executable=None):
    from .empirical_reader import EmpiricalStepReader
    from .empirical_timing import whole_layer_timing, unique_forward_crossing

    config, rows, selection_hash = _selection(manifest_path)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected_files = sorted({row['input_file_index'] for row in rows})
    sources = {i: inspect(config['input_roots'][i]) for i in selected_files}
    metadata = dict(study_scope='historical_MC_only', split_manifest_sha256=selection_hash,
        selected_csv_sha256=config['selected_events_sha256'], observed_sources=list(sources.values()),
        production_calibration_enabled=False, native_transport_calibration_verified=False,
        crossing_target='unique forward intersection of stored ordered midpoint polyline with v=12*layer',
        maximum_truth_segment_mm=2., active_pad_u_bounds_mm=[-36.95, 36.95],
        observed_gate='whole-layer positive charge>0 and peak>40ADC; no MC-tag/time-window cleaning',
        sources_grouped_by='separately audited common metadata lineage and original source entry')
    statistics = dict(events=0, possible_layers=0, observed_layers=0, truth_crossing_layers=0,
                      paired_layers=0, target_read_failures=0)
    started = time.monotonic()
    observed_path, truth_path = output/'timing_observations.root', output/'timing_truth.root'
    with StageWriter(observed_path, 'empirical_timing_observations', dict(metadata, role='observations')) as obs, \
            StageWriter(truth_path, 'empirical_timing_truth', dict(metadata, role='truth')) as truth:
        for fi in selected_files:
            info = sources[fi]
            with ExitStack() as stack:
                reader = stack.enter_context(ObservedReader(info['path'], root_executable))
                steps_reader = None
                for row in (r for r in rows if r['input_file_index'] == fi):
                    entry = row['entry']
                    raw = reader.read(entry)
                    measured, preparation = preprocess(raw)
                    q = measured['positive_waveform']
                    moments = whole_layer_timing(q, minimum_layer_charge_adc_tb=0.,
                                                minimum_layer_peak_adc=40.)
                    identity = dict(file_index=np.int64(fi), entry=np.int64(entry),
                        source_group=np.int64(row['source_entry']),
                        split_code=np.int64(SPLITS[row['split']]),
                        request_index=np.int64(row['request_index']))
                    observed = dict(identity, positive_waveform=q, raw_waveform=raw,
                        baseline=measured['baseline'], noise_sigma=measured['noise_sigma'],
                        mean_tb=np.asarray(moments['mean_tb'], dtype=float),
                        width_tb=np.asarray(moments['width_tb'], dtype=float),
                        mean_u_mm=np.asarray(moments['mean_u_mm'], dtype=float),
                        charge_adc_tb=np.asarray(moments['charge_adc_tb'], dtype=float),
                        peak_adc=np.asarray(moments['peak_adc'], dtype=float),
                        edge_fraction_low=np.asarray(moments['edge_fraction_low'], dtype=float),
                        edge_fraction_high=np.asarray(moments['edge_fraction_high'], dtype=float),
                        observed_layer_valid=np.asarray(moments['valid_layer'], dtype=np.uint8))
                    obs.write(observed, dict(observed_only=True, preprocessing=preparation,
                        statistic_definition=moments['statistic_definition']))
                    points = np.full((12, 3), np.nan); valid = np.zeros(12, np.uint8)
                    lengths = np.full(12, np.nan); crossing_counts = np.zeros(12, np.int64)
                    initial = np.full(8, np.nan); details = []; target_error = None
                    try:
                        if steps_reader is None:
                            steps_reader = stack.enter_context(EmpiricalStepReader(info['path'], root_executable))
                        initial = np.asarray(steps_reader.track(entry), dtype=float)
                        steps = np.asarray(steps_reader.steps(entry), dtype=float)
                        expected_vertex = np.array([float(row['vertex_x_mm']),
                            float(row['vertex_y_mm']), float(row['vertex_z_mm'])])
                        if initial.shape != (8,) or not np.allclose(initial[1:4], expected_vertex,
                                                                   atol=1e-7, rtol=0.):
                            raise ValueError('Actual selected alpha initial vertex disagrees with frozen lineage metadata')
                        for layer, depth in enumerate(LAYER_V_MM):
                            result = unique_forward_crossing(steps[:, 1:4], float(depth),
                                coordinate_frame='pad_uvw_mm', active_u_bounds_mm=[-36.95, 36.95],
                                max_segment_length_mm=2.)
                            points[layer] = result['point_uvw_mm']
                            valid[layer] = result['valid']
                            crossing_counts[layer] = result['crossing_count']
                            lengths[layer] = result['segment_length_mm']
                            details.append(result)
                    except TimeoutError:
                        raise  # A timeout is not permission to silently rerun an export.
                    except (ValueError, RuntimeError) as error:
                        target_error = str(error)
                        statistics['target_read_failures'] += 1
                        if steps_reader is not None:
                            steps_reader.close(); steps_reader = None
                        points[:] = np.nan; valid[:] = 0
                    momentum = np.array([-initial[4], initial[6], initial[5]])
                    norm = np.linalg.norm(momentum)
                    vertex = np.array([-initial[1], initial[3]+66., initial[2]+128.8])
                    truth.write(dict(identity, truth_crossing_uvw_mm=points,
                        truth_crossing_valid=valid, truth_segment_length_mm=lengths,
                        truth_crossing_count=crossing_counts, truth_initial_track_g4=initial,
                        truth_initial_vertex_uvw_mm=vertex,
                        truth_initial_direction_uvw=momentum/norm if norm > 0. else np.full(3, np.nan)),
                        dict(evaluation_only=True, target_error=target_error,
                             stored_steps_coordinate_frame='pad_uvw_mm_NO_transform', crossings=details))
                    statistics['events'] += 1; statistics['possible_layers'] += 12
                    statistics['observed_layers'] += int(np.sum(observed['observed_layer_valid']))
                    statistics['truth_crossing_layers'] += int(valid.sum())
                    statistics['paired_layers'] += int(np.sum(valid & observed['observed_layer_valid']))
                    if statistics['events'] % 25 == 0 or statistics['events'] == len(rows):
                        print('timing export', statistics['events'], '/', len(rows),
                              'file', fi, 'entry', entry, 'paired layers', statistics['paired_layers'], flush=True)
        obs.metrics(statistics); truth.metrics(statistics)
    receipt = dict(metadata, statistics=statistics, elapsed_seconds=time.monotonic()-started,
        observations_root=str(observed_path), observations_sha256=sha256(observed_path),
        truth_root=str(truth_path), truth_sha256=sha256(truth_path),
        truth_was_used_for_exported_observation_statistics=False,
        affine_model_fitted=False, no_120_degree_constraint=True)
    (output/'export_receipt.json').write_text(json.dumps(jsonable(receipt), indent=2)+'\n')
    return receipt


def _statistic_rows(path, fields, allow_truth=False):
    """Bounded column reads; training does not reread raw waveform branches."""
    import uproot
    meta = manifest(path)
    if meta.get('role') in ('truth', 'evaluation') and not allow_truth:
        raise ValueError('Truth/evaluation data cannot be observation statistics')
    specifications = {name: meta['numeric_fields'][name] for name in fields}
    keys = list(fields)+[name+'__shape' for name, spec in specifications.items() if not spec['scalar']]
    with uproot.open(path) as source:
        for block in source['events'].iterate(keys, step_size='8 MB', library='np'):
            for index in range(len(block['entry'])):
                row = {}
                for name, spec in specifications.items():
                    value = np.asarray(block[name][index], dtype=spec['dtype'])
                    if not spec['scalar']:
                        shape = tuple(int(x) for x in block[name+'__shape'][index])
                        if any(n < 0 for n in shape) or int(np.prod(shape)) != value.size:
                            raise ValueError('Corrupt embedded statistic shape')
                        value = value.reshape(shape)
                    row[name] = value.copy()
                yield row


def fit_exported(observations_root, truth_root, output, bootstrap_samples=200):
    """Train once and score the frozen partitions, retaining every layer row.

    The source ROOT files are the only data dependencies. No original digi,
    waveform truth attribution, external NPZ or refit on validation/test is used.
    """
    from .empirical_timing import fit_group_affine, bootstrap_group_affine, evaluate_frozen_map

    observations_meta, truth_meta = manifest(observations_root), manifest(truth_root)
    if (observations_meta.get('role') != 'observations' or truth_meta.get('role') != 'truth' or
            observations_meta.get('split_manifest_sha256') != truth_meta.get('split_manifest_sha256')):
        raise ValueError('Need independently stored paired observation/truth study ROOTs')
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = {}
    identity_fields = ['file_index', 'entry', 'source_group', 'split_code']
    for row in _statistic_rows(truth_root, identity_fields+['truth_crossing_uvw_mm', 'truth_crossing_valid'], True):
        targets[int(row['file_index']), int(row['entry'])] = row
    fields = {name: [] for name in ('mean_tb', 'truth_w', 'group', 'split', 'layer', 'mean_u',
        'width_tb', 'log_charge', 'edge_fraction_high', 'observed_valid', 'target_valid')}
    event_ids = []
    for observed in _statistic_rows(observations_root, identity_fields+['mean_tb', 'width_tb', 'mean_u_mm',
            'charge_adc_tb', 'edge_fraction_high', 'observed_layer_valid']):
        identity = int(observed['file_index']), int(observed['entry'])
        target = targets.pop(identity)
        if (int(observed['source_group']) != int(target['source_group']) or
                int(observed['split_code']) != int(target['split_code'])):
            raise ValueError('ROOT pairing disagrees in source group or frozen partition')
        event_ids.append(identity)
        values = dict(mean_tb=observed['mean_tb'], truth_w=target['truth_crossing_uvw_mm'][:, 2],
            group=np.full(12, observed['source_group'], dtype=np.int64),
            split=np.full(12, observed['split_code'], dtype=np.int64), layer=np.arange(12),
            mean_u=observed['mean_u_mm'], width_tb=observed['width_tb'],
            log_charge=np.log10(np.maximum(observed['charge_adc_tb'], 1e-300)),
            edge_fraction_high=observed['edge_fraction_high'],
            observed_valid=observed['observed_layer_valid'].astype(bool),
            target_valid=target['truth_crossing_valid'].astype(bool))
        for key in fields:
            fields[key].extend(np.asarray(values[key]).tolist())
    if targets:
        raise ValueError('Unpaired truth events remain')
    arrays = {key: np.asarray(value) for key, value in fields.items()}
    groups, split = arrays['group'].astype(np.int64), arrays['split'].astype(np.int64)
    if any(set(groups[split == i]) & set(groups[split == j]) for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Source groups leaked across stored ROOT partitions')
    gates = dict(observed_layer=arrays['observed_valid'].astype(bool),
                 unique_forward_MC_crossing=arrays['target_valid'].astype(bool))
    train = split == SPLITS['train']
    model = fit_group_affine(arrays['mean_tb'][train], arrays['truth_w'][train], groups[train],
                           gate_masks={key: value[train] for key, value in gates.items()})
    if not model['success']:
        (output/'failed_model.json').write_text(json.dumps(jsonable(model), indent=2)+'\n')
        raise ValueError('Frozen training cohort does not identify an affine model: '+model['status'])
    bootstrap = bootstrap_group_affine(arrays['mean_tb'][train], arrays['truth_w'][train], groups[train],
        gate_masks={key: value[train] for key, value in gates.items()}, n_resamples=bootstrap_samples, seed=20261027)
    (output/'frozen_affine_model.json').write_text(json.dumps(jsonable(model), indent=2)+'\n')
    (output/'training_coefficient_bootstrap.json').write_text(json.dumps(jsonable(bootstrap), indent=2)+'\n')
    diagnostic_edges = dict(layer=np.arange(-.5, 12., 2.), mean_u=np.arange(-40., 41., 10.),
        mean_tb=np.arange(0., 513., 32.), truth_w=np.arange(-20., 201., 20.),
        width_tb=np.arange(0., 129., 8.), log_charge=np.arange(1., 9.1, .5),
        edge_fraction_high=np.array([0., .0001, .001, .01, .1, 1.]))
    models = {'affine_train_only': model,
        'unadjusted_v_times_t': dict(success=True, coefficients=np.array([0., .36])),
        'delay_and_GEM_reference_only': dict(success=True, coefficients=np.array([9.-80.*.36, .36]))}
    evaluations = {}; numeric = {}
    for name, candidate in models.items():
        evaluations[name] = {}
        for partition, code in SPLITS.items():
            selected = split == code
            details = evaluate_frozen_map(candidate, arrays['mean_tb'][selected], arrays['truth_w'][selected],
                groups[selected], gate_masks={key: value[selected] for key, value in gates.items()},
                bins={key: dict(values=arrays[key][selected], edges=edges) for key, edges in diagnostic_edges.items()},
                n_bootstrap=bootstrap_samples, seed=20261027+code)
            summary = {key: value for key, value in details.items() if key not in
                ('predicted_w_mm', 'residual_true_minus_predicted_mm', 'accepted_row',
                 'outside_training_mean_tb_range')}
            evaluations[name][partition] = summary
            numeric[name, partition] = details
    report = dict(scope='historical_MC_empirical_timing_ONLY', evaluations=evaluations,
        calibrated_for_charge_mean=False, eligible_for_production=False,
        native_transport_calibration_verified=False, coefficients_fitted_on='train only',
        parameter_covariance_is_global_common_error=True, bootstrap_samples=bootstrap_samples,
        test_ci_conditions_on_frozen_coefficients=True,
        crossing_truth_is_midpoint_polyline_interpolation=True,
        layer_charge_mean_and_layer_center_crossing_are_different_definitions=True,
        no_120_degree_or_energy_sharing_constraint=True,
        complete_objective_achieved=False,
        physical_acceptance_tolerance='not supplied; diagnostic findings cannot automatically certify a physical calibration')
    (output/'timing_closure_summary.json').write_text(json.dumps(jsonable(report), indent=2)+'\n')
    with StageWriter(output/'timing_closure.root', 'historical_timing_closure', dict(role='evaluation',
            observations_root_sha256=sha256(observations_root), truth_root_sha256=sha256(truth_root),
            model=jsonable(model), scope=report['scope'])) as writer:
        for index, identity in enumerate(event_ids):
            sl = slice(index*12, (index+1)*12)
            pred = np.asarray(model['coefficients'])[0]+np.asarray(model['coefficients'])[1]*arrays['mean_tb'][sl]
            writer.write(dict(file_index=np.int64(identity[0]), entry=np.int64(identity[1]),
                source_group=np.int64(groups[sl][0]), split_code=np.int64(split[sl][0]),
                observed_mean_tb=arrays['mean_tb'][sl], reference_w_mm=arrays['truth_w'][sl],
                predicted_w_mm=pred, residual_true_minus_predicted_mm=arrays['truth_w'][sl]-pred,
                observed_layer_valid=arrays['observed_valid'][sl].astype(np.uint8),
                target_layer_valid=arrays['target_valid'][sl].astype(np.uint8)),
                dict(evaluation_only=True, model_refitted=False, production_calibration_enabled=False))
        writer.metrics({'possible_layers': len(split), 'paired_layers': int(np.sum(gates['observed_layer'] & gates['unique_forward_MC_crossing'])),
                        'production_calibration_enabled': 0.})
        for partition in SPLITS:
            details = numeric['affine_train_only', partition]
            writer.histogram('affine_'+partition+'_residual_mm',
                details['residual_true_minus_predicted_mm'][details['accepted_row']], bins=160, range=(-20., 20.))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('export', 'fit'), default='export')
    parser.add_argument('--manifest')
    parser.add_argument('--observations-root')
    parser.add_argument('--truth-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--root-executable')
    args = parser.parse_args()
    if args.stage == 'export':
        if not args.manifest:
            parser.error('--manifest is required for export')
        result = export(args.manifest, args.output, args.root_executable)
    else:
        if not args.observations_root or not args.truth_root:
            parser.error('--observations-root and --truth-root are required for fit')
        result = fit_exported(args.observations_root, args.truth_root, args.output)
    print(json.dumps(jsonable(result), indent=2))


if __name__ == '__main__':
    main()
