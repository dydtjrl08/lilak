"""Independent 3-alpha relativistic kinematics with explicit direction reference.

The existing fitter exports a tangent at the FIRST OBSERVED plane. It is not
silently called the production-vertex direction here. Without an explicit
approximation acknowledgement, finite mass/angle outputs are candidates only.
No 120-degree or equal-energy constraint is used to calculate these observables.
"""
from __future__ import annotations

import json

import numpy as np

from .algorithms.physics import four_vectors, kinematics, propagate_kinematics_uncertainty


def _decode(array):
    value = np.asarray(array)
    if value.ndim != 1 or np.any(value < 0) or np.any(value > 255):
        raise ValueError('Invalid embedded calibration UTF-8 array')
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode('utf-8'))


def _encode(value):
    return np.frombuffer(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8'),
                         dtype=np.uint8).copy()


def _empty():
    return dict(kinematics_ok=np.int64(0), kinematics_candidate_valid=np.int64(0),
                kinematics_physics_accuracy_validated=np.int64(0),
                alpha_four_vectors_mev=np.full((3, 4), np.nan),
                com_four_vectors_mev=np.full((3, 4), np.nan),
                parent_four_vector_mev=np.full(4, np.nan), parent_beta=np.full(3, np.nan),
                invariant_mass_mev=np.float64(np.nan), excitation_mev=np.float64(np.nan),
                q_three_alpha_mev=np.float64(np.nan),
                com_opening_angles_deg=np.full(3, np.nan), com_pair_indices=np.array([[0, 1], [0, 2], [1, 2]], np.int64),
                com_kinetic_energy_mev=np.full(3, np.nan), com_energy_fractions=np.full(3, np.nan),
                dalitz_xy=np.full(2, np.nan), com_momentum_residual_mev=np.float64(np.nan),
                com_energy_closure_mev=np.float64(np.nan),
                kinematics_direction_reference_code=np.int64(-1),
                kinematics_uses_first_observed_tangent=np.int64(0),
                kinematics_first_tangent_approximation_acknowledged=np.int64(0),
                kinematics_vertex_direction_provenance_valid=np.int64(0),
                kinematics_120_degree_constraint_used=np.int64(0),
                mass_alpha_mev=np.float64(np.nan), mass_carbon12_ground_mev=np.float64(np.nan),
                kinematics_mass_provenance_json_utf8=np.empty(0, dtype=np.uint8),
                kinematics_covariance_propagated=np.int64(0),
                kinematics_full_input_covariance=np.full((9, 9), np.nan),
                kinematics_output_covariance=np.full((7, 7), np.nan),
                kinematics_output_sigma=np.full(7, np.nan))


def _masses(record, config):
    supplied = config.get('masses', {})
    if supplied.get('alpha_mev') is not None and supplied.get('carbon12_ground_mev') is not None:
        return float(supplied['alpha_mev']), float(supplied['carbon12_ground_mev']), dict(
            source='explicit_config', declaration=supplied.get('provenance', 'caller supplied; not automatically validated'))
    if np.asarray(record.get('stopping_metadata_json_utf8', [])).size:
        metadata = _decode(record['stopping_metadata_json_utf8'])
        masses = metadata.get('masses_mev', {})
        if masses.get('alpha') is not None and masses.get('carbon12_ground') is not None:
            if metadata.get('schema') != 'tpcdrum-geant4-material-export-v1' or not metadata.get('calibration_sha256'):
                raise ValueError('Embedded Geant4 masses lack calibration schema/hash')
            return float(masses['alpha']), float(masses['carbon12_ground']), dict(
                source='embedded_initialized_Geant4_particle_masses',
                calibration_sha256=metadata['calibration_sha256'],
                geant4_version=metadata.get('geant4_version'),
                mass_convention='same initialized nuclear particle table as explicit transport calibration')
    if np.asarray(record.get('kinematics_mass_provenance_json_utf8', [])).size:
        provenance = _decode(record['kinematics_mass_provenance_json_utf8'])
        if not provenance.get('source'):
            raise ValueError('Embedded masses lack their original provenance')
        return float(record['mass_alpha_mev']), float(record['mass_carbon12_ground_mev']), provenance
    raise ValueError('Missing explicit or embedded initialized-Geant4 alpha/C12 ground masses')


def _directions(record, diagnostic, cfg):
    reference = cfg.get('direction_reference', 'first_observed_plane')
    if reference == 'vertex':
        directions = np.asarray(record.get('track_vertex_directions_uvw'), dtype=float)
        valid = np.asarray(record.get('vertex_direction_valid', []), dtype=bool)
        provenance = diagnostic.get('vertex_direction', {})
        if (directions.shape != (3, 3) or valid.shape != (3,) or not np.all(valid) or
                provenance.get('observation_only') is not True or not provenance.get('source')):
            raise ValueError('Vertex directions need explicit observation-only model provenance and valid per-track flags')
        return directions, 1, True, dict(status='explicit_vertex_direction', provenance=provenance)
    if reference != 'first_observed_plane':
        raise ValueError('direction_reference must be vertex or first_observed_plane')
    directions = np.asarray(record.get('track_directions_uvw'), dtype=float)
    if directions.shape != (3, 3):
        raise ValueError('Missing three fitted first-observed downstream directions')
    fit = diagnostic.get('fitting', {})
    fit_ok = np.asarray(record.get('fit_ok', []), dtype=bool)
    if not fit.get('physical_3d', False) or fit_ok.shape != (3,) or not np.all(fit_ok):
        raise ValueError('First-observed tangents lack calibrated physical-3D fit provenance')
    acknowledged = cfg.get('acknowledge_first_point_direction_as_vertex_approximation') is True
    return directions, 0, acknowledged, dict(
        status='acknowledged_first_point_direction_approximation' if acknowledged else
               'candidate_only_first_point_direction_not_vertex_direction',
        approximation_acknowledged=acknowledged,
        description='first observed downstream tangent combined with vertex kinetic energy; unobserved pre-hit deflections not recovered')


def kinematics_event(record, diagnostic, config):
    """Return numeric updates/diagnostics; optional covariance is explicitly joint.

    Full covariance ordering is (T0,tu0,tw0,T1,tu1,tw1,T2,tu2,tw2), with kinetic
    energies at the vertex and slopes in the selected direction reference. No
    block-diagonal covariance or zero cross terms are automatically fabricated.
    Candidate observables can be inspected without accepting a direction model.
    """
    values = _empty()
    result = dict(status='missing_physical_inputs', physics_accuracy_evaluated=False,
                  imposed_120_degree_constraint=False, imposed_equal_energy_constraint=False,
                  invariant_mass_is_not_excitation=True,
                  particle_hypothesis='three alpha particles; this handler does not perform PID',
                  uncertainty_status='not_supplied', direction_reference=None)
    cfg = config.get('kinematics', {})
    try:
        energies = np.asarray(record.get('track_vertex_energy_mev'), dtype=float)
        ok = np.asarray(record.get('energy_ok', []), dtype=bool)
        if energies.shape != (3,) or ok.shape != (3,) or not np.all(ok):
            raise ValueError('Need three independently resolved alpha vertex kinetic energies')
        mass, ground, provenance = _masses(record, config)
        if not np.isfinite([mass, ground]).all() or min(mass, ground) <= 0:
            raise ValueError('Nuclear masses must be positive finite explicit values')
        values.update(mass_alpha_mev=np.float64(mass), mass_carbon12_ground_mev=np.float64(ground),
                      kinematics_mass_provenance_json_utf8=_encode(provenance))
        directions, reference, accepted, direction_info = _directions(record, diagnostic, cfg)
        values.update(kinematics_direction_reference_code=np.int64(reference),
                      kinematics_uses_first_observed_tangent=np.int64(reference == 0),
                      kinematics_first_tangent_approximation_acknowledged=np.int64(reference == 0 and accepted),
                      kinematics_vertex_direction_provenance_valid=np.int64(reference == 1))
        vectors = four_vectors(energies, directions, mass)
        computed = kinematics(vectors, ground)
        values.update(kinematics_candidate_valid=np.int64(computed['angles_valid']),
                      kinematics_ok=np.int64(accepted and computed['angles_valid']),
                      alpha_four_vectors_mev=vectors, com_four_vectors_mev=computed['com_four_vectors'],
                      parent_four_vector_mev=computed['parent_four_vector'], parent_beta=computed['parent_beta'],
                      invariant_mass_mev=np.float64(computed['invariant_mass_mev']),
                      excitation_mev=np.float64(computed['excitation_mev']),
                      q_three_alpha_mev=np.float64(computed['q_three_alpha_mev']),
                      com_opening_angles_deg=computed['com_opening_angles_deg'],
                      com_kinetic_energy_mev=computed['com_kinetic_mev'],
                      com_energy_fractions=computed['com_energy_fractions'], dalitz_xy=computed['dalitz_xy'],
                      com_momentum_residual_mev=np.float64(computed['com_momentum_residual_mev']),
                      com_energy_closure_mev=np.float64(computed['com_energy_closure_mev']))
        result.update(status=('ok_with_explicit_direction_reference' if accepted else
                              'candidate_only_direction_reference_unacknowledged')
                             if computed['angles_valid'] else 'undefined_COM_observable',
                      direction_reference=direction_info, mass_provenance=provenance,
                      pair_order=[[0, 1], [0, 2], [1, 2]],
                      direction_reference_codes={'first_observed_plane': 0, 'vertex': 1})
        covariance = cfg.get('full_covariance_9x9', record.get('kinematics_full_joint_covariance_9x9'))
        if covariance is not None:
            declared_full = cfg.get('covariance_is_full_joint',
                                    bool(record.get('kinematics_covariance_is_full_joint', False)))
            if not declared_full:
                result['uncertainty_status'] = 'covariance_provided_without_full_joint_scope_declaration'
            elif np.any(directions[:, 1] <= 0):
                result['uncertainty_status'] = 'unsupported_backward_direction_slope_chart'
            else:
                try:
                    covariance = np.asarray(covariance, dtype=float)
                    propagated = propagate_kinematics_uncertainty(energies,
                        directions[:, [0, 2]]/directions[:, 1, None], covariance, mass, ground,
                        method=cfg.get('uncertainty_method', 'delta'),
                        samples=int(cfg.get('uncertainty_samples', 2000)),
                        seed=int(cfg.get('uncertainty_seed', 0)))
                    values.update(kinematics_covariance_propagated=np.int64(1),
                                  kinematics_full_input_covariance=covariance,
                                  kinematics_output_covariance=propagated['covariance'],
                                  kinematics_output_sigma=propagated['sigma'])
                    result.update(uncertainty_status='explicit_joint_covariance_propagated',
                                  uncertainty_labels=propagated['labels'],
                                  uncertainty_scope='caller-supplied full joint covariance only; physical calibration not certified')
                except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
                    result.update(uncertainty_status='invalid_explicit_covariance', uncertainty_message=str(error))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result.update(status='missing_or_invalid_physical_inputs', message=str(error))
    return values, dict(diagnostic, kinematics=result)
