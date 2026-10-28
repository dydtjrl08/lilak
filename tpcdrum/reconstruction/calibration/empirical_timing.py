"""Experimental historical-MC timing mathematics; never production calibration.

The predictor is the whole-layer positive observed-ADC charge mean. The target
is a unique crossing of an ORDERED stored-MCStep midpoint polyline, not the
response-weighted centroid that generated those ADC samples. Their discrepancy
is part of this empirical study. No velocity, event t0, truth-assisted inference,
physical drift-calibration switch, coordinate transform or extrapolated MC point
is supplied by this module. Physical target-quality gates belong to the caller.
"""
from __future__ import annotations

import numpy as np

from ..preprocessing import PAD_U_MM

STATISTIC = 'whole-layer positive observed ADC: sum(pad,TB)(TB*q)/sum(pad,TB)(q); no pad/time threshold clipping'
SCOPE = 'experimental historical-MC empirical map; not validated physical drift calibration'


def whole_layer_timing(positive_waveform, minimum_layer_charge_adc_tb=0.,
                       minimum_layer_peak_adc=0., *, edge_tb_count=1):
    """Measure all 64 pads/all 512 TB; activity gates never alter the mean.

    Input must ALREADY be observation-only pedestal-subtracted, positive ADC.
    width_tb is charge-cloud RMS width, NOT uncertainty on the mean and not
    divided by sqrt(ADC). Edge fractions are diagnostics, not selection gates.
    """
    q = np.asarray(positive_waveform, dtype=float)
    if q.shape != (12, 64, 512) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('Need finite nonnegative observed ADC (12,64,512)')
    limits = np.asarray([minimum_layer_charge_adc_tb, minimum_layer_peak_adc], float)
    if not np.isfinite(limits).all() or np.any(limits < 0):
        raise ValueError('Explicit layer gates must be finite and nonnegative')
    if (not isinstance(edge_tb_count, (int, np.integer)) or
            isinstance(edge_tb_count, (bool, np.bool_)) or not 1 <= edge_tb_count <= 256):
        raise ValueError('edge_tb_count must be an integer in [1,256]')
    charge = q.sum(axis=(1, 2))
    peak = q.max(axis=(1, 2))
    tb_charge = q.sum(axis=1)
    pad_charge = q.sum(axis=2)
    mean = np.full(12, np.nan)
    mean_u = np.full(12, np.nan)
    width = np.full(12, np.nan)
    low, high = np.full(12, np.nan), np.full(12, np.nan)
    nonzero = charge > 0
    time = np.arange(512., dtype=float)
    mean[nonzero] = tb_charge[nonzero] @ time / charge[nonzero]
    mean_u[nonzero] = pad_charge[nonzero] @ PAD_U_MM / charge[nonzero]
    width[nonzero] = np.sqrt(np.maximum(0.,
        np.sum(tb_charge[nonzero] * (time[None, :]-mean[nonzero, None])**2, axis=1) / charge[nonzero]))
    low[nonzero] = tb_charge[nonzero, :edge_tb_count].sum(axis=1) / charge[nonzero]
    high[nonzero] = tb_charge[nonzero, -edge_tb_count:].sum(axis=1) / charge[nonzero]
    charge_gate = charge > limits[0]
    peak_gate = peak > limits[1]
    valid = nonzero & charge_gate & peak_gate
    return dict(mean_tb=mean, width_tb=width, mean_u_mm=mean_u,
                charge_adc_tb=charge, peak_adc=peak,
                edge_fraction_low=low, edge_fraction_high=high,
                edge_tb_count=int(edge_tb_count), valid_layer=valid,
                gate_counts=dict(total_layers=12, positive_charge=int(nonzero.sum()),
                    charge_gate_pass=int(charge_gate.sum()), peak_gate_pass=int(peak_gate.sum()),
                    accepted_layers=int(valid.sum())),
                minimum_layer_charge_adc_tb=float(limits[0]), minimum_layer_peak_adc=float(limits[1]),
                statistic_definition=STATISTIC, width_definition='charge-weighted RMS cloud width, not centroid error',
                calibrated_for_charge_mean=False, eligible_for_production=False)


def unique_forward_crossing(points_uvw_mm, target_v_mm, *, active_u_bounds_mm,
                            max_segment_length_mm, coordinate_frame='pad_uvw_mm',
                            atol_mm=1.e-9):
    """Interpolate a unique forward crossing; never sort, transform or extrapolate.

    All duplicate/nonforward/reverse steps reject the ordered polyline. The gap
    gate applies to the segment actually interpolated. An exact saved midpoint
    at the requested plane needs no interpolation and has zero segment length.
    active-u and maximum-gap choices are explicit study target-quality gates.
    atol_mm is only numerical equality tolerance, not a detector acceptance cut.
    """
    bounds = np.asarray(active_u_bounds_mm, float)
    target, gap, tol = float(target_v_mm), float(max_segment_length_mm), float(atol_mm)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
            or not np.isfinite([target, gap, tol]).all() or gap <= 0 or tol < 0):
        raise ValueError('Explicit finite active-u bounds, positive segment gap and numerical tolerance required')
    result = dict(valid=False, status='invalid_polyline', failure_reasons=[],
                  point_uvw_mm=np.full(3, np.nan), candidate_point_uvw_mm=np.full(3, np.nan),
                  crossing_count=0, segment_indices=(), segment_length_mm=float('nan'),
                  coordinate_frame=coordinate_frame, target_v_mm=target,
                  active_u_bounds_mm=bounds.copy(), max_segment_length_mm=gap,
                  gap_gate_scope='interpolated crossing segment only', numerical_atol_mm=tol,
                  target_definition='ordered stored-MCStep midpoint-polyline crossing',
                  extrapolated=False, coordinate_transform_applied=False)
    if coordinate_frame != 'pad_uvw_mm':
        result.update(status='wrong_coordinate_frame', failure_reasons=['wrong_coordinate_frame'])
        return result
    p = np.asarray(points_uvw_mm, dtype=float)
    if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 2:
        result.update(status='insufficient_or_malformed_points', failure_reasons=['insufficient_or_malformed_points'])
        return result
    if not np.isfinite(p).all():
        result.update(status='nonfinite_points', failure_reasons=['nonfinite_points'])
        return result
    delta = np.diff(p, axis=0)
    length = np.linalg.norm(delta, axis=1)
    dv = delta[:, 1]
    reasons = []
    if np.any(length <= tol):
        reasons.append('duplicate_step_points')
    if np.any(dv < -tol):
        reasons.append('reverse_step')
    if np.any((np.abs(dv) <= tol) & (length > tol)):
        reasons.append('nonforward_step')
    on_plane = np.abs(p[:, 1]-target) <= tol
    if np.any(on_plane[:-1] & on_plane[1:]):
        reasons.append('segment_in_target_plane')
    candidates = [(p[i].copy(), (int(i), int(i)), 0.) for i in np.flatnonzero(on_plane)]
    for i in range(len(delta)):
        # Exact endpoints were included above, once per stored midpoint.
        if on_plane[i] or on_plane[i+1]:
            continue
        if (p[i, 1] < target < p[i+1, 1]) or (p[i+1, 1] < target < p[i, 1]):
            t = (target-p[i, 1]) / dv[i]
            point = p[i] + t*delta[i]
            candidates.append((point, (i, i+1), float(length[i])))
    result['crossing_count'] = len(candidates)
    if len(candidates) == 0:
        reasons.append('no_crossing_no_extrapolation')
    elif len(candidates) > 1:
        reasons.append('multiple_crossings')
    else:
        point, indices, segment_length = candidates[0]
        result.update(candidate_point_uvw_mm=point.copy(), segment_indices=indices,
                      segment_length_mm=segment_length)
        if segment_length > gap + tol:
            reasons.append('crossing_segment_gap')
        if point[0] < bounds[0]-tol or point[0] > bounds[1]+tol:
            reasons.append('outside_active_u')
    result['failure_reasons'] = reasons
    if reasons:
        result['status'] = reasons[0]
    else:
        result.update(valid=True, status='ok', point_uvw_mm=result['candidate_point_uvw_mm'].copy())
    return result


def _mask(value, n, name):
    a = np.asarray(value)
    if (a.shape != (n,) or a.dtype.kind not in 'biuf' or
            not np.isfinite(a).all() or not np.isin(a, [0, 1]).all()):
        raise ValueError(name+' must be a length-N boolean/0/1 mask')
    return a.astype(bool, copy=True)


def _prepare(mean_tb, w_mm, source_groups, eligible=None, row_weights=None, gate_masks=None):
    x, y = np.asarray(mean_tb, float), np.asarray(w_mm, float)
    group = np.asarray(source_groups)
    if x.ndim != 1 or y.shape != x.shape or group.shape != x.shape:
        raise ValueError('mean_tb, w_mm and source_groups must be matching 1D arrays')
    if group.dtype.kind not in 'iuSU' and not (group.dtype.kind == 'f' and np.isfinite(group).all()):
        raise ValueError('Source group IDs must be finite numeric IDs or strings')
    n = len(x)
    weight = np.ones(n) if row_weights is None else np.asarray(row_weights, float)
    if weight.shape != (n,):
        raise ValueError('row_weights must have length N')
    gates = dict(finite_mean_tb=np.isfinite(x), finite_truth_w=np.isfinite(y),
                 positive_row_weight=np.isfinite(weight) & (weight > 0))
    gates['eligible'] = np.ones(n, bool) if eligible is None else _mask(eligible, n, 'eligible')
    for name, values in (gate_masks or {}).items():
        if not isinstance(name, str) or not name or name in gates:
            raise ValueError('Gate names must be unique nonempty strings')
        gates[name] = _mask(values, n, name)
    accepted = np.ones(n, bool)
    coverage = dict(total_rows=n, total_groups=int(len(np.unique(group))),
                    gates={}, sequential_gates=[])
    for name, values in gates.items():
        before = int(accepted.sum())
        accepted &= values
        coverage['gates'][name] = dict(pass_rows=int(values.sum()), fail_rows=int((~values).sum()))
        coverage['sequential_gates'].append(dict(gate=name, entered_rows=before,
            rejected_here=before-int(accepted.sum()), remaining_rows=int(accepted.sum())))
    coverage.update(accepted_rows=int(accepted.sum()), rejected_rows=int((~accepted).sum()),
                    accepted_groups=int(len(np.unique(group[accepted]))),
                    group_weighting='equal total weight per accepted source group; optional row weights normalized within each group')
    return x, y, group, weight, accepted, coverage


def _group_weights(group, row_weight):
    labels, inverse = np.unique(group, return_inverse=True)
    totals = np.bincount(inverse, weights=row_weight, minlength=len(labels))
    return row_weight/totals[inverse], labels, inverse


def _affine(x, y, weight):
    """Centered/scaled weighted solve, avoiding an arbitrary raw-unit rank cut."""
    if len(x) < 2 or not np.isfinite(weight.sum()) or weight.sum() <= 0:
        return None, 'insufficient_rows'
    center = float(np.average(x, weights=weight))
    scale = float(np.sqrt(np.average((x-center)**2, weights=weight)))
    if not np.isfinite(scale) or scale <= 0:
        return None, 'nonidentifiable_timing_range'
    design = np.column_stack((np.ones(len(x)), (x-center)/scale))
    sw = np.sqrt(weight)
    solution, _, rank, singular = np.linalg.lstsq(design*sw[:, None], y*sw, rcond=None)
    if rank != 2 or not np.isfinite(solution).all():
        return None, 'rank_deficient_affine_design'
    b = float(solution[1]/scale)
    coefficients = np.array([solution[0]-b*center, b])
    if not np.isfinite(coefficients).all():
        return None, 'nonfinite_affine_solution'
    return dict(coefficients=coefficients, a_mm=float(coefficients[0]), b_mm_per_tb=b,
                centered_design_condition=float(singular[0]/singular[-1]),
                weighted_mean_tb=center, weighted_scale_tb=scale), 'ok'


def fit_group_affine(mean_tb, w_mm, source_groups, eligible=None, row_weights=None,
                     gate_masks=None):
    """Fit w=a+b*meanTB with equal total source-group weight.

    A mathematically identifiable affine map needs at least two finite timing
    values. A single-group fit may be algebraically identifiable but does not
    support group-based uncertainty; bootstrap explicitly refuses that case.
    """
    x, y, group, weight, valid, coverage = _prepare(mean_tb, w_mm, source_groups,
        eligible, row_weights, gate_masks)
    model = dict(success=False, status='insufficient_rows', coefficients=np.full(2, np.nan),
                 a_mm=float('nan'), b_mm_per_tb=float('nan'), fit_row_count=int(valid.sum()),
                 fit_group_count=coverage['accepted_groups'], coverage=coverage,
                 statistic_definition=STATISTIC, scope=SCOPE,
                 calibrated_for_charge_mean=False, eligible_for_production=False,
                 method='source_group_weighted_affine_v1', frozen=True,
                 transform='w_empirical_mm = a_mm + b_mm_per_tb * observed_mean_tb',
                 training_mean_tb_range=[float('nan'), float('nan')])
    if valid.sum() < 2:
        return model
    balanced, labels, _ = _group_weights(group[valid], weight[valid])
    result, status = _affine(x[valid], y[valid], balanced)
    model['status'] = status
    if result is None:
        return model
    model.update(result, success=True, fit_source_groups=labels.copy(),
                 training_mean_tb_range=[float(x[valid].min()), float(x[valid].max())],
                 group_uncertainty_identifiable=len(labels) >= 2)
    return model


def bootstrap_group_affine(mean_tb, w_mm, source_groups, *, n_resamples, seed,
                           eligible=None, row_weights=None, gate_masks=None):
    """Resample entire source groups, retaining every within-group row/error.

    Repeatedly selected groups retain multiplicity in the weighted fit. Invalid
    rank resamples remain recorded, rather than being redrawn until successful.
    Covariance from surviving replicates is explicitly conditional if any fail.
    """
    if (not isinstance(n_resamples, (int, np.integer)) or isinstance(n_resamples, (bool, np.bool_))
            or n_resamples < 2):
        raise ValueError('n_resamples must be an explicit integer >=2')
    x, y, group, weight, valid, coverage = _prepare(mean_tb, w_mm, source_groups,
        eligible, row_weights, gate_masks)
    result = dict(success=False, status='insufficient_source_groups', covariance=np.full((2, 2), np.nan),
                  coefficient_samples=np.empty((0, 2)), resample_coefficients=np.full((n_resamples, 2), np.nan),
                  resample_valid=np.zeros(n_resamples, bool), n_resamples=int(n_resamples), seed=seed,
                  accepted_resamples=0, rejected_resamples=int(n_resamples), coverage=coverage,
                  calibrated_for_charge_mean=False, eligible_for_production=False,
                  unit_of_resampling='whole original source group, preserving sibling/layer common errors')
    if valid.sum() < 2 or coverage['accepted_groups'] < 2:
        return result
    x, y, group, weight = x[valid], y[valid], group[valid], weight[valid]
    balanced, labels, inverse = _group_weights(group, weight)
    rng = np.random.default_rng(seed)
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(len(labels), size=len(labels)), minlength=len(labels))
        active = counts[inverse] > 0
        solution, _ = _affine(x[active], y[active], balanced[active]*counts[inverse[active]])
        if solution is not None:
            result['resample_coefficients'][i] = solution['coefficients']
            result['resample_valid'][i] = True
    accepted = int(result['resample_valid'].sum())
    samples = result['resample_coefficients'][result['resample_valid']].copy()
    result.update(coefficient_samples=samples, accepted_resamples=accepted,
                  rejected_resamples=n_resamples-accepted, source_group_count=len(labels),
                  covariance_conditioning='valid-rank bootstrap replicates only' if accepted < n_resamples else 'all requested bootstrap replicates')
    if accepted >= 2:
        result.update(success=True, status='ok' if accepted == n_resamples else 'partial_rank_valid_bootstrap',
                      covariance=np.cov(samples, rowvar=False, ddof=1))
    else:
        result['status'] = 'insufficient_valid_bootstrap_replicates'
    return result


def _metrics(residual, weights):
    bias = float(np.average(residual, weights=weights))
    rmse = float(np.sqrt(np.average(residual**2, weights=weights)))
    return dict(bias_mm=bias, rmse_mm=rmse,
                std_mm=float(np.sqrt(np.maximum(0., np.average((residual-bias)**2, weights=weights)))))


def _summary(residual, group, row_weight, n_bootstrap, rng, confidence):
    result = dict(rows=len(residual), source_groups=int(len(np.unique(group))),
                  weighting='equal total weight per source group within this cohort/bin',
                  std_definition='weighted population RMS about weighted mean, not a centroid precision',
                  group_ci_status='insufficient_source_groups', group_ci={})
    if len(residual) == 0:
        result.update(bias_mm=float('nan'), rmse_mm=float('nan'), std_mm=float('nan'))
        return result
    balanced, labels, inverse = _group_weights(group, row_weight)
    result.update(_metrics(residual, balanced))
    if n_bootstrap == 0:
        result['group_ci_status'] = 'not_requested'
        return result
    if len(labels) < 2:
        return result
    values = []
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(len(labels), size=len(labels)), minlength=len(labels))
        values.append(_metrics(residual, balanced*counts[inverse]))
    tail = (1.-confidence)/2.
    for name in ('bias_mm', 'rmse_mm', 'std_mm'):
        result['group_ci'][name] = np.quantile([value[name] for value in values], [tail, 1.-tail]).tolist()
    result.update(group_ci_status='ok', group_ci_confidence=float(confidence),
                  group_ci_resamples=int(n_bootstrap),
                  group_ci_method='percentile bootstrap of whole evaluation source groups; frozen coefficients')
    return result


def evaluate_frozen_map(model, mean_tb, w_mm, source_groups, *, eligible=None,
                        row_weights=None, gate_masks=None, bins=None,
                        n_bootstrap=0, seed=0, confidence=.95):
    """Evaluate frozen coefficients, NEVER refitting on the supplied target rows.

    bins maps a diagnostic name to {'values': length-N values, 'edges': finite
    strictly increasing edges}. Under/overflow/nonfinite counts are retained.
    Held-out group bootstrap CIs condition on the frozen map; training coefficient
    covariance is a different uncertainty and is not silently combined with it.
    """
    if model.get('success') is not True:
        raise ValueError('A successful frozen empirical map is required')
    coefficients = np.asarray(model.get('coefficients'), float)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise ValueError('Frozen coefficients must be finite [a,b]')
    if (not isinstance(n_bootstrap, (int, np.integer)) or isinstance(n_bootstrap, (bool, np.bool_))
            or n_bootstrap not in (0,) and n_bootstrap < 2):
        raise ValueError('n_bootstrap must be 0 or an explicit integer >=2')
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('confidence must be strictly between zero and one')
    x_input, y_input = np.asarray(mean_tb, float), np.asarray(w_mm, float)
    with np.errstate(over='ignore', invalid='ignore'):
        predicted = coefficients[0] + coefficients[1]*x_input
        residual = y_input-predicted
    evaluation_gates = dict(gate_masks or {})
    for name, mask in [('finite_frozen_prediction', np.isfinite(predicted)),
                       ('finite_frozen_residual', np.isfinite(residual))]:
        if name in evaluation_gates:
            raise ValueError('Reserved internal evaluation gate: '+name)
        evaluation_gates[name] = mask
    x, y, group, weight, valid, coverage = _prepare(mean_tb, w_mm, source_groups,
        eligible, row_weights, evaluation_gates)
    rng = np.random.default_rng(seed)
    result = dict(coefficients=coefficients.copy(), predicted_w_mm=predicted,
                  residual_true_minus_predicted_mm=residual, accepted_row=valid,
                  coverage=coverage, summary=_summary(residual[valid], group[valid], weight[valid],
                    n_bootstrap, rng, confidence), bins={},
                  scope=SCOPE, statistic_definition=STATISTIC, coefficients_refitted=False,
                  calibrated_for_charge_mean=False, eligible_for_production=False,
                  group_ci_conditions_on='frozen coefficients; training coefficient uncertainty excluded')
    interval = np.asarray(model.get('training_mean_tb_range', [np.nan, np.nan]), float)
    extrapolated = np.zeros(len(x), bool)
    if interval.shape == (2,) and np.isfinite(interval).all():
        extrapolated = np.isfinite(x) & ((x < interval[0]) | (x > interval[1]))
    training_groups = model.get('fit_source_groups')
    if training_groups is None:
        overlap, independent = None, None
    else:
        # Type-preserving identity tokens avoid falsely equating integer 1 with
        # string '1' across separately persisted studies.
        def token(value):
            if isinstance(value, np.generic):
                value = value.item()
            return (type(value).__name__, value)
        trained = {token(value) for value in np.asarray(training_groups).ravel()}
        evaluated = {token(value) for value in np.unique(group[valid])}
        overlap = len(trained & evaluated)
        independent = overlap == 0
    result.update(outside_training_mean_tb_range=extrapolated,
                  accepted_rows_outside_training_range=int((valid & extrapolated).sum()),
                  training_source_group_overlap_count=overlap,
                  disjoint_from_training_source_groups=independent)
    for name, specification in (bins or {}).items():
        values, edges = np.asarray(specification['values'], float), np.asarray(specification['edges'], float)
        if (values.shape != x.shape or edges.ndim != 1 or len(edges) < 2
                or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0)):
            raise ValueError('Diagnostic bins need length-N values and increasing finite edges')
        records = []
        for i in range(len(edges)-1):
            in_bin = (values >= edges[i]) & ((values <= edges[i+1]) if i == len(edges)-2 else (values < edges[i+1]))
            select = valid & in_bin
            record = _summary(residual[select], group[select], weight[select], n_bootstrap, rng, confidence)
            record.update(low=float(edges[i]), high=float(edges[i+1]),
                          right_edge_inclusive=i == len(edges)-2,
                          total_rows_in_bin=int(in_bin.sum()), rejected_rows_in_bin=int((~valid & in_bin).sum()))
            records.append(record)
        result['bins'][name] = dict(edges=edges.copy(), records=records,
            underflow_rows=int((np.isfinite(values) & (values < edges[0])).sum()),
            overflow_rows=int((np.isfinite(values) & (values > edges[-1])).sum()),
            nonfinite_rows=int((~np.isfinite(values)).sum()),
            accepted_underflow_rows=int((valid & (values < edges[0])).sum()),
            accepted_overflow_rows=int((valid & (values > edges[-1])).sum()))
    return result
