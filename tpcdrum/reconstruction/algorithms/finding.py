"""Observation-only native-pad finding and testable waveform hypotheses.

Geometric baselines are self-contained in geometry_core.py. All returned lines
are feature-coordinate proposals, never calibrated physical drift trajectories.
"""
from time import perf_counter
from itertools import combinations
import numpy as np
from .geometry_core import (infer, refine, _config, _input, _tls, _support,
    _duplicate, _distance, _iterate, _physical_lines)

PAD_U_MM = np.r_[np.arange(-36.5,-9.5), np.arange(-9.,10.,2.), np.arange(10.5,37.5)]
PAD_WIDTH_MM = np.r_[np.full(27,.9), np.full(10,1.9), np.full(27,.9)]

def _event_input(positive_waveform, config):
    q = np.asarray(positive_waveform, dtype=np.float64)
    if q.shape != (12, 64, 512) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('positive_waveform must be finite nonnegative (12,64,512)')
    cfg = dict(config or {})
    k = cfg.get('k', 3)
    if isinstance(k, bool) or int(k) != k or not 1 <= int(k) <= 6:
        raise ValueError('k must be a configured integer between 1 and 6')
    tb = cfg.get('tb_bin', 4)
    if isinstance(tb, bool) or int(tb) != tb or tb < 1 or 512 % int(tb):
        raise ValueError('tb_bin must be an integer divisor of 512')
    cfg.update(k=int(k), tb_bin=int(tb))
    threshold = float(cfg.get('activity_threshold_adc', 40.))
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('activity_threshold_adc must be finite and nonnegative')
    cfg['activity_threshold_adc'] = threshold
    scale = float(cfg.get('feature_time_scale', .36))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('feature_time_scale must be finite positive; it is NOT a drift calibration')
    cfg['feature_time_scale'] = scale
    return q, cfg


def _coarse_points(q, cfg):
    tb = cfg['tb_bin']
    coarse = q.reshape(12, 64, 512 // tb, tb)
    active = coarse.max(-1) > cfg['activity_threshold_adc']
    layer, row, time = np.nonzero(active)
    points = np.column_stack((PAD_U_MM[row], layer * 12.,
                              (time * tb + (tb - 1.) / 2.) * cfg['feature_time_scale']))
    return points, coarse.sum(-1)[active], active


def _unresolved_event(q,cfg,status):
    k,tb=cfg['k'],cfg['tb_bin']
    fractions=np.zeros((k+1,12,64,512//tb),dtype=np.float32)
    fractions[-1]=1.
    return dict(components=np.zeros((k,12,64,512),dtype=np.float32),background=q.astype(np.float32),
                fractions=fractions,feature_lines=np.full((k,6),np.nan),
                diagnostics=dict(status=status,ambiguous=True,selected_k=k,fixed_k=True,
                                 found_count=0,
                                 identifiability='insufficient_observational_evidence',
                                 physics_accuracy_evaluated=False,calibrated_physical_w=False))


def _check_prior(prior, k, nt):
    p = np.asarray(prior, dtype=np.float64)
    if p.shape != (k + 1, 12, 64, nt) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('prior must be finite nonnegative (K+1,12,64,512/tb_bin), background last')
    if not np.allclose(p.sum(0), 1., atol=2e-5):
        raise ValueError('prior fractions must sum to one; arbitrary invalid scores are not renormalized')
    return p


def _tls_baseline(points, charge, k, config):
    cfg = _config(config)
    _, metric, weights, scales = _input(points, charge, cfg)
    lines, remaining = [], weights.copy()
    for _ in range(k):
        line = _tls(metric, remaining)
        if line is None or not _support(metric, line, remaining, cfg):
            break
        if any(_duplicate(line, old, cfg) for old in lines):
            break
        lines.append(line)
        remaining *= 1. - np.exp(-.5 * (_distance(metric, *line) / cfg.assignment_width)**2)
    lines, fractions = _iterate(metric, weights, lines, k, cfg)
    return dict(lines=_physical_lines(lines, scales, k), fractions=fractions.astype(np.float32),
                found_count=sum(x is not None for x in lines), method='iterative_tls')


def _baseline(q, cfg, method, prior=None):
    points, charge, active = _coarse_points(q, cfg)
    k, tb = cfg['k'], cfg['tb_bin']
    fractions = np.zeros((k + 1, 12, 64, 512 // tb), dtype=np.float32)
    fractions[-1] = 1.
    if not len(points):
        return dict(fractions=fractions, lines=np.full((k, 6), np.nan), found_count=0,
                    active=active, method=method)
    geometry = cfg.get('geometry', {'scales_mm': (2., 6., 4.)})
    if method=='cnn':
        p=_check_prior(prior,k,512//tb)
        gcfg=_config(geometry)
        _,metric,weights,scales=_input(points,charge,gcfg)
        fitted=[]
        for slot in range(k):
            w=weights*p[slot,active]
            line=_tls(metric,w)
            fitted.append(line if line is not None and _support(metric,line,w,gcfg) else None)
        return dict(fractions=p.astype(np.float32),lines=_physical_lines(fitted,scales,k),
                    found_count=sum(x is not None for x in fitted),active=active,
                    method='cnn_unchanged_fractions_slot_tls_for_diagnostics_only')
    elif method == 'tls':
        result = _tls_baseline(points, charge, k, geometry)
    else:
        result = infer(points, charge, k, method='ransac' if method == 'ransac' else 'hough',
                       seed=int(cfg.get('seed', 20261027)), config=geometry)
    if prior is not None:
        prior = _check_prior(prior, k, 512 // tb)
        result = refine(points, charge, prior[:, active].T, result['lines'], config=geometry)
        fractions = prior.astype(np.float32).copy()
    fractions[:, active] = result['fractions'].T
    return dict(result, fractions=fractions, active=active)


def _lines_to_parameters(lines, feature_time_scale):
    """u(v), peak-TB(v), evaluated at the middle layer v=66 mm."""
    result = []
    for line in lines:
        if not np.isfinite(line).all() or abs(line[4]) < 1e-7:
            continue
        du, dt = line[3] / line[4], line[5] / line[4] / feature_time_scale
        result.append([line[0] + du * (66. - line[1]), du,
                       line[2] / feature_time_scale + dt * (66. - line[1]), dt])
    return np.asarray(result, dtype=float).reshape(-1, 4)


def _parameters_to_lines(parameters, feature_time_scale):
    a = np.asarray(parameters, dtype=float).reshape(-1, 4)
    result = np.empty((len(a), 6), dtype=float)
    for i, (u, du, t, dt) in enumerate(a):
        direction = np.array([du, 1., dt * feature_time_scale])
        result[i] = [u, 66., t * feature_time_scale, *(direction / np.linalg.norm(direction))]
    return result


def _pulse_template(q, specification=None):
    """Train-pad-only empirical electronics template, or an explicit calibration.

    Empirical alignment cannot establish a calibrated single-particle pulse
    response: unresolved double pulses can broaden the estimate. This provenance
    is returned and such model uncertainty is not hidden in a small fit error.
    """
    if specification is not None:
        offsets = np.asarray(specification['offset_tb'], dtype=float)
        values = np.asarray(specification['values'], dtype=float)
        if offsets.ndim != 1 or offsets.size < 3 or values.shape != offsets.shape:
            raise ValueError('pulse_template needs equal 1-D offset_tb/values arrays of length >=3')
        if not np.isfinite(offsets).all() or not np.isfinite(values).all() or np.any(np.diff(offsets) <= 0):
            raise ValueError('pulse_template must be finite and offsets strictly increasing')
        if np.any(values < 0) or values.max() <= 0:
            raise ValueError('pulse_template values must be nonnegative and nonempty')
        return offsets, values / values.max(), 'explicit_response_template'
    wave = q.reshape(-1, 512)
    peaks = wave.max(-1)
    ids = np.argsort(peaks)[-min(32, np.count_nonzero(peaks > 0)):]
    offsets = np.arange(-32., 97.)
    if not len(ids) or peaks[ids].max(initial=0) <= 0:
        raise ValueError('Cannot estimate a pulse shape without observed signal')
    profiles = []
    for idx in ids:
        if peaks[idx] <= 0:
            continue
        center = int(wave[idx].argmax())
        profiles.append(np.interp(center + offsets, np.arange(512), wave[idx] / peaks[idx], left=0., right=0.))
    values = np.median(profiles, axis=0)
    if values.max() <= 0:
        raise ValueError('Empirical response template has no positive support')
    return offsets, values / values.max(), 'train_pads_empirical_peak_aligned_median_NOT_calibrated'


def _basis(parameters, template, pad_sigma_mm):
    from scipy.special import erf
    if not np.isfinite(pad_sigma_mm) or pad_sigma_mm <= 0:
        raise ValueError('waveform.pad_sigma_mm must be finite positive')
    a = np.asarray(parameters).reshape(-1, 4)
    v = np.arange(12) * 12. - 66.
    u = a[:, 0, None] + a[:, 1, None] * v
    peak_t = a[:, 2, None] + a[:, 3, None] * v
    # Integrate response over each real pad's width; no invented sub-pad data.
    hi = (PAD_U_MM[None, None] + PAD_WIDTH_MM[None, None] / 2. - u[..., None]) / (np.sqrt(2.) * pad_sigma_mm)
    lo = (PAD_U_MM[None, None] - PAD_WIDTH_MM[None, None] / 2. - u[..., None]) / (np.sqrt(2.) * pad_sigma_mm)
    spatial = .5 * (erf(hi) - erf(lo))
    offset = np.arange(512)[None, None, :] - peak_t[..., None]
    temporal = np.interp(offset, template[0], template[1], left=0., right=0.)
    return spatial[..., None] * temporal[:, :, None, :]


def _amplitude_fit(parameters, q, mask, template, options, keep=False):
    """Per-layer NNLS with shared cross-layer track trajectories and response.

    Returned covariance is a conditional linear approximation at fixed geometry,
    NOT calibrated coverage and NOT a posterior for an unresolved decomposition.
    """
    from scipy.optimize import nnls
    k = len(parameters)
    basis = _basis(parameters, template, float(options.get('pad_sigma_mm', 1.2)))
    sigma = np.asarray(options.get('noise_sigma_adc', 2.), dtype=float)
    if sigma.ndim == 0:
        sigma = np.full((12, 64), float(sigma))
    if sigma.shape != (12, 64) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('waveform.noise_sigma_adc must be positive scalar or (12,64)')
    shot = float(options.get('shot_variance_scale', 1.))
    if not np.isfinite(shot) or shot < 0:
        raise ValueError('shot_variance_scale must be finite nonnegative')
    variance = sigma[..., None]**2 + shot * q
    amplitudes, backgrounds, errors, conditions, correlations = [], [], [], [], []
    train_residual, prediction = [], np.zeros_like(q)
    forward = np.zeros_like(basis) if keep else None
    forward_bg = np.zeros_like(q) if keep else None
    for layer in range(12):
        rows = np.asarray(mask[layer], dtype=bool)
        stride = int(options.get('_optimization_tb_stride', 1))
        design = basis[:, layer, rows, ::stride].reshape(k, -1).T
        target = q[layer, rows, ::stride].ravel()
        std = np.sqrt(variance[layer, rows, ::stride].ravel())
        include_bg = bool(options.get('fit_constant_background', True))
        matrix = np.column_stack((design, np.ones(len(design)))) if include_bg else design
        whitened, target_w = matrix / std[:, None], target / std
        # SciPy >=1.8 supports this active-set API. No unconstrained negatives.
        coefficients, _ = nnls(whitened, target_w, maxiter=max(100, 10 * matrix.shape[1]))
        a = coefficients[:k]
        bg = float(coefficients[-1]) if include_bg else 0.
        amplitudes.append(a)
        backgrounds.append(bg)
        residual = whitened @ coefficients - target_w
        train_residual.append(residual)
        prediction[layer] = np.einsum('k,krt->rt', a, basis[:, layer]) + bg
        gram = whitened.T @ whitened
        diagonal = np.sqrt(np.maximum(np.diag(gram), 1e-300))
        corr = gram / np.outer(diagonal, diagonal)
        track_corr = corr[:k,:k]
        correlations.append(float(np.max(track_corr[np.triu_indices(k, 1)])) if k > 1 else 0.)
        condition = float(np.linalg.cond(corr)) if np.any(gram) else float('inf')
        conditions.append(condition)
        if condition > 1e12 or not np.isfinite(condition):
            errors.append(np.full(k, np.inf))
        else:
            scale = max(1., float(residual @ residual) / max(1, len(residual) - len(coefficients)))
            errors.append(np.sqrt(np.maximum(np.diag(np.linalg.pinv(gram))[:k], 0.) * scale))
        if keep:
            forward[:, layer] = a[:, None, None] * basis[:, layer]
            forward_bg[layer] = bg
    residual = np.concatenate(train_residual)
    out = dict(amplitudes=np.asarray(amplitudes), background_amplitudes=np.asarray(backgrounds),
               conditional_amplitude_std=np.asarray(errors), template_gram_condition=np.asarray(conditions),
               template_pair_correlation=np.asarray(correlations), residual=residual,
               chi2=float(residual @ residual), n_samples=len(residual), prediction=prediction,
               variance=variance, forward_components=forward, forward_background=forward_bg)
    return out


def _fit_hypothesis(q, train_mask, parameters, template, options, optimize=True):
    from scipy.optimize import least_squares
    p = np.asarray(parameters, dtype=float).reshape(-1, 4).copy()
    low = np.tile([-60., -2., 0., -3.], len(p))
    high = np.tile([60., 2., 511., 3.], len(p))
    initial = np.clip(p.ravel(), low + 1e-6, high - 1e-6)
    fit_width=bool(options.get('fit_pad_sigma',True))
    width=float(options.get('pad_sigma_mm',1.2))
    if not np.isfinite(width) or width<=0:
        raise ValueError('pad_sigma_mm must be finite positive')
    width_bounds=np.asarray(options.get('pad_sigma_bounds_mm',[.2,4.]),dtype=float)
    if width_bounds.shape!=(2,) or not np.isfinite(width_bounds).all() or not 0<width_bounds[0]<width_bounds[1]:
        raise ValueError('pad_sigma_bounds_mm must contain ordered positive finite bounds')
    if fit_width:
        initial=np.r_[initial,np.clip(width,width_bounds[0]+1e-6,width_bounds[1]-1e-6)]
        low=np.r_[low,width_bounds[0]];high=np.r_[high,width_bounds[1]]
    nfev = 0
    success = True
    if optimize and int(options.get('nonlinear_max_nfev', 8)) > 0:
        optimization_options = dict(options)
        optimization_options['_optimization_tb_stride'] = int(options.get('optimization_tb_stride', 4))
        if optimization_options['_optimization_tb_stride'] < 1:
            raise ValueError('optimization_tb_stride must be a positive integer')
        def residual(x):
            opts=dict(optimization_options)
            if fit_width:opts['pad_sigma_mm']=float(x[-1])
            return _amplitude_fit(x[:4*len(p)].reshape(-1,4),q,train_mask,template,opts)['residual']
        fit = least_squares(residual, initial, bounds=(low, high), method='trf',
                            max_nfev=int(options.get('nonlinear_max_nfev', 8)),
                            diff_step=2e-4,x_scale=np.r_[np.tile([3.,.05,8.,.05],len(p)),.5] if fit_width else np.tile([3.,.05,8.,.05],len(p)),
                            ftol=1e-4, xtol=1e-4, gtol=1e-4)
        p = fit.x[:4*len(p)].reshape(-1,4)
        if fit_width:width=float(fit.x[-1])
        nfev, success = int(fit.nfev), bool(fit.success)
    else:
        p = initial[:4*len(p)].reshape(-1,4)
        if fit_width:width=float(initial[-1])
    fitted_options=dict(options,pad_sigma_mm=width)
    result = _amplitude_fit(p,q,train_mask,template,fitted_options)
    # The holdout comprises entire neighboring pads, not interleaved samples of
    # an already-fitted pulse. Candidate geometry and template also exclude it.
    held = ~train_mask
    held_residual = ((result['prediction'] - q) / np.sqrt(result['variance']))[held].ravel()
    n_parameters = 4 * len(p) + int(fit_width) + 12 * (len(p) + bool(options.get('fit_constant_background', True)))
    train_bic = result['chi2'] + n_parameters * np.log(max(2, result['n_samples']))
    held_chi2 = float(held_residual @ held_residual)
    score = held_chi2 + float(options.get('complexity_penalty', 1.)) * n_parameters * np.log(max(2, held_residual.size))
    return dict(parameters=p, train_chi2=result['chi2'], train_n=result['n_samples'], train_bic=float(train_bic),
                heldout_chi2=held_chi2, heldout_n=held_residual.size, selection_score=float(score),
                parameter_count=n_parameters,optimizer_success=success,nfev=nfev,pad_sigma_mm=width,
                pad_sigma_at_bound=bool(fit_width and min(abs(width-width_bounds[0]),abs(width-width_bounds[1]))<.002))


def _candidate_parameters(q_train, cfg, options, prior=None):
    k_requested = cfg['k']
    compare = bool(options.get('compare_k', False))
    ks = options.get('k_candidates', list(range(1, k_requested + 1))) if compare else [k_requested]
    ks = sorted(set(int(k) for k in ks))
    if not ks or any(k < 1 or k > 6 for k in ks):
        raise ValueError('waveform.k_candidates must contain integer multiplicities in [1,6]')
    kmax = max(ks)
    gc = dict(cfg)
    gc['k'] = kmax
    seeds = []
    all_results = {}
    for method in ('hough', 'ransac', 'tls'):
        result = _baseline(q_train, gc, method)
        all_results[method] = result
        p = _lines_to_parameters(result['lines'], cfg['feature_time_scale'])
        for k in ks:
            if len(p) >= k:
                seeds.append((method, k, p[:k].copy()))
    if prior is not None:
        # This prior is produced only from training pads below. An externally
        # supplied full-event prior must be explicitly marked as heldout-blind.
        if kmax > cfg['k']:
            raise ValueError('CNN-assisted K comparison cannot exceed the checkpoint K')
        points,charge,active=_coarse_points(q_train,cfg)
        if len(points):
            result=refine(points,charge,prior[:,active].T,lines=None,
                          config=cfg.get('geometry',{'scales_mm':(2.,6.,4.)}))
            p=_lines_to_parameters(result['lines'],cfg['feature_time_scale'])
            all_results['cnn_prior_tls']=result
            for k in ks:
                if len(p)>=k:
                    # PIT slots have no preferred order; testing a smaller K
                    # cannot silently equate "first slot" with brightest track.
                    for subset in combinations(range(len(p)),k):
                        seeds.append(('cnn_prior_tls_'+','.join(map(str,subset)),k,p[list(subset)].copy()))
    # Explicit nearby-overlap hypotheses prevent lack of separated maxima from
    # being confused with proof of a single track. They must earn holdout support.
    tls = _lines_to_parameters(all_results['tls']['lines'], cfg['feature_time_scale'])
    if not len(tls):
        pts, charge, _ = _coarse_points(q_train, cfg)
        if len(pts) >= 2:
            line = _tls(pts, np.sqrt(charge))
            if line is not None:
                tls = _lines_to_parameters(np.r_[line[0], line[1]][None], cfg['feature_time_scale'])
    if len(tls):
        for k in ks:
            if k == 1:
                seeds.append(('global_tls_single', k, tls[:1].copy()))
            else:
                for mode in ('pad', 'time', 'diagonal'):
                    p = np.repeat(tls[:1], k, axis=0)
                    spread = np.linspace(-1., 1., k)
                    if mode in ('pad', 'diagonal'):
                        p[:, 0] += spread * float(options.get('split_seed_pad_mm', 1.5))
                        p[:, 1] += spread * float(options.get('split_seed_slope', .015))
                    if mode in ('time', 'diagonal'):
                        p[:, 2] += spread * float(options.get('split_seed_time_tb', 4.))
                    seeds.append(('neighbor_' + mode + '_split', k, p))
    limit = int(options.get('max_hypotheses', 6))
    if limit < len(ks):
        raise ValueError('max_hypotheses must allow at least one candidate for every compared K')
    # Round-robin multiplicities: a cap must never silently discard all K=1 tests.
    selected = []
    for index in range(max((sum(s[1] == k for s in seeds) for k in ks), default=0)):
        for k in ks:
            available = [s for s in seeds if s[1] == k]
            if index < len(available):
                candidate = available[index]
                if not any(candidate[1] == old[1] and np.allclose(candidate[2], old[2], atol=1e-7) for old in selected):
                    selected.append(candidate)
            if len(selected) >= limit:
                return selected, all_results
    return selected, all_results


def _waveform_hypotheses(q, cfg, prior=None):
    options = dict(cfg.get('waveform', {}))
    modulo, fold = int(options.get('holdout_modulo', 4)), int(options.get('holdout_fold', 0))
    if modulo < 2 or not 0 <= fold < modulo:
        raise ValueError('whole-pad holdout needs modulo>=2 and 0<=fold<modulo')
    rows = np.arange(64)[None, :]
    layers = np.arange(12)[:, None]
    train_mask = (rows + 2 * layers) % modulo != fold
    q_train = q * train_mask[..., None]
    if not np.any(q_train>cfg['activity_threshold_adc']):
        return _unresolved_event(q,cfg,'no_train_pad_support_for_holdout')
    if cfg.get('checkpoint') is not None:
        from .waveform_models import predict_checkpoint
        prior=predict_checkpoint(cfg['checkpoint'],q_train,device=cfg.get('device','cpu'),
                                 expected_k=cfg['k'],expected_tb_bin=cfg['tb_bin'],
                                 trusted_checkpoint=bool(cfg.get('trusted_checkpoint',False)))['fractions']
    elif prior is not None and not options.get('prior_is_heldout_blind',False):
        raise ValueError('External prior may contain heldout-pad information. Use an explicit checkpoint, or prior_is_heldout_blind=True only for a prior built from the training-pad mask.')
    template = _pulse_template(q_train, options.get('pulse_template'))
    seeds, geometry = _candidate_parameters(q_train, cfg, options,prior)
    if not seeds:
        return _unresolved_event(q,cfg,'insufficient_trajectory_evidence')
    evidence = []
    for name, k, p in seeds:
        fitted = _fit_hypothesis(q, train_mask, p, template, options)
        fitted.update(seed_method=name, k=k)
        evidence.append(fitted)
    evidence.sort(key=lambda x: x['selection_score'])
    best = evidence[0]
    selected_k = best['k']
    # Refit amplitudes on every observed pad only AFTER testing held-out neighbors.
    final_options=dict(options,pad_sigma_mm=best['pad_sigma_mm'])
    final = _amplitude_fit(best['parameters'],q,np.ones((12,64),dtype=bool),template,final_options,keep=True)
    forward = final['forward_components']
    forward_bg = final['forward_background']
    model = forward.sum(0) + forward_bg
    unexplained = np.maximum(q - model, 0.)
    denom = model + unexplained
    full_fraction = np.divide(forward, denom[None], out=np.zeros_like(forward), where=denom[None] > 0)
    background_fraction = np.maximum(1. - full_fraction.sum(0), 0.)
    components = (full_fraction * q[None]).astype(np.float32)
    background = (background_fraction * q).astype(np.float32)
    tb = cfg['tb_bin']
    assigned = np.concatenate((components, background[None]))
    coarse_q = q.reshape(12, 64, 512 // tb, tb).sum(-1)
    fractions = np.divide(assigned.reshape(selected_k + 1, 12, 64, 512 // tb, tb).sum(-1),
                          coarse_q[None], out=np.zeros((selected_k + 1, 12, 64, 512 // tb)), where=coarse_q[None] > 0)
    fractions[-1, coarse_q == 0] = 1.
    comparable = [x for x in evidence if x['k'] != selected_k]
    delta = float(comparable[0]['selection_score'] - best['selection_score']) if comparable else None
    active_layers = q.sum((1, 2)) > 0
    corr = final['template_pair_correlation'][active_layers]
    condition = final['template_gram_condition'][active_layers]
    local_bad = ((final['template_pair_correlation'] > float(options.get('max_template_correlation', .995)))
                 | (final['template_gram_condition'] > float(options.get('max_gram_condition',1e8)))) & active_layers
    local_ambiguity = bool(selected_k > 1 and np.any(local_bad))
    indistinguishable = bool(selected_k > 1 and np.any(active_layers) and np.all(local_bad[active_layers]))
    weak_model_selection = bool(delta is not None and delta < float(options.get('min_evidence_gap', 10.)))
    actual_mass = components.sum((1, 2, 3), dtype=np.float64)
    tiny_slot = bool(np.any(actual_mass < max(1e-8, q.sum() * float(options.get('min_component_fraction', .005)))))
    relative_l1 = float(np.abs(model-q).sum()/max(q.sum(),1e-12))
    model_misfit = relative_l1 > float(options.get('max_relative_waveform_l1', .25))
    optimization_incomplete = not best['optimizer_success']
    ambiguous = bool(local_ambiguity or weak_model_selection or tiny_slot or model_misfit or optimization_incomplete or best['pad_sigma_at_bound'])
    descriptions = []
    for item in evidence:
        descriptions.append({key: (value.tolist() if isinstance(value, np.ndarray) else value)
                             for key, value in item.items()})
    diagnostics = dict(
        status='ambiguous' if ambiguous else 'hypothesis_supported_NOT_physics_validated',
        ambiguous=ambiguous, identifiability=('not_identifiable' if indistinguishable else 'locally_unresolved_charge' if local_ambiguity else 'conditional_on_response_model'),
        requested_k=cfg['k'], selected_k=selected_k, fixed_k=not bool(options.get('compare_k', False)),
        found_count=int(np.count_nonzero(actual_mass>=max(1e-8,q.sum()*float(options.get('min_component_fraction',.005))))),
        found_count_note='Non-negligible conditional slots; not proof of independently resolved physical particles',
        hypotheses=descriptions, heldout_pad_rule={'modulo': modulo, 'fold': fold, 'formula': '(row+2*layer)%modulo'},
        heldout_evidence_gap=delta, weak_model_selection=weak_model_selection,
        heldout_note='Geometry and empirical pulse template exclude heldout whole pads; selection consumes their predictive residual. Not a dataset test split.',
        response_template_source=template[2], response_is_measured_calibration=False,
        fitted_pad_sigma_mm=best['pad_sigma_mm'],pad_sigma_at_bound=best['pad_sigma_at_bound'],
        response_calibration_note='An explicit array alone is not evidence of measured detector calibration; Gaussian PRF width remains a modelling assumption.',
        response_model='Gaussian pad-integrated PRF times observed/explicit electronics pulse, straight u(v),t(v), NNLS per-layer amplitudes',
        likelihood='Working heteroscedastic squared residual on nonnegative ADC; not exact electronics likelihood, clipped-noise model or calibrated significance',
        selection_score_note='Heldout composite chi2 + parameter_count*log(n_heldout); BIC-like penalty, NOT a Bayes factor or calibrated detection probability',
        conditional_amplitudes=final['amplitudes'].tolist(), conditional_amplitude_std=final['conditional_amplitude_std'].tolist(),
        amplitude_uncertainty_note='Conditional linear Hessian at fitted geometry and response; boundary/geometry/template uncertainty omitted; singular fits return inf',
        layer_template_correlation=final['template_pair_correlation'].tolist(), layer_gram_condition=final['template_gram_condition'].tolist(),
        coincident_track_warning=indistinguishable, locally_unresolved_layers=np.flatnonzero(local_bad).tolist(),negligible_slot=tiny_slot,
        waveform_model_misfit=model_misfit, optimization_incomplete=optimization_incomplete,
        optimization_tb_stride=int(options.get('optimization_tb_stride',4)),
        optimization_note='Nonlinear proposal refinement may subsample TBs; final NNLS amplitudes and heldout/full waveform scores use all 512 TBs. No global optimum guarantee.',
        full_waveform_chi2=float(np.sum((model-q)**2/final['variance'])),
        full_waveform_relative_l1=relative_l1,
        fractions_note='512-TB forward-model responsibilities; coarse output is charge-weighted diagnostic, not a physical purity measurement',
        forward_model_note='forward_components+forward_background predict observations; components+background partition measured ADC and conserve charge by construction',
        prior_note='CNN prior provides candidate geometry only, from training pads. NNLS waveform evidence fits observed ADC without truth.' if prior is not None else 'none',
        physics_accuracy_evaluated=False, calibrated_physical_w=False)
    return dict(components=components, background=background, fractions=fractions.astype(np.float32),
                feature_lines=_parameters_to_lines(best['parameters'], cfg['feature_time_scale']),
                forward_components=forward.astype(np.float32), forward_background=forward_bg.astype(np.float32),
                residual_waveform=(q-model).astype(np.float32), diagnostics=diagnostics)


def find_event(positive_waveform, config, prior=None):
    """Reconstruct observed native-pad waveforms without MC/truth arguments.

    Methods: hough, ransac, tls, cnn, cnn_hough, waveform_hypotheses (default).
    K is configured (default 3); waveform.compare_k explicitly enables competing
    multiplicities and exports every tested hypothesis's predictive evidence.
    A missing/untrained checkpoint never silently becomes a trained CNN.
    Feature-line columns are (u,v,feature_time,du,dv,dfeature_time), NOT physical w.
    """
    started = perf_counter()
    q, cfg = _event_input(positive_waveform, config)
    method = cfg.get('method', 'waveform_hypotheses')
    if method not in {'hough', 'ransac', 'tls','cnn','cnn_hough','waveform_hypotheses'}:
        raise ValueError('Unsupported finding method: ' + str(method))
    if cfg.get('checkpoint') is not None and prior is not None:
        raise ValueError('Supply either checkpoint or prior, not both')
    if cfg.get('checkpoint') is not None and method != 'waveform_hypotheses':
        from .waveform_models import predict_checkpoint
        if prior is not None:
            raise ValueError('Supply either checkpoint or prior, not both')
        prior = predict_checkpoint(cfg['checkpoint'], q, device=cfg.get('device', 'cpu'), expected_k=cfg['k'],
                                   expected_tb_bin=cfg['tb_bin'],trusted_checkpoint=bool(cfg.get('trusted_checkpoint',False)))['fractions']
    if method in {'cnn','cnn_hough'} and prior is None:
        raise ValueError(method+' requires explicit trained checkpoint or validated prior')
    if prior is not None:
        prior = _check_prior(prior, cfg['k'], 512 // cfg['tb_bin'])
    if not np.any(q > cfg['activity_threshold_adc']):
        result = _unresolved_event(q,cfg,'no_active_signal')
    elif method == 'waveform_hypotheses':
        result = _waveform_hypotheses(q, cfg, prior)
    else:
        baseline = _baseline(q, cfg, method, prior)
        assigned = np.repeat(baseline['fractions'], cfg['tb_bin'], axis=-1) * q[None]
        result = dict(components=assigned[:-1].astype(np.float32), background=assigned[-1].astype(np.float32),
                      fractions=baseline['fractions'], feature_lines=baseline['lines'],
                      diagnostics=dict(status='cnn_prediction_NOT_physics_validated' if method=='cnn' else 'geometric_proximity_only', ambiguous=True,
                                       identifiability='not_assessed_by_geometric_baseline',
                                       found_count=baseline['found_count'], selected_k=cfg['k'], fixed_k=True,
                                       fractions_note='CNN soft assignment, not calibrated error probability' if method=='cnn' else 'proximity scores, not calibrated charge purity',
                                       physics_accuracy_evaluated=False, calibrated_physical_w=False))
    forward_available='forward_components' in result
    result.setdefault('forward_components',np.full_like(result['components'],np.nan))
    result.setdefault('forward_background',np.full_like(result['background'],np.nan))
    result.setdefault('residual_waveform',np.full_like(result['background'],np.nan))
    result['diagnostics'].update(method=method, seconds=perf_counter()-started,
        forward_waveform_model_available=forward_available,
        coordinate_note='u PAD_U_MM mm; v layer*12 mm; feature_time coarse-TB center*feature_time_scale. NOT a physical drift coordinate.',
        k_note='Configured or explicit tested hypotheses; fixed-K output is not automatic multiplicity estimation.',
        max_charge_conservation_error_adc=float(np.max(np.abs(result['components'].sum(0)+result['background']-q))))
    return result
