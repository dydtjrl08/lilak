"""Kalman-family geometry fits in physical (u,v,w) millimetres, with RTS.

State order is (u,w,du/dv,dw/dv,T). There is no magnetic field or imposed
three-alpha angle. A mean trajectory is straight between measurements;
distributed angular process noise permits its local tangent to change.

``linear_rts`` is an explicitly labelled 4D geometry baseline. ``ekf_rts`` and
``ukf_rts`` propagate a 5D state through an explicitly supplied stopping table.
No gas density, radiation length, drift calibration or particle energy is
invented here. ``energy_mev`` is a prior AT THE FIRST OBSERVED v, not at a
production vertex or at a Si surface. Missing physical inputs return a status.

RTS is a backward information pass after a forward filter, not an independent
backwards track and not a guarantee of recovering the pre-scattering direction.
This implementation does not reproduce the full Geant4 MSC distribution.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np


class _FitError(ValueError):
    def __init__(self, status, message, missing=None):
        super().__init__(message)
        self.status = status
        self.missing = list(missing or [])


def _require(condition, message, status="invalid_input", missing=None):
    if not condition:
        raise _FitError(status, message, missing)


def _number(mapping, key, minimum=None, strictly=False):
    if key not in mapping or mapping[key] is None:
        raise _FitError("missing_calibration", "Explicit input required: " + key, [key])
    value = float(mapping[key])
    _require(np.isfinite(value), key + " must be finite")
    if minimum is not None:
        _require(value > minimum if strictly else value >= minimum,
                 key + " is outside its allowed range")
    return value


def _psd(matrix, name="covariance", positive=False):
    matrix = np.asarray(matrix, dtype=float)
    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and
             np.all(np.isfinite(matrix)), name + " must be a finite square matrix")
    diagonal = np.diag(matrix)
    _require(np.all(diagonal >= -1e-14), name + " has a negative diagonal variance",
             "numerical_failure")
    scales = np.sqrt(np.maximum(np.abs(diagonal), 1e-30))
    normalized = matrix / scales[:, None] / scales[None, :]
    _require(np.max(np.abs(normalized - normalized.T)) <=
             1e-7 * max(float(np.max(np.abs(normalized))), 1.), name + " is not symmetric")
    matrix = (matrix + matrix.T) * .5
    normalized = (normalized + normalized.T) * .5
    values, vectors = np.linalg.eigh(normalized)
    _require(values[0] >= -1e-8, name + " is not positive semidefinite",
             "numerical_failure")
    if positive:
        _require(values[0] > 0., name + " must be positive definite")
    # Only remove round-off negative eigenvalues, never hide an indefinite fit.
    return ((vectors * np.maximum(values, 0.)) @ vectors.T) * scales[:, None] * scales[None, :]


def _solve(a, b):
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError as error:
        # RTS also has a well-defined conditional Gaussian limit for exact
        # components (zero variance). Use a dimensionless Moore-Penrose inverse
        # there; scaling avoids discarding a small energy variance merely because
        # a position variance is large. Non-symmetric singular systems still fail.
        if not np.allclose(a, a.T, rtol=1e-9, atol=1e-12):
            raise _FitError("numerical_failure", "Singular matrix") from error
        diagonal = np.sqrt(np.maximum(np.diag(a), 0.))
        diagonal[diagonal == 0.] = 1.
        normalized = a / diagonal[:, None] / diagonal[None, :]
        eigen = np.linalg.eigvalsh(normalized)
        _require(eigen[0] >= -1e-9, "Singular matrix is indefinite", "numerical_failure")
        inverse = np.linalg.pinv(normalized, rcond=1e-12) / diagonal[:, None] / diagonal[None, :]
        return inverse @ b


class _Table:
    def __init__(self, mapping, value_key, unit_key, unit_value):
        _require(isinstance(mapping, Mapping), "A tabulated material input is required",
                 "missing_calibration", [value_key + "_table"])
        self.energy = np.asarray(mapping.get("energy_mev", []), dtype=float)
        self.values = np.asarray(mapping.get(value_key, []), dtype=float)
        units = mapping.get("units", {})
        _require(units.get("energy") == "MeV" and units.get(unit_key) == unit_value,
                 "Table units must explicitly declare MeV and " + unit_value,
                 "missing_calibration", ["table.units"])
        _require(self.energy.ndim == 1 and len(self.energy) >= 2 and
                 self.values.shape == self.energy.shape and
                 np.all(np.isfinite(self.energy)) and np.all(np.isfinite(self.values)) and
                 np.all(np.diff(self.energy) > 0.) and self.energy[0] >= 0. and
                 np.all(self.values >= 0.), "Invalid tabulated material data")
        self.mapping = copy.deepcopy(dict(mapping))

    def at(self, energy):
        _require(np.isfinite(energy) and self.energy[0] <= energy <= self.energy[-1],
                 "Energy %.9g MeV outside material table [%.9g, %.9g]; no extrapolation" %
                 (energy, self.energy[0], self.energy[-1]), "transport_domain_error")
        return float(np.interp(energy, self.energy, self.values))


class _Transport:
    def __init__(self, config, energy_mev, stopping_table):
        self.config = config
        method = config.get("method", "linear_rts")
        aliases = {"daf_ekf": "ekf_rts", "daf_ukf": "ukf_rts",
                   "robust_linear_rts": "linear_rts"}
        self.method = aliases.get(method, method)
        _require(self.method in ("linear_rts", "ekf_rts", "ukf_rts"), "Unknown fitting method")
        self.energy_aware = self.method != "linear_rts"
        self.dimension = 5 if self.energy_aware else 4
        _require("process_noise_mode" in config,
                 "Choose process_noise_mode explicitly (none/tuned/highland/table)",
                 "missing_calibration", ["process_noise_mode"])
        self.noise_mode = config["process_noise_mode"]
        _require(self.noise_mode in ("none", "tuned", "highland", "table"),
                 "Unknown process_noise_mode")
        self.stopping = None
        self.scattering = None
        self.straggling = 0.
        self.energy = None
        self.energy_sigma = None
        self.max_step = float(config.get("max_transport_step_mm", 1.))
        _require(np.isfinite(self.max_step) and self.max_step > 0.,
                 "max_transport_step_mm must be positive")
        if self.energy_aware:
            missing = []
            if energy_mev is None:
                missing.append("energy_mev_at_first_observed_point")
            if config.get("energy_reference") != "first_point":
                missing.append("energy_reference=first_point")
            if stopping_table is None:
                missing.append("stopping_table")
            if config.get("initial_covariance") is None:
                missing.append("initial_covariance_5x5")
            if missing:
                raise _FitError("missing_calibration", "Energy-aware transport needs " +
                                ", ".join(missing), missing)
            _require(np.ndim(energy_mev) == 0 and np.isfinite(energy_mev) and energy_mev > 0.,
                     "energy_mev must be a positive scalar prior at first observed v")
            self.energy = float(energy_mev)
            self.energy_sigma = _number(config, "energy_sigma_mev", 0.)
            self.straggling = _number(config, "straggling_variance_mev2_per_mm", 0.)
            self.stopping = _Table(stopping_table, "dedx_mev_per_mm", "dedx", "MeV/mm")
            self.stopping.at(self.energy)
        else:
            _require(self.noise_mode in ("none", "tuned"),
                     "The 4D baseline has no energy state: choose none or tuned scattering")
        if self.noise_mode == "tuned":
            self.tuned_q = _number(config, "angular_variance_rad2_per_mm", 0.)
        if self.noise_mode == "table":
            self.scattering = _Table(config.get("scattering_table"),
                                     "projected_angular_variance_rad2_per_mm",
                                     "angular_variance", "rad2/mm")
        if self.noise_mode == "highland":
            material = config.get("material", {})
            self.radiation_length = _number(material, "radiation_length_mm", 0., True)
            self.mass = _number(material, "mass_mev", 0., True)
            self.charge = _number(material, "charge_e", 0., True)
        self.warnings = set()

    def loss(self, kinetic, length):
        """RK4 continuous stopping; never clamp energy/table bounds."""
        if not self.energy_aware:
            return kinetic
        self.stopping.at(kinetic)
        count = max(1, int(math.ceil(abs(length) / self.max_step)))
        h = length / count
        result = float(kinetic)
        for _ in range(count):
            a = -self.stopping.at(result)
            b = -self.stopping.at(result + .5 * h * a)
            c = -self.stopping.at(result + .5 * h * b)
            d = -self.stopping.at(result + h * c)
            result += h * (a + 2. * b + 2. * c + d) / 6.
            _require(result > 0., "Particle stopped before next observed plane",
                     "stopped_before_last_measurement")
            self.stopping.at(result)
        return result

    def mean(self, state, delta_v):
        result = state.copy()
        result[:2] += state[2:4] * delta_v
        if self.energy_aware:
            length = delta_v * math.sqrt(1. + state[2] ** 2 + state[3] ** 2)
            result[4] = self.loss(state[4], length)
        return result

    def jacobian(self, state, delta_v):
        size = self.dimension
        f = np.eye(size)
        f[0, 2] = f[1, 3] = delta_v
        if self.energy_aware:
            base = self.mean(state, delta_v)[4]
            for j in (2, 3, 4):
                h = max(abs(state[j]) * 1e-5, 1e-6)
                plus, minus = state.copy(), state.copy()
                plus[j] += h
                minus[j] -= h
                yp = ym = None
                try:
                    yp = self.mean(plus, delta_v)[4]
                except _FitError as error:
                    if error.status != "transport_domain_error":
                        raise
                try:
                    ym = self.mean(minus, delta_v)[4]
                except _FitError as error:
                    if error.status != "transport_domain_error":
                        raise
                _require(yp is not None or ym is not None,
                         "Cannot linearize transport at table boundary", "transport_domain_error")
                f[4, j] = ((yp - ym) / (2. * h) if yp is not None and ym is not None else
                           (yp - base) / h if yp is not None else (base - ym) / h)
        return f

    def noise(self, state, delta_v):
        length = delta_v * math.sqrt(1. + state[2] ** 2 + state[3] ** 2)
        q = np.zeros((self.dimension, self.dimension))
        theta2 = 0.
        mid_energy = self.loss(state[4], length * .5) if self.energy_aware else None
        if self.noise_mode == "tuned":
            theta2 = self.tuned_q * length
        elif self.noise_mode == "table":
            theta2 = self.scattering.at(mid_energy) * length
        elif self.noise_mode == "highland":
            fraction = length / self.radiation_length
            if not 1e-3 <= fraction <= 100.:
                _require(self.config.get("allow_highland_outside_valid_range") is True,
                         "Highland segment thickness outside documented 1e-3 <= s/X0 <= 100; "
                         "supply a scattering table or explicitly allow this approximation",
                         "outside_scattering_model_domain")
                self.warnings.add("highland_outside_documented_thickness_range")
            momentum = math.sqrt(mid_energy * (mid_energy + 2. * self.mass))
            beta = momentum / (mid_energy + self.mass)
            correction = 1. + .038 * math.log(fraction)
            _require(correction > 0., "Highland logarithmic correction is nonpositive",
                     "outside_scattering_model_domain")
            theta2 = (13.6 * self.charge / (beta * momentum)) ** 2 * fraction * correction ** 2
            self.warnings.add("highland_gaussian_core_approximation_not_geant4_ion_msc")
        tu, tw = state[2:4]
        # Jacobian from isotropic kicks in the plane perpendicular to the tangent
        # to the two slopes. Correlation is essential away from small slopes.
        slope_cov = theta2 * (1. + tu * tu + tw * tw) * np.array(
            [[1. + tu * tu, tu * tw], [tu * tw, 1. + tw * tw]])
        q[:2, :2] = delta_v ** 2 / 3. * slope_cov
        q[:2, 2:4] = delta_v / 2. * slope_cov
        q[2:4, :2] = delta_v / 2. * slope_cov
        q[2:4, 2:4] = slope_cov
        if self.energy_aware:
            q[4, 4] = self.straggling * length
        return q

    def predict(self, state, covariance, delta_v):
        q = self.noise(state, delta_v)
        if self.method != "ukf_rts":
            f = self.jacobian(state, delta_v)
            return self.mean(state, delta_v), _psd(f @ covariance @ f.T + q), covariance @ f.T, q
        size = self.dimension
        settings = self.config.get("unscented", {})
        alpha, beta, kappa = (float(settings.get("alpha", 1.)),
                              float(settings.get("beta", 2.)),
                              float(settings.get("kappa", 0.)))
        scale = alpha * alpha * (size + kappa)
        _require(np.isfinite(scale) and scale > 0. and alpha > 0. and
                 np.isfinite(beta) and beta >= 0.,
                 "Invalid unscented parameters")
        values, vectors = np.linalg.eigh(_psd(covariance))
        root = (vectors * np.sqrt(np.maximum(values, 0.) * scale)) @ vectors.T
        sigma = np.concatenate((state[None], state + root.T, state - root.T), axis=0)
        wm = np.full(2 * size + 1, 1. / (2. * scale))
        wm[0] = 1. - size / scale
        wc = wm.copy()
        wc[0] += 1. - alpha * alpha + beta
        propagated = np.array([self.mean(s, delta_v) for s in sigma])
        predicted = wm @ propagated
        d0, d1 = sigma - state, propagated - predicted
        predicted_cov = (d1.T * wc) @ d1 + q
        cross = (d0.T * wc) @ d1
        return predicted, _psd(predicted_cov), cross, q


def _seed(points, size, config, transport):
    v = points[:, 1]
    slopes = []
    for coordinate in (0, 2):
        slopes.append(float(np.median([(points[j, coordinate] - points[i, coordinate]) /
                                      (v[j] - v[i]) for i in range(len(v))
                                      for j in range(i + 1, len(v))])))
    seed = np.zeros(size)
    seed[2:4] = slopes
    seed[:2] = np.median(points[:, (0, 2)] - (v - v[0])[:, None] * np.array(slopes), axis=0)
    if transport.energy_aware:
        seed[4] = transport.energy
    if config.get("initial_state") is not None:
        supplied = np.asarray(config["initial_state"], dtype=float)
        _require(supplied.shape == (size,) and np.all(np.isfinite(supplied)), "Invalid initial_state")
        seed = supplied.copy()
        if transport.energy_aware:
            _require(seed[4] == transport.energy, "initial_state energy disagrees with energy_mev prior")
    initial_cov = config.get("initial_covariance")
    if initial_cov is not None:
        initial_cov = _psd(initial_cov, "initial_covariance")
        _require(initial_cov.shape == (size, size), "Wrong initial_covariance shape")
        if transport.energy_aware:
            _require(np.isclose(initial_cov[4, 4], transport.energy_sigma ** 2),
                     "initial_covariance energy variance disagrees with energy_sigma_mev")
    else:
        # Numerical diffuse seed, not a claim about detector resolution. The seed
        # location is data derived; its deliberately broad prior is documented.
        initial_cov = np.diag([1e6, 1e6, 100., 100.] +
                              ([transport.energy_sigma ** 2] if size == 5 else []))
    return seed, initial_cov


def _measurement_covariance(raw, state):
    if raw.shape == (2, 2):
        return raw
    # First-order errors-in-v correction at the current tangent, including uv,
    # vw and uw correlations. Treating v as exact would overstate precision.
    j = np.array([[1., -state[2], 0.], [0., -state[3], 1.]])
    return _psd(j @ raw @ j.T, "effective measurement covariance", positive=True)


def _pass(points, covariance, config, transport, weights):
    count, size = len(points), transport.dimension
    h = np.zeros((2, size)); h[0, 0] = h[1, 1] = 1.
    xp = np.zeros((count, size)); pp = np.zeros((count, size, size))
    xf, pf = xp.copy(), pp.copy()
    residual = np.zeros((count, 2)); innovation_cov = np.zeros((count, 2, 2))
    pulls = residual.copy(); effective_r = innovation_cov.copy()
    chi2 = np.zeros(count); cross = np.zeros((count - 1, size, size))
    process = np.zeros((count, size, size))
    seed, seed_cov = _seed(points, size, config, transport)
    identity = np.eye(size)
    for i in range(count):
        if i == 0:
            xp[i], pp[i] = seed, seed_cov
        else:
            xp[i], pp[i], cross[i - 1], process[i] = transport.predict(
                xf[i - 1], pf[i - 1], points[i, 1] - points[i - 1, 1])
        r = _measurement_covariance(covariance[i], xp[i])
        effective_r[i] = r / weights[i]
        residual[i] = points[i, (0, 2)] - h @ xp[i]
        innovation_cov[i] = _psd(h @ pp[i] @ h.T + effective_r[i], positive=True)
        pulls[i] = _solve(np.linalg.cholesky(innovation_cov[i]), residual[i])
        chi2[i] = float(pulls[i] @ pulls[i])
        gain = _solve(innovation_cov[i], h @ pp[i]).T
        xf[i] = xp[i] + gain @ residual[i]
        a = identity - gain @ h
        pf[i] = _psd(a @ pp[i] @ a.T + gain @ effective_r[i] @ gain.T)
        if transport.energy_aware:
            _require(xf[i, 4] > 0., "Filtered energy became nonpositive", "invalid_filtered_energy")
    xs, ps = xf.copy(), pf.copy()
    smoothing_gain = np.zeros_like(cross)
    for i in range(count - 2, -1, -1):
        smoothing_gain[i] = _solve(pp[i + 1], cross[i].T).T
        xs[i] = xf[i] + smoothing_gain[i] @ (xs[i + 1] - xp[i + 1])
        ps[i] = _psd(pf[i] + smoothing_gain[i] @ (ps[i + 1] - pp[i + 1]) @ smoothing_gain[i].T)
    return dict(predicted=xp, predicted_cov=pp, filtered=xf, filtered_cov=pf,
                smoothed=xs, smoothed_cov=ps, innovations=residual,
                innovation_cov=innovation_cov, innovation_pulls=pulls,
                innovation_chi2=chi2, effective_r=effective_r, process_cov=process,
                smoothing_gain=smoothing_gain)


def _external_chi2(points, covariance, result):
    """Remove each hit's information from its smoothed 2D marginal (cavity)."""
    count = len(points)
    chi2, pulls, residuals = np.zeros(count), np.zeros((count, 2)), np.zeros((count, 2))
    for i in range(count):
        b = result["smoothed_cov"][i, :2, :2]
        effective_r = result["effective_r"][i]
        information = _solve(b, np.eye(2)) - _solve(effective_r, np.eye(2))
        information = _psd(information, "leave-one-out information", positive=True)
        cavity_cov = _solve(information, np.eye(2))
        cavity_mean = cavity_cov @ (_solve(b, result["smoothed"][i, :2]) -
                                    _solve(effective_r, points[i, (0, 2)]))
        raw_r = _measurement_covariance(covariance[i], result["smoothed"][i])
        residuals[i] = points[i, (0, 2)] - cavity_mean
        s = _psd(raw_r + cavity_cov, positive=True)
        pulls[i] = _solve(np.linalg.cholesky(s), residuals[i])
        chi2[i] = pulls[i] @ pulls[i]
    return chi2, pulls, residuals


def _padded_states(array, size):
    if size == 5:
        return array
    out = np.full((array.shape[0], 5), np.nan)
    out[:, :4] = array
    return out


def _padded_covariance(array, size):
    if size == 5:
        return array
    out = np.full((array.shape[0], 5, 5), np.nan)
    out[:, :4, :4] = array
    return out


def fit_track(points_uvw_mm, covariance_mm2, config, energy_mev=None, stopping_table=None):
    """Fit one observation-only track; returns a serializable diagnostics dict.

    Covariances are N x 3 x 3 in (u,v,w), or N x 2 x 2 in (u,w) with v exact.
    A single 2x2/3x3 covariance is broadcast. Points are sorted by increasing v;
    repeated v is rejected rather than silently merging independent hits.
    Missing physics inputs and out-of-table transport are inspectable failures.

    Robust options: config['robust']={'kind':'daf'|'huber'|'none',...}. DAF uses
    an explicit Gaussian-versus-cutoff weight on leave-one-out residuals, followed
    by refitting with R_i / w_i at decreasing temperatures. This is DAF-inspired
    fixed-assignment rejection, not GENFIT's competing-hit implementation.
    """
    try:
        _require(isinstance(config, Mapping), "config must be a mapping")
        config = copy.deepcopy(dict(config))
        points = np.asarray(points_uvw_mm, dtype=float)
        _require(points.ndim == 2 and points.shape[1] == 3 and len(points) >= 3 and
                 np.all(np.isfinite(points)), "At least three finite physical (u,v,w) points required")
        _require(config.get("coordinate_system", "physical_uvw_mm") == "physical_uvw_mm",
                 "Time/scaled feature coordinates cannot be fitted as physical 3D",
                 "missing_calibration", ["calibrated_physical_w_mm"])
        order = np.argsort(points[:, 1], kind="stable")
        points = points[order]
        _require(np.all(np.diff(points[:, 1]) > 0.), "v coordinates must be distinct")
        _require(covariance_mm2 is not None, "Space-point measurement covariance is required",
                 "missing_calibration", ["space_point_covariance_mm2"])
        covariance = np.asarray(covariance_mm2, dtype=float)
        if covariance.shape in ((2, 2), (3, 3)):
            covariance = np.broadcast_to(covariance, (len(points),) + covariance.shape).copy()
        _require(covariance.shape in ((len(points), 2, 2), (len(points), 3, 3)),
                 "covariance_mm2 must be N x 2 x 2 or N x 3 x 3")
        covariance = covariance[order]
        covariance = np.array([_psd(c, "measurement covariance", positive=c.shape == (2, 2))
                               for c in covariance])
        transport = _Transport(config, energy_mev, stopping_table)
        robust = dict(config.get("robust", {}))
        kind = robust.get("kind", "daf" if config.get("method") in
                          ("daf_ekf", "daf_ukf", "robust_linear_rts") else "none")
        _require(kind in ("none", "daf", "huber"), "Unknown robust treatment")
        temperatures = list(robust.get("annealing_temperatures", [16., 8., 4., 2., 1.])) if kind == "daf" else [1.]
        _require(temperatures and np.all(np.isfinite(temperatures)) and min(temperatures) > 0. and
                 all(a >= b for a, b in zip(temperatures, temperatures[1:])),
                 "Annealing temperatures must be positive and nonincreasing")
        if kind == "daf":
            _require(temperatures[-1] == 1., "DAF schedule must end at temperature 1")
        cutoff = float(robust.get("chi2_cutoff", 9.210340371976184))
        minimum = float(robust.get("minimum_weight", 1e-4))
        _require(cutoff > 0. and np.isfinite(cutoff) and 0. < minimum < 1., "Invalid robust cutoff/weight")
        iterations = int(robust.get("iterations_per_temperature", 3 if kind == "daf" else 12))
        _require(iterations >= 1, "Positive robust iteration count required")
        weights = np.ones(len(points)); history = []
        for temperature in temperatures:
            for iteration in range(1 if kind == "none" else iterations):
                result = _pass(points, covariance, config, transport, weights)
                if kind == "none":
                    break
                distances, _, _ = _external_chi2(points, covariance, result)
                if kind == "daf":
                    argument = np.clip((distances - cutoff) / (2. * temperature), -700., 700.)
                    updated = 1. / (1. + np.exp(argument))
                else:
                    updated = np.minimum(1., np.sqrt(cutoff / np.maximum(distances, 1e-300)))
                updated = np.maximum(updated, minimum)
                delta = float(np.max(np.abs(updated - weights)))
                weights = updated
                history.append(dict(temperature=float(temperature), iteration=iteration,
                                    max_weight_change=delta, weights=weights.copy()))
                if delta < float(robust.get("weight_tolerance", 1e-4)):
                    break
        result = _pass(points, covariance, config, transport, weights)
        _require(float(np.sum(weights)) >= 2., "Too little retained information after robust fit",
                 "insufficient_inliers")
        xs, ps = result["smoothed"], result["smoothed_cov"]
        if transport.energy_aware:
            _require(np.all(xs[:, 4] > 0.), "Smoothed kinetic energy is nonpositive",
                     "invalid_smoothed_energy")
        tu, tw = xs[0, 2:4]
        tangent = np.array([tu, 1., tw]); norm = float(np.linalg.norm(tangent))
        direction = tangent / norm
        jac = (np.eye(3) - np.outer(direction, direction))[:, (0, 2)] / norm
        direction_cov = _psd(jac @ ps[0, 2:4, 2:4] @ jac.T)
        fitted = np.column_stack((xs[:, 0], points[:, 1], xs[:, 1]))
        post_residual = points[:, (0, 2)] - xs[:, :2]
        data_chi2 = sum(float(r @ _solve(_measurement_covariance(c, s), r))
                        for r, c, s in zip(post_residual, covariance, xs))
        warnings = sorted(transport.warnings | {
            "first_observed_plane_tangent_is_not_necessarily_production_direction",
            "synthetic_software_validation_is_not_detector_resolution_evidence"})
        if covariance.shape[-1] == 3:
            warnings.append("uncertainty_in_v_uses_first_order_tangent_projection")
        if config.get("initial_covariance") is None:
            warnings.append("data_derived_seed_with_numerically_diffuse_geometry_prior")
        elif config.get("initial_state") is None:
            warnings.append("finite_prior_covariance_with_data_derived_initial_mean_requires_independent_validation")
        if kind != "none":
            warnings.append("robust_weights_are_model_membership_weights_not_calibrated_probabilities")
        external = None
        weight_residual = 0.
        if kind != "none":
            external = _external_chi2(points, covariance, result)
            if kind == "daf":
                implied = 1. / (1. + np.exp(np.clip((external[0] - cutoff) / 2., -700., 700.)))
            else:
                implied = np.minimum(1., np.sqrt(cutoff / np.maximum(external[0], 1e-300)))
            implied = np.maximum(implied, minimum)
            weight_residual = float(np.max(np.abs(implied - weights)))
            if weight_residual >= float(robust.get("weight_tolerance", 1e-4)):
                warnings.append("robust_weights_not_converged_within_configured_iterations")
        output = dict(status="ok", success=True, method=transport.method,
                      robust_method=kind, state_dimension=transport.dimension,
                      n_points=len(points),
                      energy_treatment="tabulated_mean_loss_with_explicit_straggling" if transport.energy_aware
                      else "not_fitted_4D_geometry_energy_columns_are_nan",
                      process_noise_mode=transport.noise_mode, input_order=order,
                      points_uvw_mm=points, covariance_mm2=covariance,
                      states=_padded_states(xs, transport.dimension),
                      covariances=_padded_covariance(ps, transport.dimension),
                      filtered_states=_padded_states(result["filtered"], transport.dimension),
                      filtered_covariances=_padded_covariance(result["filtered_cov"], transport.dimension),
                      predicted_states=_padded_states(result["predicted"], transport.dimension),
                      predicted_covariances=_padded_covariance(result["predicted_cov"], transport.dimension),
                      fitted_points_uvw_mm=fitted, initial_downstream_direction=direction,
                      initial_direction_covariance=direction_cov,
                      initial_reference_v_mm=float(points[0, 1]),
                      theta_pad_deg=np.degrees(np.arctan(xs[:, 2])),
                      initial_theta_pad_deg=float(np.degrees(np.arctan(tu))),
                      innovation_residuals=result["innovations"], innovation_pulls=result["innovation_pulls"],
                      innovation_covariances=result["innovation_cov"], innovation_chi2=result["innovation_chi2"],
                      smoothed_residuals=post_residual, effective_measurement_covariances=result["effective_r"],
                      process_covariances=_padded_covariance(result["process_cov"], transport.dimension),
                      daf_weights=weights, annealing_history=history,
                      robust_converged=weight_residual < float(robust.get("weight_tolerance", 1e-4)),
                      robust_weight_fixed_point_error=weight_residual,
                      sum_measurement_weights=float(np.sum(weights)),
                      outlier_mask=weights < .5,
                      innovation_chi2_sum=float(np.sum(result["innovation_chi2"])),
                      measurement_residual_chi2=float(data_chi2), nominal_geometric_ndof=2 * len(points) - 4,
                      chi2_p_value=None,
                      chi2_interpretation="Forward innovations and posterior measurement residuals are separately stored; "
                      "process noise, fitted states and robust weights require calibration before goodness-of-fit use.",
                      physical_uncertainty_validated=False, physics_accuracy_evaluated=False,
                      initial_mean_source="provided" if config.get("initial_state") is not None
                      else "observation_derived_robust_seed",
                      initial_prior_independence_verified=False,
                      warnings=warnings, config=config,
                      stopping_table_provenance=copy.deepcopy(stopping_table) if transport.energy_aware else None)
        if kind != "none":
            output["leave_one_out_chi2"], output["leave_one_out_pulls"], output["leave_one_out_residuals"] = \
                external
        return output
    except _FitError as error:
        return dict(status=error.status, success=False, message=str(error), missing_inputs=error.missing,
                    physics_accuracy_evaluated=False, physical_uncertainty_validated=False)
    except (TypeError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        return dict(status="invalid_input", success=False, message=str(error), missing_inputs=[],
                    physics_accuracy_evaluated=False, physical_uncertainty_validated=False)
