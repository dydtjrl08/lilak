"""Observation-only Si association, material loss inversion and 3-alpha kinematics.

Lengths are mm, kinetic/total energies and masses are numerical MeV (c=1).
Four-vectors have columns (p_u,p_v,p_w,E). No MC tags, initial truth momenta,
120-degree constraints, or equal-energy constraints are accepted by these APIs.
CSDA inversion estimates a mean loss; it cannot undo event-wise straggling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import csv
import json
from pathlib import Path

import numpy as np


def _finite(value, shape=None, name="value"):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError("%s must have shape %s" % (name, shape))
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _covariance(value, n, name="covariance"):
    c = _finite(value, (n, n), name)
    if not np.allclose(c, c.T, rtol=1e-8, atol=1e-12):
        raise ValueError(name + " must be symmetric")
    if np.linalg.eigvalsh(c)[0] < -1e-10 * max(1., np.linalg.norm(c)):
        raise ValueError(name + " must be positive semidefinite")
    return (c + c.T) * .5


def _interp_strict(x, xp, fp, name):
    x = _finite(x, name=name)
    tolerance = 1e-12 * max(1., abs(float(xp[-1])))
    if np.any(x < xp[0]-tolerance) or np.any(x > xp[-1]+tolerance):
        raise ValueError("%s is outside exported table [%g,%g]; extrapolation forbidden" %
                         (name, xp[0], xp[-1]))
    return np.interp(np.clip(x, xp[0], xp[-1]), xp, fp)


@dataclass
class StoppingTable:
    """One particle in one actual material, with strict monotone CSDA range."""
    energy_mev: np.ndarray
    range_mm: np.ndarray
    dedx_mev_per_mm: np.ndarray
    material: str
    provenance: dict = field(default_factory=dict)
    units: dict = field(default_factory=lambda: {
        "energy": "MeV", "range": "mm", "length": "mm", "dedx": "MeV/mm"})

    def __post_init__(self):
        self.energy_mev = _finite(self.energy_mev, name="energy_mev")
        self.range_mm = _finite(self.range_mm, name="range_mm")
        self.dedx_mev_per_mm = _finite(self.dedx_mev_per_mm, name="dedx_mev_per_mm")
        if (self.energy_mev.ndim != 1 or len(self.energy_mev) < 2 or
                self.range_mm.shape != self.energy_mev.shape or
                self.dedx_mev_per_mm.shape != self.energy_mev.shape):
            raise ValueError("Stopping arrays must be matching 1D arrays with >=2 points")
        if (np.any(np.diff(self.energy_mev) <= 0) or np.any(np.diff(self.range_mm) <= 0) or
                np.any(self.energy_mev < 0) or np.any(self.range_mm < 0) or
                np.any(self.dedx_mev_per_mm <= 0)):
            raise ValueError("Energy/range must increase strictly; stopping power must be positive")
        if not self.material:
            raise ValueError("An actual material identity is required")
        expected = {"energy": "MeV", "range": "mm", "dedx": "MeV/mm"}
        if any(self.units.get(k) != v for k, v in expected.items()):
            raise ValueError("Unsupported stopping-table units")

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, cls):
            return value
        return cls(np.asarray(value["energy_mev"]), np.asarray(value["range_mm"]),
                   np.asarray(value["dedx_mev_per_mm"]), str(value["material"]),
                   dict(value.get("provenance", {})), dict(value["units"]))

    def range_at(self, energy):
        return _interp_strict(energy, self.energy_mev, self.range_mm, "energy [MeV]")

    def energy_at_range(self, distance):
        return _interp_strict(distance, self.range_mm, self.energy_mev, "CSDA range [mm]")

    def dedx_at(self, energy):
        return _interp_strict(energy, self.energy_mev, self.dedx_mev_per_mm, "energy [MeV]")

    def mapping(self):
        return dict(energy_mev=self.energy_mev.tolist(), range_mm=self.range_mm.tolist(),
                    dedx_mev_per_mm=self.dedx_mev_per_mm.tolist(), material=self.material,
                    provenance=dict(self.provenance), units=dict(self.units))

    def consistency_audit(self):
        """Compare independent range and stopping definitions, without correcting them.

        Geant4 ComputeTotalDEDX includes nuclear stopping when applicable, whereas
        GetCSDARange queries the built energy-loss-process CSDA table. Their
        equality must be checked in the actual energy interval used by the fit.
        """
        implied = np.diff(self.energy_mev)/np.diff(self.range_mm)
        measured = .5*(self.dedx_mev_per_mm[1:]+self.dedx_mev_per_mm[:-1])
        relative = (measured-implied)/implied
        return dict(energy_midpoints_mev=.5*(self.energy_mev[1:]+self.energy_mev[:-1]),
                    range_derivative_dedx_mev_per_mm=implied,
                    dedx_relative_difference=relative,
                    max_abs_relative_difference=float(np.max(np.abs(relative))),
                    definitions_automatically_reconciled=False)


def load_stopping_tables(path, particle_pdg=1000020040):
    """Load exported JSON; materials stay keyed by exact initialized material ID."""
    obj = json.loads(Path(path).expanduser().read_text())
    if obj.get("schema") != "tpcdrum-geant4-material-export-v1":
        raise ValueError("Unsupported material export schema")
    tables = {}
    for raw in obj.get("tables", []):
        if int(raw["particle_pdg"]) != int(particle_pdg):
            continue
        raw = dict(raw)
        raw["provenance"] = dict(raw.get("provenance", {}), source=str(Path(path).resolve()),
                                 particle_pdg=int(particle_pdg),
                                 geant4_version=obj.get("geant4_version"),
                                 physics_list=obj.get("physics_list"))
        table = StoppingTable.from_mapping(raw)
        if table.material in tables:
            raise ValueError("Ambiguous duplicate material/particle table")
        tables[table.material] = table
    if not tables:
        raise ValueError("No stopping tables for requested particle")
    return tables, obj


def load_material_paths(path, expected_segment_lengths_mm=None, length_tolerance_mm=1e-6):
    """Read G4Navigator paths for independently reconstructed event/track IDs.

    ``expected_segment_lengths_mm`` maps (path_id, segment_id) to the Euclidean
    length of the ORIGINAL submitted reconstructed segment. Supplying it checks
    completeness against that input. IDs must be unique to an event/track, and
    segment IDs must increase in vertex->Si order. No MC trajectory is needed.
    The caller must additionally verify which geometry/parameter export was used.
    """
    required = ['path_id', 'segment_id', 'part', 'material', 'length_mm', 'volume']
    pieces = {}
    with Path(path).expanduser().open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required:
            raise ValueError('Unexpected material-path CSV header')
        for row in reader:
            key = (int(row['path_id']), int(row['segment_id']))
            part = int(row['part'])
            length = float(row['length_mm'])
            if (part < 0 or key[1] < 0 or not row['material'] or
                    not np.isfinite(length) or length <= 0):
                raise ValueError('Invalid navigated material-path row')
            parts = pieces.setdefault(key, {})
            if part in parts:
                raise ValueError('Duplicate material-path part')
            parts[part] = dict(material=row['material'], length_mm=length,
                               volume=row['volume'], segment_id=key[1], part=part)
    if not pieces:
        raise ValueError('No material paths exported')
    paths = {}
    for key, parts in sorted(pieces.items()):
        if sorted(parts) != list(range(len(parts))):
            raise ValueError('Missing/noncontiguous navigated parts')
        paths.setdefault(key[0], []).extend(parts[i] for i in range(len(parts)))
    if expected_segment_lengths_mm is not None:
        expected = {(int(k[0]), int(k[1])): float(v)
                    for k, v in expected_segment_lengths_mm.items()}
        nonzero = {k: v for k, v in expected.items() if v != 0}
        if any(not np.isfinite(v) or v < 0 for v in expected.values()):
            raise ValueError('Expected path length must be finite and nonnegative')
        if set(nonzero) != set(pieces):
            raise ValueError('Navigated paths do not match all submitted nonzero segments')
        for key, length in nonzero.items():
            if abs(sum(p['length_mm'] for p in pieces[key].values())-length) > length_tolerance_mm:
                raise ValueError('Navigated material lengths do not close the submitted segment')
    return paths


def _path(path_segments, tables, require_provenance):
    result = []
    for i, segment in enumerate(path_segments):
        material = str(segment["material"])
        length = float(segment["length_mm"])
        if not np.isfinite(length) or length < 0:
            raise ValueError("Path length must be finite and nonnegative")
        if material not in tables:
            raise ValueError("Missing stopping table for path material " + material)
        table = StoppingTable.from_mapping(tables[material])
        if table.material != material:
            raise ValueError("Path/table material identity mismatch")
        if require_provenance and (not table.provenance.get("source") or
                table.provenance.get("particle_pdg") != 1000020040):
            raise ValueError("Explicit alpha stopping-table provenance is required")
        result.append((material, length, table, i))
    return result


def inverse_energy(energy_at_si_mev, path_segments, tables, energy_sigma_mev=0.,
                   require_provenance=True, path_validated=False,
                   path_covariance_mm2=None):
    """Invert vertex->Si-entrance path in reverse material order using R(E).

    ``energy_at_si_mev`` is kinetic energy at the Si entrance, NOT unqualified
    deposited energy. Include dead layers explicitly in the ordered path, or
    undo them before this call. A material identity for every segment is required.
    The optional path covariance is in the original vertex->Si segment order.
    The returned sigma propagates supplied boundary-energy and path uncertainty;
    unprovided material/straggling uncertainty is explicitly not invented.
    """
    energy, sigma = float(energy_at_si_mev), float(energy_sigma_mev)
    if not np.isfinite(energy) or energy < 0 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Energy and uncertainty must be finite nonnegative values")
    if not path_validated:
        return dict(status="unvalidated_material_path", vertex_energy_mev=None,
                    message="Supply the material inventory along the reconstructed path")
    segments = _path(path_segments, tables, require_provenance)
    derivative = 1.
    path_derivative = np.zeros(len(segments))
    history = []
    for material, length, table, index in reversed(segments):
        downstream = energy
        r = float(table.range_at(downstream))
        energy = float(table.energy_at_range(r + length))
        # Derivative of the actual piecewise-linear interpolant (not an assumed
        # equality between independently exported dE/dx and the range derivative).
        j0 = int(np.clip(np.searchsorted(table.energy_mev, downstream, side="right")-1,
                         0, len(table.energy_mev)-2))
        j1 = int(np.clip(np.searchsorted(table.range_mm, r+length, side="right")-1,
                         0, len(table.range_mm)-2))
        dR_dE = ((table.range_mm[j0+1]-table.range_mm[j0]) /
                 (table.energy_mev[j0+1]-table.energy_mev[j0]))
        dE_dR = ((table.energy_mev[j1+1]-table.energy_mev[j1]) /
                 (table.range_mm[j1+1]-table.range_mm[j1]))
        local_derivative = dR_dE*dE_dR
        derivative *= local_derivative
        path_derivative *= local_derivative
        path_derivative[index] = dE_dR
        history.append(dict(path_index=index, material=material, length_mm=length,
                            downstream_energy_mev=downstream, upstream_energy_mev=energy,
                            mean_loss_mev=energy-downstream))
    path_variance = 0.
    if path_covariance_mm2 is not None:
        covariance = _covariance(path_covariance_mm2, len(segments), 'path covariance [mm^2]')
        path_variance = float(path_derivative@covariance@path_derivative)
    return dict(status="ok", vertex_energy_mev=energy,
                boundary_only_sigma_mev=abs(derivative)*sigma,
                supplied_uncertainty_sigma_mev=float(np.sqrt(max(0., (derivative*sigma)**2+path_variance))),
                boundary_energy_jacobian=derivative, reverse_history=history,
                path_length_jacobian_mev_per_mm=path_derivative,
                treatment="mean_CSDA_range_inversion",
                uncertainty_scope="supplied independent boundary energy and optional correlated path lengths; no unprovided straggling/material variance",
                eventwise_energy_loss_recovered_exactly=False)


def forward_energy(energy_at_vertex_mev, path_segments, tables, require_provenance=True):
    """Mean CSDA forward transport, used for inverse closure and simulation tests."""
    energy = float(energy_at_vertex_mev)
    if not np.isfinite(energy) or energy < 0:
        raise ValueError("Invalid kinetic energy")
    for material, length, table, _ in _path(path_segments, tables, require_provenance):
        remaining = float(table.range_at(energy)) - length
        if remaining < table.range_mm[0]:
            return dict(status="stopped_or_below_table", energy_mev=None, material=material)
        energy = float(table.energy_at_range(remaining))
    return dict(status="ok", energy_mev=energy)


def silicon_entrance_energy(deposit_mev, thickness_mm, incidence_cosine, si_table,
                            stopping_verified=False, kinetic_upper_bound_mev=None):
    """Interpret deposit as kinetic energy only when stopping is established.

    An adequate range for Edep alone does not rule out a higher-energy punch-
    through solution. A validated kinetic upper bound can establish stopping.
    Dead layers and escaping secondary energy must be accounted for separately.
    """
    deposit, thickness, cosine = map(float, (deposit_mev, thickness_mm, incidence_cosine))
    if (not np.isfinite([deposit, thickness, cosine]).all() or deposit < 0 or
            thickness <= 0 or not 0 < abs(cosine) <= 1):
        raise ValueError("Invalid Si deposit, active thickness or incidence cosine")
    table = StoppingTable.from_mapping(si_table)
    path = thickness/abs(cosine)
    proof = "caller_supplied_stopping_verification" if stopping_verified else None
    if kinetic_upper_bound_mev is not None:
        upper = float(kinetic_upper_bound_mev)
        if upper < deposit:
            raise ValueError("Kinetic upper bound is below measured deposited energy")
        if float(table.range_at(upper)) <= path:
            proof = "range_of_validated_kinetic_upper_bound_fits_active_Si"
    if proof is None:
        return dict(status="ambiguous_stopping_or_punchthrough", energy_at_si_mev=None,
                    active_path_mm=path)
    if float(table.range_at(deposit)) > path:
        return dict(status="stopping_assumption_inconsistent", energy_at_si_mev=None,
                    active_path_mm=path)
    return dict(status="ok", energy_at_si_mev=deposit, active_path_mm=path, proof=proof,
                assumption="full active-energy collection; entrance dead-layer loss handled separately")


def intersect_detector(origin_uvw_mm, direction_uvw, detector):
    """Intersect a downstream ray with an exported planar rectangular Si module."""
    origin = _finite(origin_uvw_mm, (3,), "track origin")
    direction = _finite(direction_uvw, (3,), "track direction").copy()
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError("Zero track direction")
    direction /= norm
    center = _finite(detector["center_uvw_mm"], (3,), "Si center")
    normal = _finite(detector["normal_uvw"], (3,), "Si normal")
    au = _finite(detector["axis_u_uvw"], (3,), "Si local u")
    av = _finite(detector["axis_v_uvw"], (3,), "Si local v")
    basis = np.stack([au, av, normal])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6):
        raise ValueError("Si local axes and normal must be orthonormal")
    widths = _finite([detector["width_mm"], detector["height_mm"]], (2,), "Si dimensions")
    if np.any(widths <= 0):
        raise ValueError("Si dimensions must be positive")
    denominator = float(direction @ normal)
    if abs(denominator) < 1e-12:
        return dict(valid=False, status="parallel_to_Si")
    distance = float((center-origin) @ normal/denominator)
    if distance < 0:
        return dict(valid=False, status="Si_behind_track")
    point = origin + distance*direction
    local = np.array([(point-center) @ au, (point-center) @ av])
    outside = np.maximum(np.abs(local)-widths/2., 0.)
    return dict(valid=bool(np.all(outside <= 1e-9)), status="ok" if np.all(outside <= 1e-9)
                else "outside_active_rectangle", center_plane_point_uvw_mm=point,
                local_mm=local, outside_mm=outside, path_distance_mm=distance,
                incidence_cosine=abs(denominator))


def associate_si(origins_uvw_mm, directions_uvw, detectors, observed_energy_mev,
                 position_sigma_mm=1., gate_sigma=3., energy_threshold_mev=0.,
                 max_hypotheses=256):
    """Enumerate geometry-supported Si assignments without track-specific MC Edep.

    Module totals alone cannot assign individual energies when tracks share a
    module. Such hypotheses are retained with an explicit unresolved status.
    No preference for a module center is imposed on intersections inside it.
    """
    origins = _finite(origins_uvw_mm, name="origins")
    directions = _finite(directions_uvw, origins.shape, "directions")
    if origins.ndim != 2 or origins.shape[1] != 3 or not 1 <= len(origins) <= 3:
        raise ValueError("Expected 1..3 physical track origins/directions")
    energies = _finite(observed_energy_mev, (len(detectors),), "observed per-module energies")
    if np.any(energies < 0) or not np.isfinite(position_sigma_mm) or position_sigma_mm <= 0:
        raise ValueError("Nonnegative energy and positive position uncertainty required")
    if not np.isfinite(gate_sigma) or gate_sigma < 0 or max_hypotheses < 1:
        raise ValueError("Invalid association gate or hypothesis limit")
    hit_modules = np.flatnonzero(energies > float(energy_threshold_mev)).tolist()
    if not hit_modules:
        return dict(status="no_Si_energy_observation", hypotheses=[], unique=False)
    intersections, candidates = [], []
    for origin, direction in zip(origins, directions):
        row = [intersect_detector(origin, direction, d) for d in detectors]
        choices = [j for j in hit_modules if row[j].get("outside_mm") is not None and
                   np.linalg.norm(row[j]["outside_mm"]) <= gate_sigma*position_sigma_mm]
        intersections.append(row)
        candidates.append(choices)
    if any(not row for row in candidates):
        return dict(status="track_without_supported_Si", hypotheses=[], unique=False,
                    intersections=intersections)
    hypotheses = []
    truncated = False
    for assignment in product(*candidates):
        # All positive module totals need an explanation within this selected
        # K-alpha model. Extra background/pileup requires an explicit model.
        if set(assignment) != set(hit_modules):
            continue
        if len(hypotheses) >= max_hypotheses:
            truncated = True
            break
        multiplicity = np.bincount(assignment, minlength=len(detectors))
        shared = any(multiplicity[j] > 1 for j in hit_modules)
        track_energy = np.array([energies[j] if multiplicity[j] == 1 else np.nan
                                 for j in assignment])
        cost = sum(float(np.sum(intersections[i][j]["outside_mm"]**2))/position_sigma_mm**2
                   for i, j in enumerate(assignment))
        hypotheses.append(dict(track_detector_ids=list(assignment), geometry_cost=cost,
                               track_deposit_mev=track_energy,
                               shared_detector=shared,
                               status="unresolved_shared_detector" if shared else "unique_module_energies"))
    hypotheses.sort(key=lambda h: (h["geometry_cost"], h["track_detector_ids"]))
    unique = len(hypotheses) == 1 and not truncated and not hypotheses[0]["shared_detector"]
    return dict(status="ok" if unique else "ambiguous_Si_association" if hypotheses else
                "unexplained_positive_Si_module", hypotheses=hypotheses, unique=unique,
                hypothesis_search_truncated=truncated, intersections=intersections,
                input_scope="observed per-detector totals only; no MC track Edep")


def shared_si_energy_fit(module_totals_mev, module_covariance_mev2, track_detector_ids,
                         track_prior_mev=None, prior_covariance_mev2=None):
    """Constrained Gaussian energy fit using explicitly calibrated TPC priors.

    With no independent prior, a shared module's sum is rank deficient. This
    function then returns unresolved instead of dividing its total equally.
    A prior here must come from observed, calibrated TPC information, not MC.
    """
    y = _finite(module_totals_mev, name="Si module totals")
    assignment = np.asarray(track_detector_ids, dtype=int)
    if y.ndim != 1 or np.any(y < 0) or assignment.ndim != 1 or not 1 <= len(assignment) <= 3:
        raise ValueError("Invalid totals or track assignment")
    if np.any(assignment < 0) or np.any(assignment >= len(y)):
        raise ValueError("Assigned module is outside observed module array")
    k = len(assignment)
    v = _covariance(module_covariance_mev2, len(y), "Si covariance")
    if np.linalg.eigvalsh(v)[0] <= 0:
        raise ValueError("Si covariance must be positive definite")
    a = np.zeros((len(y), k))
    a[assignment, np.arange(k)] = 1.
    rank = int(np.linalg.matrix_rank(a))
    if track_prior_mev is None and rank < k:
        return dict(status="unresolved_shared_detector", track_energy_mev=None,
                    measured_rank=rank, unidentifiable_dimensions=k-rank,
                    equal_split_imposed=False)
    vi = np.linalg.inv(v)
    h, b = a.T @ vi @ a, a.T @ vi @ y
    prior_term = np.zeros((k, k))
    mu = np.zeros(k)
    if track_prior_mev is not None:
        mu = _finite(track_prior_mev, (k,), "observed TPC energy prior")
        c = _covariance(prior_covariance_mev2, k, "TPC prior covariance")
        if np.any(mu < 0) or np.linalg.eigvalsh(c)[0] <= 0:
            raise ValueError("TPC priors need positive-definite calibrated covariance")
        prior_term = np.linalg.inv(c)
        h, b = h+prior_term, b+prior_term @ mu
    best = None
    # For K<=3 exhaustive nonnegative quadratic active sets are inexpensive.
    for bits in product((False, True), repeat=k):
        free = np.flatnonzero(bits)
        x, cov = np.zeros(k), np.zeros((k, k))
        if len(free):
            try:
                x[free] = np.linalg.solve(h[np.ix_(free, free)], b[free])
            except np.linalg.LinAlgError:
                continue
            if np.any(x[free] < -1e-10):
                continue
            cov[np.ix_(free, free)] = np.linalg.inv(h[np.ix_(free, free)])
        residual = y-a @ x
        score = float(residual @ vi @ residual + (x-mu) @ prior_term @ (x-mu))
        if best is None or score < best[0]:
            best = score, x, cov
    if best is None:
        raise ValueError("No feasible Si energy solution")
    score, x, cov = best
    return dict(status="ok", track_energy_mev=np.maximum(x, 0), covariance_mev2=cov,
                chi2=score, module_residual_mev=y-a @ x, measured_rank=rank,
                prior_used=track_prior_mev is not None, equal_split_imposed=False,
                uncertainty_scope="supplied Gaussian module/TPC priors; boundary-truncated Gaussian approximation")


def four_vectors(kinetic_mev, directions_uvw, mass_mev):
    t = _finite(kinetic_mev, name="kinetic energies")
    directions = _finite(directions_uvw, (len(t), 3), "directions")
    masses = np.broadcast_to(_finite(mass_mev, name="daughter masses"), t.shape)
    norms = np.linalg.norm(directions, axis=1)
    if t.ndim != 1 or np.any(t < 0) or np.any(masses <= 0) or np.any(norms <= 0):
        raise ValueError("Nonnegative kinetic energy, positive mass and nonzero direction required")
    momentum = np.sqrt(t*(t+2*masses))
    return np.column_stack([directions/norms[:, None]*momentum[:, None], t+masses])


def boost(vectors, beta):
    """Active ROOT-convention boost; boost_to_com calls this with -P/E."""
    vectors = _finite(vectors, name="four-vectors")
    beta = _finite(beta, (3,), "beta")
    if vectors.ndim != 2 or vectors.shape[1] != 4:
        raise ValueError("Four-vectors must have shape (K,4)")
    b2 = float(beta @ beta)
    if b2 >= 1:
        raise ValueError("Boost speed must be strictly below c")
    if b2 == 0:
        return vectors.copy()
    gamma = 1/np.sqrt(1-b2)
    projection = vectors[:, :3] @ beta
    spatial = vectors[:, :3] + (((gamma-1)/b2)*projection + gamma*vectors[:, 3])[:, None]*beta
    energy = gamma*(vectors[:, 3]+projection)
    return np.column_stack([spatial, energy])


def kinematics(vectors, parent_ground_mass_mev):
    vectors = _finite(vectors, name="four-vectors")
    if vectors.shape != (3, 4):
        raise ValueError("This kinematics task requires exactly three alpha four-vectors")
    ground = float(parent_ground_mass_mev)
    if not np.isfinite(ground) or ground <= 0 or np.any(vectors[:, 3] <= 0):
        raise ValueError("Positive parent ground mass and total energies required")
    daughter_m2 = vectors[:, 3]**2-np.sum(vectors[:, :3]**2, axis=1)
    if np.any(daughter_m2 <= 0):
        raise ValueError("Daughter vectors must be timelike")
    parent = vectors.sum(axis=0)
    m2 = parent[3]**2-float(parent[:3] @ parent[:3])
    if m2 <= 0:
        raise ValueError("Parent four-vector must be timelike")
    mass = float(np.sqrt(m2))
    beta = parent[:3]/parent[3]
    com = boost(vectors, -beta)
    masses = np.sqrt(daughter_m2)
    kinetic = com[:, 3]-masses
    if np.any(kinetic < -1e-8):
        raise ValueError("Negative COM kinetic energy beyond floating roundoff")
    kinetic = np.maximum(kinetic, 0.)
    pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=int)
    norms = np.linalg.norm(com[:, :3], axis=1)
    angles = []
    for i, j in pairs:
        cosine = float(com[i, :3] @ com[j, :3]/(norms[i]*norms[j])) if norms[i]*norms[j] else np.nan
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1., 1.)))))
    total_kinetic = float(kinetic.sum())
    fractions = kinetic/total_kinetic if total_kinetic > 0 else np.full(3, np.nan)
    dalitz = np.array([(fractions[1]-fractions[0])/np.sqrt(3.), fractions[2]-1/3.])
    angles_valid = bool(np.isfinite(angles).all())
    return dict(status="ok" if angles_valid else "undefined_COM_angle",
                angles_valid=angles_valid, parent_four_vector=parent, invariant_mass_mev=mass,
                excitation_mev=mass-ground, q_three_alpha_mev=mass-float(masses.sum()),
                parent_beta=beta, com_four_vectors=com, com_kinetic_mev=kinetic,
                com_energy_fractions=fractions, pair_indices=pairs,
                com_opening_angles_deg=np.asarray(angles), dalitz_xy=dalitz,
                com_momentum_residual_mev=float(np.linalg.norm(com[:, :3].sum(axis=0))),
                com_energy_closure_mev=float(com[:, 3].sum()-mass),
                imposed_120_degree_constraint=False, imposed_equal_energy_constraint=False,
                rest_mass_convention="caller supplied; use masses exported from the actual simulation")


def propagate_kinematics_uncertainty(kinetic_mev, slopes_uv_wv, covariance,
                                     mass_mev, parent_ground_mass_mev,
                                     method="delta", samples=2000, seed=0):
    """Correlated uncertainty in (T,du/dv,dw/dv) for each of three alphas.

    The supplied 9x9 covariance must already include the uncertainties the
    caller wants to propagate. This does not fabricate straggling or calibration
    errors. Downstream +v tracks only; backward tracks need a different chart.
    """
    energies = _finite(kinetic_mev, (3,), "kinetic energies")
    slopes = _finite(slopes_uv_wv, (3, 2), "physical slopes")
    c = _covariance(covariance, 9)
    x = np.column_stack([energies, slopes]).reshape(-1)

    def evaluate(p):
        p = np.asarray(p).reshape(3, 3)
        directions = np.column_stack([p[:, 1], np.ones(3), p[:, 2]])
        result = kinematics(four_vectors(p[:, 0], directions, mass_mev), parent_ground_mass_mev)
        return np.r_[result["invariant_mass_mev"], result["excitation_mev"],
                     result["com_opening_angles_deg"], result["dalitz_xy"]]

    center = evaluate(x)
    if not np.isfinite(center).all():
        raise ValueError("Undefined central COM angles or Dalitz coordinates")
    labels = ["invariant_mass_mev", "excitation_mev", "angle_01_deg", "angle_02_deg",
              "angle_12_deg", "dalitz_x", "dalitz_y"]
    if method == "delta":
        jac = np.zeros((7, 9))
        for j in range(9):
            h = max(1e-6, abs(x[j])*1e-5)
            plus, minus = x.copy(), x.copy()
            plus[j] += h
            minus[j] -= h
            if j % 3 == 0 and minus[j] < 0:
                jac[:, j] = (evaluate(plus)-center)/h
            else:
                jac[:, j] = (evaluate(plus)-evaluate(minus))/(2*h)
        out = jac @ c @ jac.T
        extra = dict(jacobian=jac)
    elif method == "mc":
        if samples < 2:
            raise ValueError("Monte Carlo propagation needs >=2 samples")
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(x, c, size=int(samples))
        draws = draws[np.all(draws[:, [0, 3, 6]] >= 0, axis=1)]
        if len(draws) < 2:
            raise ValueError("Fewer than two physically valid uncertainty draws")
        values = np.stack([evaluate(draw) for draw in draws])
        out = np.cov(values, rowvar=False, ddof=1)
        extra = dict(valid_draw_fraction=len(draws)/float(samples),
                     propagated_mean=values.mean(axis=0), samples=len(draws))
    else:
        raise ValueError("method must be delta or mc")
    return dict(status="ok", labels=labels, central_values=center,
                covariance=(out+out.T)*.5, sigma=np.sqrt(np.maximum(np.diag(out), 0)),
                method=method, uncertainty_scope="supplied covariance only", **extra)
