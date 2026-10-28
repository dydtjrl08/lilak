"""Observed pedestal/noise and explicit time-to-distance calibration."""
from __future__ import annotations
import numpy as np

PAD_U_MM = np.r_[np.arange(27)-36.5, np.arange(10)*2.-9., np.arange(27)+10.5]
PAD_WIDTH_MM = np.r_[np.full(27, .9), np.full(10, 1.9), np.full(27, .9)]
LAYER_V_MM = np.arange(12)*12.


def preprocess(raw, config=None):
    config = dict(config or {})
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (12,64,512) or not np.isfinite(raw).all():
        raise ValueError('Expected finite observed waveform (12,64,512)')
    first, stop = config.get('pedestal_tb_window', [0,30])
    if not 0 <= first < stop <= 512:
        raise ValueError('Invalid observed pedestal window')
    quiet = raw[..., first:stop]
    baseline = np.median(quiet, axis=-1)
    sigma = 1.4826*np.median(np.abs(quiet-baseline[...,None]), axis=-1)
    signal = raw-baseline[...,None]
    positive = np.maximum(signal, 0.)
    threshold = float(config.get('activity_threshold_adc', 40.))
    peaks = positive.max(-1)
    suspicious = np.count_nonzero(quiet > baseline[...,None]+max(threshold,1.))
    if config.get('require_zero_front', False) and np.count_nonzero(quiet):
        raise ValueError('Noise-free-MC front-window assumption failed')
    return {'positive_waveform': positive, 'baseline': baseline.astype(np.float32),
            'noise_sigma': sigma.astype(np.float32), 'native_peak': peaks,
            'active_pad': (peaks > threshold).astype(np.uint8)}, {
            'pedestal_method': 'per-pad observed median', 'pedestal_tb_window': [first,stop],
            'front_nonzero': int(np.count_nonzero(quiet)),
            'front_samples_above_signal_threshold': int(suspicious),
            'pedestal_window_contains_possible_signal': bool(suspicious),
            'baseline_is_measured_from_current_event': True,
            'positive_clipping': 'negative baseline residual retained only in source raw; positive ADC used downstream'}


def drift_coordinate(tb, calibration):
    """Charge-mean timing calibration only. Return physical w[mm], never guessed TB scale."""
    calibration = dict(calibration or {})
    names = ['velocity_mm_per_ns','t0_tb','reference_w_mm','sign','tb_ns']
    if calibration.get('calibrated_for_charge_mean') is not True:
        raise ValueError('Missing validated charge-mean timing calibration')
    if any(n not in calibration for n in names):
        raise ValueError('Drift calibration requires '+', '.join(names))
    velocity, t0, reference, sign, step = (float(calibration[n]) for n in names)
    if not np.isfinite([velocity,t0,reference,sign,step]).all() or velocity <= 0 or step <= 0 or sign not in (-1.,1.):
        raise ValueError('Invalid drift calibration units/sign/velocity')
    return reference+sign*(np.asarray(tb)-t0)*step*velocity


def space_points(waveform, config=None, calibration=None):
    """Per-native-pad pulse islands -> measured centroids + sampling covariance.

    This route is explicitly a lossy space-point representation. Covariance
    includes pad quantization and measured pulse-time spread, not a verified
    detector response model. physical_3d=false when calibration is unavailable.
    """
    from scipy.ndimage import label
    config = dict(config or {})
    waveform = np.asarray(waveform, dtype=float)
    if waveform.shape != (12,64,512) or not np.isfinite(waveform).all() or np.any(waveform < 0):
        raise ValueError('Space points require nonnegative prepared ADC')
    threshold = float(config.get('activity_threshold_adc',40.))
    relative = float(config.get('island_fraction_of_peak', .02))
    records = []
    physical = bool((calibration or {}).get('calibrated_for_charge_mean') is True)
    for layer in range(12):
        for pad in range(64):
            values = waveform[layer,pad]
            if values.max() <= threshold:
                continue
            components, count = label(values > max(relative*values.max(), 0.))
            for index in range(1,count+1):
                ids = np.flatnonzero(components == index)
                q = values[ids]
                total = q.sum()
                if not len(ids) or q.max() <= threshold or total <= 0:
                    continue
                t = float(ids @ q/total)
                variance = float(((ids-t)**2) @ q/total)+1./12.
                records.append([layer,pad,PAD_U_MM[pad],LAYER_V_MM[layer],t,total,variance])
    r = np.asarray(records, dtype=float).reshape(-1,7)
    coords = np.column_stack((r[:,2],r[:,3],np.full(len(r),np.nan)))
    covariance = np.zeros((len(r),3,3))
    if len(r):
        covariance[:,0,0] = PAD_WIDTH_MM[r[:,1].astype(int)]**2/12.
        covariance[:,1,1] = 12.**2/12.
        covariance[:,2,2] = np.nan
        if physical:
            coords[:,2] = drift_coordinate(r[:,4],calibration)
            scale = float(calibration['tb_ns'])*float(calibration['velocity_mm_per_ns'])
            covariance[:,2,2] = r[:,6]*scale**2
    return {'points_uvw_mm': coords, 'point_covariance_mm2': covariance,
            'point_charge': r[:,5], 'point_tb': r[:,4],
            'point_layer': r[:,0].astype(np.int64),'point_pad':r[:,1].astype(np.int64),
            'physical_3d': np.asarray(physical, dtype=np.uint8)}, {
            'representation':'per-pad contiguous pulse-island centroids',
            'physical_3d':physical, 'covariance_model':'sampling/pulse-spread proxy, requires data validation',
            'point_count':len(r), 'within_pad_overlap_can_merge':True}
