"""Explicit space-point route; full pad-waveform finding is a separate method."""
import numpy as np
from .preprocessing import PAD_U_MM


def find_spacepoint_event(record, config):
    from .algorithms.geometry_core import infer
    points=np.asarray(record['points_uvw_mm'],dtype=float)
    charge=np.asarray(record['point_charge'],dtype=float)
    physical=bool(record['physical_3d'])
    time_scale=float(config.get('feature_time_scale',.36))
    metric_points=points.copy()
    if not physical:
        metric_points[:,2]=np.asarray(record['point_tb'])*time_scale
    k=int(config.get('k',3));method=config.get('method','hough')
    if method not in ('hough','ransac'):
        raise ValueError('Space-point route currently supports hough or ransac proposals')
    result=infer(metric_points,charge,k,method=method,seed=int(config.get('seed',20261027)),
                 config=config.get('geometry',{'scales_mm':(2.,6.,4.)}))
    fractions=result['fractions']
    q=np.asarray(record['positive_waveform'])
    # Map an observed voxel to its nearest measured pulse island on that pad.
    # This explicitly loses within-island separation, unlike direct waveform fits.
    native=np.zeros((k+1,12,64,512),dtype=np.float32);native[-1]=1.
    layers=np.asarray(record['point_layer'],dtype=int)
    pads=np.asarray(record['point_pad'],dtype=int)
    times=np.asarray(record['point_tb'])
    for layer in range(12):
        for pad in range(64):
            indices=np.flatnonzero((layers==layer)&(pads==pad))
            if not len(indices):continue
            nearest=indices[np.argmin(abs(np.arange(512)[:,None]-times[indices][None,:]),axis=1)]
            native[:,layer,pad]=fractions[nearest].T
    components=native*q[None]
    return {'components':components[:-1], 'background':components[-1],
            'fractions':native, 'point_fractions':fractions,
            'feature_lines':result['lines'],
            'diagnostics':{'representation':'space_points','method':method,
                'physical_3d':physical,'line_coordinates':'physical_uvw_mm' if physical else 'u,v,unphysical_time_feature',
                'found_count':result['found_count'],'ambiguity_assessed':False,
                'voxel_assignment':'nearest observed pulse island on same native pad',
                'within_point_overlap_unresolved':True,'physics_accuracy_evaluated':False}}
