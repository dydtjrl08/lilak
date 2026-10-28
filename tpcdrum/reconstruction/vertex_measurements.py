"""First-observed local line states from ORIGINAL measured layer clouds.

Used to seed vertex-region searches, not to treat fitted vertices as new hits.
Covariances are conditional on the supplied measurement/spread proxy and do not
include unmeasured drift common offsets or charge-assignment systematics.
"""
import numpy as np


def local_state(points,covariances,first_layers=4):
    points=np.asarray(points,float);covariances=np.asarray(covariances,float)
    if points.ndim!=2 or points.shape[1] not in (2,3):raise ValueError('Need uv or uvw points')
    n,d=points.shape
    if n<3 or covariances.shape!=(n,d,d):raise ValueError('Need >=3 original measured layers and aligned covariance')
    if not np.isfinite(points).all() or not np.isfinite(covariances).all():raise ValueError('Nonfinite local observations')
    order=np.argsort(points[:,1],kind='stable')[:int(first_layers)]
    p=points[order];c=covariances[order]
    if len(p)<3 or np.any(np.diff(p[:,1])<=0):raise ValueError('Need distinct increasing layer depths')
    if any(not np.allclose(x,x.T,atol=1e-10) or np.linalg.eigvalsh(x).min()<-1e-10 for x in c):
        raise ValueError('Measurement covariance must be symmetric positive semidefinite')
    transverse=[0] if d==2 else [0,2]
    m=len(transverse);vref=float(p[0,1]);dv=p[:,1]-vref
    linear=np.c_[np.ones(len(p)),dv]
    initial=np.linalg.lstsq(linear,p[:,transverse],rcond=None)[0]
    state=np.r_[initial[0],initial[1]]
    for _ in range(8):
        normal=np.zeros((2*m,2*m));rhs=np.zeros(2*m)
        slope=state[m:]
        for point,cov,delta in zip(p,c,dv):
            projection=np.zeros((m,d));projection[:,1]=-slope
            projection[np.arange(m),transverse]=1.
            noise=projection@cov@projection.T
            if np.linalg.eigvalsh(noise).min()<=0:
                raise ValueError('Transverse uncertainty must be positive; no precision floor invented')
            design=np.c_[np.eye(m),delta*np.eye(m)]
            normal+=design.T@np.linalg.solve(noise,design)
            rhs+=design.T@np.linalg.solve(noise,point[transverse])
        if np.linalg.matrix_rank(normal)<2*m:raise ValueError('Singular first-layer direction support')
        updated=np.linalg.solve(normal,rhs)
        if np.max(np.abs(updated-state))<1e-10:
            state=updated;break
        state=updated
    origin=np.array([state[0],vref] if d==2 else [state[0],vref,state[1]])
    direction=np.array([state[m],1.] if d==2 else [state[m],1.,state[m+1]])
    direction/=np.linalg.norm(direction)
    return dict(origin=origin,direction=direction,state=state,covariance=np.linalg.inv(normal),
                reference_v=vref,layers_used=order,
                covariance_scope='supplied per-layer measurement covariance; no invented common systematic or assignment precision')


def observed_tracks(record,config,dimension):
    """Extract UV diagnostics or calibrated UVW observations, never previous fits."""
    from .algorithms.track_fit import _drift,_waveform_measurements,_point_measurements
    if dimension not in (2,3):raise ValueError('dimension must be 2 or 3')
    calibration,error=_drift(config.get('drift'),record)
    physical=calibration is not None and error is None
    if dimension==3 and not physical:raise ValueError('Physical 3D vertex requires explicit charge-mean drift calibration')
    key='fit_measurement_points_uvw_mm' if dimension==3 else 'fit_measurement_points_uv_mm'
    covkey='fit_measurement_covariance_mm2' if dimension==3 else 'fit_measurement_covariance_uv_mm2'
    maskkey='fit_valid_layer' if dimension==3 else 'fit_observed_layer_mask'
    if key in record and covkey in record and maskkey in record:
        points=np.asarray(record[key],float);covariance=np.asarray(record[covkey],float)
        mask=np.asarray(record[maskkey],bool)
    else:
        fitting=dict(config.get('fitting',{}))
        if 'point_fractions' in record:
            points,covariance,_,mask,point_physical=_point_measurements(record,calibration,fitting)
            if dimension==3 and not point_physical:raise ValueError('Space-point coordinates are not calibrated physical 3D')
        else:points,covariance,_,mask=_waveform_measurements(record,fitting,calibration)
        if dimension==2:points=points[:,:,:2];covariance=covariance[:,:,:2,:2]
    k=len(points)
    if points.shape!=(k,12,dimension) or covariance.shape!=(k,12,dimension,dimension) or mask.shape!=(k,12):
        raise ValueError('Embedded original measured track arrays have inconsistent shapes')
    mask=mask&np.isfinite(points).all(-1)&np.isfinite(covariance).all((-1,-2))
    tracks=[];covariances=[];layers=[];slots=[]
    minimum=int(config.get('vertex',{}).get('joint',{}).get('minimum_layers',3))
    for slot in range(k):
        ids=np.flatnonzero(mask[slot])
        if len(ids)<minimum:continue
        tracks.append(points[slot,ids].copy());covariances.append(covariance[slot,ids].copy())
        layers.append(ids);slots.append(slot)
    return tracks,covariances,layers,np.asarray(slots,dtype=np.int64)
